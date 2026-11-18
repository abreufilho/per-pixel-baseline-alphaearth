#!/usr/bin/env python3
"""Passo 5 — objetos por watershed sobre P(fronteira) + métricas de objeto no TESTE.

Desenho v2: o único knob (limiar t dos marcadores do watershed) é escolhido nas células de
VALIDAÇÃO (maximiza mean_iou média) e congelado; as métricas reportadas vêm só das células de
TESTE. Harness de métricas mantido da v1: por tile, só talhões inteiramente no interior
(borda erodida em BUFFER_M — clipar talhões parciais criaria fronteira artificial coincidente
com a borda dos candidatos), perímetro do tile removido das fronteiras de candidato.

Métricas por talhão de referência r vs melhor candidato c* (maior interseção) — Clinton et al.
(2010): IoU, over_seg, under_seg (ambos só sobre talhões pareados, ia>0), erro de área (só
detectados IoU≥0.5). A referência avaliada exclui slivers sub-pixel (<1 px = 100 m²;
fragmentos da delineação semi-automática, indetectáveis com MIN_SEG_PX=5). Boundary F1 (tol 1 px,
precision restrita ao footprint do café — referência esparsa). Extras v2: AP de fronteira
pixel-a-pixel (livre de limiar, dentro do footprint) e Wilcoxon pareado por tile de teste,
com correção de Holm-Bonferroni dentro de cada contraste pareado (coluna p_holm).

Saídas (não-versionadas): data/metrics/{rf,lin}/metrics_E1_E4*_per_tile.csv (teste),
metrics_E1_E4*.csv (média por experimento), metrics_E1_E4*_tests.csv (Wilcoxon),
thresholds*.json (t* por experimento). Com --variant lin, avalia prob_*_lin.tif
(probe LogisticRegression do passo 4) com o MESMO protocolo, sufixando as saídas com _lin.

Uso: python code/05_validate.py [--variant rf|lin]
"""
import argparse
import json
import warnings
from concurrent.futures import ProcessPoolExecutor

import geopandas as gpd
import numpy as np
import pandas as pd

import objects
from paths import (
    REF,
    SPLIT,
    TILES,
    label_path,
    metrics_dir,
    metrics_path,
    prob_path_for_experiment,
    staged_outputs,
    thresholds_path,
)
from stats import TEST_METRICS, paired_tests

warnings.filterwarnings(
    "ignore",
    message="Setting the shape on a NumPy array has been deprecated.*",
    category=DeprecationWarning,
)

EXPERIMENTS = ("E1", "E2", "E3", "E4")  # sobrescrito em main() com --variant lin
T_GRID = [round(t, 2) for t in np.arange(0.10, 0.65, 0.05)]


def load_ref() -> gpd.GeoDataFrame:
    return objects.load_ref(REF)


def eval_tile(tid: str, exp: str, t: float, ref_all: gpd.GeoDataFrame) -> dict | None:
    return objects.eval_prob_file(prob_path_for_experiment(tid, exp), t, ref_all,
                                  objects.watershed_decoder)


REF_ALL: gpd.GeoDataFrame | None = None  # carregada no main (fork) ou via _init_worker (spawn)


def _init_worker(variant: str) -> None:
    """Estado global dos workers: sob fork herda do pai; sob spawn precisa recriar."""
    global REF_ALL, EXPERIMENTS
    suf = "" if variant == "rf" else "_lin"
    EXPERIMENTS = tuple(f"{e}{suf}" for e in ("E1", "E2", "E3", "E4"))
    if REF_ALL is None:
        REF_ALL = load_ref()


def _sweep_task(args: tuple) -> tuple:
    exp, t, val_ids = args
    ious = [m["mean_iou"] for tid in val_ids
            if (m := eval_tile(tid, exp, t, REF_ALL)) and "mean_iou" in m]
    return exp, t, (float(np.mean(ious)) if ious else float("nan"))


def _test_task(args: tuple) -> list[dict]:
    tid, thresholds, props = args
    rows = []
    for exp in EXPERIMENTS:
        m = eval_tile(tid, exp, thresholds[exp], REF_ALL)
        if m is None:
            continue
        m["boundary_ap"] = round(boundary_ap(tid, exp), 3)
        p = props[tid]
        rows.append({"tile": tid, "experiment": exp, "region": p["region"],
                     "size_class": p["size_class"], "dens_class": p["dens_class"], **m})
    print(f"· {tid} ok", flush=True)
    return rows


def boundary_ap(tid: str, exp: str) -> float:
    return objects.boundary_ap(prob_path_for_experiment(tid, exp), label_path(tid))


def main() -> None:
    global REF_ALL, EXPERIMENTS
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=("rf", "lin"), default="rf")
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()
    suf = "" if args.variant == "rf" else "_lin"
    EXPERIMENTS = tuple(f"{e}{suf}" for e in ("E1", "E2", "E3", "E4"))
    REF_ALL = load_ref()
    split = json.loads(SPLIT.read_text())
    props = {f["properties"]["tile_id"]: f["properties"]
             for f in json.loads(TILES.read_text())["features"]}
    val_ids = [t for t, p in split.items() if p == "val"]
    test_ids = [t for t, p in split.items() if p == "test"]
    if not val_ids:
        raise SystemExit("split sem tiles de validação")
    if not test_ids:
        raise SystemExit("split sem tiles de teste")
    missing = []
    for tid in val_ids + test_ids:
        for exp in EXPERIMENTS:
            f = prob_path_for_experiment(tid, exp)
            if not f.exists():
                missing.append(str(f))
    for tid in test_ids:
        f = label_path(tid)
        if not f.exists():
            missing.append(str(f))
    if missing:
        preview = "\n  ".join(missing[:40])
        extra = f"\n  ... +{len(missing) - 40}" if len(missing) > 40 else ""
        raise SystemExit(f"artefatos ausentes:\n  {preview}{extra}")

    tasks = [(exp, t, val_ids) for exp in EXPERIMENTS for t in T_GRID]
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_init_worker,
                             initargs=(args.variant,)) as pool:
        results = list(pool.map(_sweep_task, tasks))
    thresholds = {}
    for exp in EXPERIMENTS:
        cand = [(t, iou) for e, t, iou in results if e == exp and not np.isnan(iou)]
        if not cand:
            raise SystemExit(f"{exp}: nenhum resultado válido na validação")
        best_t, best_iou = max(cand, key=lambda x: x[1])
        thresholds[exp] = best_t
        print(f"· {exp}: t*={best_t} (mean_iou val={best_iou:.3f}, {len(val_ids)} tiles)", flush=True)
    metrics_dir(args.variant).mkdir(parents=True, exist_ok=True)

    with ProcessPoolExecutor(max_workers=args.workers, initializer=_init_worker,
                             initargs=(args.variant,)) as pool:
        per_tile_rows = list(pool.map(_test_task, [(tid, thresholds, props) for tid in test_ids]))
    rows = [r for rs in per_tile_rows for r in rs]

    if not rows:
        print("sem tiles de teste avaliáveis (rode os passos 3-4)")
        return

    per_tile = pd.DataFrame(rows)

    metric_cols = ["object_recall", "mean_iou", "over_seg", "under_seg",
                   "rel_area_error_detected", "boundary_precision", "boundary_recall",
                   "boundary_f1", "boundary_ap"]
    agg = per_tile.groupby("experiment")[metric_cols].mean().round(3)

    print(f"\n=== Média por experimento ({per_tile.tile.nunique()} tiles de TESTE) ===")
    print(agg.to_string())
    for strat in ("region", "size_class"):
        print(f"\n=== Por {strat} × experimento (mean_iou / boundary_f1 / boundary_ap) ===")
        s = (per_tile.groupby([strat, "experiment"])[["mean_iou", "boundary_f1", "boundary_ap"]]
             .mean().round(3))
        print(s.to_string())

    pairs = tuple((f"{a}{suf}", f"{b}{suf}")
                  for a, b in (("E3", "E1"), ("E4", "E2"), ("E4", "E1"), ("E2", "E1")))
    tests_df = paired_tests(per_tile, pairs, TEST_METRICS)
    with staged_outputs() as stage:
        stage(thresholds_path(args.variant)).write_text(json.dumps(thresholds, indent=1))
        per_tile.to_csv(stage(metrics_path("per_tile", args.variant)), index=False)
        agg.to_csv(stage(metrics_path("summary", args.variant)))
        tests_df.to_csv(stage(metrics_path("tests", args.variant)), index=False)
    print("\n=== Wilcoxon pareado por tile de teste (median_diff = primeiro - segundo) ===")
    print(tests_df.to_string(index=False))


if __name__ == "__main__":
    main()
