#!/usr/bin/env python3
"""Passo 10 — objetos, métricas e testes para E1-E4 e E1c-E4c (watershed ou fusão RAG).

O limiar de cada experimento × decodificador × probe é escolhido nos 13 tiles de VALIDAÇÃO
originais (máximo de mean_iou) e congelado; no escopo confirm ele é apenas lido.

  --scope original  teste = 24 tiles de teste originais (pós-hoc) -> data/ctx/metrics/
  --scope confirm   teste = tiles confirmatórios                  -> data/confirm/metrics/

Avaliação estrita (§6): tile avaliável sem objetos conta como recall/IoU/F1 = 0, inclusive na
validação (denominador fixo); AP é calculado para todo tile, independente do decodificador;
precisão total até a inferência. Empate na validação: menor limiar.

Saídas: {decoder}_{variant}_{per_tile,summary,tests}.csv e, no escopo original,
thresholds_{decoder}_{variant}.json + {decoder}_{variant}_sweep.csv. Testes descritivos
(Wilcoxon + Holm por contraste); a análise confirmatória fica em 12_confirmatory.py. Uso:
  python code/10_context_eval.py --scope original --variant rf --decoder rag
"""
import argparse
import hashlib
import json
import multiprocessing as mp
import time
import warnings
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

import objects
from paths import (
    REF,
    atomic_output,
    atomic_write_text,
    check_disk,
    layout,
    scoped_metrics_dir,
    scoped_prob_path,
)
from stats import TEST_METRICS, paired_tests

warnings.filterwarnings("ignore", message="Setting the shape on a NumPy array has been deprecated.*",
                        category=DeprecationWarning)

BASE_EXPS = ("E1", "E2", "E3", "E4", "E1c", "E2c", "E3c", "E4c")
GRIDS = {"ws": [round(t, 2) for t in np.arange(0.10, 0.65, 0.05)],
         "rag": [round(t, 2) for t in np.arange(0.20, 0.85, 0.05)]}
RAW_PAIRS = (("E3", "E1"), ("E4", "E2"))
CTX_PAIRS = (("E3c", "E1c"), ("E3c", "E2c"), ("E4c", "E2c"))
METRIC_COLS = ["object_recall", "mean_iou", "over_seg", "under_seg", "rel_area_error_detected",
               "boundary_precision", "boundary_recall", "boundary_f1", "boundary_ap"]
FORK = mp.get_context("fork")
# docs/protocol/confirmatory.md, Addendum A: limiares congelados antes do download confirmatório
FROZEN_SHA256 = {
    "thresholds_rag_lin.json": "6334f3b0a4b62b4a926e2977c0c0185efb9f67a568d838be5002f5f17cf3237d",
    "thresholds_rag_rf.json": "6453893d9faf5a44b89ce8f8a3e27120213801c5976517745e51919a91e00313",
    "thresholds_ws_lin.json": "941343b4bdfd4e33901e95ac5c681c57cfdc07339aa95a0712a3f47d839cdbfb",
    "thresholds_ws_rf.json": "6d53a493b3b5c678eb3dddc56031d173925f6c11632fe01077434a7426d490d2",
}

REF_ALL = None  # herdado pelos workers via fork


def evaluate(scope: str, decoder: str, tid: str, exp: str, t: float) -> dict | None:
    return objects.eval_prob_file(scoped_prob_path(scope, tid, exp), t, REF_ALL,
                                  objects.DECODERS[decoder], strict=True, digits=None)


def sweep_task(args) -> tuple:
    decoder, exp, t, val_ids = args
    ious = [m["mean_iou"] for tid in val_ids
            if (m := evaluate("original", decoder, tid, exp, t)) and "mean_iou" in m]
    return exp, t, float(np.mean(ious)) if ious else float("nan")


def test_task(args) -> list[dict]:
    scope, decoder, tid, thresholds, props = args
    label = layout(scope).label(tid)
    rows = []
    for exp, t in thresholds.items():
        m = evaluate(scope, decoder, tid, exp, t) or {}
        m["boundary_ap"] = objects.boundary_ap(scoped_prob_path(scope, tid, exp), label)
        p = props[tid]
        rows.append({"tile": tid, "experiment": exp, "region": p["region"],
                     "size_class": p["size_class"], "dens_class": p["dens_class"], **m})
    print(f"  · {tid} ok", flush=True)
    return rows


def select_thresholds(decoder: str, exps: list[str], val_ids: list[str], workers: int):
    tasks = [(decoder, e, t, val_ids) for e in exps for t in GRIDS[decoder]]
    with ProcessPoolExecutor(workers, mp_context=FORK) as pool:
        sweep = list(pool.map(sweep_task, tasks))
    best = {}
    for exp in exps:
        cand = [(t, iou) for e, t, iou in sweep if e == exp and not np.isnan(iou)]
        if not cand:
            raise SystemExit(f"{exp}: nenhum resultado válido na validação")
        top = max(iou for _, iou in cand)
        best[exp] = float(min(t for t, iou in cand if iou == top))
    return best, pd.DataFrame(sweep, columns=["experiment", "threshold", "val_mean_iou"])


def write_csv(df: pd.DataFrame, path, **kw) -> None:
    with atomic_output(path) as tmp:
        df.to_csv(tmp, **kw)


def main() -> None:
    global REF_ALL
    ap = argparse.ArgumentParser()
    ap.add_argument("--scope", choices=("original", "confirm"), required=True)
    ap.add_argument("--variant", choices=("rf", "lin"), default="rf")
    ap.add_argument("--decoder", choices=tuple(objects.DECODERS), default="ws")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--reselect", action="store_true",
                    help="refaz a seleção de limiares no escopo original (sobrescreve os congelados)")
    args = ap.parse_args()
    suf = "" if args.variant == "rf" else "_lin"
    exps = [f"{e}{suf}" for e in BASE_EXPS]
    tag = f"{args.decoder}_{args.variant}"
    out_dir = scoped_metrics_dir(args.scope)
    out_dir.mkdir(parents=True, exist_ok=True)
    thr_path = scoped_metrics_dir("original") / f"thresholds_{tag}.json"
    print(f"· {args.scope} {tag} · livre={check_disk():.0f} GB", flush=True)

    REF_ALL = objects.load_ref(REF)
    lay = layout(args.scope)
    split = json.loads(lay.split.read_text())
    props = {f["properties"]["tile_id"]: f["properties"]
             for f in json.loads(lay.tiles.read_text())["features"]}
    test_ids = [t for t, p in split.items() if p == "test"]
    check_ids = test_ids + ([t for t, p in split.items() if p == "val"]
                            if args.scope == "original" else [])
    missing = [str(scoped_prob_path(args.scope, t, e)) for t in check_ids for e in exps
               if not scoped_prob_path(args.scope, t, e).exists()]
    if missing:
        raise SystemExit(f"{len(missing)} probabilidades ausentes, ex.: {missing[:3]}")

    t0 = time.time()
    if args.scope == "original" and (args.reselect or not thr_path.exists()):
        val_ids = [t for t, p in split.items() if p == "val"]
        thresholds, sweep = select_thresholds(args.decoder, exps, val_ids, args.workers)
        atomic_write_text(thr_path, json.dumps(thresholds, indent=1))
        write_csv(sweep, out_dir / f"{tag}_sweep.csv", index=False)
        print(f"· limiares (validação, {time.time() - t0:.0f}s): {thresholds}", flush=True)
    else:
        if not thr_path.exists():
            raise SystemExit(f"limiares congelados ausentes: {thr_path} (rode o escopo original)")
        digest = hashlib.sha256(thr_path.read_bytes()).hexdigest()
        if digest != FROZEN_SHA256[thr_path.name]:
            raise SystemExit(f"{thr_path.name} difere do congelado no protocolo (sha256 {digest})")
        thresholds = json.loads(thr_path.read_text())
        print(f"· limiares congelados: {thresholds}", flush=True)

    tasks = [(args.scope, args.decoder, tid, thresholds, props) for tid in test_ids]
    with ProcessPoolExecutor(args.workers, mp_context=FORK) as pool:
        per_tile = pd.DataFrame([r for rs in pool.map(test_task, tasks) for r in rs])
    summary = per_tile.groupby("experiment")[METRIC_COLS].mean().round(3)

    e = {b: f"{b}{suf}" for b in BASE_EXPS}
    pairs = [(e[a], e[b]) for a, b in RAW_PAIRS + CTX_PAIRS]
    tests = paired_tests(per_tile, pairs, TEST_METRICS)

    write_csv(per_tile, out_dir / f"{tag}_per_tile.csv", index=False)
    write_csv(summary, out_dir / f"{tag}_summary.csv")
    write_csv(tests, out_dir / f"{tag}_tests.csv", index=False)

    print(f"\n=== {args.scope} {tag}: {per_tile.tile.nunique()} tiles ({time.time() - t0:.0f}s) ===")
    print(summary[["boundary_ap", "mean_iou", "object_recall", "boundary_f1"]].to_string())
    print(tests[tests.metric.isin(["boundary_ap", "mean_iou"])].to_string(index=False))
    print(f"· livre={check_disk():.0f} GB", flush=True)


if __name__ == "__main__":
    main()
