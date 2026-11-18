#!/usr/bin/env python3
"""Passo 9 — probes com contexto espacial (E1c-E4c) e predição nos tiles confirmatórios.

Mesmo probe do passo 4 (RF + linear, hiperparâmetros fixos, mesmos pixels de treino), sempre
treinado nos 43 tiles de TREINO originais. Nada é ajustado nos tiles confirmatórios.

  --scope original  E1c-E4c nos tiles de validação/teste originais -> data/ctx/probabilities/
  --scope confirm   E1-E4 e E1c-E4c nos tiles confirmatórios     -> data/confirm/probabilities/
                    (E1-E4 são retreinados; a predição de um tile de validação original é
                    comparada com data/probabilities para provar que o modelo é o mesmo)

Checkpoint: experimentos com todas as saídas presentes são pulados (use --force para refazer).
Protocolo: docs/protocol/confirmatory.md. Uso:
  python code/09_context_probe.py --scope original [--exps E4c] [--force]
"""
import argparse
import json
import time

import numpy as np
import rasterio

from paths import (
    ORIGINAL,
    exp_name,
    atomic_output,
    atomic_write_text,
    check_disk,
    layout,
    scoped_metrics_dir,
    scoped_prob_path,
)
from probing import (
    CTX_EXP_SOURCES,
    EXP_SOURCES,
    PROBE_VARIANTS,
    load_stack,
    prob_profile,
    train_experiment,
)

DEFAULT_EXPS = {"original": list(CTX_EXP_SOURCES),
                "confirm": list(EXP_SOURCES) + list(CTX_EXP_SOURCES)}


def read_split(scope: str) -> dict[str, str]:
    return json.loads(layout(scope).split.read_text())


def read_label(tid: str) -> np.ndarray:
    with rasterio.open(ORIGINAL.label(tid)) as ds:
        return ds.read(1)


def outputs(scope: str, tid: str, exp: str, variants):
    return [scoped_prob_path(scope, tid, exp_name(exp, v)) for v in variants]


def write_probs(scope: str, tid: str, exp: str, probs: dict[str, np.ndarray]) -> None:
    lay = layout(scope)
    profile = prob_profile(lay.raster(tid, "s2"), *next(iter(probs.values())).shape)
    for variant, prob in probs.items():
        out = scoped_prob_path(scope, tid, exp_name(exp, variant))
        out.parent.mkdir(parents=True, exist_ok=True)
        with atomic_output(out) as tmp:
            with rasterio.open(tmp, "w", **profile) as dst:
                dst.write(prob, 1)


def reproduction_check(exp: str, probes, tid: str) -> dict:
    """Retreino de E1-E4 deve reproduzir data/probabilities (mesmo modelo, mesma seed)."""
    pred = probes.predict(load_stack(ORIGINAL, tid, exp))
    out = {}
    for variant, prob in pred.items():
        stored_file = scoped_prob_path("original", tid, exp_name(exp, variant))
        if not stored_file.exists():
            continue
        with rasterio.open(stored_file) as ds:
            stored = ds.read(1)
        out[variant] = float(np.nanmax(np.abs(prob - stored)))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scope", choices=("original", "confirm"), required=True)
    ap.add_argument("--exps", nargs="+", default=None)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--n-jobs", type=int, default=-1)
    ap.add_argument("--variants", nargs="+", default=list(PROBE_VARIANTS),
                    choices=("rf", "lin", "rf8"), help="rf8 = sensibilidade max_features=8")
    args = ap.parse_args()
    exps = args.exps or DEFAULT_EXPS[args.scope]
    if args.scope == "original" and set(exps) & set(EXP_SOURCES) and set(args.variants) & {"rf", "lin"}:
        # E1-E4 originais vivem em data/probabilities (passo 4) e não são sobrescritos aqui
        raise SystemExit("no escopo original só E1c-E4c; E1-E4 vêm do passo 4")

    orig_split = read_split("original")
    train_ids = [t for t, p in orig_split.items() if p == "train"]
    if args.scope == "original":
        predict_ids = [t for t, p in orig_split.items() if p in ("val", "test")]
    else:
        predict_ids = [t for t, p in read_split("confirm").items() if p == "test"]
    if not predict_ids:
        raise SystemExit(f"nenhum tile para predizer no escopo {args.scope}")
    labels_by_tile = {tid: read_label(tid) for tid in train_ids}
    val_probe = next(t for t, p in orig_split.items() if p == "val")
    repro = {}

    print(f"· escopo={args.scope} · treino={len(train_ids)} tiles · predição={len(predict_ids)} "
          f"tiles · exps={exps} · livre={check_disk():.0f} GB", flush=True)
    t_all = time.time()
    for i, exp in enumerate(exps, start=1):
        pending = [t for t in predict_ids
                   if args.force or not all(p.exists() for p in outputs(args.scope, t, exp, args.variants))]
        check_repro = args.scope == "confirm"
        if not pending and not check_repro:
            print(f"· [{i}/{len(exps)}] {exp}: saídas presentes (pulado)", flush=True)
            continue
        check_disk()
        t0 = time.time()
        probes, shape_ = train_experiment(ORIGINAL, exp, train_ids, labels_by_tile, args.n_jobs,
                                          tuple(args.variants))
        print(f"· [{i}/{len(exps)}] {exp}: treinado ({shape_[0]} px, {shape_[1]} features) "
              f"em {time.time() - t0:.0f}s", flush=True)
        if check_repro:
            repro[exp] = reproduction_check(exp, probes, val_probe)
            print(f"  {exp}: reprodução vs probabilidades originais ({val_probe}) "
                  f"max|Δ|={repro[exp]}", flush=True)
            if not repro[exp] or max(repro[exp].values()) > 1e-5:
                raise SystemExit(f"{exp}: retreino não reproduz o modelo original ({repro[exp]})")
        for j, tid in enumerate(pending, start=1):
            write_probs(args.scope, tid, exp, probes.predict(load_stack(layout(args.scope), tid, exp)))
            if j % 8 == 0 or j == len(pending):
                el = time.time() - t_all
                print(f"  {exp}: {j}/{len(pending)} tiles · {el / 60:.1f} min decorridos", flush=True)
        print(f"  {exp}: pronto em {time.time() - t0:.0f}s · livre={check_disk():.0f} GB", flush=True)

    if repro:
        out = scoped_metrics_dir(args.scope) / "reproduction_check.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_text(out, json.dumps({"tile": val_probe, "max_abs_diff": repro}, indent=1))
        worst = max(v for d in repro.values() for v in d.values())
        if worst > 1e-5:
            raise SystemExit(f"retreino NÃO reproduz data/probabilities (max|Δ|={worst:.2e})")
        print(f"✓ retreino reproduz data/probabilities (max|Δ|={worst:.1e})")


if __name__ == "__main__":
    main()
