#!/usr/bin/env python3
"""Passo 13 — checagens pós-hoc (não confirmatórias) pedidas pela revisão do manuscrito.

Todas com o probe RF principal, treinado nos 43 tiles de treino originais, avaliadas por
Boundary AP nos tiles de teste originais e confirmatórios (sem limiar, sem decodificador):

  seed      E1c e E3c com outras sementes (amostra de pixels + floresta): 1 e 2
  pca16     E3c com dimensionalidade igual a E1c: PCA do embedding (16 componentes, ajustada
            nos pixels de treino) + os mesmos 4 filtros -> 80 features
  ablation  E1 e E3 com subconjuntos dos filtros: bruto+média 3x3; bruto+média+DPs 3x3/5x5

Nada é gravado em data/probabilities; saídas em data/confirm/metrics/posthoc/.
Uso: python code/13_posthoc_checks.py [--checks seed pca16 ablation]
"""
import argparse
import json
import time
from collections.abc import Callable

import numpy as np
import pandas as pd
import rasterio
from sklearn.decomposition import PCA
from sklearn.metrics import average_precision_score

from paths import (
    CONFIRM_ROOT,
    ORIGINAL,
    atomic_output,
    atomic_write_text,
    check_disk,
    layout,
    scoped_metrics_dir,
)
from probing import SEED, context_features, make_rf, sample_training
from stats import hl_interval

OUT_DIR = CONFIRM_ROOT / "metrics" / "posthoc"
SEEDS = (1, 2)
PCA_COMPONENTS = 16
ABLATIONS = {"m3": (0,), "m3sd": (0, 1, 2)}  # índices em context_features: mean3, std3, std5, grad

StackFn = Callable[[object, str], np.ndarray]


def read_raw(lay, tid: str, source: str) -> np.ndarray:
    with rasterio.open(lay.raster(tid, source)) as ds:
        return ds.read().astype("float32")


def filtered(data: np.ndarray, keep=(0, 1, 2, 3)) -> np.ndarray:
    layers = []
    for band in data:
        feats = context_features(band)
        layers.append(np.nan_to_num(band))
        layers.extend(np.nan_to_num(feats[k]) for k in keep)
    return np.stack(layers).astype("float32")


def source_stack(source: str, keep=(0, 1, 2, 3)) -> StackFn:
    return lambda lay, tid: filtered(read_raw(lay, tid, source), keep)


def pca_stack(pca: PCA) -> StackFn:
    def build(lay, tid):
        ae = np.nan_to_num(read_raw(lay, tid, "ae"))
        b, h, w = ae.shape
        comps = pca.transform(ae.reshape(b, -1).T).T.reshape(-1, h, w).astype("float32")
        return filtered(comps)
    return build


def read_label(lay, tid: str) -> np.ndarray:
    with rasterio.open(lay.label(tid)) as ds:
        return ds.read(1)


def fit(stack_fn: StackFn, train_ids, labels, seed: int):
    rng = np.random.default_rng(seed)
    X, y = [], []
    for tid in train_ids:
        xt, yt = sample_training(stack_fn(ORIGINAL, tid), labels[tid], rng)
        if xt is not None:
            X.append(xt)
            y.append(yt)
    X, y = np.concatenate(X), np.concatenate(y)
    return make_rf(-1, seed=seed).fit(X, y), X.shape[1]


def tile_ap(model, stack_fn: StackFn, lay, tid: str) -> float:
    stack = stack_fn(lay, tid)
    lab = read_label(lay, tid)
    h, w = min(stack.shape[1], lab.shape[0]), min(stack.shape[2], lab.shape[1])
    stack, lab = stack[:, :h, :w], lab[:h, :w]
    mask = lab > 0
    if mask.sum() == 0 or (lab[mask] == 2).all() or (lab[mask] == 1).all():
        return float("nan")
    prob = model.predict_proba(stack[:, mask].T)[:, 1]
    return float(average_precision_score(lab[mask] == 2, prob))


def fit_pca(train_ids, labels) -> PCA:
    rng = np.random.default_rng(SEED)
    X = []
    for tid in train_ids:
        xt, _ = sample_training(np.nan_to_num(read_raw(ORIGINAL, tid, "ae")), labels[tid], rng)
        if xt is not None:
            X.append(xt)
    return PCA(n_components=PCA_COMPONENTS, random_state=SEED).fit(np.concatenate(X))


def configs(checks, train_ids, labels) -> list[tuple[str, StackFn, int]]:
    out = []
    if "seed" in checks:
        for s in SEEDS:
            out += [(f"E1c_seed{s}", source_stack("s2"), s), (f"E3c_seed{s}", source_stack("ae"), s)]
    if "pca16" in checks:
        pca = fit_pca(train_ids, labels)
        print(f"· PCA {PCA_COMPONENTS} comps: variância explicada "
              f"{pca.explained_variance_ratio_.sum():.3f}", flush=True)
        out.append(("E3c_pca16", pca_stack(pca), SEED))
    if "ablation" in checks:
        for tag, keep in ABLATIONS.items():
            out += [(f"E1_{tag}", source_stack("s2", keep), SEED),
                    (f"E3_{tag}", source_stack("ae", keep), SEED)]
    return out


def summarize(ap: pd.DataFrame) -> dict:
    contrasts = {"E3c_seed1-E1c_seed1": ("E3c_seed1", "E1c_seed1"),
                 "E3c_seed2-E1c_seed2": ("E3c_seed2", "E1c_seed2"),
                 "E3c_pca16-E1c": ("E3c_pca16", "E1c"),
                 "E3_m3-E1_m3": ("E3_m3", "E1_m3"),
                 "E3_m3sd-E1_m3sd": ("E3_m3sd", "E1_m3sd"),
                 "E1c-E3": ("E1c", "E3")}
    out = {}
    for scope, df in ap.groupby("scope"):
        wide = df.pivot(index="tile_id", columns="exp", values="ap")
        res = {"mean_ap": {k: round(float(v), 4) for k, v in wide.mean().items()}}
        for name, (a, b) in contrasts.items():
            if a in wide and b in wide:
                d = (wide[a] - wide[b]).dropna()
                est, lo, hi = hl_interval(d.to_numpy(), 0.95)
                res[name] = {"n": int(len(d)), "hl": round(est, 4), "ci95": [round(lo, 4), round(hi, 4)],
                             "n_negative": int((d < 0).sum())}
        out[scope] = res
    return out


def main() -> None:
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("--checks", nargs="+", default=["seed", "pca16", "ablation"],
                     choices=("seed", "pca16", "ablation"))
    args = ap_.parse_args()
    orig_split = json.loads(ORIGINAL.split.read_text())
    train_ids = [t for t, p in orig_split.items() if p == "train"]
    eval_sets = {"original": [t for t, p in orig_split.items() if p == "test"],
                 "confirm": [t for t, p in json.loads(layout("confirm").split.read_text()).items()
                             if p == "test"]}
    labels = {tid: read_label(ORIGINAL, tid) for tid in train_ids}
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"· checks={args.checks} · livre={check_disk():.0f} GB", flush=True)

    base = []
    for scope in eval_sets:
        df = pd.read_csv(scoped_metrics_dir(scope) / "ap_per_tile.csv")
        df = df[df["experiment"].isin(["E1", "E3", "E1c", "E3c"])]
        base.append(df.rename(columns={"tile": "tile_id", "experiment": "exp", "boundary_ap": "ap"})
                    .assign(scope=scope)[["scope", "tile_id", "exp", "ap"]])

    rows = []
    for name, stack_fn, seed in configs(args.checks, train_ids, labels):
        t0 = time.time()
        model, nfeat = fit(stack_fn, train_ids, labels, seed)
        print(f"· {name}: treinado ({nfeat} features) em {time.time() - t0:.0f}s", flush=True)
        for scope, ids in eval_sets.items():
            lay = layout(scope)
            rows += [{"scope": scope, "tile_id": t, "exp": name, "ap": tile_ap(model, stack_fn, lay, t)}
                     for t in ids]
        print(f"  {name}: avaliado em {time.time() - t0:.0f}s", flush=True)
        ap = pd.concat([*base, pd.DataFrame(rows)], ignore_index=True)
        with atomic_output(OUT_DIR / "posthoc_ap_per_tile.csv") as tmp:
            ap.to_csv(tmp, index=False)
        atomic_write_text(OUT_DIR / "posthoc_summary.json", json.dumps(summarize(ap), indent=1))
    print("✓ pós-hoc concluído", flush=True)


if __name__ == "__main__":
    main()
