#!/usr/bin/env python3
"""Passo 14 — sensibilidades pós-hoc (não confirmatórias) pedidas pela 2ª rodada de revisão.

Probe RF principal (e linear onde indicado), sempre treinado nos 43 tiles de treino originais
com os mesmos pixels (rng SEED) e hiperparâmetros; avaliado nos 24 tiles de teste originais e
nos 48 confirmatórios. Nada em data/ é sobrescrito; saídas em data/posthoc2/.

  c1  operador multivariado (dissimilaridade cosseno média 3x3/5x5 do vetor de features;
      S2 padronizado com estatísticas de treino, AE unitário) + varredura de escala do
      operador por banda (janelas 3/5/7/9; w=3 reproduz E1c/E3c)
  c2  proxy de falsa fronteira com as probabilidades salvas: P média e fração acima do limiar
      congelado no exterior não rotulado (>= 3 px do footprint) e no interior profundo
      (>= 3 px da faixa de fronteira); AP de fronteira com o exterior como negativo
  c3  variantes do desfecho: AP por fronteira externa vs interna (outro talhão a <= 2 px);
      AP com tolerância de ±1 px (máximo 3x3 de P; negativos a >= 3 px da faixa) e faixa de 2 px
  c4  índices S2 recalculados localmente (EVI recortado em [-1, 1]); E1/E2/E4 e variantes c
      retreinados (rf + lin); mais a versão barata: AP excluindo pixels a <= 2 px de índice zero
  c5  curvas mean IoU x limiar (ws e RAG) nos tiles confirmatórios, limiar oráculo, ablação
      do sigma do RAG (0; 0,5; 1; 2)

Uso: python code/14_posthoc_sensitivity.py --checks c2 c3 [--workers 12]
"""
import argparse
import importlib
import json
import multiprocessing as mp
import time
from concurrent.futures import ProcessPoolExecutor
from functools import partial

import numpy as np
import pandas as pd
import rasterio
from rasterio.features import rasterize
from scipy import ndimage as ndi
from scipy.stats import wilcoxon
from shapely.geometry import box
import geopandas as gpd
from sklearn.metrics import average_precision_score

import objects
from paths import DATA_ROOT, ORIGINAL, REF, atomic_output, atomic_write_text, check_disk, layout, scoped_metrics_dir, scoped_prob_path
from probing import SEED, context_features, fit_probes, sample_training
from stats import MARGIN, diff_in_diff, hl_interval, non_superiority, paired, tost

ph = importlib.import_module("13_posthoc_checks")
ctx_eval = importlib.import_module("10_context_eval")
clean = importlib.import_module("12_confirmatory").clean

OUT_DIR = DATA_ROOT / "posthoc2"
FORK = mp.get_context("fork")
SCOPES = ("original", "confirm")
RF_EXPS = ("E1", "E2", "E3", "E4", "E1c", "E2c", "E3c", "E4c")
S2_INDEX_BANDS = (5, 6, 7, 13, 14, 15)  # NDVI NDWI EVI x {wet, dry}
EXCL_PX = 2           # C2/C4: raio (Chebyshev) de exclusão em torno de pixels com índice zero
FAR_PX = 3            # C2: exterior/interior "profundo"
ADJ_PX = 2            # C3: outro talhão a <= 2 px => fronteira interna
SWEEP_WINDOWS = (3, 5, 7, 9)
SIGMAS = (0.0, 0.5, 1.0, 2.0)
C5_EXPS = ("E1c", "E2c", "E3c", "E4c")
REF_ALL = None


# ---------------------------------------------------------------- comuns
def eval_sets() -> dict[str, list[str]]:
    out = {}
    for scope in SCOPES:
        split = json.loads(layout(scope).split.read_text())
        out[scope] = [t for t, p in split.items() if p == "test"]
    return out


def thresholds(decoder: str) -> dict[str, float]:
    return json.loads((scoped_metrics_dir("original") / f"thresholds_{decoder}_rf.json").read_text())


def read_prob(scope: str, tid: str, exp: str) -> np.ndarray:
    with rasterio.open(scoped_prob_path(scope, tid, exp)) as ds:
        return ds.read(1)


def crop(*arrs):
    h = min(a.shape[-2] for a in arrs)
    w = min(a.shape[-1] for a in arrs)
    return [a[..., :h, :w] for a in arrs]


def safe_ap(y: np.ndarray, score: np.ndarray) -> float:
    if y.size == 0 or y.all() or not y.any():
        return float("nan")
    return float(average_precision_score(y, score))


def zero_index_mask(lay, tid: str) -> np.ndarray:
    """Pixels com algum índice S2 exatamente 0 (fora do AOI: mascarados no GEE, ver c4)."""
    return (ph.read_raw(lay, tid, "s2")[list(S2_INDEX_BANDS)] == 0).any(axis=0)


def near(mask: np.ndarray, px: int) -> np.ndarray:
    return ndi.binary_dilation(mask, structure=np.ones((2 * px + 1, 2 * px + 1), bool))


def r4(x) -> float | None:
    return None if x is None or not np.isfinite(x) else round(float(x), 4)


def contrast(wide: pd.DataFrame, a: str, b: str) -> dict | None:
    """a - b por tile: HL + IC95, média, vitórias (n_neg = b melhor se maior é melhor)."""
    if a not in wide or b not in wide:
        return None
    d = (wide[a] - wide[b]).dropna().round(12)
    if len(d) < 5:
        return None
    hl, lo, hi = hl_interval(d.to_numpy(), 0.95)
    p = 1.0 if d.abs().sum() == 0 else float(wilcoxon(d).pvalue)
    return {"n": int(len(d)), "hl": r4(hl), "ci95": [r4(lo), r4(hi)], "mean": r4(d.mean()),
            "n_neg": int((d < 0).sum()), "n_pos": int((d > 0).sum()), "p_wilcoxon": round(p, 6)}


def contrasts_by_scope(df: pd.DataFrame, value: str, pairs, keys=("scope",)) -> dict:
    out = {}
    for key, g in df.groupby(list(keys)):
        wide = g.pivot_table(index="tile", columns="experiment", values=value)
        res = {"mean": {k: r4(v) for k, v in wide.mean().items()}}
        for a, b in pairs:
            c = contrast(wide, a, b)
            if c:
                res[f"{a}-{b}"] = c
        out["|".join(map(str, key if isinstance(key, tuple) else (key,)))] = res
    return out


def base_ap() -> pd.DataFrame:
    rows = []
    for scope in SCOPES:
        df = pd.read_csv(scoped_metrics_dir(scope) / "ap_per_tile.csv")
        rows.append(df.assign(scope=scope)[["scope", "tile", "experiment", "boundary_ap"]])
    return pd.concat(rows, ignore_index=True)


def write_csv(df: pd.DataFrame, name: str) -> None:
    with atomic_output(OUT_DIR / name) as tmp:
        df.to_csv(tmp, index=False)


def update_summary(*keys_and_value) -> None:
    """update_summary("c4", "retrain", {...}) -> summary["c4"]["retrain"] = {...}."""
    *keys, value = keys_and_value
    path = OUT_DIR / "summary.json"
    cur = json.loads(path.read_text()) if path.exists() else {}
    node = cur
    for k in keys[:-1]:
        node = node.setdefault(k, {})
    node[keys[-1]] = clean(value)
    atomic_write_text(path, json.dumps(cur, indent=1))


def run_pool(fn, tasks, workers: int) -> list[dict]:
    with ProcessPoolExecutor(workers, mp_context=FORK) as pool:
        return [r for rs in pool.map(fn, tasks) for r in rs]


# ---------------------------------------------------------------- probes retreinados (c1, c4)
def fit(stack_fn, train_ids, labels, variants):
    rng = np.random.default_rng(SEED)
    X, y = [], []
    for tid in train_ids:
        xt, yt = sample_training(stack_fn(ORIGINAL, tid), labels[tid], rng)
        if xt is not None:
            X.append(xt)
            y.append(yt)
    X, y = np.concatenate(X), np.concatenate(y)
    return fit_probes(X, y, -1, variants), X.shape


def tile_aps(probes, stack_fn, lay, tid: str) -> dict[str, float]:
    stack, lab = crop(stack_fn(lay, tid), ph.read_label(lay, tid))
    mask = lab > 0
    X = stack[:, mask].T
    out = {}
    for variant, model in probes.models.items():
        x = (X - probes.mu) / probes.sd if variant == "lin" else X
        out[variant] = safe_ap(lab[mask] == 2, model.predict_proba(x)[:, 1])
    return out


def run_models(configs, train_ids, labels, sets, csv_name: str, base: pd.DataFrame, summarize):
    rows = []
    for name, stack_fn, variants in configs:
        t0 = time.time()
        probes, shape_ = fit(stack_fn, train_ids, labels, variants)
        print(f"· {name}: treinado ({shape_[0]} px, {shape_[1]} features) em {time.time() - t0:.0f}s",
              flush=True)
        for scope, ids in sets.items():
            lay = layout(scope)
            for tid in ids:
                for variant, ap in tile_aps(probes, stack_fn, lay, tid).items():
                    exp = name if variant == "rf" else f"{name}_{variant}"
                    rows.append({"scope": scope, "tile": tid, "experiment": exp, "boundary_ap": ap})
        print(f"  {name}: avaliado em {time.time() - t0:.0f}s", flush=True)
        new = pd.DataFrame(rows)
        write_csv(new, csv_name)
        summarize(pd.concat([base, new], ignore_index=True))


# ---------------------------------------------------------------- c1
def cosine_dissimilarity(data: np.ndarray) -> list[np.ndarray]:
    """Média de 1 - cos(v_i, v_j) sobre os 8 vizinhos (3x3) e os 24 vizinhos (5x5).
    Borda: reflexão simétrica (= modo 'reflect' do scipy usado nos filtros por banda)."""
    v = np.nan_to_num(data).astype("float32")
    u = v / np.maximum(np.linalg.norm(v, axis=0), 1e-6)
    _, h, w = u.shape
    pad = np.pad(u, ((0, 0), (2, 2), (2, 2)), mode="symmetric")
    s3 = np.zeros((h, w), "float64")
    s5 = np.zeros((h, w), "float64")
    for dy in range(-2, 3):
        for dx in range(-2, 3):
            if dy == dx == 0:
                continue
            d = 1.0 - (u * pad[:, 2 + dy:2 + dy + h, 2 + dx:2 + dx + w]).sum(axis=0)
            s5 += d
            if max(abs(dy), abs(dx)) == 1:
                s3 += d
    return [(s3 / 8).astype("float32"), (s5 / 24).astype("float32")]


def filtered_w(data: np.ndarray, w: int) -> np.ndarray:
    """Operador por banda em escala w: bruto, média w, DP w, DP w+2, Sobel da média w."""
    layers = []
    for band in data:
        layers.append(np.nan_to_num(band))
        layers.extend(np.nan_to_num(f) for f in context_features(band, w, w + 2))
    return np.stack(layers).astype("float32")


def train_moments(source: str, train_ids, labels):
    rng = np.random.default_rng(SEED)
    X = [xt for tid in train_ids
         if (xt := sample_training(np.nan_to_num(ph.read_raw(ORIGINAL, tid, source)), labels[tid], rng)[0])
         is not None]
    X = np.concatenate(X)
    sd = X.std(axis=0)
    sd[sd == 0] = 1.0
    return X.mean(axis=0)[:, None, None], sd[:, None, None]


def c1_configs(train_ids, labels):
    mu, sd = train_moments("s2", train_ids, labels)
    print(f"· c1: padronização S2 com {mu.size} médias/DPs de treino", flush=True)

    def raw(src):
        return lambda lay, tid: np.nan_to_num(ph.read_raw(lay, tid, src))

    def mv(src):
        if src == "s2":
            return lambda lay, tid: np.stack(cosine_dissimilarity((raw("s2")(lay, tid) - mu) / sd))
        return lambda lay, tid: np.stack(cosine_dissimilarity(raw("ae")(lay, tid)))

    def cat(*fns):
        return lambda lay, tid: np.concatenate([f(lay, tid) for f in fns]).astype("float32")

    ctx = {"s2": ph.source_stack("s2"), "ae": ph.source_stack("ae")}
    cfg = []
    for e, src in (("E1", "s2"), ("E3", "ae")):
        cfg += [(f"{e}mvonly", mv(src), ("rf",)),
                (f"{e}mv", cat(raw(src), mv(src)), ("rf",)),
                (f"{e}cmv", cat(ctx[src], mv(src)), ("rf",))]
    for w in SWEEP_WINDOWS:
        for e, src in (("E1c", "s2"), ("E3c", "ae")):
            cfg.append((f"{e}_w{w}", lambda lay, tid, s=src, w=w: filtered_w(ph.read_raw(lay, tid, s), w),
                        ("rf",)))
    return cfg


def c1_summary(ap: pd.DataFrame) -> None:
    pairs = [("E3mvonly", "E1mvonly"), ("E3mv", "E1mv"), ("E3cmv", "E1cmv"), ("E3", "E1"), ("E3c", "E1c"),
             ("E1mv", "E1"), ("E3mv", "E3"), ("E1cmv", "E1c"), ("E3cmv", "E3c"),
             *[(f"E3c_w{w}", f"E1c_w{w}") for w in SWEEP_WINDOWS],
             *[(f"E1c_w{w}", "E1c") for w in SWEEP_WINDOWS], *[(f"E3c_w{w}", "E3c") for w in SWEEP_WINDOWS]]
    res = contrasts_by_scope(ap, "boundary_ap", pairs)
    wide = ap.pivot_table(index=["scope", "tile"], columns="experiment", values="boundary_ap")
    res["reproduction_w3_max_abs_diff"] = {
        e: r4((wide[f"{e}_w3"] - wide[e]).abs().max()) for e in ("E1c", "E3c") if f"{e}_w3" in wide}
    update_summary("c1", res)


# ---------------------------------------------------------------- c2
def c2_task(args) -> list[dict]:
    scope, tid, thr_ws, thr_rag = args
    lay = layout(scope)
    lab = ph.read_label(lay, tid)
    excl = near(zero_index_mask(lay, tid), EXCL_PX)
    lab, excl = crop(lab, excl)
    ext = (lab == 0) & (ndi.distance_transform_edt(lab == 0) >= FAR_PX) & ~excl
    deep = (lab == 1) & (ndi.distance_transform_edt(lab == 1) >= FAR_PX) & ~excl
    pos = (lab == 2) & ~excl
    rows = []
    for exp in RF_EXPS:
        p = read_prob(scope, tid, exp)[:lab.shape[0], :lab.shape[1]]
        row = {"scope": scope, "tile": tid, "experiment": exp, "n_ext": int(ext.sum()), "n_deep": int(deep.sum())}
        for region, m in (("ext", ext), ("deep", deep)):
            row[f"{region}_mean_p"] = float(p[m].mean()) if m.any() else float("nan")
            row[f"{region}_frac_ws"] = float((p[m] >= thr_ws[exp]).mean()) if m.any() else float("nan")
            row[f"{region}_frac_rag"] = float((p[m] >= thr_rag[exp]).mean()) if m.any() else float("nan")
        neg = (lab == 1) & ~excl | ext
        row["ap_open"] = safe_ap(pos[pos | neg], p[pos | neg])
        row["ap_vs_exterior"] = safe_ap(pos[pos | ext], p[pos | ext])
        rows.append(row)
    return rows


C_PAIRS = [("E3c", "E1c"), ("E3", "E1"), ("E3c", "E2c"), ("E4c", "E2c")]


def run_c2(sets, workers: int) -> None:
    tw, tr = thresholds("ws"), thresholds("rag")
    df = pd.DataFrame(run_pool(c2_task, [(s, t, tw, tr) for s, ids in sets.items() for t in ids], workers))
    write_csv(df, "c2_false_boundary_per_tile.csv")
    metrics = [c for c in df.columns if c.startswith(("ext_", "deep_", "ap_"))]
    update_summary("c2", {m: contrasts_by_scope(df, m, C_PAIRS) for m in metrics})


# ---------------------------------------------------------------- c3
def parcel_ids(lay, tid: str) -> np.ndarray:
    """Todos os talhões da referência que tocam o tile, rasterizados com id próprio."""
    with rasterio.open(lay.raster(tid, "s2")) as ds:
        shape_, transform, crs, bounds = ds.shape, ds.transform, ds.crs, ds.bounds
    bb = gpd.GeoSeries([box(*bounds)], crs=crs).to_crs(REF_ALL.crs).total_bounds
    ref_t = REF_ALL.cx[bb[0]:bb[2], bb[1]:bb[3]].to_crs(crs)
    if ref_t.empty:
        return np.zeros(shape_, "int32")
    return rasterize([(g, i + 1) for i, g in enumerate(ref_t.geometry)], out_shape=shape_,
                     transform=transform, fill=0, dtype="int32")


def c3_task(args) -> list[dict]:
    scope, tid = args
    lay = layout(scope)
    lab, ids = crop(ph.read_label(lay, tid), parcel_ids(lay, tid))
    k = 2 * ADJ_PX + 1
    big = np.iinfo("int32").max
    mx = ndi.maximum_filter(ids, k)
    mn = ndi.minimum_filter(np.where(ids == 0, big, ids), k)
    other = ((mx != ids) & (mx > 0)) | ((mn != ids) & (mn != big))
    band = lab == 2
    internal, external = band & other, band & ~other
    interior = lab == 1
    tol_neg = interior & ~near(band, 2)
    band2 = (lab > 0) & near(band, 1)
    rows = []
    for exp in RF_EXPS:
        p = read_prob(scope, tid, exp)[:lab.shape[0], :lab.shape[1]]
        pmax = ndi.maximum_filter(p, 3)
        m_ext, m_int, m_tol = external | interior, internal | interior, band | tol_neg
        rows.append({
            "scope": scope, "tile": tid, "experiment": exp,
            "n_band": int(band.sum()), "frac_internal": float(internal.sum() / max(band.sum(), 1)),
            "ap": safe_ap(band[lab > 0], p[lab > 0]),
            "ap_external": safe_ap(external[m_ext], p[m_ext]),
            "ap_internal": safe_ap(internal[m_int], p[m_int]),
            "ap_tol1": safe_ap(band[m_tol], pmax[m_tol]),
            "ap_band2": safe_ap(band2[lab > 0], p[lab > 0]),
        })
    return rows


def run_c3(sets, workers: int) -> None:
    df = pd.DataFrame(run_pool(c3_task, [(s, t) for s, ids in sets.items() for t in ids], workers))
    write_csv(df, "c3_endpoint_per_tile.csv")
    b = base_ap()
    chk = df.merge(b, on=["scope", "tile", "experiment"])
    res = {m: contrasts_by_scope(df, m, C_PAIRS) for m in ("ap_external", "ap_internal", "ap_tol1", "ap_band2")}
    res["frac_internal_median"] = {s: r4(g.drop_duplicates("tile").frac_internal.median())
                                   for s, g in df.groupby("scope")}
    res["reproduction_ap_max_abs_diff"] = r4((chk.ap - chk.boundary_ap).abs().max())
    update_summary("c3", res)


# ---------------------------------------------------------------- c4
def s2_fixed(data: np.ndarray) -> np.ndarray:
    """NDVI/NDWI/EVI recalculados das reflectâncias exportadas; EVI recortado em [-1, 1]."""
    out = data.copy()
    with np.errstate(divide="ignore", invalid="ignore"):
        for off in (0, 8):
            b2, b3, b4, b8 = data[off], data[off + 1], data[off + 2], data[off + 3]
            out[off + 5] = (b8 - b4) / (b8 + b4)
            out[off + 6] = (b3 - b8) / (b3 + b8)
            out[off + 7] = np.clip(2.5 * (b8 - b4) / (b8 + 6 * b4 - 7.5 * b2 + 1), -1, 1)
    return np.nan_to_num(out).astype("float32")


def c4_configs():
    def src(name, context):
        def build(lay, tid):
            data = ph.read_raw(lay, tid, name)
            data = s2_fixed(data) if name == "s2" else np.nan_to_num(data)
            return ph.filtered(data) if context else data
        return build

    def cat(*fns):
        return lambda lay, tid: np.concatenate([f(lay, tid) for f in fns]).astype("float32")

    sources = {"E1": ("s2",), "E2": ("s1", "s2"), "E4": ("s1", "s2", "ae")}
    cfg = []
    for context in (False, True):
        for e, srcs in sources.items():
            name = f"{e}{'c' if context else ''}F"
            cfg.append((name, cat(*[src(s, context) for s in srcs]), ("rf", "lin")))
    return cfg


def c4_summary(ap: pd.DataFrame) -> None:
    res = {}
    for scope, g in ap.groupby("scope"):
        g = g.dropna(subset=["boundary_ap"])
        r = {}
        if {"E1cF", "E1F"} <= set(g.experiment):
            r["H1_fix"] = non_superiority(paired(g, "E3c", "E1cF", "boundary_ap"), MARGIN)
            r["raw_E3_vs_E1F"] = non_superiority(paired(g, "E3", "E1F", "boundary_ap"), MARGIN)
            r["H2_fix"] = diff_in_diff(g, ("E3", "E1F"), ("E3c", "E1cF"))
        if {"E2cF", "E4cF"} <= set(g.experiment):
            r["E3c_vs_E2cF"] = non_superiority(paired(g, "E3c", "E2cF", "boundary_ap"), MARGIN)
            r["E4cF_vs_E2cF_tost"] = tost(g, "E4cF", "E2cF", margin=MARGIN)
            r["H2_E4F_vs_E2F"] = diff_in_diff(g, ("E4F", "E2F"), ("E4cF", "E2cF"))
        if "E1cF_lin" in set(g.experiment):
            r["H3_fix_tost"] = tost(g, "E3c_lin", "E1cF_lin", margin=MARGIN)
        wide = g.pivot_table(index="tile", columns="experiment", values="boundary_ap")
        r["fix_minus_orig"] = {f"{e}F-{e}": contrast(wide, f"{e}F", e)
                               for e in ("E1", "E2", "E4", "E1c", "E2c", "E4c") if f"{e}F" in wide}
        r["mean"] = {k: r4(v) for k, v in wide.mean().items() if k.endswith(("F", "F_lin"))}
        res[scope] = r
    update_summary("c4", "retrain", res)


def c4_excl_task(args) -> list[dict]:
    scope, tid = args
    lay = layout(scope)
    lab, zero = crop(ph.read_label(lay, tid), zero_index_mask(lay, tid))
    excl = near(zero, EXCL_PX)
    keep = (lab > 0) & ~excl
    rows = []
    for exp in RF_EXPS + tuple(f"{e}_lin" for e in RF_EXPS):
        p = read_prob(scope, tid, exp)[:lab.shape[0], :lab.shape[1]]
        rows.append({"scope": scope, "tile": tid, "experiment": exp,
                     "n_labeled": int((lab > 0).sum()), "n_excluded": int(((lab > 0) & excl).sum()),
                     "frac_zero_tile": float(zero.mean()), "boundary_ap": safe_ap(lab[keep] == 2, p[keep])})
    return rows


def run_c4_exclusion(sets, workers: int) -> None:
    df = pd.DataFrame(run_pool(c4_excl_task, [(s, t) for s, ids in sets.items() for t in ids], workers))
    write_csv(df, "c4_exclusion_ap_per_tile.csv")
    res = {}
    for scope, g in df.groupby("scope"):
        res[scope] = {
            "frac_labeled_excluded_mean": r4((g.n_excluded / g.n_labeled).mean()),
            "frac_labeled_excluded_max": r4((g.n_excluded / g.n_labeled).max()),
            "frac_zero_tile_mean": r4(g.drop_duplicates("tile").frac_zero_tile.mean()),
            "H1_excl": non_superiority(paired(g, "E3c", "E1c", "boundary_ap"), MARGIN),
            "raw_excl": non_superiority(paired(g, "E3", "E1", "boundary_ap"), MARGIN),
            "H2_excl": diff_in_diff(g, ("E3", "E1"), ("E3c", "E1c")),
            "H3_excl_tost": tost(g, "E3c_lin", "E1c_lin", margin=MARGIN),
            "E4c_vs_E2c_excl_tost": tost(g, "E4c", "E2c", margin=MARGIN),
        }
    update_summary("c4", "exclusion", res)


# ---------------------------------------------------------------- c5
def c5_task(args) -> list[dict]:
    """Mesma lógica de objects.eval_prob_file (strict, precisão total), referência lida uma vez."""
    tid, exp, decoder, sigma, grid = args
    with rasterio.open(scoped_prob_path("confirm", tid, exp)) as ds:
        prob, transform, crs, bounds = ds.read(1), ds.transform, ds.crs, ds.bounds
    ref_t, interior = objects.evaluable_reference(bounds, crs, REF_ALL)
    if ref_t.empty:
        return []
    fn = objects.watershed_decoder if decoder == "ws" else partial(objects.rag_decoder, sigma=sigma)
    rows = []
    for t in grid:
        labels = fn(prob, t)
        m = (objects.failed_tile_metrics(len(ref_t)) if labels is None else
             objects.tile_metrics(objects.to_polygons(labels, transform, crs), ref_t, interior,
                                  strict=True, digits=None))
        rows.append({"tile": tid, "experiment": exp, "decoder": decoder, "sigma": sigma, "threshold": t,
                     **{k: m.get(k) for k in ("n_cand", "object_recall", "mean_iou", "over_seg",
                                              "under_seg", "boundary_f1")}})
    return rows


def run_c5(sets, workers: int) -> None:
    ids = sets["confirm"]
    tasks = [(t, e, "ws", None, ctx_eval.GRIDS["ws"]) for t in ids for e in C5_EXPS]
    tasks += [(t, e, "rag", s, ctx_eval.GRIDS["rag"]) for s in SIGMAS for t in ids for e in C5_EXPS]
    tasks.sort(key=lambda x: (x[2] != "rag", x[3] == 0.0))  # RAG primeiro (mais lento)
    df = pd.DataFrame(run_pool(c5_task, tasks, workers))
    df["sigma"] = df.sigma.fillna(-1)
    df["threshold"] = df.threshold.round(2)
    write_csv(df, "c5_threshold_sigma_per_tile.csv")
    frozen = {"ws": thresholds("ws"), "rag": thresholds("rag")}
    pairs = [("E3c", "E1c"), ("E3c", "E2c"), ("E4c", "E2c")]
    res = {}
    for (dec, sig), g in df.groupby(["decoder", "sigma"]):
        key = f"{dec}" + (f"_sigma{sig:g}" if dec == "rag" else "")
        curve = g.groupby(["experiment", "threshold"]).mean_iou.mean().unstack(0)
        oracle_t = {e: float(curve[e].idxmax()) for e in curve}
        r = {"oracle_threshold": oracle_t, "oracle_mean_iou": {e: r4(curve[e].max()) for e in curve},
             "frozen_threshold": {e: frozen[dec][e] for e in C5_EXPS}}
        for label, tmap in (("frozen", frozen[dec]), ("oracle", oracle_t)):
            sel = pd.concat([g[(g.experiment == e) & np.isclose(g.threshold, tmap[e])] for e in C5_EXPS])
            for metric in ("mean_iou", "object_recall"):
                wide = sel.pivot_table(index="tile", columns="experiment", values=metric)
                r[f"{label}_{metric}"] = {"mean": {k: r4(v) for k, v in wide.mean().items()},
                                          **{f"{a}-{b}": contrast(wide, a, b) for a, b in pairs}}
        per_tile_best = g.groupby(["tile", "experiment"]).mean_iou.max().unstack()
        r["per_tile_oracle_mean_iou"] = {f"{a}-{b}": contrast(per_tile_best, a, b) for a, b in pairs}
        common = {}
        for t, gt in g.groupby("threshold"):
            wide = gt.pivot_table(index="tile", columns="experiment", values="mean_iou")
            common[f"{t:.2f}"] = {f"{a}-{b}": contrast(wide, a, b) for a, b in pairs}
        r["common_threshold"] = common
        res[key] = r
    chk = df[(df.decoder == "rag") & (df.sigma == 1.0)]
    stored = pd.read_csv(scoped_metrics_dir("confirm") / "rag_rf_per_tile.csv")
    m = chk.merge(stored[["tile", "experiment", "mean_iou"]], on=["tile", "experiment"], suffixes=("", "_stored"))
    m = m[np.isclose(m.threshold, m.experiment.map(frozen["rag"]))]
    res["reproduction_rag_sigma1_frozen_max_abs_diff"] = r4((m.mean_iou - m.mean_iou_stored).abs().max())
    stored = pd.read_csv(scoped_metrics_dir("confirm") / "ws_rf_per_tile.csv")
    w = df[df.decoder == "ws"].merge(stored[["tile", "experiment", "mean_iou"]], on=["tile", "experiment"],
                                     suffixes=("", "_stored"))
    w = w[np.isclose(w.threshold, w.experiment.map(frozen["ws"]))]
    res["reproduction_ws_frozen_max_abs_diff"] = r4((w.mean_iou - w.mean_iou_stored).abs().max())
    update_summary("c5", res)


# ---------------------------------------------------------------- main
def main() -> None:
    global REF_ALL
    ap = argparse.ArgumentParser()
    ap.add_argument("--checks", nargs="+", required=True, choices=("c1", "c2", "c3", "c4", "c5"))
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    sets = eval_sets()
    print(f"· checks={args.checks} · tiles={ {k: len(v) for k, v in sets.items()} } · livre={check_disk():.0f} GB",
          flush=True)
    if {"c3", "c5"} & set(args.checks):
        REF_ALL = objects.load_ref(REF)
    orig_split = json.loads(ORIGINAL.split.read_text())
    train_ids = [t for t, p in orig_split.items() if p == "train"]
    for check in args.checks:
        t0 = time.time()
        print(f"=== {check} ===", flush=True)
        if check == "c2":
            run_c2(sets, args.workers)
        elif check == "c3":
            run_c3(sets, args.workers)
        elif check == "c5":
            run_c5(sets, args.workers)
        else:
            labels = {tid: ph.read_label(ORIGINAL, tid) for tid in train_ids}
            if check == "c1":
                run_models(c1_configs(train_ids, labels), train_ids, labels, sets, "c1_ap_per_tile.csv",
                           base_ap(), c1_summary)
            else:
                run_c4_exclusion(sets, args.workers)
                run_models(c4_configs(), train_ids, labels, sets, "c4_index_fix_ap_per_tile.csv",
                           base_ap(), c4_summary)
        dt = time.time() - t0
        print(f"✓ {check} em {dt / 60:.1f} min", flush=True)
        update_summary("runtime_min", check, round(dt / 60, 1))


if __name__ == "__main__":
    main()
