#!/usr/bin/env python3
"""Peças compartilhadas do probe: rótulos, pilhas de features, amostragem e ajuste.

Usado por 04_probe.py (E1-E4, pixel a pixel) e 09_context_probe.py (E1c-E4c com filtros de
janela, e predição nos tiles confirmatórios). Hiperparâmetros fixos e idênticos para todos os
experimentos; ver docs/protocol/confirmatory.md §3-4.
"""
from dataclasses import dataclass
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import rasterize
from scipy import ndimage as ndi
from shapely.geometry import box
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from paths import Layout

EXP_SOURCES = {"E1": ["s2"], "E2": ["s1", "s2"], "E3": ["ae"], "E4": ["s1", "s2", "ae"]}
CTX_EXP_SOURCES = {f"{e}c": s for e, s in EXP_SOURCES.items()}
BOUNDARY_M = 10.0            # meia-largura da classe fronteira (~1 px)
MAX_PX = {"boundary": 4000, "interior": 8000}  # cap de pixels de treino por tile/classe
SEED = 42


def experiment_spec(exp: str) -> tuple[list[str], bool]:
    """Fontes e se usa filtros de janela: 'E3c' -> (['ae'], True)."""
    if exp in EXP_SOURCES:
        return EXP_SOURCES[exp], False
    if exp in CTX_EXP_SOURCES:
        return CTX_EXP_SOURCES[exp], True
    raise ValueError(f"experimento desconhecido: {exp}")


def rasterize_labels(raster_file: Path, ref_all: gpd.GeoDataFrame):
    """0=fora do footprint, 1=interior de talhão, 2=fronteira. Retorna (labels, profile)
    ou (None, None) se nenhum talhão cabe inteiro no interior do tile."""
    with rasterio.open(raster_file) as ds:
        shape_, transform, crs, bounds = ds.shape, ds.transform, ds.crs, ds.bounds
    interior_box = box(*bounds).buffer(-BOUNDARY_M)
    bb = gpd.GeoSeries([box(*bounds)], crs=crs).to_crs(ref_all.crs).total_bounds
    ref_t = ref_all.cx[bb[0]:bb[2], bb[1]:bb[3]].to_crs(crs)
    ref_t = ref_t[ref_t.within(interior_box)]
    if ref_t.empty:
        return None, None
    parcels = rasterize([(g, 1) for g in ref_t.geometry], out_shape=shape_,
                        transform=transform, fill=0, dtype="uint8")
    edges = rasterize([(g.boundary.buffer(BOUNDARY_M), 1) for g in ref_t.geometry],
                      out_shape=shape_, transform=transform, fill=0, dtype="uint8")
    labels = np.where((edges == 1) & (parcels == 1), 2, parcels).astype("uint8")
    profile = {"driver": "GTiff", "height": shape_[0], "width": shape_[1], "count": 1,
               "dtype": "uint8", "crs": crs, "transform": transform, "compress": "lzw"}
    return labels, profile


def context_features(band: np.ndarray, small: int = 3, large: int = 5) -> list[np.ndarray]:
    """Média 3x3, DP 3x3, DP 5x5 e magnitude de Sobel da média 3x3; NaN-aware.
    small/large ≠ 3/5 só na varredura de escala pós-hoc (14_posthoc_sensitivity.py)."""
    valid = np.isfinite(band)
    filled = np.where(valid, band, 0.0).astype("float64")
    mask = valid.astype("float64")

    def window(size: int):
        den = np.maximum(ndi.uniform_filter(mask, size), 1e-6)
        mean = ndi.uniform_filter(filled, size) / den
        var = ndi.uniform_filter(filled ** 2, size) / den - mean ** 2
        return mean, np.sqrt(np.maximum(var, 0.0))

    mean3, std3 = window(small)
    _, std5 = window(large)
    grad = np.hypot(ndi.sobel(mean3, axis=0), ndi.sobel(mean3, axis=1))
    return [mean3, std3, std5, grad]


def load_stack(layout: Layout, tid: str, exp: str) -> np.ndarray:
    """Pilha (bandas, h, w) float32 do experimento; NaN -> 0 nas bandas brutas."""
    sources, context = experiment_spec(exp)
    arrs = []
    for s in sources:
        with rasterio.open(layout.raster(tid, s)) as ds:
            data = ds.read().astype("float32")
        if context:
            layers = []
            for band in data:
                layers.append(np.nan_to_num(band))
                layers.extend(context_features(band))
            data = np.stack(layers).astype("float32")
        else:
            data = np.nan_to_num(data)
        arrs.append(data)
    if len({a.shape[1:] for a in arrs}) > 1:
        # shapes divergentes quebrariam a garantia de "mesmos pixels de treino p/ E1-E4"
        raise ValueError(f"{tid}: shapes divergentes entre fontes {sources}: "
                         f"{[a.shape[1:] for a in arrs]}")
    return np.concatenate(arrs, axis=0)


def sample_training(stack: np.ndarray, labels: np.ndarray, rng: np.random.Generator):
    h, w = stack.shape[1], stack.shape[2]
    labels = labels[:h, :w]
    xs, ys = [], []
    for cls, name in ((2, "boundary"), (1, "interior")):
        r, c = np.nonzero(labels == cls)
        if len(r) == 0:
            continue
        if len(r) > MAX_PX[name]:
            idx = rng.choice(len(r), MAX_PX[name], replace=False)
            r, c = r[idx], c[idx]
        xs.append(stack[:, r, c].T)
        ys.append(np.full(len(r), 1 if cls == 2 else 0, dtype="int8"))
    if not xs:
        return None, None
    return np.concatenate(xs), np.concatenate(ys)


PROBE_VARIANTS = ("rf", "lin")
SENSITIVITY_VARIANTS = ("rf8",)  # RF com max_features=8 fixo (≈ sqrt(80) do E1c)


def make_rf(n_jobs: int, max_features: int | str = "sqrt", seed: int = SEED) -> RandomForestClassifier:
    return RandomForestClassifier(n_estimators=200, min_samples_leaf=5, max_features=max_features,
                                  class_weight="balanced_subsample",
                                  n_jobs=n_jobs, random_state=seed)


@dataclass
class Probes:
    models: dict
    mu: np.ndarray
    sd: np.ndarray

    def predict(self, stack: np.ndarray) -> dict[str, np.ndarray]:
        h, w = stack.shape[1], stack.shape[2]
        flat = stack.reshape(stack.shape[0], -1).T
        out = {}
        for variant, model in self.models.items():
            x = (flat - self.mu) / self.sd if variant == "lin" else flat
            out[variant] = model.predict_proba(x)[:, 1].reshape(h, w).astype("float32")
        return out


def fit_probes(X: np.ndarray, y: np.ndarray, n_jobs: int = -1,
               variants: tuple[str, ...] = PROBE_VARIANTS) -> Probes:
    mu, sd = X.mean(axis=0), X.std(axis=0)
    sd[sd == 0] = 1.0
    models = {}
    for variant in variants:
        if variant == "rf":
            models[variant] = make_rf(n_jobs).fit(X, y)
        elif variant == "rf8":
            models[variant] = make_rf(n_jobs, max_features=min(8, X.shape[1])).fit(X, y)
        elif variant == "lin":
            lin = LogisticRegression(C=1.0, max_iter=1000, class_weight="balanced")
            models[variant] = lin.fit((X - mu) / sd, y)
        else:
            raise ValueError(f"variante desconhecida: {variant}")
    return Probes(models, mu, sd)


def train_experiment(layout: Layout, exp: str, train_ids: list[str],
                     labels_by_tile: dict[str, np.ndarray], n_jobs: int = -1,
                     variants: tuple[str, ...] = PROBE_VARIANTS):
    """rng recriado por experimento: mesma seed + mesma sequência de sorteios por tile
    ⇒ todos os experimentos treinam exatamente nos mesmos pixels."""
    rng = np.random.default_rng(SEED)
    X, y = [], []
    for tid in train_ids:
        xt, yt = sample_training(load_stack(layout, tid, exp), labels_by_tile[tid], rng)
        if xt is not None:
            X.append(xt)
            y.append(yt)
    X, y = np.concatenate(X), np.concatenate(y)
    return fit_probes(X, y, n_jobs, variants), X.shape


def prob_profile(raster_file: Path, h: int, w: int) -> dict:
    with rasterio.open(raster_file) as ds:
        return {"driver": "GTiff", "height": h, "width": w, "count": 1, "dtype": "float32",
                "crs": ds.crs, "transform": ds.transform, "compress": "lzw"}
