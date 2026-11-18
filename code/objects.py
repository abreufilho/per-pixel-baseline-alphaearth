#!/usr/bin/env python3
"""Decodificadores de objetos sobre P(fronteira) e métricas de objeto por tile.

Compartilhado por 05_validate.py (watershed, desenho original) e 10_context_eval.py
(controles de contexto e tiles confirmatórios). Métricas por talhão de referência r vs
melhor candidato c* (maior interseção), Clinton et al. (2010). A referência avaliada exclui
slivers sub-pixel (< 1 px = 100 m²) e talhões que cruzam a margem de BUFFER_M do tile.
"""
from pathlib import Path
from typing import Callable

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import shapes, sieve
from scipy import ndimage as ndi
from shapely.geometry import box, shape
from shapely.ops import unary_union
from skimage import graph
from skimage.filters import gaussian
from skimage.morphology import local_minima
from skimage.segmentation import watershed
from sklearn.metrics import average_precision_score

IOU_HIT = 0.5
BUFFER_M = 10.0  # tolerância de fronteira (~1 px a 10 m)
MIN_SEG_PX = 5
MIN_REF_AREA_M2 = 100.0  # 1 px a 10 m — exclui slivers sub-pixel da referência avaliada
RAG_SIGMA = 1.0

Decoder = Callable[[np.ndarray, float], np.ndarray | None]


def load_ref(path: Path) -> gpd.GeoDataFrame:
    df = pd.read_parquet(path)
    return gpd.GeoDataFrame(df, geometry=gpd.GeoSeries.from_wkb(df.geom_wkb), crs=4674)


def _round(x: float, digits: int | None) -> float:
    return round(x, digits) if digits is not None else float(x)


def boundary_f1(cand: gpd.GeoDataFrame, ref: gpd.GeoDataFrame, interior,
                strict: bool = False, digits: int | None = 3) -> tuple[float, float, float]:
    """strict=True: referência com fronteira e nenhuma fronteira prevista perto dela conta
    como recall 0 e F1 0 (falha), em vez de NaN (dado ausente)."""
    nan = float("nan")
    ref_b = ref.geometry.boundary.union_all()
    if ref_b.is_empty:
        return nan, nan, nan
    miss = (nan, 0.0, 0.0) if strict else (nan, nan, nan)
    pred_b = cand.geometry.boundary.union_all().intersection(interior)
    if pred_b.is_empty:
        return miss
    footprint = unary_union(ref.geometry.values).buffer(BUFFER_M)
    pred_near = pred_b.intersection(footprint)
    if pred_near.is_empty or pred_near.length == 0:
        return miss
    prec = pred_near.intersection(ref_b.buffer(BUFFER_M)).length / pred_near.length
    rec = ref_b.intersection(pred_near.buffer(BUFFER_M)).length / ref_b.length
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else (0.0 if strict else nan)
    return _round(prec, digits), _round(rec, digits), _round(f1, digits)


def failed_tile_metrics(n_ref: int) -> dict:
    """Tile avaliável sem nenhum objeto previsto: falha pontuada, não removida (§6)."""
    nan = float("nan")
    return {"n_ref": n_ref, "n_cand": 0, "object_recall": 0.0, "mean_iou": 0.0,
            "over_seg": nan, "under_seg": nan, "rel_area_error_detected": nan,
            "boundary_precision": nan, "boundary_recall": 0.0, "boundary_f1": 0.0}


def tile_metrics(cand: gpd.GeoDataFrame, ref: gpd.GeoDataFrame, interior,
                 strict: bool = False, digits: int | None = 3) -> dict:
    """Métricas de um experimento dentro de um tile. cand/ref já em CRS métrico.
    digits=None mantém precisão total (inferência); 3 reproduz o desenho original."""
    n_ref = len(ref)
    if n_ref == 0 or cand.empty:
        if strict and n_ref > 0:
            return failed_tile_metrics(n_ref)
        return {"n_ref": n_ref, "n_cand": len(cand)}

    cand = cand.reset_index(drop=True)
    cand["cand_id"] = range(len(cand))
    cand["cand_area"] = cand.area
    ref = ref.reset_index(drop=True)
    ref["ref_id"] = range(len(ref))
    ref["ref_area"] = ref.area

    inter = gpd.overlay(
        cand[["geometry", "cand_id", "cand_area"]],
        ref[["geometry", "ref_id", "ref_area"]],
        how="intersection", keep_geom_type=True,
    )
    inter["ia"] = inter.area
    best_int = inter.sort_values("ia").drop_duplicates(subset=["ref_id"], keep="last")
    best = ref[["ref_id", "ref_area"]].merge(
        best_int[["ref_id", "cand_area", "ia"]], on="ref_id", how="left"
    )
    best["ia"] = best["ia"].fillna(0.0)
    union = best["ref_area"] + best["cand_area"] - best["ia"]
    iou = (best["ia"] / union).fillna(0.0)
    # over/under de Clinton sobre o MESMO conjunto: só talhões com interseção > 0
    # (sem isso, over incluiria não-pareados como 1.0 e under os descartaria)
    matched = best["ia"] > 0
    over = (1 - best["ia"] / best["ref_area"])[matched]
    under = (1 - best["ia"] / best["cand_area"])[matched]
    detected = iou >= IOU_HIT
    area_err = ((best["cand_area"] - best["ref_area"]).abs() / best["ref_area"])[detected]
    area_err = area_err.replace([np.inf], np.nan).dropna()

    prec, rec, f1 = boundary_f1(cand, ref, interior, strict, digits)
    r = lambda x: _round(float(x), digits)  # noqa: E731
    return {
        "n_ref": n_ref, "n_cand": len(cand),
        "object_recall": r(detected.mean()),
        "mean_iou": r(iou.mean()),
        "over_seg": r(over.mean()) if matched.any() else float("nan"),
        "under_seg": r(under.mean()) if matched.any() else float("nan"),
        "rel_area_error_detected": r(area_err.median()) if len(area_err) else float("nan"),
        "boundary_precision": prec, "boundary_recall": rec, "boundary_f1": f1,
    }


def watershed_decoder(prob: np.ndarray, t: float) -> np.ndarray | None:
    """Desenho original: marcadores = componentes de P < t, watershed sobre P."""
    markers, n = ndi.label(prob < t)
    if n == 0:
        return None
    return sieve(watershed(prob, markers).astype("int32"), size=MIN_SEG_PX)


def _weight_boundary(g, src, dst, n):
    default = {"weight": 0.0, "count": 0}
    a, b = g[src].get(n, default), g[dst].get(n, default)
    count = a["count"] + b["count"]
    return {"count": count, "weight": (a["count"] * a["weight"] + b["count"] * b["weight"]) / count}


def _merge_boundary(g, src, dst):
    pass


def rag_decoder(prob: np.ndarray, t: float, sigma: float = RAG_SIGMA) -> np.ndarray | None:
    """Fusão hierárquica de regiões: oversegmentação por watershed (mínimos locais de P
    suavizada) e fusão pelo RAG enquanto a P média da fronteira compartilhada < t."""
    prob = np.nan_to_num(prob.astype("float64"), nan=1.0)
    smooth = gaussian(prob, sigma=sigma, preserve_range=True) if sigma else prob
    markers, n = ndi.label(local_minima(smooth))
    if n == 0:
        return None
    labels = watershed(smooth, markers)
    rag = graph.rag_boundary(labels, prob)
    merged = graph.merge_hierarchical(labels, rag, thresh=t, rag_copy=False, in_place_merge=True,
                                      merge_func=_merge_boundary, weight_func=_weight_boundary)
    return sieve((merged + 1).astype("int32"), size=MIN_SEG_PX)


DECODERS: dict[str, Decoder] = {"ws": watershed_decoder, "rag": rag_decoder}


def to_polygons(labels: np.ndarray, transform, crs) -> gpd.GeoDataFrame:
    geoms = [shape(g) for g, _ in shapes(labels, transform=transform)]
    return gpd.GeoDataFrame({"cand": range(len(geoms))}, geometry=geoms, crs=crs)


def evaluable_reference(bounds, crs, ref_all: gpd.GeoDataFrame):
    """Talhões inteiramente no interior do tile (margem BUFFER_M) com área >= 1 px."""
    interior = box(*bounds).buffer(-BUFFER_M)
    # pré-filtro por bbox no CRS da referência ANTES do to_crs (reprojetar 384k é o gargalo)
    bb = gpd.GeoSeries([box(*bounds)], crs=crs).to_crs(ref_all.crs).total_bounds
    ref_t = ref_all.cx[bb[0]:bb[2], bb[1]:bb[3]].to_crs(crs)
    ref_t = ref_t[ref_t.within(interior)].copy()
    return ref_t[ref_t.geometry.area >= MIN_REF_AREA_M2], interior


def eval_prob_file(prob_file: Path, t: float, ref_all: gpd.GeoDataFrame,
                   decoder: Decoder = watershed_decoder, strict: bool = False,
                   digits: int | None = 3) -> dict | None:
    """strict=False reproduz 05_validate (tile sem marcadores sai da média); strict=True pontua
    a falha como recall/IoU/F1 = 0 (protocolo confirmatório)."""
    if not prob_file.exists():
        return None
    with rasterio.open(prob_file) as ds:
        prob, transform, crs, bounds = ds.read(1), ds.transform, ds.crs, ds.bounds
    labels = decoder(prob, t)
    if labels is None and not strict:
        return None
    ref_t, interior = evaluable_reference(bounds, crs, ref_all)
    if ref_t.empty:
        return None
    if labels is None:
        return failed_tile_metrics(len(ref_t))
    cand = to_polygons(labels, transform, crs)
    return tile_metrics(cand, ref_t, interior, strict, digits)


def boundary_ap(prob_file: Path, label_file: Path) -> float:
    """AP de fronteira pixel a pixel, livre de limiar, só dentro do footprint da referência."""
    with rasterio.open(prob_file) as ds:
        prob = ds.read(1)
    with rasterio.open(label_file) as ds:
        lab = ds.read(1)
    h, w = min(prob.shape[0], lab.shape[0]), min(prob.shape[1], lab.shape[1])
    prob, lab = prob[:h, :w], lab[:h, :w]
    mask = lab > 0
    if mask.sum() == 0 or (lab[mask] == 2).all() or (lab[mask] == 1).all():
        return float("nan")
    return float(average_precision_score(lab[mask] == 2, prob[mask]))
