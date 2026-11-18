#!/usr/bin/env python3
"""Valida artefatos do layout canonico sem recalcular resultados."""
import argparse
import json
import warnings
from collections import Counter

import geopandas as gpd
import pandas as pd
import rasterio
from rasterio.features import rasterize
from shapely.geometry import box

from paths import (
    EXPECTED_BANDS,
    EXPERIMENTS,
    REF,
    SOURCES,
    SPLIT,
    TILES,
    label_path,
    metrics_path,
    prob_path,
    raster_path,
    thresholds_path,
)

BOUNDARY_M = 10.0

warnings.filterwarnings(
    "ignore",
    message="Setting the shape on a NumPy array has been deprecated.*",
    category=DeprecationWarning,
)


def fail(errors: list[str]) -> None:
    if errors:
        print("ERROS:")
        for err in errors[:80]:
            print(f"  - {err}")
        if len(errors) > 80:
            print(f"  ... +{len(errors) - 80}")
        raise SystemExit(1)


def raster_meta(path):
    with rasterio.open(path) as ds:
        return {
            "count": ds.count,
            "width": ds.width,
            "height": ds.height,
            "crs": ds.crs.to_string() if ds.crs else None,
            "transform": tuple(round(x, 9) for x in ds.transform[:6]),
        }


def load_ref() -> gpd.GeoDataFrame:
    df = pd.read_parquet(REF)
    return gpd.GeoDataFrame(df, geometry=gpd.GeoSeries.from_wkb(df.geom_wkb), crs=4674)


def label_violations(tid: str, ref_all: gpd.GeoDataFrame) -> int:
    with rasterio.open(raster_path(tid, "s2")) as ds:
        shape_, transform, crs, bounds = ds.shape, ds.transform, ds.crs, ds.bounds
    with rasterio.open(label_path(tid)) as ds:
        labels = ds.read(1)[:shape_[0], :shape_[1]]

    bb = gpd.GeoSeries([box(*bounds)], crs=crs).to_crs(ref_all.crs).total_bounds
    ref_t = ref_all.cx[bb[0]:bb[2], bb[1]:bb[3]].to_crs(crs)
    ref_t = ref_t[ref_t.within(box(*bounds).buffer(-BOUNDARY_M))]
    if ref_t.empty:
        return int((labels == 2).sum())

    parcels = rasterize(
        [(g, 1) for g in ref_t.geometry],
        out_shape=shape_,
        transform=transform,
        fill=0,
        dtype="uint8",
    )
    return int(((labels == 2) & (parcels == 0)).sum())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-lin", action="store_true", help="nao exige artefatos da variante linear")
    args = ap.parse_args()

    errors: list[str] = []
    for path in (REF, TILES, SPLIT):
        if not path.exists():
            errors.append(f"arquivo ausente: {path}")
    fail(errors)

    tiles = json.loads(TILES.read_text())["features"]
    tile_ids = [ft["properties"]["tile_id"] for ft in tiles]
    split = json.loads(SPLIT.read_text())
    counts = Counter(split.values())
    eval_ids = [tid for tid, part in split.items() if part in ("val", "test")]
    test_ids = [tid for tid, part in split.items() if part == "test"]

    for tid in tile_ids:
        metas = {}
        for source in SOURCES:
            path = raster_path(tid, source)
            if not path.exists():
                errors.append(f"raster ausente: {path}")
                continue
            meta = raster_meta(path)
            metas[source] = meta
            if meta["count"] != EXPECTED_BANDS[source]:
                errors.append(f"{path}: {meta['count']} bandas; esperado {EXPECTED_BANDS[source]}")
        if len(metas) == len(SOURCES):
            grid = {k: (v["width"], v["height"], v["crs"], v["transform"]) for k, v in metas.items()}
            if len(set(grid.values())) != 1:
                errors.append(f"{tid}: grades S1/S2/AE divergentes: {grid}")

    ref_all = load_ref()
    for tid, part in split.items():
        if part == "skip":
            continue
        if not label_path(tid).exists():
            errors.append(f"label ausente: {label_path(tid)}")
        elif (n := label_violations(tid, ref_all)) > 0:
            errors.append(f"{label_path(tid)}: {n} px de fronteira fora do footprint dos talhoes")

    variants = ("rf",) if args.skip_lin else ("rf", "lin")
    for variant in variants:
        for kind in ("summary", "per_tile", "tests"):
            if not metrics_path(kind, variant).exists():
                errors.append(f"metrica ausente: {metrics_path(kind, variant)}")
        if not thresholds_path(variant).exists():
            errors.append(f"thresholds ausente: {thresholds_path(variant)}")
        for tid in eval_ids:
            for exp in EXPERIMENTS:
                path = prob_path(tid, exp, variant)
                if not path.exists():
                    errors.append(f"probabilidade ausente: {path}")

    fail(errors)
    print("✓ artefatos OK")
    print(f"  tiles={len(tile_ids)} split={dict(counts)} val/test={len(eval_ids)} test={len(test_ids)}")


if __name__ == "__main__":
    main()
