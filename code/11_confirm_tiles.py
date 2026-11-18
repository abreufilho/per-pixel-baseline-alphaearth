#!/usr/bin/env python3
"""Passo 11 — tiles confirmatórios: amostragem, elegibilidade, rótulos e deriva dos rasters.

docs/protocol/confirmatory.md §7. Mesma grade angular e estratos de 02_tiles.py.

  --stage sample   lista ordenada de N_LIST candidatos -> docs/protocol/confirm_tiles.{geojson,csv}
                   (versionada com o protocolo, antes de qualquer arquivo em data/confirm/)
  --stage materialize  copia a lista congelada para data/confirm/tiles/tiles_eval.geojson
  (download)       ALPHAEARTH_MG_DATA=/work/data/confirm python code/03_gee_stacks.py --allow-partial
                   (até 3 tentativas; o 03 só baixa o que falta)
  --stage labels   elegibilidade na ordem da lista, antes de qualquer predição; os primeiros
                   N_TEST elegíveis viram "test" em data/confirm/tiles/split.json
  --stage drift-setup / drift-compare
                   re-download de DRIFT_N tiles originais em data/confirm/drift/ e comparação
                   com data/rasters (coleções do Earth Engine podem mudar)
  --stage coverage relatório de cobertura (pixels inválidos por fonte) dos tiles originais

Exclusão geométrica: candidato não pode tocar nenhuma célula de 0,5° tocada por um tile
original, nem ficar a menos de MIN_DIST_M de um tile original. Ordem: regiões embaralhadas
(seed 2026); 1ª rodada = 1 tile por região; 2ª rodada = 2º tile de estrato diferente, se houver.
"""
import argparse
import importlib
import json
import math

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from shapely.geometry import box

from objects import evaluable_reference, load_ref
from paths import (
    CONFIRM,
    CONFIRM_ROOT,
    EXPECTED_BANDS,
    ORIGINAL,
    REF,
    ROOT,
    SOURCES,
    Layout,
    atomic_output,
    atomic_write_text,
    check_disk,
    scoped_metrics_dir,
)
from probing import rasterize_labels

tiles02 = importlib.import_module("02_tiles")

SEED = 2026
N_LIST = 72
N_TEST = 48
MAX_PER_REGION = 2
PX = 500
CELL_DEG = 0.5
MIN_DIST_M = 5000.0
MIN_EVAL_PARCELS = 10
MAX_INVALID_FRAC = 0.10
METRIC_CRS = 5880  # SIRGAS 2000 / Brazil Polyconic
DRIFT = Layout(CONFIRM_ROOT / "drift")
DRIFT_N = 3
PROTOCOL_LIST = ROOT / "docs" / "protocol" / "confirm_tiles.csv"
PROTOCOL_GEOJSON = ROOT / "docs" / "protocol" / "confirm_tiles.geojson"


def cells_touched(geoms) -> set[tuple[float, float]]:
    cells = set()
    for g in geoms:
        minx, miny, maxx, maxy = g.bounds
        for lat in np.arange(math.floor(miny / CELL_DEG) * CELL_DEG, maxy, CELL_DEG):
            for lon in np.arange(math.floor(minx / CELL_DEG) * CELL_DEG, maxx, CELL_DEG):
                if g.intersects(box(lon, lat, lon + CELL_DEG, lat + CELL_DEG)):
                    cells.add((round(lat, 1), round(lon, 1)))
    return cells


def sample() -> None:
    if PROTOCOL_GEOJSON.exists():
        raise SystemExit(f"{PROTOCOL_GEOJSON} já existe; a amostra é congelada")
    ref = load_ref(REF)
    grid, agg = tiles02.candidate_tiles(ref, PX)
    cand = grid.set_index("tile").loc[agg.index].join(agg)
    orig = gpd.read_file(ORIGINAL.tiles).to_crs(cand.crs)
    banned = cells_touched(orig.geometry)
    touches = cand.geometry.apply(lambda g: bool(cells_touched([g]) & banned))
    dist = cand.to_crs(METRIC_CRS).geometry.apply(
        lambda g: orig.to_crs(METRIC_CRS).distance(g).min())
    cand["min_dist_km"] = (dist / 1000).round(2)
    pool = cand[~touches & (dist >= MIN_DIST_M)].copy()

    rng = np.random.default_rng(SEED)
    regions = sorted(pool.region.astype(str).unique())
    rng.shuffle(regions)
    picks, used = [], {}
    for _ in range(MAX_PER_REGION):
        for region in regions:
            block = pool[(pool.region.astype(str) == region) & ~pool.index.isin(picks)]
            if block.empty:
                continue
            fresh = block[~block.stratum.isin(used.get(region, set()))]
            block = fresh if not fresh.empty else block
            tile = block.index[rng.integers(len(block))]
            picks.append(tile)
            used.setdefault(region, set()).add(pool.loc[tile, "stratum"])
            if len(picks) == N_LIST:
                break
        if len(picks) == N_LIST:
            break
    out = pool.loc[picks].copy()
    out["tile_id"] = [f"c{i:02d}" for i in range(len(out))]
    out["order"] = range(len(out))
    out = gpd.GeoDataFrame(out, geometry="geometry", crs=cand.crs)
    with atomic_output(PROTOCOL_GEOJSON) as tmp:
        out.to_file(tmp, driver="GeoJSON")
    cols = ["tile_id", "order", "region", "stratum", "n_parcels", "med_area", "min_dist_km"]
    table = out[cols].copy()
    table["wkt"] = out.geometry.to_wkt()
    with atomic_output(PROTOCOL_LIST) as tmp:
        table.to_csv(tmp, index=False)
    print(f"✓ {len(out)} candidatos ordenados de {out.region.nunique()} regiões "
          f"(pool {len(pool)} tiles / {pool.region.nunique()} regiões; "
          f"{len(banned)} células banidas; distância mín. {out.min_dist_km.min()} km)")
    print(f"✓ {PROTOCOL_LIST}")


def materialize() -> None:
    if CONFIRM.tiles.exists():
        raise SystemExit(f"{CONFIRM.tiles} já existe")
    CONFIRM.tiles.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(CONFIRM.tiles, PROTOCOL_GEOJSON.read_text())
    print(f"✓ {CONFIRM.tiles}")


def invalid_fraction(lay: Layout, tid: str, source: str) -> float:
    with rasterio.open(lay.raster(tid, source)) as ds:
        data = ds.read()
    bad = ~np.isfinite(data).all(axis=0) | (data == 0).all(axis=0)
    return float(bad.mean())


def rasters_ok(lay: Layout, tid: str) -> tuple[bool, str]:
    grids = set()
    for source in SOURCES:
        path = lay.raster(tid, source)
        if not path.exists():
            return False, f"{source} ausente"
        with rasterio.open(path) as ds:
            if ds.count != EXPECTED_BANDS[source]:
                return False, f"{source}: {ds.count} bandas"
            if ds.crs is None:
                return False, f"{source}: sem CRS"
            grids.add((ds.crs.to_string(), ds.width, ds.height,
                       tuple(round(x, 9) for x in ds.transform[:6])))
    if len(grids) != 1:
        return False, "grades divergentes"
    return True, "ok"


def labels() -> None:
    check_disk()
    ref = load_ref(REF)
    feats = sorted(json.loads(CONFIRM.tiles.read_text())["features"],
                   key=lambda f: f["properties"]["order"])
    split, report = {}, []
    for ft in feats:
        tid = ft["properties"]["tile_id"]
        row = {"tile": tid, "region": ft["properties"]["region"]}
        if sum(1 for p in split.values() if p == "test") >= N_TEST:
            split[tid] = "unused"
            report.append({**row, "status": "unused", "reason": "cota preenchida"})
            continue
        ok, why = rasters_ok(CONFIRM, tid)
        cover = {}
        if ok:
            cover = {f"invalid_{s}": invalid_fraction(CONFIRM, tid, s) for s in SOURCES}
            worst = max(cover.values())
            if worst > MAX_INVALID_FRAC:
                ok, why = False, f"cobertura: {worst:.1%} pixels inválidos"
        lab = profile = None
        if ok:
            lab, profile = rasterize_labels(CONFIRM.raster(tid, "s2"), ref)
            if lab is None or not ((lab == 1).any() and (lab == 2).any()):
                ok, why = False, "sem as duas classes de rótulo"
        n_eval = 0
        if ok:
            with rasterio.open(CONFIRM.raster(tid, "s2")) as ds:
                n_eval = len(evaluable_reference(ds.bounds, ds.crs, ref)[0])
            if n_eval < MIN_EVAL_PARCELS:
                ok, why = False, f"{n_eval} talhões avaliáveis (< {MIN_EVAL_PARCELS})"
        row.update(cover, n_eval=n_eval)
        if not ok:
            split[tid] = "invalid"
            report.append({**row, "status": "invalid", "reason": why})
            continue
        CONFIRM.label(tid).parent.mkdir(parents=True, exist_ok=True)
        with atomic_output(CONFIRM.label(tid)) as tmp:
            with rasterio.open(tmp, "w", **profile) as dst:
                dst.write(lab, 1)
        split[tid] = "test"
        report.append({**row, "status": "test", "reason": "ok"})
    atomic_write_text(CONFIRM.split, json.dumps(split, indent=1))
    rep = pd.DataFrame(report)
    out = scoped_metrics_dir("confirm") / "tile_selection.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with atomic_output(out) as tmp:
        rep.to_csv(tmp, index=False)
    print(rep.status.value_counts().to_string())
    print(rep[rep.status == "invalid"][["tile", "reason"]].to_string(index=False))
    n_test = int((rep.status == "test").sum())
    print(f"· {n_test} tiles de teste em {rep[rep.status == 'test'].region.nunique()} regiões")
    if n_test < N_TEST:
        print(f"! só {n_test} tiles elegíveis (< {N_TEST}); reportar como desvio, sem estender a lista")


def drift_setup() -> None:
    split = json.loads(ORIGINAL.split.read_text())
    ids = [t for t, p in split.items() if p == "test"][:DRIFT_N]
    orig = gpd.read_file(ORIGINAL.tiles)
    sub = orig[orig.tile_id.isin(ids)]
    DRIFT.tiles.parent.mkdir(parents=True, exist_ok=True)
    with atomic_output(DRIFT.tiles) as tmp:
        sub.to_file(tmp, driver="GeoJSON")
    print(f"✓ {DRIFT.tiles}: {ids}")


def drift_compare() -> None:
    ids = [f["properties"]["tile_id"] for f in json.loads(DRIFT.tiles.read_text())["features"]]
    rows = []
    for tid in ids:
        for s in SOURCES:
            with rasterio.open(ORIGINAL.raster(tid, s)) as a, rasterio.open(DRIFT.raster(tid, s)) as b:
                x, y = a.read(), b.read()
                same_grid = a.transform == b.transform and x.shape == y.shape
            diff = np.abs(x - y) if same_grid else np.array([np.nan])
            rows.append({"tile": tid, "source": s, "same_grid": same_grid,
                         "max_abs_diff": float(np.nanmax(diff)),
                         "frac_changed": float((diff > 1e-6).mean()) if same_grid else None})
    rep = pd.DataFrame(rows)
    out = scoped_metrics_dir("confirm") / "raster_drift.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with atomic_output(out) as tmp:
        rep.to_csv(tmp, index=False)
    print(rep.to_string(index=False))


def coverage_original() -> None:
    ids = [f["properties"]["tile_id"] for f in json.loads(ORIGINAL.tiles.read_text())["features"]]
    rep = pd.DataFrame([{"tile": t, **{f"invalid_{s}": invalid_fraction(ORIGINAL, t, s)
                                        for s in SOURCES}} for t in ids])
    out = scoped_metrics_dir("original") / "coverage.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with atomic_output(out) as tmp:
        rep.to_csv(tmp, index=False)
    print(rep.describe().round(4).to_string())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True,
                    choices=("sample", "materialize", "labels", "drift-setup", "drift-compare",
                             "coverage"))
    stage = ap.parse_args().stage
    {"sample": sample, "materialize": materialize, "labels": labels, "drift-setup": drift_setup,
     "drift-compare": drift_compare, "coverage": coverage_original}[stage]()


if __name__ == "__main__":
    main()
