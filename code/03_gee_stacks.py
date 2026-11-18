#!/usr/bin/env python3
"""Passo 3 — BAIXA os insumos (S2/S1/AlphaEarth, 2017) de cada tile do GEE para disco.

GEE é apenas FONTE DE DADOS: exportamos rasters; o processamento (segmentação) é LOCAL (passo 4).
Padrão de download adaptado de detecta-colheita (download_embeddings.py / download_s2.py):
getDownloadURL(GEO_TIFF) + requests + ThreadPoolExecutor. AlphaEarth (64 bandas > 48 MB no tile
de 5 km) é baixado em CHUNKS de bandas e remontado. S2 usa TOA (L1C) — cobertura 2017 completa.
Rasters em UTM por tile (pixel 10×10 m real; em EPSG:4326 o pixel não é quadrado em MG).

Bandas por fonte (compostas em E1-E4 no passo 4) — S1/S2 SAZONAIS (chuvosa/seca, fenologia
    café/MG; ver docs/legacy/STUDY-DESIGN.md):
  s2 (16): {B2 B3 B4 B8 B11 NDVI NDWI EVI} × {wet, dry} · s1 (8): {VV_med VH_med VV_sd VH_sd} × {wet, dry}
  ae (64): A00..A63

Requer: earthengine-api + requests + EE_PROJECT. Uso: python code/03_gee_stacks.py [--limit K]
"""
import argparse
import io
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import ee
import numpy as np
import rasterio
import requests

from paths import EXPECTED_BANDS, RASTER_DIR, SOURCES as SOURCE_NAMES, TILES, atomic_output, check_disk, raster_path

YEAR = 2017
START, END = f"{YEAR}-01-01", f"{YEAR}-12-31"
# estações (café/MG, ano-base 2017): chuvosa = jan-mar + out-dez; seca = abr-set
SEASONS = {
    "wet": [(f"{YEAR}-01-01", f"{YEAR}-04-01"), (f"{YEAR}-10-01", f"{YEAR + 1}-01-01")],
    "dry": [(f"{YEAR}-04-01", f"{YEAR}-10-01")],
}
SCALE, CHUNK, WORKERS = 10, 12, 8  # CHUNK<=~20 bandas p/ caber no limite de 48MB


def utm_epsg(lon: float, lat: float) -> str:
    """CRS métrico do tile: pixel 10×10 m real (em EPSG:4326 o pixel não é quadrado)."""
    zone = int((lon + 180) // 6) + 1
    return f"EPSG:{(32600 if lat >= 0 else 32700) + zone}"


def _season_col(col: "ee.ImageCollection", season: str) -> "ee.ImageCollection":
    return col.filter(ee.Filter.Or(*[ee.Filter.date(a, b) for a, b in SEASONS[season]]))


def s2_features(aoi: "ee.Geometry") -> "ee.Image":
    def mask(img):
        qa = img.select("QA60")
        clear = qa.bitwiseAnd(1 << 10).eq(0).And(qa.bitwiseAnd(1 << 11).eq(0))
        return img.updateMask(clear).divide(10000)

    col = (ee.ImageCollection("COPERNICUS/S2_HARMONIZED")  # TOA (L1C)
           .filterBounds(aoi).filterDate(START, END)
           .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 60)))
    imgs = []
    for season in SEASONS:
        # filtro de data ANTES do map: divide() no mask descarta system:time_start
        med = _season_col(col, season).map(mask).median()
        ndvi = med.normalizedDifference(["B8", "B4"]).rename("NDVI")
        ndwi = med.normalizedDifference(["B3", "B8"]).rename("NDWI")
        evi = med.expression("2.5*((N-R)/(N+6*R-7.5*B+1))",
                             {"N": med.select("B8"), "R": med.select("B4"),
                              "B": med.select("B2")}).rename("EVI")
        img = med.select(["B2", "B3", "B4", "B8", "B11"]).addBands([ndvi, ndwi, evi])
        names = [f"{b}_{season}" for b in ["B2", "B3", "B4", "B8", "B11", "NDVI", "NDWI", "EVI"]]
        imgs.append(img.rename(names))
    return ee.Image.cat(imgs).clip(aoi)


def s1_features(aoi: "ee.Geometry") -> "ee.Image":
    col = (ee.ImageCollection("COPERNICUS/S1_GRD")
           .filterBounds(aoi).filterDate(START, END)
           .filter(ee.Filter.eq("instrumentMode", "IW"))
           .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
           .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
           .select(["VV", "VH"]))
    imgs = []
    for season in SEASONS:
        sc = _season_col(col, season)
        med = sc.median().rename([f"VV_med_{season}", f"VH_med_{season}"])
        sd = sc.reduce(ee.Reducer.stdDev()).rename([f"VV_sd_{season}", f"VH_sd_{season}"])
        imgs.append(med.addBands(sd))
    return ee.Image.cat(imgs).clip(aoi)


def ae_features(aoi: "ee.Geometry") -> "ee.Image":
    return (ee.ImageCollection("GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL")
            .filterBounds(aoi).filterDate(START, f"{YEAR + 1}-01-01").mosaic().clip(aoi))


SOURCES = {"s2": s2_features, "s1": s1_features, "ae": ae_features}


def raster_valid(path, source: str) -> tuple[bool, str]:
    try:
        with rasterio.open(path) as ds:
            if ds.count != EXPECTED_BANDS[source]:
                return False, f"{ds.count} bandas; esperado {EXPECTED_BANDS[source]}"
            return True, "ok"
    except Exception as e:  # noqa: BLE001
        return False, str(e)


def raster_grid(path):
    with rasterio.open(path) as ds:
        return ds.width, ds.height, ds.crs.to_string() if ds.crs else None, tuple(round(x, 9) for x in ds.transform[:6])


def validate_tile_rasters(tile_ids: list[str]) -> list[str]:
    errors = []
    for tid in tile_ids:
        grids = {}
        for source in SOURCE_NAMES:
            path = raster_path(tid, source)
            if not path.exists():
                errors.append(f"{tid} {source}: ausente ({path})")
                continue
            ok, why = raster_valid(path, source)
            if not ok:
                errors.append(f"{tid} {source}: invalido ({why})")
                continue
            grids[source] = raster_grid(path)
        if len(grids) == len(SOURCE_NAMES) and len(set(grids.values())) != 1:
            errors.append(f"{tid}: grades S1/S2/AE divergentes")
    return errors


def _fetch(image: "ee.Image", bands: list, aoi: "ee.Geometry", crs: str):
    url = image.select(bands).getDownloadURL(
        {"scale": SCALE, "region": aoi, "format": "GEO_TIFF", "crs": crs})
    r = requests.get(url, timeout=300)
    r.raise_for_status()
    with rasterio.open(io.BytesIO(r.content)) as ds:
        return ds.read().astype("float32"), ds.profile


def download_source(tid: str, name: str, aoi: "ee.Geometry", crs: str) -> str:
    out = raster_path(tid, name)
    if out.exists():
        ok, why = raster_valid(out, name)
        if ok:
            return f"{tid} {name} (existe)"
        print(f"  {tid} {name}: cache invalido sera substituido ({why})", flush=True)
    try:
        img = SOURCES[name](aoi)
        bands = img.bandNames().getInfo()
        arrs, profile = [], None
        for i in range(0, len(bands), CHUNK):       # chunk de bandas (AlphaEarth > 48 MB)
            arr, prof = _fetch(img, bands[i:i + CHUNK], aoi, crs)
            arrs.append(arr)
            profile = profile or prof
        data = np.concatenate(arrs, axis=0)
        profile.update(count=data.shape[0], dtype="float32", compress="lzw", tiled=True)
        with atomic_output(out) as tmp_out:
            with rasterio.open(tmp_out, "w", **profile) as dst:
                dst.write(data)
                for j, b in enumerate(bands, start=1):
                    dst.set_band_description(j, b)
        return f"{tid} {name} OK ({out.stat().st_size / 1e6:.1f} MB)"
    except Exception as e:  # noqa: BLE001 — log e segue (1 fonte ruim não derruba o lote)
        return f"{tid} {name} ERRO: {str(e)[:80]}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="nº máx de tiles (0 = todos)")
    ap.add_argument("--allow-partial", action="store_true",
                    help="mantem exit code 0 mesmo se algum download falhar")
    args = ap.parse_args()
    check_disk()
    ee.Initialize(project=os.environ.get("EE_PROJECT"))
    RASTER_DIR.mkdir(parents=True, exist_ok=True)

    tiles = json.loads(TILES.read_text())["features"]
    if args.limit:
        tiles = tiles[:args.limit]
    tile_ids = [ft["properties"]["tile_id"] for ft in tiles]
    def tile_crs(ft: dict) -> str:
        coords = ft["geometry"]["coordinates"][0]
        lon = sum(c[0] for c in coords) / len(coords)
        lat = sum(c[1] for c in coords) / len(coords)
        return utm_epsg(lon, lat)

    tasks = [(ft["properties"]["tile_id"], name, ee.Geometry(ft["geometry"]), tile_crs(ft))
             for ft in tiles for name in SOURCE_NAMES]
    errors = []
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = [pool.submit(download_source, *t) for t in tasks]
        for fut in as_completed(futures):
            msg = fut.result()
            print(f"  {msg}")
            if " ERRO:" in msg:
                errors.append(msg)
    if errors and not args.allow_partial:
        raise SystemExit(f"{len(errors)} downloads falharam; use --allow-partial para continuar mesmo assim")
    validation_errors = validate_tile_rasters(tile_ids)
    if validation_errors:
        preview = "\n  ".join(validation_errors[:40])
        extra = f"\n  ... +{len(validation_errors) - 40}" if len(validation_errors) > 40 else ""
        msg = f"rasters invalidos:\n  {preview}{extra}"
        if args.allow_partial:
            print(msg)
        else:
            raise SystemExit(msg)


if __name__ == "__main__":
    main()
