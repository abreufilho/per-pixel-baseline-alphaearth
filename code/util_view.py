#!/usr/bin/env python3
"""Utilitário — exporta versão VISÍVEL (QGIS) da referência de café e dos tiles.

A referência fica em GeoParquet WKB (não abre direto no QGIS). Aqui geramos GPKG:
  data/view/cafe_in_tiles.gpkg  — talhões de café DENTRO dos tiles de avaliação
  data/view/tiles.gpkg          — os tiles de avaliação

Uso: python code/util_view.py
"""
import geopandas as gpd
import pandas as pd

from paths import REF, TILES, VIEW_DIR, atomic_output


def load_ref() -> gpd.GeoDataFrame:
    df = pd.read_parquet(REF)
    return gpd.GeoDataFrame(df.drop(columns=["geom_wkb"]),
                            geometry=gpd.GeoSeries.from_wkb(df.geom_wkb), crs=4674)


def main() -> None:
    VIEW_DIR.mkdir(parents=True, exist_ok=True)
    ref = load_ref()
    tiles = gpd.read_file(TILES).to_crs(ref.crs)

    inside = gpd.sjoin(ref, tiles[["tile_id", "geometry"]], predicate="within")
    inside = inside.drop(columns=[c for c in ("index_right",) if c in inside.columns])
    with atomic_output(VIEW_DIR / "cafe_in_tiles.gpkg") as tmp_view:
        inside.to_file(tmp_view, driver="GPKG")
    with atomic_output(VIEW_DIR / "tiles.gpkg") as tmp_tiles:
        tiles.to_file(tmp_tiles, driver="GPKG")

    print(f"✓ {VIEW_DIR}/cafe_in_tiles.gpkg — {len(inside)} talhões de café (dentro dos {len(tiles)} tiles)")
    print(f"✓ {VIEW_DIR}/tiles.gpkg — {len(tiles)} tiles")
    print("  abra os dois no QGIS (cafe_in_tiles sobre tiles).")


if __name__ == "__main__":
    main()
