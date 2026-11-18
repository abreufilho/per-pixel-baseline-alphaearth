#!/usr/bin/env python3
"""Passo 2 — amostra N tiles de avaliação (default 20) de PX×PX px, estratificados.

Por que tiles: não dá pra segmentar MG inteira. Avaliamos o método em janelas pequenas onde
conhecemos a verdade (polígonos CONAB de café). Os tiles são ESTRATIFICADOS (região × tamanho ×
densidade dos talhões) p/ o resultado não enviesar por uma condição só. Os tiles são os LOCAIS
DE TESTE, não o resultado.

Entrada: data/reference/ref_cafe_mg_2017.parquet (passo 1).
Saída: data/tiles/tiles_eval.geojson (não-versionado).

Uso: python code/02_tiles.py [--n N] [--px PX]   (default: N=20 tiles, PX=500 → 5 km @ 10 m)
"""
import argparse

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import box

from paths import REF, TILES, atomic_output

SCALE_M = 10
MIN_PARCELS = 10


def load_ref() -> gpd.GeoDataFrame:
    df = pd.read_parquet(REF)
    return gpd.GeoDataFrame(df, geometry=gpd.GeoSeries.from_wkb(df.geom_wkb), crs=4674)


def candidate_tiles(ref: gpd.GeoDataFrame, px: int) -> tuple[gpd.GeoDataFrame, pd.DataFrame]:
    """Grade de tiles px×px sobre a referência e estratos dos tiles com >= MIN_PARCELS."""
    tile_deg = px * SCALE_M / 111320.0  # ~0.045° p/ 500 px
    minx, miny, maxx, maxy = ref.total_bounds
    cells = [box(x, y, x + tile_deg, y + tile_deg)
             for x in np.arange(minx, maxx, tile_deg)
             for y in np.arange(miny, maxy, tile_deg)]
    grid = gpd.GeoDataFrame({"tile": range(len(cells))}, geometry=cells, crs=ref.crs)

    joined = gpd.sjoin(ref[["area_ha", "region_cell", "geometry"]], grid, predicate="within")
    agg = joined.groupby("tile").agg(
        n_parcels=("area_ha", "size"),
        med_area=("area_ha", "median"),
        region=("region_cell", lambda s: s.mode().iat[0]),
    )
    agg = agg[agg.n_parcels >= MIN_PARCELS]
    agg["size_class"] = np.where(agg.med_area < 1.0, "small",
                                 np.where(agg.med_area < 3.0, "mid", "large"))
    agg["dens_class"] = np.where(agg.n_parcels < 50, "low",
                                 np.where(agg.n_parcels < 200, "mid", "high"))
    agg["stratum"] = agg.region.astype(str) + "|" + agg.size_class + "|" + agg.dens_class
    return grid, agg


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20, help="nº de tiles (default 20)")
    ap.add_argument("--px", type=int, default=500, help="lado do tile em px (default 500 → 5 km)")
    args = ap.parse_args()
    tile_deg = args.px * SCALE_M / 111320.0

    grid, agg = candidate_tiles(load_ref(), args.px)
    if agg.empty:
        print("· nenhum tile com café suficiente (rode o passo 1 completo)")
        return

    one = agg.sample(frac=1, random_state=42).groupby("stratum").head(1)  # espalha por estrato
    if len(one) >= args.n:
        sample = one.sample(args.n, random_state=42)
    else:
        rest = agg.drop(one.index)
        extra = rest.sample(min(args.n - len(one), len(rest)), random_state=42)
        sample = pd.concat([one, extra])

    out = grid.set_index("tile").loc[sample.index].join(sample)
    out["tile_id"] = [f"t{i:02d}" for i in range(len(out))]
    with atomic_output(TILES) as tmp_tiles:
        out.to_file(tmp_tiles, driver="GeoJSON")
    print(f"✓ {TILES}")
    print(f"  {len(out)} tiles de {args.px}px (~{tile_deg * 111.32:.1f} km) · "
          f"{out.stratum.nunique()} estratos")
    print(out[["tile_id", "region", "size_class", "dens_class", "n_parcels"]].to_string(index=False))


if __name__ == "__main__":
    main()
