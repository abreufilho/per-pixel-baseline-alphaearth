#!/usr/bin/env python3
"""Passo 1 — extrai a referência (café MG 2017) do fiboa br_conab → GeoParquet local.

Adiciona área (da geometria; metrics:area é nulo), centróide e um rótulo de região-proxy
(célula de 0,5° do centróide) p/ a amostragem estratificada do passo 2.
Saída: data/reference/ref_cafe_mg_2017.parquet (não-versionado).

Uso: python code/01_extract_reference.py [LIMIT]   (LIMIT opcional p/ teste)
"""
import sys

import duckdb

from paths import REF, REFERENCE_DIR, atomic_output

URL = "https://data.source.coop/fiboa/data/br_conab/br_conab.parquet"


def main() -> None:
    limit = f"LIMIT {int(sys.argv[1])}" if len(sys.argv) > 1 else ""
    REFERENCE_DIR.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    con.sql("INSTALL httpfs; LOAD httpfs; INSTALL spatial; LOAD spatial;")
    with atomic_output(REF) as tmp_ref:
        con.sql(f"""
            COPY (
                SELECT
                    id,
                    ST_Area_Spheroid(geometry) / 10000.0 AS area_ha,
                    ST_X(ST_Centroid(geometry)) AS lon,
                    ST_Y(ST_Centroid(geometry)) AS lat,
                    CAST(floor(ST_Y(ST_Centroid(geometry)) * 2) / 2 AS VARCHAR) || '_' ||
                    CAST(floor(ST_X(ST_Centroid(geometry)) * 2) / 2 AS VARCHAR) AS region_cell,
                    ST_AsWKB(geometry) AS geom_wkb
                FROM read_parquet('{URL}')
                WHERE id LIKE 'MG\\_CAFE\\_%' ESCAPE '\\'
                {limit}
            ) TO '{tmp_ref}' (FORMAT PARQUET);
        """)
        n, regions = con.sql(
            f"SELECT count(*), count(DISTINCT region_cell) FROM read_parquet('{tmp_ref}')"
        ).fetchone()
        stats = con.sql(
            f"SELECT round(min(area_ha),2) min, round(median(area_ha),2) med, "
            f"round(max(area_ha),2) max FROM read_parquet('{tmp_ref}')"
        ).df().to_string(index=False)
    print(f"✓ {REF}")
    print(f"  {n} talhões de café · {regions} células de região (0,5°)")
    print(stats)


if __name__ == "__main__":
    main()
