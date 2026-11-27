#!/usr/bin/env python3
"""Passo 7 — figuras de contexto do manuscrito (não derivadas de métricas).

Gera em manuscript/figures/:
- fig_study_area.png — mapa da área de estudo: (a) densidade hex-binned de
  parcelas de referência com a grade de 0.5° clipada a MG, as células de 0.5° tocadas
  pelos 80 tiles originais (excluídas do sorteio confirmatório), os 80 tiles originais
  por partição train/val/test e os candidatos confirmatórios
  (docs/protocol/confirm_tiles.geojson; com data/confirm/tiles/split.json, "test" cheio e
  não usados vazados); (b) histograma da área das parcelas.
Sem painel de localização no Brasil: nenhum painel excede a extensão de MG (exigência
de jurisdictional claims da Elsevier).

Contorno de MG em code/assets/: API de malhas v3 do IBGE (qualidade "minima",
estado 31 = MG), obtidos em 2026-07-05; dados públicos do IBGE, CRS EPSG:4674
(graus, compatível com os tiles).

Uso: python code/07_context_figures.py
"""
import json
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import PathPatch, Rectangle
from matplotlib.path import Path as MplPath
from matplotlib.ticker import FuncFormatter

from figstyle import AXIS, FULL_W, GRID, INK, INK2, MUTED, SPLIT_COLORS, SPLIT_LABELS, save
from paths import CONFIRM, REF, ROOT, SPLIT, TILES

ASSETS = Path(__file__).resolve().parent / "assets"
CONFIRM_LIST = ROOT / "docs" / "protocol" / "confirm_tiles.geojson"
HIST_GREY = "#b3b1ab"
CELL_FILL = "#dcdad3"


def panel_label(ax, letter: str) -> None:
    ax.text(0.0, 1.0, letter, transform=ax.transAxes, fontsize=8,
            fontweight="bold", va="bottom", ha="left", color=INK)


def original_cells(tiles: gpd.GeoDataFrame) -> set[tuple[float, float]]:
    """Células de 0.5° (canto SW) intersectadas por algum tile original (§7 do protocolo)."""
    cells = set()
    for minx, miny, maxx, maxy in tiles.geometry.bounds.itertuples(index=False):
        for x in np.arange(np.floor(minx * 2) / 2, maxx, 0.5):
            for y in np.arange(np.floor(miny * 2) / 2, maxy, 0.5):
                cells.add((float(x), float(y)))
    return cells


def centroids(gdf: gpd.GeoDataFrame) -> tuple[np.ndarray, np.ndarray]:
    b = gdf.geometry.bounds
    return ((b["minx"] + b["maxx"]) / 2).to_numpy(), ((b["miny"] + b["maxy"]) / 2).to_numpy()


def draw_confirm(ax) -> None:
    conf = gpd.read_file(CONFIRM_LIST)
    cx, cy = centroids(conf)
    split = json.loads(CONFIRM.split.read_text()) if CONFIRM.split.exists() else None
    kw = dict(s=20, marker="D", linewidths=0.8, zorder=5)
    if split is None:
        ax.scatter(cx, cy, facecolors="white", edgecolors=INK, **kw,
                   label=f"confirmatory candidates (n={len(conf)})")
        return
    used = conf["tile_id"].map(split).eq("test").to_numpy()
    ax.scatter(cx[used], cy[used], color=INK, edgecolors="white", **kw,
               label=f"confirmatory test (n={int(used.sum())})")
    ax.scatter(cx[~used], cy[~used], facecolors="white", edgecolors=MUTED, **kw,
               label=f"candidates not used (n={int((~used).sum())})")


def geometry_patch(geoseries, ax) -> PathPatch:
    geom = geoseries.union_all()
    polys = geom.geoms if geom.geom_type == "MultiPolygon" else [geom]
    paths = []
    for poly in polys:
        paths.append(MplPath(np.asarray(poly.exterior.coords)))
        paths.extend(MplPath(np.asarray(ring.coords)) for ring in poly.interiors)
    patch = PathPatch(MplPath.make_compound_path(*paths), transform=ax.transData,
                      facecolor="none", edgecolor="none")
    ax.add_patch(patch)
    return patch


def fig_study_area() -> None:
    tiles = gpd.read_file(TILES)
    split = json.loads(Path(SPLIT).read_text())
    tiles["split"] = tiles["tile_id"].map(split)
    parcels = pd.read_parquet(REF, columns=["lon", "lat", "area_ha"])
    mg = gpd.read_file(ASSETS / "ibge_mg_uf.geojson")

    fig = plt.figure(figsize=(FULL_W, 4.4))
    gs = fig.add_gridspec(2, 2, width_ratios=(2.5, 1.0), height_ratios=(0.85, 1.15),
                          wspace=0.18, hspace=0.34)
    ax = fig.add_subplot(gs[:, 0])
    ax_leg = fig.add_subplot(gs[0, 1])
    ax_leg.set_axis_off()
    ax_hist = fig.add_subplot(gs[1, 1])

    # (a) mapa principal
    minx, miny, maxx, maxy = mg.total_bounds
    pad = 0.35
    ax.set_xlim(minx - pad, maxx + pad)
    ax.set_ylim(miny - pad, maxy + pad)
    ax.set_aspect("equal")

    mg_clip = geometry_patch(mg, ax)
    for lon in np.arange(np.floor(minx * 2) / 2, maxx + 0.5, 0.5):
        ax.axvline(lon, color=GRID, linewidth=0.4, zorder=1).set_clip_path(mg_clip)
    for lat in np.arange(np.floor(miny * 2) / 2, maxy + 0.5, 0.5):
        ax.axhline(lat, color=GRID, linewidth=0.4, zorder=1).set_clip_path(mg_clip)

    cells = original_cells(tiles)
    for x, y in cells:
        ax.add_patch(Rectangle((x, y), 0.5, 0.5, facecolor=CELL_FILL, edgecolor="none",
                               zorder=1.5))
    hb = ax.hexbin(parcels["lon"], parcels["lat"], gridsize=70, cmap="Purples",
                   bins="log", mincnt=1, linewidths=0.0, zorder=2)
    mg.boundary.plot(ax=ax, color=INK2, linewidth=0.9, zorder=3)

    cx, cy = centroids(tiles)
    for part in ("train", "val", "test"):
        sel = (tiles["split"] == part).to_numpy()
        ax.scatter(cx[sel], cy[sel], s=20, marker="s",
                   color=SPLIT_COLORS[part], edgecolors="white", linewidths=0.6,
                   zorder=4, label=f"original {SPLIT_LABELS[part]} (n={int(sel.sum())})")
    draw_confirm(ax)
    handles, labels = ax.get_legend_handles_labels()
    handles.append(Rectangle((0, 0), 1, 1, facecolor=CELL_FILL, edgecolor="none"))
    labels.append(f"0.5° cells with original tiles (n={len(cells)})")
    ax_leg.legend(handles, labels, loc="upper left", bbox_to_anchor=(-0.08, 1.02),
                  frameon=False, fontsize=6.8, handletextpad=0.4, labelspacing=0.45,
                  borderaxespad=0.0)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{abs(v):g}°W"))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{abs(v):g}°S"))
    ax.tick_params(labelsize=7)

    cax = ax.inset_axes([0.03, 0.085, 0.30, 0.028])
    cbar = fig.colorbar(hb, cax=cax, orientation="horizontal", ticks=[1, 10, 100, 1000])
    cax.xaxis.set_label_position("top")
    cax.set_xlabel("reference parcels\nper hex bin", fontsize=7, color=INK2,
                   labelpad=3, loc="left", ma="left")
    cax.set_xticklabels(["1", "10", "100", "1000"])
    cbar.ax.tick_params(labelsize=6.5, colors=MUTED)
    cbar.ax.minorticks_off()
    cbar.outline.set_edgecolor(AXIS)
    cbar.outline.set_linewidth(0.6)
    panel_label(ax, "(a)")

    # (b) histograma de área — distribuição completa (modo de slivers visível),
    # corte de 1 px marcado e mediana dos talhões avaliados (≥ 1 px)
    kept = parcels.loc[parcels["area_ha"] >= 0.01, "area_ha"]
    med = float(kept.median())
    bins = np.logspace(np.log10(max(parcels["area_ha"].min(), 1e-3)),
                       np.log10(parcels["area_ha"].max()), 60)
    ax_hist.hist(parcels["area_ha"], bins=bins, color=HIST_GREY, edgecolor="none")
    ax_hist.set_xscale("log")
    ax_hist.axvline(0.01, color=INK2, linewidth=0.9, linestyle=(0, (3, 2)))
    ax_hist.text(0.01 * 0.72, ax_hist.get_ylim()[1] * 0.96,
                 "1-px cut (slivers excluded)", fontsize=6.5, color=INK,
                 rotation=90, ha="right", va="top")
    ax_hist.axvline(med, color=INK, linewidth=1.0)
    ax_hist.text(med * 1.5, ax_hist.get_ylim()[1] * 0.95,
                 f"median {med:.2f} ha\n(parcels ≥ 1 px)", fontsize=7.5, color=INK, va="top")
    ax_hist.set_xlabel("parcel area (ha, log scale)", fontsize=7.5)
    ax_hist.set_ylabel("parcels", fontsize=7.5, labelpad=1)
    ax_hist.yaxis.set_major_formatter(
        FuncFormatter(lambda v, _: f"{v / 1000:g}k" if v >= 1000 else f"{v:g}"))
    ax_hist.tick_params(labelsize=7)
    for side in ("top", "right"):
        ax_hist.spines[side].set_visible(False)
    panel_label(ax_hist, "(b)")

    save(fig, "fig_study_area")


def main() -> None:
    fig_study_area()


if __name__ == "__main__":
    main()
