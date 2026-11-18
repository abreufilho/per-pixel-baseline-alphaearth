#!/usr/bin/env python3
"""Passo 8 — figura pictórica do pipeline (fig_pipeline.png).

Desenho experimental com artefatos reais do tile qualitativo (06_figures.load_scene: tile
original de diferença AP E3c−E1c mediana): referência, labels esparsos, mapa dos tiles
(originais por partição + candidatos confirmatórios), representações cruas E1–E4 (pilhas,
1 lâmina = 8 bandas), passo de filtros de janela com miniaturas reais de uma banda S2
(NDVI seca), probe idêntico, P(fronteira) de E3c, os dois decodificadores (watershed e
fusão de regiões, limiares congelados na validação) e os dois conjuntos de avaliação.

Uso: python code/08_pipeline_figure.py
"""
import importlib
import json

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from matplotlib.colors import ListedColormap
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
from scipy import ndimage as ndi

import objects
from figstyle import (
    AXIS, CAND_EDGE, EXP_FILLS, FULL_W, INK, INK2, MUTED, PLAIN_BG, PROB_CMAP, REF_RED,
    SPLIT_COLORS, add_scale_bar, boundary_mask, communication_crop, draw_contour_mask, save,
)
from paths import CONFIRM, ROOT, SPLIT, TILES, layout, scoped_metrics_dir

m06 = importlib.import_module("06_figures")
EXP_COLORS = m06.EXP_COLORS
ASSETS = ROOT / "code" / "assets"
CONFIRM_LIST = ROOT / "docs" / "protocol" / "confirm_tiles.geojson"
BANDS = {"E1": 16, "E2": 24, "E3": 64, "E4": 88}
BANDS_PER_SLAB = 8
BOX_FILL = "#f0efe9"
FIG_W, FIG_H = FULL_W, 7.0
S2_BAND = 13  # NDVI_dry


def img_axes(fig, cx: float, y0: float, h: float):
    w = h * FIG_H / FIG_W
    ax = fig.add_axes([cx - w / 2, y0, w, h])
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_edgecolor(AXIS)
        s.set_linewidth(0.6)
    return ax


def half_w(h: float) -> float:
    return h * FIG_H / FIG_W / 2


def arrow(ax, a, b, color: str = INK2, lw: float = 1.0, rad: float = 0.0) -> None:
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=8, color=color, lw=lw,
                                 shrinkA=0, shrinkB=0, connectionstyle=f"arc3,rad={rad}",
                                 zorder=6))


def stage_box(ax, x0, y0, w, h, lines, fill: str = BOX_FILL, edge: str = INK2,
              ls="-") -> None:
    ax.add_patch(FancyBboxPatch((x0, y0), w, h, boxstyle="round,pad=0.005", facecolor=fill,
                                edgecolor=edge, linewidth=0.9, linestyle=ls, zorder=4))
    for i, (text, kw) in enumerate(lines):
        ax.text(x0 + w / 2, y0 + h * (len(lines) - i - 0.5) / len(lines), text, ha="center",
                va="center", zorder=5, **kw)


def title(ax, x, y, text, sub=None, ha="center") -> None:
    ax.text(x, y, text, ha=ha, fontsize=7.2, fontweight="bold", color=INK)
    if sub:
        ax.text(x, y - 0.008, sub, ha=ha, va="top", fontsize=6.2, color=INK2, linespacing=1.3)


def band_stack(ax, cx, y_base, exp, slab_w=0.062, slab_h=0.010, dy=0.0062, dx=0.0016) -> None:
    parts = [("E2", 3), ("E3", 8)] if exp == "E4" else [(exp, BANDS[exp] // BANDS_PER_SLAB)]
    n = sum(k for _, k in parts)
    x0 = cx - (slab_w + dx * (n - 1)) / 2
    i = 0
    for src, k in parts:
        for _ in range(k):
            ax.add_patch(Rectangle((x0 + i * dx, y_base + i * dy), slab_w, slab_h,
                                   facecolor=EXP_FILLS[src], edgecolor=EXP_COLORS[src],
                                   linewidth=0.6, zorder=3 + i * 0.001))
            i += 1


def draw_tile_map(ax) -> None:
    mg = gpd.read_file(ASSETS / "ibge_mg_uf.geojson")
    mg.boundary.plot(ax=ax, color=MUTED, linewidth=0.5)
    tiles = gpd.read_file(TILES)
    split = json.loads(SPLIT.read_text())
    b = tiles.geometry.bounds
    cx, cy = (b.minx + b.maxx) / 2, (b.miny + b.maxy) / 2
    for part in ("train", "val", "test"):
        sel = (tiles.tile_id.map(split) == part).to_numpy()
        ax.scatter(cx[sel], cy[sel], s=5, marker="s", color=SPLIT_COLORS[part],
                   edgecolors="white", linewidths=0.25, zorder=3)
    conf = gpd.read_file(CONFIRM_LIST)
    if CONFIRM.split.exists():
        csplit = json.loads(CONFIRM.split.read_text())
        conf = conf[conf.tile_id.map(csplit) == "test"]
    b = conf.geometry.bounds
    ax.scatter((b.minx + b.maxx) / 2, (b.miny + b.maxy) / 2, s=6, marker="D",
               facecolors="white", edgecolors=INK, linewidths=0.5, zorder=4)
    ax.set_aspect("equal")
    ax.set_axis_off()


def window_features(band: np.ndarray) -> list[tuple[str, np.ndarray]]:
    x = np.nan_to_num(band.astype("float64"))

    def sd(size):
        m = ndi.uniform_filter(x, size)
        return np.sqrt(np.clip(ndi.uniform_filter(x * x, size) - m * m, 0, None))

    mean3 = ndi.uniform_filter(x, 3)
    sobel = np.hypot(ndi.sobel(mean3, 0), ndi.sobel(mean3, 1))
    return [("raw", x), ("mean 3×3", mean3), ("SD 3×3", sd(3)), ("SD 5×5", sd(5)),
            ("Sobel of\nmean 3×3", sobel)]


def show(ax, img: np.ndarray, cmap="Greys_r") -> None:
    lo, hi = np.percentile(img, (2, 98))
    ax.imshow(img, cmap=cmap, vmin=lo, vmax=hi, interpolation="nearest")


def objects_view(ax, labels: np.ndarray) -> None:
    tones = ListedColormap(["#f7f6f2", "#edebe5", "#e3e1da", "#d9d7cf"])
    perm = np.random.default_rng(0).permutation(int(labels.max()) + 1)
    ax.imshow(perm[labels] % 4, cmap=tones, vmin=0, vmax=3, interpolation="nearest")
    draw_contour_mask(ax, boundary_mask(labels), CAND_EDGE, linewidth=0.35, alpha=0.85)


def main() -> None:
    sc = m06.load_scene(exps=("E3c",))
    rs, cs, _ = communication_crop(sc["lab"])
    lab, prob, rag = sc["lab"][rs, cs], sc["probs"]["E3c"][rs, cs], sc["rag"]["E3c"][rs, cs]
    thr_ws = json.loads((scoped_metrics_dir("original") / "thresholds_ws_rf.json").read_text())
    ws = objects.watershed_decoder(sc["probs"]["E3c"], thr_ws["E3c"])[rs, cs]
    with rasterio.open(layout("original").raster(sc["tid"], "s2")) as ds:
        s2 = ds.read(S2_BAND + 1)[:sc["lab"].shape[0], :sc["lab"].shape[1]][rs, cs][2:-2, 2:-2]

    fig = plt.figure(figsize=(FIG_W, FIG_H))
    bg = fig.add_axes([0, 0, 1, 1])
    bg.set_xlim(0, 1)
    bg.set_ylim(0, 1)
    bg.axis("off")

    # 1 — referência → labels → tiles
    th, ty = 0.14, 0.81
    xs = (0.22, 0.50, 0.78)
    ax = img_axes(fig, xs[0], ty, th)
    ax.imshow(np.ones_like(lab), cmap=ListedColormap([PLAIN_BG]), vmin=0, vmax=1)
    draw_contour_mask(ax, lab == 2, REF_RED, linewidth=0.5)
    add_scale_bar(ax, sc["pixel_m"], lab.shape[1], lab.shape[0], fontsize=5.8)
    ax = img_axes(fig, xs[1], ty, th)
    ax.imshow(np.asarray(ListedColormap([PLAIN_BG, "#cfcdc6", REF_RED])(lab)))
    draw_tile_map(img_axes(fig, xs[2], ty, th))
    title(bg, xs[0], ty - 0.022, "Reference coffee parcels",
          "CONAB, Minas Gerais 2017\n384,496 polygons")
    title(bg, xs[1], ty - 0.022, "Sparse raster labels",
          "red: inner 10 m boundary band\ngrey: parcel interior")
    title(bg, xs[2], ty - 0.022, "Tiles and spatial split",
          "■ original: 43 train / 13 val / 24 test\n"
          "◇ confirmatory: 48 test tiles, unused cells")
    for a, b in zip(xs[:-1], xs[1:]):
        arrow(bg, (a + half_w(th) + 0.012, ty + th / 2), (b - half_w(th) - 0.012, ty + th / 2))

    # 2 — representações cruas → filtros de janela
    c0, c1 = 0.425, 0.715
    bg.add_patch(Rectangle((0.012, c0), 0.976, c1 - c0, facecolor="none", edgecolor=AXIS,
                           linewidth=0.7, linestyle=(0, (4, 3)), zorder=1))
    arrow(bg, (0.5, ty - 0.066), (0.5, c1 + 0.003))
    bg.text(0.026, c1 - 0.022, "Representations — same pixels, same labels", fontsize=7.2,
            fontweight="bold", color=INK)
    bg.text(0.026, c1 - 0.042, "one slab = 8 bands", fontsize=5.8, color=MUTED, style="italic")
    stack_x = {"E1": 0.07, "E2": 0.16, "E3": 0.25, "E4": 0.34}
    sub = {"E1": "S2\nseasonal", "E2": "S1+S2\nseasonal", "E3": "AlphaEarth\nannual",
           "E4": "E2 + E3"}
    for e, x in stack_x.items():
        band_stack(bg, x, 0.515, e)
        bg.text(x, 0.495, e, ha="center", fontsize=7, fontweight="bold", color=INK)
        bg.text(x, 0.482, f"{sub[e]}\n{BANDS[e]} bands", ha="center", va="top", fontsize=5.6,
                color=INK2, linespacing=1.15)
    feats = window_features(s2)
    fx = np.linspace(0.475, 0.795, len(feats))
    fh, fy = 0.075, 0.535
    bg.add_patch(FancyBboxPatch((0.43, 0.46), 0.41, 0.20, boxstyle="round,pad=0.005",
                                facecolor="white", edgecolor=INK2, linewidth=0.8, zorder=2))
    bg.text(0.635, 0.640, "+ identical neighbourhood (window) features, every band",
            ha="center", fontsize=6.6, fontweight="bold", color=INK, zorder=5)
    for (name, img), x in zip(feats, fx):
        show(img_axes(fig, x, fy, fh), img)
        bg.text(x, fy - 0.010, name, ha="center", va="top", fontsize=5.8, color=INK2,
                linespacing=1.1, zorder=5)
    bg.text(0.635, 0.468, "example: one Sentinel-2 band (dry-season NDVI)", ha="center",
            fontsize=5.6, color=MUTED, style="italic", zorder=5)
    arrow(bg, (0.392, 0.56), (0.425, 0.56))
    arrow(bg, (0.845, 0.56), (0.872, 0.56))
    bg.text(0.925, 0.60, "E1c–E4c", ha="center", fontsize=7.2, fontweight="bold", color=INK)
    bg.text(0.925, 0.585, "5 features\nper band\n\n80 / 120 /\n320 / 440\nfeatures",
            ha="center", va="top", fontsize=5.8, color=INK2, linespacing=1.2)

    # 3 — probe
    py0, py1 = 0.345, 0.395
    arrow(bg, (0.5, c0 - 0.003), (0.5, py1 + 0.006))
    stage_box(bg, 0.22, py0, 0.56, py1 - py0, [
        ("Identical probe for every input: Random Forest (+ linear control)",
         dict(fontsize=7, fontweight="bold", color=INK)),
        ("trained once on the 43 original training tiles · same pixels · no retraining",
         dict(fontsize=6, color=INK2, style="italic"))])

    # 4 — P(fronteira) → dois decodificadores
    qh, qy = 0.12, 0.16
    qx = (0.22, 0.56, 0.80)
    show_ax = img_axes(fig, qx[0], qy, qh)
    show_ax.imshow(prob, cmap=PROB_CMAP, vmin=0, vmax=1, interpolation="nearest")
    objects_view(img_axes(fig, qx[1], qy, qh), ws)
    objects_view(img_axes(fig, qx[2], qy, qh), rag)
    arrow(bg, (0.40, py0 - 0.006), (qx[0], qy + qh + 0.008), rad=-0.2)
    title(bg, qx[0], qy - 0.022, "P(boundary) per pixel", "illustrated with E3c")
    title(bg, qx[1], qy - 0.022, "Watershed", f"markers P < t\n(t = {thr_ws['E3c']:.2f})")
    title(bg, qx[2], qy - 0.022, "Region merging",
          f"RAG merge while boundary\nP < τ (τ = {sc['thr']['E3c']:.2f})")
    x_from = qx[0] + half_w(qh) + 0.01
    arrow(bg, (x_from, qy + qh * 0.5), (qx[1] - half_w(qh) - 0.01, qy + qh * 0.5))
    arrow(bg, (x_from, qy + qh * 0.9), (qx[2] - half_w(qh) - 0.01, qy + qh * 0.9), rad=-0.12)
    bg.text((x_from + qx[1] - half_w(qh)) / 2, qy + qh * 0.44, "thresholds frozen on\n13 "
            "original validation tiles", ha="center", va="top", fontsize=5.8, color=INK2,
            linespacing=1.2)

    # 5 — avaliação
    ey0, ey1 = 0.008, 0.068
    stage_box(bg, 0.06, ey0, 0.42, ey1 - ey0, [
        ("Original test tiles · n = 24 · post hoc", dict(fontsize=6.8, fontweight="bold",
                                                         color=INK)),
        ("Boundary AP, object metrics; Wilcoxon, HL + 95% CI", dict(fontsize=6, color=INK2))])
    stage_box(bg, 0.52, ey0, 0.42, ey1 - ey0, [
        ("Confirmatory tiles · n = 48 · pre-specified", dict(fontsize=6.8, fontweight="bold",
                                                            color=INK)),
        ("H1 non-superiority, H2, H3 TOST (δ = 0.03 AP)", dict(fontsize=6, color=INK2))],
        fill="white", ls=(0, (3, 2)))
    save(fig, "fig_pipeline")


if __name__ == "__main__":
    main()
