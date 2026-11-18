#!/usr/bin/env python3
"""Estilo, cores e utilitários compartilhados pelas figuras (06, 07, 08).

Cor segue a representação (E1 azul, E2 aqua, E3 âmbar, E4 verde); a variante com
filtros de janela (E1c–E4c) usa a mesma cor com marcador cheio, a crua marcador vazado.
Tamanhos IEEE: coluna simples 3.5 in, largura total 7.16 in; PNG a 600 dpi.
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import patheffects as pe  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
from scipy import ndimage as ndi  # noqa: E402

from paths import FIGURES_DIR, atomic_output  # noqa: E402

COL_W, FULL_W = 3.5, 7.16
DPI = 600

INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, BAND = "#e1e0d9", "#c3c2b7", "#f4f3ee"
REF_RED, CAND_EDGE = "#e34948", "#111111"
PROB_CMAP = "Blues"
PLAIN_BG = "#faf9f4"

BASE_COLORS = {"E1": "#2a78d6", "E2": "#1baf7a", "E3": "#eda100", "E4": "#008300"}
EXP_COLORS = {**BASE_COLORS, **{f"{k}c": v for k, v in BASE_COLORS.items()}}
EXP_FILLS = {"E1": "#eaf1fb", "E2": "#e6f6ef", "E3": "#fdf3dc", "E4": "#e4f2e4"}
SOURCE_LABELS = {"E1": "S2 seasonal", "E2": "S1+S2 seasonal", "E3": "AlphaEarth",
                 "E4": "S1+S2+AlphaEarth"}
EXP_LABELS = {**{k: f"{k} ({v})" for k, v in SOURCE_LABELS.items()},
              **{f"{k}c": f"{k}c ({v} + windows)" for k, v in SOURCE_LABELS.items()}}
SPLIT_COLORS = {"train": "#2a78d6", "val": "#eda100", "test": "#e34948"}
SPLIT_LABELS = {"train": "training", "val": "validation", "test": "test"}
CONFIRM_MARK = dict(marker="D", color=INK, edgecolor="white")

plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 7.5, "text.color": INK,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "axes.linewidth": 0.7,
    "axes.labelsize": 7.5, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "xtick.color": MUTED, "ytick.color": MUTED, "figure.facecolor": "white",
    "axes.facecolor": "white", "savefig.dpi": DPI, "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02, "legend.fontsize": 7,
})


def save(fig, name: str) -> None:
    with atomic_output(FIGURES_DIR / f"{name}.png") as tmp:
        fig.savefig(tmp, format="png")
    plt.close(fig)
    print(f"· {name}.png")


def panel_label(ax, text: str, fontsize: float = 8) -> None:
    ax.set_title(text, loc="left", fontsize=fontsize, fontweight="bold", color=INK, pad=3)


def clean_axes(ax, grid_axis: str = "x") -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def pending_panel(ax, text: str) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_edgecolor(AXIS)
        s.set_linestyle((0, (3, 2)))
    ax.text(0.5, 0.5, text, transform=ax.transAxes, ha="center", va="center",
            fontsize=7.5, color=MUTED, style="italic", linespacing=1.5)


def image_axes_off(ax) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def boundary_mask(labels: np.ndarray) -> np.ndarray:
    edge = np.zeros(labels.shape, dtype=bool)
    edge[:-1, :] |= labels[:-1, :] != labels[1:, :]
    edge[1:, :] |= labels[:-1, :] != labels[1:, :]
    edge[:, :-1] |= labels[:, :-1] != labels[:, 1:]
    edge[:, 1:] |= labels[:, :-1] != labels[:, 1:]
    return edge & (labels > 0)


def draw_contour_mask(ax, mask: np.ndarray, color: str, linewidth: float, alpha: float = 1.0,
                      halo: float = 0.0) -> None:
    if mask.any():
        cs = ax.contour(mask.astype(float), levels=[0.5], colors=[color], linewidths=linewidth,
                        alpha=alpha)
        if halo:
            cs.set_path_effects([pe.Stroke(linewidth=linewidth + halo, foreground="white"),
                                 pe.Normal()])


def add_scale_bar(ax, pixel_m: float, width_px: int, height_px: int, align: str = "right",
                  margin: float = 12, text_dy: float = 6, fontsize: float = 6.5) -> None:
    span = width_px * pixel_m
    bar_m = 500 if span >= 1800 else 250 if span >= 900 else 100
    bar_px = bar_m / pixel_m
    x0 = margin if align == "left" else width_px - bar_px - margin
    y0 = height_px - margin
    ax.plot([x0, x0 + bar_px], [y0, y0], color="white", linewidth=3.4, solid_capstyle="butt",
            zorder=8)
    ax.plot([x0, x0 + bar_px], [y0, y0], color=INK, linewidth=1.6, solid_capstyle="butt",
            zorder=9)
    ax.text(x0 + bar_px / 2, y0 - text_dy, f"{bar_m} m", ha="center", va="bottom",
            fontsize=fontsize, color=INK,
            path_effects=[pe.withStroke(linewidth=2.2, foreground="white")], zorder=10)


def communication_crop(lab: np.ndarray, size: int = 190) -> tuple[slice, slice, dict]:
    """Recorte derivado só da REFERÊNCIA (densidade de fronteira + footprint), neutro quanto
    às predições."""
    rows, cols = min(size, lab.shape[0]), min(size, lab.shape[1])
    signal = (lab == 2).astype(np.int32)
    integ = np.pad(signal, ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    best = None
    for y in range(0, lab.shape[0] - rows + 1, 35):
        for x in range(0, lab.shape[1] - cols + 1, 35):
            bpx = int(integ[y + rows, x + cols] - integ[y, x + cols] - integ[y + rows, x]
                      + integ[y, x])
            density = bpx / (rows * cols)
            if not 0.030 <= density <= 0.085:
                continue
            fpx = int((lab[y:y + rows, x:x + cols] > 0).sum())
            cand = (bpx + 0.08 * fpx - 22000 * abs(density - 0.055), y, x, bpx)
            if best is None or cand > best:
                best = cand
    if best is None:
        y, x = max((lab.shape[0] - rows) // 2, 0), max((lab.shape[1] - cols) // 2, 0)
        best = (0, y, x, int(signal[y:y + rows, x:x + cols].sum()))
    _, y, x, bpx = best
    return slice(y, y + rows), slice(x, x + cols), {"y": y, "x": x, "boundary_px": bpx}


def count_objects(labels: np.ndarray, min_px: int = 25) -> int:
    _, counts = np.unique(labels[labels > 0], return_counts=True)
    return int((counts >= min_px).sum())


def zoom_window(lab: np.ndarray, size: int = 64, step: int = 8) -> tuple[int, int, int]:
    """Janela de zoom só da REFERÊNCIA: máximo de talhões distintos, desempate por pixels de
    fronteira."""
    ref_objs = ndi.label(lab == 1)[0]
    best = None
    for y in range(0, lab.shape[0] - size + 1, step):
        for x in range(0, lab.shape[1] - size + 1, step):
            cand = (count_objects(ref_objs[y:y + size, x:x + size]),
                    int((lab[y:y + size, x:x + size] == 2).sum()), -y, -x)
            if best is None or cand > best:
                best = cand
    n_ref, _, ny, nx = best
    return -ny, -nx, n_ref


def draw_zoom_box(ax, y: int, x: int, size: int = 64) -> None:
    ax.add_patch(Rectangle((x, y), size, size, fill=False, edgecolor="white", linewidth=1.8,
                           zorder=11))
    ax.add_patch(Rectangle((x, y), size, size, fill=False, edgecolor=INK, linewidth=0.8,
                           linestyle=(0, (3, 2)), zorder=12))


def markdown_table(rows: list[dict], right_aligned: set[str]) -> str:
    headers = list(rows[0])
    widths = {h: max(3, len(h), *(len(str(r[h])) for r in rows)) for h in headers}

    def cell(h, text):
        return f"{text:>{widths[h]}}" if h in right_aligned else f"{text:<{widths[h]}}"

    sep = ["-" * (widths[h] - 1) + ":" if h in right_aligned else ":" + "-" * (widths[h] - 1)
           for h in headers]
    lines = ["| " + " | ".join(cell(h, h) for h in headers) + " |",
             "| " + " | ".join(sep) + " |"]
    lines += ["| " + " | ".join(cell(h, str(r[h])) for h in headers) + " |" for r in rows]
    return "\n".join(lines)
