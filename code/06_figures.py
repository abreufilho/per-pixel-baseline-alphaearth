#!/usr/bin/env python3
"""Passo 6 — figuras e tabelas de resultados (desenho com filtros de janela, J-STARS).

Lê data/ctx/metrics/ (24 tiles de teste originais, pós-hoc) e, quando
data/confirm/metrics/confirmatory_summary.json existir, data/confirm/metrics/ (tiles
confirmatórios); sem ele, os painéis confirmatórios saem marcados como pendentes.

Gera em manuscript/figures/:
- fig_reversal.png — diferenças pareadas de Boundary AP por tile (RF), HL + IC 95%
  (stats.hl_interval), margem ±0.03; HL + IC do probe linear como marcador secundário.
- fig_filter_gain.png — Boundary AP cru → com janelas (E1→E1c ... E4→E4c), por tile e média.
- fig_size_class.png — mean IoU (RAG) e Boundary AP por classe de tamanho, E1/E3/E1c/E3c.
- fig_qualitative.png — P(fronteira) de E1/E3/E1c/E3c (RF) com referência e candidatos RAG
  no tile original de diferença AP E3c−E1c mediana; recorte e zoom só da referência.
- table_main.{csv,md} (original) e table_confirm.{csv,md} (confirmatório, se disponível).

Uso: python code/06_figures.py
"""
import json
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from matplotlib import patheffects as pe
from matplotlib.lines import Line2D

import objects
from figstyle import (
    AXIS, BAND, CAND_EDGE, EXP_COLORS, EXP_LABELS, FULL_W, GRID, INK, INK2, MUTED, PROB_CMAP,
    REF_RED, add_scale_bar, boundary_mask, clean_axes, communication_crop, count_objects,
    draw_contour_mask, draw_zoom_box, image_axes_off, markdown_table, panel_label,
    pending_panel, save, zoom_window,
)
from paths import FIGURES_DIR, atomic_output, layout, scoped_metrics_dir, scoped_prob_path
from stats import MARGIN, hl_interval

warnings.filterwarnings("ignore", message="Setting the shape on a NumPy array has been deprecated.*",
                        category=DeprecationWarning)

__all__ = ["EXP_COLORS", "load_scene"]

BASES = ("E1", "E2", "E3", "E4")
EXPS = BASES + tuple(f"{b}c" for b in BASES)
N_FEATURES = {"E1": 16, "E2": 24, "E3": 64, "E4": 88}
N_FEATURES |= {f"{k}c": 5 * v for k, v in N_FEATURES.items()}
CONTRASTS = (("E3", "E1", "raw bands, per pixel"), ("E3c", "E1c", "identical window features"),
             ("E3c", "E2c", "identical window features"), ("E4c", "E2c", "identical window features"))
SIZE_ORDER = ("small", "mid", "large")
SIZE_DEFS = {"small": "<1 ha", "mid": "1–3 ha", "large": "≥3 ha"}
SCOPE_TITLES = {"original": "original test tiles, post hoc", "confirm": "confirmatory tiles"}
PENDING = "Confirmatory tiles (n = 48)\npending: re-run after\n12_confirmatory.py --scope confirm"
QUAL_EXPS = ("E1", "E3", "E1c", "E3c")
SHORT = {"E1": "S2 seasonal", "E2": "S1+S2 seasonal", "E3": "AlphaEarth", "E4": "fusion"}
ZOOM = 64
STROKE = [pe.withStroke(linewidth=2.2, foreground="white")]


# --- dados -----------------------------------------------------------------------------------
def available(scope: str) -> bool:
    return (scoped_metrics_dir(scope) / "confirmatory_summary.json").exists()


def load(scope: str) -> dict | None:
    if not available(scope):
        return None
    d = scoped_metrics_dir(scope)
    data = {f"{dec}_{v}": pd.read_csv(d / f"{dec}_{v}_per_tile.csv")
            for dec in ("ws", "rag") for v in ("rf", "lin")}
    data["ap"] = pd.read_csv(d / "ap_per_tile.csv").pivot(
        index="tile", columns="experiment", values="boundary_ap")
    data["thr"] = {k: json.loads((scoped_metrics_dir("original") / f"thresholds_{k}.json")
                                 .read_text()) for k in ("ws_rf", "ws_lin", "rag_rf", "rag_lin")}
    return data


def jitter_for(tiles, width: float, seed: int = 0) -> dict:
    tiles = sorted(tiles)
    return dict(zip(tiles, np.random.default_rng(seed).uniform(-width, width, len(tiles))))


def fmt_ci(hl: float, lo: float, hi: float) -> str:
    return f"{hl:+.3f} [{lo:+.3f}, {hi:+.3f}]".replace("-", "−")


# --- fig_reversal ----------------------------------------------------------------------------
def reversal_panel(ax, ap: pd.DataFrame) -> list[str]:
    jit = jitter_for(ap.index, 0.13)
    notes = []
    ax.axvspan(-MARGIN, MARGIN, color=BAND, zorder=0, lw=0)
    ax.axvline(0, color=AXIS, linewidth=0.8, zorder=1)
    for i, (a, b, _) in enumerate(CONTRASTS):
        y = len(CONTRASTS) - 1 - i
        d = (ap[a] - ap[b]).dropna()
        ax.scatter(d, [y + 0.12 + jit[t] for t in d.index], s=9, color=MUTED, alpha=0.55,
                   edgecolors="white", linewidths=0.4, zorder=2)
        hl, lo, hi = hl_interval(d, 0.95)
        ax.plot([lo, hi], [y - 0.22, y - 0.22], color=INK, linewidth=1.6,
                solid_capstyle="butt", zorder=4)
        ax.scatter([hl], [y - 0.22], marker="D", s=26, color=EXP_COLORS[a], edgecolors=INK,
                   linewidths=0.6, zorder=5)
        notes.append(f"{a}−{b} RF HL {fmt_ci(hl, lo, hi)} (n={len(d)})")
        la, lb = f"{a}_lin", f"{b}_lin"
        if la in ap and lb in ap:
            dl = (ap[la] - ap[lb]).dropna()
            hl2, lo2, hi2 = hl_interval(dl, 0.95)
            ax.plot([lo2, hi2], [y - 0.40, y - 0.40], color=INK2, linewidth=0.8, zorder=4)
            ax.scatter([hl2], [y - 0.40], marker="o", s=14, facecolors="white",
                       edgecolors=INK2, linewidths=0.8, zorder=5)
            notes.append(f"{a}−{b} lin HL {fmt_ci(hl2, lo2, hi2)}")
        ax.text(0.99, y + 0.40, f"RF HL {fmt_ci(hl, lo, hi)}", transform=ax.get_yaxis_transform(),
                ha="right", va="center", fontsize=6.2, color=INK2, path_effects=STROKE, zorder=6)
    ax.set_ylim(-0.6, len(CONTRASTS) - 0.4)
    ax.set_yticks(range(len(CONTRASTS)))
    ax.set_yticklabels([f"{a} − {b}\n{lab}" for a, b, lab in reversed(CONTRASTS)])
    ax.tick_params(axis="y", labelcolor=INK2, length=0)
    clean_axes(ax)
    ax.set_xlabel("Paired difference in Boundary AP per tile")
    return notes


def fig_reversal(data: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(FULL_W, 3.05), constrained_layout=True)
    all_d = []
    for scope, ax, letter in zip(data, axes, "ab"):
        if data[scope] is None:
            pending_panel(ax, PENDING)
            panel_label(ax, f"({letter}) {SCOPE_TITLES[scope]}")
            continue
        ap = data[scope]["ap"]
        for line in reversal_panel(ax, ap):
            print(f"  [{scope}] {line}")
        all_d += [(ap[a] - ap[b]).dropna() for a, b, _ in CONTRASTS]
        panel_label(ax, f"({letter}) {SCOPE_TITLES[scope]} (n = {len(ap)})")
    d = pd.concat(all_d)
    lim = np.ceil(max(abs(d.min()), abs(d.max())) * 20 + 0.2) / 20
    for scope, ax in zip(data, axes):
        if data[scope] is not None:
            ax.set_xlim(-lim, lim)
    if data["confirm"] is not None:
        axes[1].tick_params(axis="y", labelleft=False)
    handles = [
        Line2D([], [], ls="none", marker="o", ms=3.5, mfc=MUTED, mec="white", label="one tile (RF)"),
        Line2D([], [], color=INK, lw=1.6, marker="D", ms=4.5, mfc=MUTED, mec=INK,
               label="Hodges–Lehmann estimate, 95% CI (RF)"),
        Line2D([], [], color=INK2, lw=0.8, marker="o", ms=3.5, mfc="white", mec=INK2,
               label="same, linear probe"),
        plt.Rectangle((0, 0), 1, 1, color=BAND, label=f"±{MARGIN} AP margin"),
    ]
    fig.legend(handles=handles, loc="outside upper center", ncols=4, frameon=False,
               handletextpad=0.4, columnspacing=1.2)
    save(fig, "fig_reversal")


# --- fig_filter_gain -------------------------------------------------------------------------
def fig_filter_gain(data: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(FULL_W, 2.7), constrained_layout=True)
    for scope, ax, letter in zip(data, axes, "ab"):
        if data[scope] is None:
            pending_panel(ax, PENDING)
            panel_label(ax, f"({letter}) {SCOPE_TITLES[scope]}")
            continue
        ap = data[scope]["ap"]
        for i, b in enumerate(BASES):
            x0, x1 = i - 0.17, i + 0.17
            pair = ap[[b, f"{b}c"]].dropna()
            for _, r in pair.iterrows():
                ax.plot([x0, x1], r.values, color=MUTED, lw=0.45, alpha=0.45, zorder=2)
            m0, m1 = pair[b].mean(), pair[f"{b}c"].mean()
            ax.plot([x0, x1], [m0, m1], color=EXP_COLORS[b], lw=2.2, zorder=4)
            ax.scatter([x0], [m0], s=30, facecolors="white", edgecolors=EXP_COLORS[b],
                       linewidths=1.4, zorder=5)
            ax.scatter([x1], [m1], s=30, color=EXP_COLORS[b], edgecolors="white",
                       linewidths=0.6, zorder=5)
            ax.text(i, 0.985, f"{m0:.3f} → {m1:.3f}\n(+{m1 - m0:.3f})", ha="center", va="top",
                    fontsize=6.3, color=INK2, transform=ax.get_xaxis_transform())
            print(f"  [{scope}] {b}: {m0:.3f} -> {m1:.3f} (+{m1 - m0:.3f})")
        ax.set_xticks(range(len(BASES)))
        ax.set_xticklabels([f"{b} → {b}c\n{SHORT[b]}" for b in BASES])
        ax.tick_params(axis="x", labelcolor=INK2, length=0)
        ax.set_xlim(-0.55, len(BASES) - 0.45)
        clean_axes(ax, "y")
        panel_label(ax, f"({letter}) {SCOPE_TITLES[scope]} (n = {len(ap)})")
    axes[0].set_ylabel("Boundary AP per tile (RF)")
    for scope, ax in zip(data, axes):
        if data[scope] is not None:
            ax.set_ylim(0.1, 1.0)
    handles = [
        Line2D([], [], ls="none", marker="o", ms=5, mfc="white", mec=INK2, mew=1.2,
               label="raw bands (mean)"),
        Line2D([], [], ls="none", marker="o", ms=5, mfc=INK2, mec="white",
               label="+ window features (mean)"),
        Line2D([], [], color=MUTED, lw=0.6, label="one tile"),
    ]
    fig.legend(handles=handles, loc="outside upper center", ncols=3, frameon=False)
    save(fig, "fig_filter_gain")


# --- fig_size_class --------------------------------------------------------------------------
def size_panel(ax, df: pd.DataFrame, metric: str, show_groups: bool) -> None:
    exps = ("E1", "E1c", "E3", "E3c")
    pos, y = {}, 0.0
    for sc in SIZE_ORDER:
        for e in exps:
            pos[(sc, e)] = y
            y += 1
        y += 0.9
    jit = jitter_for(df.tile.unique(), 0.18, seed=1)
    for k, sc in enumerate(SIZE_ORDER):
        top, bot = pos[(sc, exps[0])] - 0.5, pos[(sc, exps[-1])] + 0.5
        if k % 2 == 0:
            ax.axhspan(top, bot, color=BAND, zorder=0, lw=0)
        sub_sc = df[df.size_class == sc]
        if show_groups:
            ax.text(0.01, top - 0.08, f"{sc} ({SIZE_DEFS[sc]}), n = {sub_sc.tile.nunique()}",
                    transform=ax.get_yaxis_transform(), fontsize=6.4, color=INK, va="bottom")
        for e in exps:
            sub = sub_sc[sub_sc.experiment == e].dropna(subset=[metric])
            yc = pos[(sc, e)]
            raw = not e.endswith("c")
            c = EXP_COLORS[e]
            ax.scatter(sub[metric], [yc + jit[t] for t in sub.tile], s=8, linewidths=0.6,
                       facecolors="white" if raw else c, edgecolors=c, alpha=0.6, zorder=2)
            if len(sub):
                ax.scatter([sub[metric].mean()], [yc], marker="D", s=30, zorder=4,
                           facecolors="white" if raw else c, edgecolors=c if raw else INK,
                           linewidths=1.2 if raw else 0.6)
    ax.set_yticks([pos[k] for k in pos], [k[1] for k in pos])
    ax.set_ylim(y - 0.9 + 0.1, -1.3)
    ax.tick_params(axis="y", labelcolor=INK2, length=0)
    clean_axes(ax)


def fig_size_class(data: dict) -> None:
    fig = plt.figure(figsize=(FULL_W, 3.9), constrained_layout=True)
    gs = fig.add_gridspec(1, 4)
    specs = (("mean_iou", "Mean IoU (region merging)"), ("boundary_ap", "Boundary AP"))
    for s, scope in enumerate(data):
        if data[scope] is None:
            ax = fig.add_subplot(gs[0, 2 * s:2 * s + 2])
            pending_panel(ax, PENDING)
            panel_label(ax, f"({'abcd'[2 * s]}) {SCOPE_TITLES[scope]}")
            continue
        df = data[scope]["rag_rf"]
        for j, (metric, name) in enumerate(specs):
            ax = fig.add_subplot(gs[0, 2 * s + j])
            size_panel(ax, df, metric, show_groups=j == 0)
            ax.set_xlim(0, 1)
            ax.set_xticks(np.arange(0, 1.01, 0.25))
            ax.xaxis.set_major_formatter(lambda v, _: f"{v:g}")
            ax.set_xlabel(f"{name} per tile")
            if j or s:
                ax.tick_params(axis="y", labelleft=False)
            short = "original" if scope == "original" else "confirmatory"
            panel_label(ax, f"({'abcd'[2 * s + j]}) {name.split(' (')[0]}, {short}",
                        fontsize=7.2)
            means = df[df.experiment.isin(QUAL_EXPS)].groupby(["size_class", "experiment"])[metric]
            print(f"  [{scope}] {metric}: " + ", ".join(
                f"{k[0]}/{k[1]}={v:.3f}" for k, v in means.mean().items()))
    handles = [Line2D([], [], ls="none", marker="o", ms=5, mfc=mf, mec=EXP_COLORS[e], mew=1.2,
                      label=EXP_LABELS[e])
               for e, mf in (("E1", "white"), ("E1c", EXP_COLORS["E1"]), ("E3", "white"),
                             ("E3c", EXP_COLORS["E3"]))]
    handles.append(Line2D([], [], ls="none", marker="D", ms=4.5, mfc=MUTED, mec=INK,
                          label="stratum mean (RF probe)"))
    fig.legend(handles=handles, loc="outside upper center", ncols=5, frameon=False,
               handletextpad=0.3, columnspacing=1.0)
    save(fig, "fig_size_class")


# --- fig_qualitative -------------------------------------------------------------------------
def load_scene(exps=QUAL_EXPS) -> dict:
    """Tile original de diferença AP RF E3c−E1c mediana (determinístico, não escolhido a dedo);
    probabilidades dos experimentos pedidos e candidatos RAG nos limiares congelados."""
    ap = pd.read_csv(scoped_metrics_dir("original") / "ap_per_tile.csv").pivot(
        index="tile", columns="experiment", values="boundary_ap")
    diff = (ap["E3c"] - ap["E1c"]).dropna().sort_values()
    tid = diff.index[(len(diff) - 1) // 2]
    with rasterio.open(layout("original").label(tid)) as ds:
        lab = ds.read(1)
    thr = json.loads((scoped_metrics_dir("original") / "thresholds_rag_rf.json").read_text())
    probs, pixel_m = {}, 10.0
    for e in exps:
        with rasterio.open(scoped_prob_path("original", tid, e)) as ds:
            probs[e] = ds.read(1).astype("float32")
            pixel_m = float(abs(ds.transform.a))
    h = min(lab.shape[0], *(p.shape[0] for p in probs.values()))
    w = min(lab.shape[1], *(p.shape[1] for p in probs.values()))
    probs = {e: p[:h, :w] for e, p in probs.items()}
    rag = {e: objects.rag_decoder(p, thr[e]) for e, p in probs.items()}
    rag = {e: (o[:h, :w] if o is not None else np.zeros((h, w), "int32")) for e, o in rag.items()}
    return dict(tid=tid, lab=lab[:h, :w], probs=probs, rag=rag, thr=thr, ap=ap.loc[tid],
                diff=float(diff.loc[tid]), pixel_m=pixel_m)


def fig_qualitative() -> dict:
    sc = load_scene()
    rs, cs, crop = communication_crop(sc["lab"])
    lab = sc["lab"][rs, cs]
    zy, zx, n_zoom = zoom_window(lab, ZOOM)
    zsl = (slice(zy, zy + ZOOM), slice(zx, zx + ZOOM))
    fig, axes = plt.subplots(2, 4, figsize=(FULL_W, 4.25), constrained_layout=True)
    im = None
    for j, e in enumerate(QUAL_EXPS):
        prob, obj = sc["probs"][e][rs, cs], sc["rag"][e][rs, cs]
        top, bot = axes[0, j], axes[1, j]
        im = top.imshow(prob, cmap=PROB_CMAP, vmin=0, vmax=1, interpolation="nearest")
        draw_contour_mask(top, boundary_mask(obj), CAND_EDGE, linewidth=0.35, alpha=0.75)
        draw_contour_mask(top, lab == 2, REF_RED, linewidth=0.55, alpha=0.98, halo=0.8)
        draw_zoom_box(top, zy, zx, ZOOM)
        top.text(0.03, 0.97, f"AP {sc['ap'][e]:.3f}", transform=top.transAxes, fontsize=6.6,
                 ha="left", va="top", color=INK, path_effects=STROKE)
        panel_label(top, f"({'abcd'[j]}) {e}: {'S2' if e[:2] == 'E1' else 'AlphaEarth'}{' + windows' if e.endswith('c') else ''}",
                    fontsize=6.6)
        bot.imshow(prob[zsl], cmap=PROB_CMAP, vmin=0, vmax=1, interpolation="nearest")
        draw_contour_mask(bot, boundary_mask(obj[zsl]), CAND_EDGE, linewidth=0.8, alpha=0.85)
        band = np.zeros((*lab[zsl].shape, 4))
        band[lab[zsl] == 2] = (0.89, 0.29, 0.28, 0.55)
        bot.imshow(band, interpolation="nearest", zorder=3)
        bot.text(0.04, 0.96, f"{count_objects(obj[zsl])} candidates · {n_zoom} parcels",
                 transform=bot.transAxes, fontsize=6.3, ha="left", va="top", color=INK,
                 path_effects=STROKE)
        panel_label(bot, f"({'efgh'[j]}) zoom, {e}", fontsize=6.8)
    add_scale_bar(axes[0, 0], sc["pixel_m"], lab.shape[1], lab.shape[0],
                  align="left" if zx + ZOOM / 2 > lab.shape[1] / 2 else "right")
    add_scale_bar(axes[1, 0], sc["pixel_m"], ZOOM, ZOOM, margin=5, text_dy=3)
    for ax in axes.ravel():
        image_axes_off(ax)
    for ax in axes[1]:
        for s in ax.spines.values():
            s.set_visible(True)
            s.set_edgecolor(INK)
            s.set_linewidth(0.8)
            s.set_linestyle((0, (3, 2)))
    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), orientation="horizontal", fraction=0.04,
                        pad=0.02, aspect=50, shrink=0.6)
    cbar.set_label("P(boundary), RF probe", color=INK2)
    cbar.outline.set_visible(False)
    handles = [Line2D([], [], color=REF_RED, lw=1.3, label="reference boundary band (10 m)"),
               Line2D([], [], color=CAND_EDGE, lw=0.9, alpha=0.8,
                      label="region-merging candidate (frozen τ)"),
               Line2D([], [], color=INK, lw=0.9, ls=(0, (3, 2)), label="zoom window")]
    fig.legend(handles=handles, loc="outside upper center", ncols=3, frameon=False)
    save(fig, "fig_qualitative")
    print(f"  tile qualitativo: {sc['tid']} (AP E3c−E1c = {sc['diff']:+.3f}; "
          f"crop y={crop['y']} x={crop['x']}; zoom y={zy} x={zx}, {n_zoom} talhões; "
          + ", ".join(f"{e} AP={sc['ap'][e]:.3f} τ={sc['thr'][e]}" for e in QUAL_EXPS) + ")")
    return sc


# --- tabelas ---------------------------------------------------------------------------------
TABLE_COLS = (  # (coluna, fonte, métrica, decimais, melhor)
    ("Boundary AP", "ws", "boundary_ap", 3, "max"),
    ("WS t", "ws", "t", 2, None),
    ("WS object recall", "ws", "object_recall", 3, "max"),
    ("WS mean IoU", "ws", "mean_iou", 3, "max"),
    ("WS Boundary F1", "ws", "boundary_f1", 3, "max"),
    ("RAG τ", "rag", "t", 2, None),
    ("RAG mean IoU", "rag", "mean_iou", 3, "max"),
)


def write_table(scope: str, data: dict, name: str) -> None:
    rows = []
    for probe, variant in (("RF", "rf"), ("Linear", "lin")):
        suf = "" if variant == "rf" else "_lin"
        means = {dec: data[f"{dec}_{variant}"].groupby("experiment").mean(numeric_only=True)
                 for dec in ("ws", "rag")}
        for e in EXPS:
            key = e + suf
            row = {"scope": scope, "probe": probe, "experiment": e, "features": N_FEATURES[e],
                   "n_tiles": int(data[f"ws_{variant}"].query("experiment == @key").tile.nunique())}
            for col, dec, metric, _, _ in TABLE_COLS:
                row[col] = (data["thr"][f"{dec}_{variant}"][key] if metric == "t"
                            else float(means[dec].loc[key, metric]))
            rows.append(row)
    table = pd.DataFrame(rows)
    with atomic_output(FIGURES_DIR / f"{name}.csv") as tmp:
        table.round(4).to_csv(tmp, index=False)
    md_rows = []
    for _, r in table.iterrows():
        block = table[table.probe == r.probe]
        out = {"Probe": r.probe, "Experiment": EXP_LABELS[r.experiment], "Features": r.features}
        for col, _, _, digits, best in TABLE_COLS:
            text = f"{r[col]:.{digits}f}"
            if best and text == f"{block[col].max():.{digits}f}":
                text = f"**{text}**"
            out[col] = text
        md_rows.append(out)
    md = markdown_table(md_rows, {c for c in md_rows[0]} - {"Probe", "Experiment"})
    with atomic_output(FIGURES_DIR / f"{name}.md") as tmp:
        tmp.write_text(md + "\n")
    print(f"· {name}.csv/.md ({scope})")


def main() -> None:
    data = {"original": load("original"), "confirm": load("confirm")}
    if data["original"] is None:
        raise SystemExit("data/ctx/metrics/confirmatory_summary.json ausente (rode 09/10/12)")
    if data["confirm"] is None:
        print("· confirmatório ausente: painéis marcados como pendentes")
    write_table("original", data["original"], "table_main")
    if data["confirm"] is not None:
        write_table("confirm", data["confirm"], "table_confirm")
    fig_reversal(data)
    fig_filter_gain(data)
    fig_size_class(data)
    fig_qualitative()


if __name__ == "__main__":
    main()
