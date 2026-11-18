#!/usr/bin/env python3
"""Inferência pareada por tile (docs/protocol/confirmatory.md §8).

Unidade = tile. Estimando = locação simétrica das diferenças por tile, estimada pelo
pseudo-mediano de Hodges-Lehmann (mediana das médias de Walsh), com IC obtido pela inversão
do teste de postos sinalizados de Wilcoxon (aproximação normal). Testes de Wilcoxon com o
mesmo estimando: bilateral (diferença), unilateral deslocado (não superioridade) e TOST
(equivalência). Holm-Bonferroni dentro de cada contraste para as famílias descritivas.
"""
import numpy as np
import pandas as pd
from scipy.stats import norm, wilcoxon

TEST_METRICS = ("object_recall", "mean_iou", "over_seg", "under_seg", "boundary_f1", "boundary_ap")
MARGIN = 0.03            # margem de não superioridade/equivalência (AP)
SENSITIVITY_MARGIN = 0.02
ALPHA = 0.05


def holm(p: np.ndarray) -> np.ndarray:
    """p ajustados por Holm-Bonferroni preservando a ordem original."""
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adj[i] = min(running, 1.0)
    return adj


def paired(per_tile: pd.DataFrame, a: str, b: str, metric: str) -> pd.Series:
    """Diferenças por tile (a - b) para tiles com ambos os valores definidos."""
    pa = per_tile[per_tile.experiment == a].set_index("tile")[metric]
    pb = per_tile[per_tile.experiment == b].set_index("tile")[metric]
    common = pa.index.intersection(pb.index)
    # arredonda ruído de ponto flutuante para que empates reais (ex.: 2/11) empatem no posto
    diff = (pa.loc[common] - pb.loc[common]).round(12)
    return diff[diff.notna()]


def hodges_lehmann(d) -> tuple[float, float, float]:
    """Pseudo-mediano de HL e IC 95% bilateral por inversão do signed-rank.

    O limite superior do IC 90% bilateral é o limite unilateral de 95% (não superioridade,
    TOST); ver hl_interval(level=0.90)."""
    return hl_interval(d, 0.95)


def hl_interval(d, level: float) -> tuple[float, float, float]:
    x = np.asarray(d, dtype="float64")
    n = len(x)
    i, j = np.triu_indices(n)
    walsh = np.sort((x[i] + x[j]) / 2.0)
    m = len(walsh)
    z = norm.ppf(1 - (1 - level) / 2)
    k = int(np.floor(n * (n + 1) / 4 - z * np.sqrt(n * (n + 1) * (2 * n + 1) / 24)))
    k = max(k, 0)
    return float(np.median(walsh)), float(walsh[k]), float(walsh[m - 1 - k])


def paired_tests(per_tile: pd.DataFrame, pairs, metrics=TEST_METRICS) -> pd.DataFrame:
    """Wilcoxon bilateral por contraste e métrica, Holm dentro de cada contraste."""
    rows = []
    for a, b in pairs:
        block = []
        for metric in metrics:
            diff = paired(per_tile, a, b, metric)
            if len(diff) < 5:
                raise ValueError(f"{a}-{b} {metric}: só {len(diff)} tiles pareados")
            # contraste sem nenhuma diferença fica na família com p = 1 (não encolhe o Holm)
            p = 1.0 if diff.abs().sum() == 0 else float(wilcoxon(diff).pvalue)
            block.append({"pair": f"{a}-{b}", "metric": metric, "n_tiles": len(diff),
                          "median_diff": round(float(diff.median()), 4), "p_value": p})
        if block:
            adj = holm(np.array([r["p_value"] for r in block]))
            rows += [{**r, "p_holm": round(float(p), 6)} for r, p in zip(block, adj)]
    out = pd.DataFrame(rows)
    if not out.empty:
        out["p_value"] = out["p_value"].round(6)
    return out


def tost(per_tile: pd.DataFrame, a: str, b: str, metric: str = "boundary_ap",
         margin: float = MARGIN) -> dict:
    """Equivalência: dois testes unilaterais de Wilcoxon sobre d = a - b, margem ±margin;
    p da hipótese = max(p1, p2); IC 90% de HL reportado (decisão coerente com o IC)."""
    diff = paired(per_tile, a, b, metric)
    p_lower = float(wilcoxon(diff + margin, alternative="greater").pvalue)
    p_upper = float(wilcoxon(diff - margin, alternative="less").pvalue)
    hl, lo, hi = hl_interval(diff, 0.90)
    p_tost = max(p_lower, p_upper)
    return {"pair": f"{a}-{b}", "metric": metric, "n_tiles": len(diff), "margin": margin,
            "hl": hl, "median": float(diff.median()), "ci90_low": lo, "ci90_high": hi,
            "p_lower": p_lower, "p_upper": p_upper,
            "p_tost": p_tost, "equivalent": bool(p_tost < ALPHA)}


def non_superiority(diff: pd.Series, margin: float = MARGIN) -> dict:
    """Desfecho primário para d = embedding - sazonal (§8).

    seasonal_superior  : IC 95% inteiro < 0
    embedding_superior : IC 95% inteiro > 0
    not_superior       : H0 d >= margin rejeitada (Wilcoxon unilateral em d - margin), sem
                         superioridade sazonal demonstrada
    inconclusive       : nenhum dos anteriores
    """
    hl, lo, hi = hl_interval(diff, 0.95)
    _, _, hi90 = hl_interval(diff, 0.90)
    p_two = float(wilcoxon(diff).pvalue)
    p_ns = float(wilcoxon(diff - margin, alternative="less").pvalue)
    if hi < 0:
        outcome = "seasonal_superior"
    elif lo > 0:
        outcome = "embedding_superior"
    elif p_ns < ALPHA:
        outcome = "not_superior"
    else:
        outcome = "inconclusive"
    return {"n_tiles": len(diff), "hl": hl, "median": float(np.median(diff)), "ci95_low": lo, "ci95_high": hi,
            "upper_one_sided_95": hi90, "margin": margin, "p_two_sided": p_two,
            "p_non_superiority": p_ns, "outcome": outcome,
            "supported": outcome in ("seasonal_superior", "not_superior")}


def diff_in_diff(per_tile: pd.DataFrame, raw: tuple[str, str], ctx: tuple[str, str],
                 metric: str = "boundary_ap") -> dict:
    """H2: (raw_a - raw_b) - (ctx_a - ctx_b) por tile; > 0 = vantagem relativa do primeiro
    termo encolhe quando todos recebem os mesmos filtros de janela."""
    d_raw = paired(per_tile, *raw, metric)
    d_ctx = paired(per_tile, *ctx, metric)
    common = d_raw.index.intersection(d_ctx.index)
    dd = d_raw.loc[common] - d_ctx.loc[common]
    hl, lo, hi = hl_interval(dd, 0.95)
    hl_raw, lo_raw, _ = hl_interval(d_raw.loc[common], 0.95)
    return {"contrast": f"({raw[0]}-{raw[1]})-({ctx[0]}-{ctx[1]})", "metric": metric,
            "n_tiles": len(dd), "hl_raw": hl_raw, "ci95_low_raw": lo_raw,
            "hl_ctx": hl_interval(d_ctx.loc[common], 0.95)[0],
            "hl_dd": hl, "median_dd": float(dd.median()), "ci95_low": lo, "ci95_high": hi,
            "p_value": float(wilcoxon(dd).pvalue),
            "raw_advantage_replicated": bool(lo_raw > 0)}


def gatekeep(primary_supported: bool, secondary: dict[str, float]) -> dict[str, dict]:
    """H2/H3 só são testados se o primário for suportado; Holm entre eles."""
    names = list(secondary)
    if not primary_supported:
        return {k: {"p": secondary[k], "p_holm": None, "tested": False} for k in names}
    adj = holm(np.array([secondary[k] for k in names]))
    return {k: {"p": secondary[k], "p_holm": float(p), "tested": True,
                "rejected": bool(p < ALPHA)} for k, p in zip(names, adj)}
