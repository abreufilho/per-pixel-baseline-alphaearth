#!/usr/bin/env python3
"""Passo 12 — análise confirmatória pré-especificada (docs/protocol/confirmatory.md §8).

Boundary AP é livre de decodificador: calculado aqui para todo tile de teste do escopo, com
precisão total, para os probes rf, lin e (se presente) rf8.

  H1 (primário, único teste confirmatório sem correção): não superioridade do embedding,
     AP RF E3c - E1c, margem 0,03; desfecho em 4 categorias (stats.non_superiority).
  H2 (gatekeeping): (AP E3 - AP E1) - (AP E3c - AP E1c), RF, Wilcoxon bilateral.
  H3 (gatekeeping): TOST AP linear E3c - E1c, margem ±0,03.
  H2/H3 testados só se H1 for suportado; Holm entre os dois.
  Sensibilidades (descritivas): H1 sob lin e rf8, margem 0,02, agregação por região,
  E3c - E2c, E4c - E2c.

Saídas em data/{ctx,confirm}/metrics/: ap_per_tile.csv, confirmatory_summary.json.
Uso: python code/12_confirmatory.py --scope confirm
"""
import argparse
import json
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

import objects
from paths import atomic_output, atomic_write_text, exp_name, layout, scoped_metrics_dir, scoped_prob_path
from stats import MARGIN, SENSITIVITY_MARGIN, diff_in_diff, gatekeep, non_superiority, paired, tost

AP_EXPS = {"rf": ("E1", "E2", "E3", "E4", "E1c", "E2c", "E3c", "E4c"),
           "lin": ("E1", "E2", "E3", "E4", "E1c", "E2c", "E3c", "E4c"),
           "rf8": ("E1c", "E2c", "E3c", "E4c")}


def ap_task(args) -> list[dict]:
    scope, tid, region = args
    label = layout(scope).label(tid)
    rows = []
    for variant, bases in AP_EXPS.items():
        for base in bases:
            exp = exp_name(base, variant)
            f = scoped_prob_path(scope, tid, exp)
            if f.exists():
                rows.append({"tile": tid, "region": region, "experiment": exp,
                             "boundary_ap": objects.boundary_ap(f, label)})
    return rows


def region_means(ap: pd.DataFrame, a: str, b: str) -> pd.Series:
    d = paired(ap, a, b, "boundary_ap").rename("d").to_frame()
    d["region"] = ap.drop_duplicates("tile").set_index("tile").loc[d.index, "region"]
    return d.groupby("region").d.mean()


def clean(obj):
    if isinstance(obj, dict):
        return {k: clean(v) for k, v in obj.items()}
    if isinstance(obj, (np.floating, float)):
        return None if np.isnan(obj) else float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


def main() -> None:
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("--scope", choices=("original", "confirm"), required=True)
    ap_.add_argument("--workers", type=int, default=12)
    args = ap_.parse_args()
    lay = layout(args.scope)
    split = json.loads(lay.split.read_text())
    props = {f["properties"]["tile_id"]: f["properties"]
             for f in json.loads(lay.tiles.read_text())["features"]}
    test_ids = [t for t, p in split.items() if p == "test"]

    with ProcessPoolExecutor(args.workers) as pool:
        rows = pool.map(ap_task, [(args.scope, t, props[t]["region"]) for t in test_ids])
    ap = pd.DataFrame([r for rs in rows for r in rs])
    variants = ("rf", "lin") + (("rf8",) if ap.experiment.str.endswith("_rf8").any() else ())
    expected = {exp_name(b, v) for v in variants for b in AP_EXPS[v]}
    counts = ap[ap.experiment.isin(expected)].groupby("experiment").boundary_ap.count()
    bad = {e: int(counts.get(e, 0)) for e in expected if counts.get(e, 0) != len(test_ids)}
    if bad or ap.boundary_ap.isna().any():
        raise SystemExit(f"AP incompleto (esperado {len(test_ids)} tiles por experimento): {bad}")
    out_dir = scoped_metrics_dir(args.scope)
    out_dir.mkdir(parents=True, exist_ok=True)
    with atomic_output(out_dir / "ap_per_tile.csv") as tmp:
        ap.to_csv(tmp, index=False)

    h1 = non_superiority(paired(ap, "E3c", "E1c", "boundary_ap"), MARGIN)
    h2 = diff_in_diff(ap, ("E3", "E1"), ("E3c", "E1c"))
    h3 = tost(ap, "E3c_lin", "E1c_lin", margin=MARGIN)
    gate = gatekeep(h1["supported"], {"H2": h2["p_value"], "H3": h3["p_tost"]})
    h2_supported = bool(gate["H2"].get("rejected") and h2["hl_dd"] > 0)
    h3_supported = bool(gate["H3"].get("rejected"))

    sens = {
        "H1_lin": non_superiority(paired(ap, "E3c_lin", "E1c_lin", "boundary_ap"), MARGIN),
        "H1_margin_0.02": non_superiority(paired(ap, "E3c", "E1c", "boundary_ap"),
                                          SENSITIVITY_MARGIN),
        "H1_region_means": non_superiority(region_means(ap, "E3c", "E1c"), MARGIN),
        "H3_margin_0.02": tost(ap, "E3c_lin", "E1c_lin", margin=SENSITIVITY_MARGIN),
        "E3c_vs_E2c_rf": non_superiority(paired(ap, "E3c", "E2c", "boundary_ap"), MARGIN),
        "E4c_vs_E2c_rf_tost": tost(ap, "E4c", "E2c", margin=MARGIN),
        "E4c_vs_E2c_lin_tost": tost(ap, "E4c_lin", "E2c_lin", margin=MARGIN),
        "H2_E4_vs_E2": diff_in_diff(ap, ("E4", "E2"), ("E4c", "E2c")),
        "H2_lin": diff_in_diff(ap, ("E3_lin", "E1_lin"), ("E3c_lin", "E1c_lin")),
        "raw_E3_vs_E1_rf": non_superiority(paired(ap, "E3", "E1", "boundary_ap"), MARGIN),
    }
    if "E3c_rf8" in set(ap.experiment):
        sens["H1_rf8"] = non_superiority(paired(ap, "E3c_rf8", "E1c_rf8", "boundary_ap"), MARGIN)

    means = ap.groupby("experiment").boundary_ap.mean().round(4).to_dict()
    summary = clean({
        "scope": args.scope, "n_tiles": len(test_ids),
        "n_regions": len({props[t]["region"] for t in test_ids}),
        "mean_ap": means,
        "H1": h1, "H2": {**h2, **gate["H2"], "supported": h2_supported},
        "H3": {**h3, **gate["H3"], "supported": h3_supported},
        "sensitivity": sens,
    })
    atomic_write_text(out_dir / "confirmatory_summary.json", json.dumps(summary, indent=1))
    print(json.dumps({k: summary[k] for k in ("scope", "n_tiles", "n_regions", "H1", "H2", "H3")},
                     indent=1))


if __name__ == "__main__":
    main()
