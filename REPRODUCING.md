# Reproducing the study

This document describes how to reproduce every table and figure of the manuscript
*"A Per-Pixel Baseline Overstates the AlphaEarth Embedding's Advantage: A Context-Matched Test on Small-Parcel Coffee Boundary Delineation"* from public data. Everything under
`data/` is regenerable; only code and text are versioned.

The confirmatory analysis follows a written protocol,
[`docs/protocol/confirmatory.md`](docs/protocol/confirmatory.md) (hypotheses H1–H3, margin,
decoders, frozen thresholds in Addendum A). The same folder holds the ordered list of
confirmatory candidate tiles (`confirm_tiles.csv`, `confirm_tiles.geojson`) and the container
package versions (`pip-freeze.txt`).

## Requirements

- Docker with Compose v2 (recommended), or Python 3.11+ with a local virtualenv.
- A Google account with Earth Engine access and a Google Cloud project with the
  Earth Engine API enabled (needed for the download steps only).
- About 10 GB for the regenerated artifacts; steps 09–12 refuse to start with less than
  80 GB free (`MIN_FREE_GB` in `code/paths.py`).

## Setup (Docker, recommended)

```bash
cp .env.example .env        # set EE_PROJECT to your Google Cloud project id
docker compose build
docker compose run --rm pipeline earthengine authenticate   # one-time browser login
docker compose run --rm pipeline python code/gee_check.py   # validates credentials
```

Alternative without Docker: `python3 -m venv .venv && .venv/bin/pip install -r
code/requirements.txt`, then replace `docker compose run --rm pipeline` with
`.venv/bin/python` in the commands below (see `code/README.md`).

## Pipeline

Run the steps in order from the repository root. Only the `03_gee_stacks.py` calls touch
the network. Below, `R="docker compose run --rm pipeline"`.

### Original tiles (steps 01–05)

80 stratified 5 km tiles, spatial split by cell (43 train / 13 validation / 24 test),
per-pixel inputs E1 (Sentinel-2 seasonal), E2 (Sentinel-1+2 seasonal), E3 (AlphaEarth),
E4 (fusion).

```bash
$R python code/01_extract_reference.py   # CONAB/fiboa reference -> GeoParquet
$R python code/02_tiles.py --n 80        # stratified 5 km tiles
$R python code/03_gee_stacks.py          # S1/S2 seasonal + AlphaEarth per tile
$R python code/04_probe.py               # labels, spatial split, RF + linear probes (E1-E4)
$R python code/05_validate.py            # watershed, thresholds, metrics, tests (RF)
$R python code/05_validate.py --variant lin
```

### Neighborhood features and confirmatory tiles (steps 09–12)

E1c–E4c add the same four neighborhood features to every input (3×3 mean, 3×3 and 5×5 standard
deviation, Sobel magnitude of the 3×3 mean). Probes are always trained on the 43 original
training tiles; thresholds are selected on the 13 original validation tiles and only read in
the confirmatory scope. Order as in section 11 of the protocol:

```bash
# post hoc analysis on the original tiles -> data/ctx/
$R python code/09_context_probe.py --scope original
for v in rf lin; do for d in ws rag; do
  $R python code/10_context_eval.py --scope original --variant $v --decoder $d
done; done
$R python code/12_confirmatory.py --scope original

# raster drift check (re-downloads 3 original test tiles)
$R python code/11_confirm_tiles.py --stage drift-setup
docker compose run --rm -e ALPHAEARTH_MG_DATA=/work/data/confirm/drift pipeline \
  python code/03_gee_stacks.py
$R python code/11_confirm_tiles.py --stage drift-compare

# confirmatory tiles -> data/confirm/
$R python code/11_confirm_tiles.py --stage materialize   # copies the committed tile list
docker compose run --rm -e ALPHAEARTH_MG_DATA=/work/data/confirm pipeline \
  python code/03_gee_stacks.py --allow-partial
$R python code/11_confirm_tiles.py --stage labels        # eligibility; first 48 eligible = test
$R python code/09_context_probe.py --scope confirm
$R python code/09_context_probe.py --scope confirm --variants rf8   # sensitivity
for v in rf lin; do for d in ws rag; do
  $R python code/10_context_eval.py --scope confirm --variant $v --decoder $d
done; done
$R python code/12_confirmatory.py --scope confirm

# post hoc checks on the confirmatory tiles (not part of the protocol)
$R python code/13_posthoc_checks.py   # --checks seed pca16 ablation
# boundary tolerance, cosine operator and window scale, false boundaries,
# Sentinel-2 index recomputation, decoder threshold and sigma -> data/posthoc2/
$R python code/14_posthoc_sensitivity.py --checks c1 c2 c3 c4 c5
```

The order in which the protocol, the frozen thresholds and the confirmatory data were
produced is documented in `docs/protocol/PROVENANCE.md`.

`11_confirm_tiles.py --stage sample` regenerates the ordered candidate list in
`docs/protocol/` (seed 2026); it is not needed to reproduce, since the list is committed with
the protocol. `--stage coverage` reports invalid-pixel fractions of the original tiles.

### Figures and tables

```bash
make figures   # code/06_figures.py, 07_context_figures.py, 08_pipeline_figure.py
$R python code/check_artifacts.py
```

Shared modules: `code/probing.py` (features and probes), `code/objects.py` (decoders and
metrics), `code/stats.py` (paired tests, Hodges–Lehmann intervals, non-superiority, TOST),
`code/figstyle.py`, `code/paths.py`.

## Determinism and expected outputs

- All sampling, the spatial split, and the probes use fixed seeds; reruns are idempotent
  (atomic replaces, per-raster download cache, checkpointed step 09).
- Reference features smaller than one 10 m pixel (0.01 ha) are excluded from evaluation
  (`MIN_REF_AREA_M2`); see Methods.
- Key outputs: `data/metrics/{rf,lin}/` (steps 04–05); `data/ctx/metrics/` (original tiles,
  including the frozen `thresholds_*.json`); `data/confirm/metrics/confirmatory_summary.json`,
  `ap_per_tile.csv`, `raster_drift.csv`; `manuscript/figures/table_main.{csv,md}`,
  `table_confirm.{csv,md}` and `fig_*.png`.
- Headline values on the 48 confirmatory tiles (36 regions): Boundary AP per pixel E3 0.579
  vs E1 0.449; with neighborhood features E1c 0.664 vs E3c 0.610 (Hodges–Lehmann E3c − E1c −0.052,
  95% CI −0.063 to −0.043; H1 outcome "seasonal superior"); H2 supported; E4c ≈ E2c.

Earth Engine collections can change over time. The Sentinel-2 composites differed between
the original (July 2026) and confirmatory (November 2026) downloads; the effect on Boundary
AP was at most 0.004 on the tiles checked (see the manuscript). A rerun of step 03 today may
therefore not reproduce the stored rasters bit for bit. Minor floating-point differences
across platforms are also possible.

## Data sources

- Reference parcels: CONAB crop mapping, distributed as `br_conab` by the fiboa project
  (https://source.coop/fiboa/data/br_conab).
- Sentinel-1 GRD, Sentinel-2 L1C (`S2_HARMONIZED`), and AlphaEarth Satellite Embedding V1,
  all accessed through Google Earth Engine.
