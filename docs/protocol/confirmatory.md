# Confirmatory protocol: identical neighbourhood features for AlphaEarth and seasonal inputs

Status: **frozen** at the commit that adds this file together with `confirm_tiles.csv` /
`confirm_tiles.geojson` and the analysis code listed in §11, before any confirmatory tile is
downloaded, labelled, or evaluated. Date: 2026-11-13. Revision 2 incorporates two independent
methods reviews of a draft (Codex gpt-6-astra and Claude Fable 5.2). Any later deviation is
reported as a deviation in the paper.

## 1. Background and status of earlier analyses

The original design (`code/04_probe.py`, `code/05_validate.py`) compares per-pixel inputs:
seasonal Sentinel-2 (E1), seasonal Sentinel-1 + Sentinel-2 (E2), the AlphaEarth annual
embedding (E3) and E2 + E3 (E4). On the 24 original test tiles the embedding ranked boundary
pixels better (Boundary AP, RF probe: 0.524 vs 0.406).

Exploratory controls (2026-11-06) added the same window filters to every input (E1c–E3c).
The advantage reversed under the RF probe (AP 0.579 for E3c vs 0.617 for E1c) and became a
near tie under the linear probe (0.597 vs 0.600). These controls reused already-inspected test
tiles, and the filter set and decoder smoothing were chosen by the author. **Every analysis of
the original 80 tiles beyond the original E1–E4 watershed design is post hoc.** This protocol
fixes the analysis before an unseen set of tiles is drawn.

Interpretation is limited to the combination *representation × identical additional
neighbourhood features × fixed probe*. AlphaEarth already encodes spatial and temporal context
internally; the filters equalise the *external* operation applied to each input, not total
receptive field or information content.

## 2. Hypotheses and estimand

The estimand is the symmetric location of per-tile paired differences, estimated by the
Hodges–Lehmann (HL) pseudo-median, with confidence intervals obtained by inverting the
Wilcoxon signed-rank test (normal approximation). Symmetry of the differences is assumed;
the sample median is reported alongside.

- **H1 (primary).** With identical neighbourhood features, the embedding does not rank
  parcel-boundary pixels better than seasonal Sentinel-2 by more than δ = 0.03 AP:
  d = AP(E3c) − AP(E1c), RF probe.
- **H2 (key secondary).** The embedding's relative standing changes when both inputs receive
  the same neighbourhood features: per tile, (AP E3 − AP E1) − (AP E3c − AP E1c) > 0, RF.
  The wording "the per-pixel advantage shrinks" is used only if the raw advantage itself
  replicates (95% CI of AP E3 − AP E1 above 0); otherwise H2 is reported as a differential
  response only.
- **H3 (key secondary).** Under the linear probe, E3c and E1c are equivalent in AP within
  ±δ (δ = 0.03).

**Margin.** δ = 0.03 AP is one quarter of the per-pixel embedding advantage reported by the
original design (+0.12) and small relative to the AP gain that the same filters bring to the
seasonal input (E1 → E1c: +0.21, exploratory). A representation difference
below 0.03 AP is treated as practically negligible for choosing between inputs. δ = 0.02 is
reported as a sensitivity analysis. Disclosure: δ was set after the exploratory data showed
that TOST at δ = 0.02 is underpowered (exploratory linear TOST p = 0.094 at δ = 0.02,
0.011 at δ = 0.03); it is fixed here, before any confirmatory data exist.

**Sample size.** Bootstrap simulation from the exploratory per-tile differences
(independent tiles, true location 0, 1,000 replicates) gave, at n = 48 and δ = 0.03, power
≥ 0.98 for the non-superiority test and for TOST under both probes; at δ = 0.02 TOST power
under the linear probe was 0.64. Tiles in the same region are not fully independent; the
margin of power above 0.8 is kept to absorb this, and a region-level sensitivity is reported.

## 3. Representations

Rasters, bands and compositing are those of `code/03_gee_stacks.py` (2017; Sentinel-2 L1C
seasonal medians, wet = Jan–Mar + Oct–Dec, dry = Apr–Sep; Sentinel-1 GRD IW seasonal median
and SD in dB; AlphaEarth `GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL`). AlphaEarth itself ingests
Sentinel-2 L1C, so L1C does not create a processing-level asymmetry for Sentinel-2.

| Exp. | Sources | Features per source band | Features |
|---|---|---|---|
| E1 / E2 / E3 / E4 | s2 / s1+s2 / ae / s1+s2+ae | raw band | 16 / 24 / 64 / 88 |
| E1c / E2c / E3c / E4c | same | raw + mean 3×3 + SD 3×3 + SD 5×5 + Sobel magnitude of mean 3×3 | 80 / 120 / 320 / 440 |

Filters use `scipy.ndimage` defaults (reflect boundary mode). Window statistics are
NaN-aware; Earth Engine exports masked pixels as 0, so in practice zero-filled pixels enter
as observations exactly as in the original design (the raw layer applies NaN → 0). The
filter set is frozen and not tuned.

**Missing data and coverage.** A pixel is invalid for a source if all its bands are 0 or
any band is non-finite. A confirmatory tile is ineligible if more than 10% of its pixels are
invalid in any source (§7). Coverage is also reported for the original tiles.

## 4. Probes

As in `code/04_probe.py`, now explicit in `code/probing.py`: Random Forest (200 trees,
`min_samples_leaf=5`, `max_features="sqrt"`, `class_weight="balanced_subsample"`,
`random_state=42`) and logistic regression (`C=1`, `class_weight="balanced"`, lbfgs,
`max_iter=1000`, per-feature standardisation with training statistics). Training pixels:
boundary class = inner 10 m band of each reference parcel fully inside the tile (outline
buffered by 10 m, clipped to the parcel), interior = rest of the parcel; capped at
4,000/8,000 pixels per tile and class with `default_rng(42)` re-created per experiment, so
every experiment trains on the same pixels. Only the 43 original training tiles are used;
no model is trained, tuned or recalibrated on confirmatory tiles. Retraining is deterministic;
`09_context_probe.py --scope confirm` verifies that retrained E1–E4 reproduce the stored
original probabilities (max |Δ| ≤ 1e-5) before predicting.

**Dimensionality sensitivity (rf8).** The RF draws √p candidate features per split (8 for
E1c, 17 for E3c). A sensitivity RF with `max_features=8` for all context experiments is
reported for AP (descriptive).

## 5. Object extraction (decoders)

- **Watershed** (as `05_validate.py`): markers = connected components (`scipy.ndimage.label`,
  4-connectivity) of P < t; watershed on P; sieve 5 px.
  t ∈ {0.10, 0.15, …, 0.60}.
- **Hierarchical region merging (RAG)**: Gaussian smoothing σ = 1 px of P; markers = local
  minima of the smoothed P; watershed on the smoothed P; region adjacency graph weighted by
  the mean **unsmoothed** P along shared boundaries (`skimage.graph.rag_boundary`);
  hierarchical merging while the weight is below τ; sieve 5 px. τ ∈ {0.20, 0.25, …, 0.80}.
  σ = 1 is fixed (the σ ablation is supplementary and post hoc).

For each experiment × probe × decoder, the threshold maximising mean IoU over the 13 original
validation tiles is selected (ties → smallest threshold) and frozen
(`data/ctx/metrics/thresholds_*.json`, SHA-256 recorded in a protocol addendum before any
confirmatory download). The same thresholds apply to original test and confirmatory tiles.

## 6. Metrics and failure handling

Boundary AP: pixel-wise average precision of P inside the labelled footprint (classes of
§4), computed for every eligible tile, **independently of any decoder**. The AP label
population is the training-label population (all parcels fully inside the tile), as in the
original design; object metrics additionally exclude sub-pixel parcels (< 100 m²). Object
metrics as in `05_validate.py` (object recall at IoU ≥ 0.5, mean IoU, over-/under-segmentation
of Clinton et al. on matched parcels, Boundary F1 at 1 px tolerance).

Failures are scored, not dropped (`code/10_context_eval.py`, strict mode): a tile with
evaluable reference parcels and no decoded object scores object recall = mean IoU = 0 and
Boundary F1 = boundary recall = 0; the same applies to Boundary F1 when no predicted boundary
lies near the reference. This applies to validation (fixed denominator) and test tiles.
Metrics are kept at full precision until inference.

## 7. Confirmatory tiles

- **Grid.** The angular grid of `code/02_tiles.py` (cell side 5,000/111,320 degrees, origin
  at the reference bounds), i.e. approximately 5 km cells; rasters are exported in the tile's
  UTM zone at 10 m and are about 500 × 490 px. Candidates have ≥ 10 reference parcels fully
  inside.
- **Exclusion (geometric).** A candidate is excluded if it intersects any 0.5° region cell
  intersected by any of the 80 original tiles (69 cells), or lies within 5 km of an original
  tile.
- **Ordering.** Remaining regions (by the candidate's modal parcel region) are shuffled with
  `default_rng(2026)`; round 1 takes one random tile per region, round 2 a second tile per
  region from a different stratum when available (strata as in `02_tiles.py`). Maximum 2 tiles
  per region. The resulting ordered list (62 candidates, 37 regions; nearest original tile
  9.4 km) is committed with this protocol as `confirm_tiles.csv` / `.geojson`.
- **Eligibility** (in list order, before any prediction): all three sources download with the
  expected band count on a common grid (up to 3 download attempts); invalid-pixel fraction
  ≤ 10% in every source; both label classes present; ≥ 10 evaluable reference parcels
  (≥ 100 m², fully inside). The first **48** eligible tiles form the confirmatory set. If fewer
  than 48 are eligible, all eligible tiles are used and the shortfall is reported; the list is
  never extended or re-drawn after seeing any result.
- **Raster drift.** Before the confirmatory download, 3 original test tiles are re-downloaded
  and compared with the stored rasters (`11_confirm_tiles.py --stage drift-*`); differences
  are reported. The inference target is eligible tiles in previously unused regions under
  this sampling design; covariates (parcel size, density, boundary prevalence, coverage) are
  reported for original and confirmatory sets without reweighting.

## 8. Statistical analysis

Unit: tile. α = 0.05. Implemented in `code/stats.py` and `code/12_confirmatory.py`.

- **H1 (single primary test, no multiplicity adjustment).** Non-superiority: one-sided
  Wilcoxon signed-rank test of H0: θ ≥ δ on d − δ. Outcomes, from the HL 95% CI:
  *seasonal superior* (CI entirely < 0); *embedding superior* (CI entirely > 0);
  *not superior* (non-superiority p < 0.05 and CI includes 0); *inconclusive* (otherwise).
  H1 is supported by *seasonal superior* or *not superior*; contradicted by *embedding
  superior*.
- **Gatekeeping.** H2 and H3 are tested only if H1 is supported, with Holm adjustment across
  the two. H2: two-sided Wilcoxon on the per-tile difference-in-differences, supported if
  adjusted p < 0.05 and HL > 0. H3: TOST (two one-sided shifted Wilcoxon tests, p = max of
  the two), supported if adjusted p < 0.05; the HL 90% CI is reported.
- **Sensitivity (descriptive, no claims on their own).** H1 under the linear probe and under
  rf8; H1 and H3 with δ = 0.02; H1 on region means (mean per-tile difference per region,
  Wilcoxon across regions); E3c vs E2c; TOST E4c vs E2c (RF, linear); H2 for E4/E2 and for
  the linear probe; raw E3 vs E1.
- **Descriptive object-level comparisons.** For each probe (RF, linear) × decoder
  (watershed, RAG): E3 − E1, E4 − E2, E3c − E1c, E3c − E2c, E4c − E2c over object recall,
  mean IoU, over-/under-segmentation and Boundary F1 (plus AP), Wilcoxon with Holm within each
  contrast, as in the original paper. These are exploratory and interpreted by effect size and
  direction. Size-class strata are descriptive.

**Conclusion table.**

| H1 \ H3 | H3 supported (equivalent, linear) | H3 not supported |
|---|---|---|
| seasonal superior | seasonal + filters ≥ embedding; linear tie | seasonal + filters better |
| not superior | embedding not better; linear tie | embedding not better |
| inconclusive | — report as inconclusive | inconclusive |
| embedding superior | conclusion of the paper reverses | conclusion reverses |

## 9. Reporting

- Original 80 tiles: all context-matched analyses (E1c–E4c, RAG, TOST, rf8) labelled post hoc.
- Confirmatory tiles: H1, H2, H3 and the eligibility report are presented whatever the
  outcome; if H1 is contradicted the paper's conclusion changes accordingly.
- Absolute quality is modest (mean IoU ≤ ~0.44); claims concern relative representation
  quality under a fixed probe and decoder, not operational delineation.

## 10. Software

Docker image from the repository `Dockerfile` (Python 3.12; scikit-learn 1.9.0,
scikit-image 0.26.0, scipy 1.18.0 at freeze time); the image's `pip freeze` is archived with
the results.

## 11. Implementation

`code/probing.py`, `code/objects.py`, `code/stats.py` (shared); `code/09_context_probe.py`
(probes, prediction, reproduction check), `code/10_context_eval.py` (decoders, thresholds,
object metrics, descriptive tests), `code/11_confirm_tiles.py` (sampling, eligibility, drift,
coverage), `code/12_confirmatory.py` (H1–H3, sensitivities). Outputs: `data/ctx/` (original
tiles) and `data/confirm/` (confirmatory tiles).

Execution order after the freeze commit: `09 --scope original` → `10 --scope original` (all
probe × decoder) → `12 --scope original` → threshold hashes committed as an addendum →
drift check → `11 --stage materialize` → download → `11 --stage labels` →
`09 --scope confirm` (+ rf8) → `10 --scope confirm` → `12 --scope confirm`.

## Addendum A (2026-11-13): frozen thresholds and software

Recorded after `09/10/12 --scope original` and before any confirmatory download (no file in
`data/confirm/` exists at this commit). Validation-selected thresholds (13 original
validation tiles, strict scoring, ties → smallest):

| File | SHA-256 | Thresholds |
|---|---|---|
| `thresholds_rag_lin.json` | `6334f3b0a4b62b4a926e2977c0c0185efb9f67a568d838be5002f5f17cf3237d` | `{"E1_lin":0.5,"E2_lin":0.5,"E3_lin":0.4,"E4_lin":0.4,"E1c_lin":0.4,"E2c_lin":0.4,"E3c_lin":0.45,"E4c_lin":0.4}` |
| `thresholds_rag_rf.json` | `6453893d9faf5a44b89ce8f8a3e27120213801c5976517745e51919a91e00313` | `{"E1":0.45,"E2":0.45,"E3":0.45,"E4":0.45,"E1c":0.3,"E2c":0.35,"E3c":0.35,"E4c":0.3}` |
| `thresholds_ws_lin.json` | `941343b4bdfd4e33901e95ac5c681c57cfdc07339aa95a0712a3f47d839cdbfb` | `{"E1_lin":0.4,"E2_lin":0.4,"E3_lin":0.3,"E4_lin":0.3,"E1c_lin":0.3,"E2c_lin":0.3,"E3c_lin":0.3,"E4c_lin":0.35}` |
| `thresholds_ws_rf.json` | `6d53a493b3b5c678eb3dddc56031d173925f6c11632fe01077434a7426d490d2` | `{"E1":0.35,"E2":0.35,"E3":0.4,"E4":0.45,"E1c":0.25,"E2c":0.2,"E3c":0.25,"E4c":0.2}` |

Retrained context probes reproduce the exploratory probabilities of 2026-11-06 exactly
(max |Δ| = 0 for E1c/E2c/E3c, RF and linear). The refactored `05_validate.py` reproduces
`data/metrics/rf` exactly (all metrics and tests). Container package versions:
`docs/protocol/pip-freeze.txt`.

## Addendum B (2026-11-13): deviations and post hoc checks

Recorded after the confirmatory results. Sections 1–11 and Addendum A are unchanged.

Implementation corrections after the freeze commit and before any confirmatory metric
(confirmatory rasters downloaded 05:58–06:09; first confirmatory metrics 06:57, UTC−03):

- `4d1ff84` (06:04): `12` aborts on a missing probability or NaN AP; reproduction check
  fails fast and also covers E1c–E4c; `10` verifies the SHA-256 of the frozen thresholds.
- `dda2502` (06:38): Holm keeps contrasts without differences in the family (p = 1); paired
  differences rounded to 1e-12 to remove floating-point ties; sample medians added;
  completeness required for rf8 and validation tiles; CRS in the common-grid check.
- `6732d77` (06:38): `10_context_eval.py` reads the frozen thresholds for the original scope.

After the confirmatory results were known, the tests were rerun on the same per-tile metrics
with the statistics module as frozen (`code/stats.py` at `df53860`); the H1–H3 decisions and
the significance classification of all 120 confirmatory object-level tests were unchanged
(see `FROZEN-STATS-CHECK.md`). This paragraph was reworded on 2026-11-18 to state when the
check was run; the check itself is unchanged.

Reporting gaps against §5/§7: the σ ablation of the region-merging decoder was not run.

Post hoc checks requested by the manuscript reviews (`code/13_posthoc_checks.py`, RF, Boundary
AP; outputs in `data/confirm/metrics/posthoc/`), not confirmatory: E1c and E3c with seeds 1
and 2; E3c on the first 16 principal components of the embedding (80 features, as E1c);
filter ablation (raw + 3×3 mean; raw + 3×3 mean + 3×3/5×5 standard deviations) for E1 and E3.
