# Check of the confirmatory tests against the frozen statistics module

After the protocol freeze (`df53860`) and before any confirmatory metric was computed, the
statistics module `code/stats.py` was corrected (`dda2502`, Addendum B). This file records
whether those corrections change any confirmatory decision.

## Method

- Frozen module: `git show df53860:code/stats.py`.
- Current module: `code/stats.py`.
- Input: the same confirmatory per-tile metrics (`results/confirm/*_per_tile.csv` and
  `ap_per_tile.csv` in the public package).
- Both modules were run on these inputs for H1–H3 and for every object-level contrast
  (raw E3 vs E1 and E4 vs E2; context-matched E3c vs E1c, E3c vs E2c, and E4c vs E2c), under
  both probes (Random Forest, linear) and both decoders (watershed, region merging), with Holm
  adjustment per contrast as in the protocol.

## Result

| Item | Frozen module | Current module |
| :-- | :-- | :-- |
| H1 (E3c − E1c, Random Forest) | seasonal superior | seasonal superior |
| H2 (difference in differences) | p = 3.6e-14 | p = 3.6e-14 |
| H3 (linear TOST) | −0.0261, not equivalent | −0.0261, not equivalent |
| Object-level tests (120) | — | 0 changes in significance classification |

## Provenance

The comparison was first run on 2026-11-13, after the confirmatory results were known, and
rerun on 2026-11-18 with the same outcome. It is self-reported by the authors and covers the
statistics module only; the other corrections in Addendum B add checks (guards, threshold
checksum, CRS check) and are not covered by this comparison.
