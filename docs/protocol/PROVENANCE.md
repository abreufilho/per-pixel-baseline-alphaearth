# Provenance of the confirmatory protocol

This file lets a reader check the order in which the confirmatory protocol, the frozen
thresholds and the confirmatory data were produced. All times are local (UTC−3) on
2026-11-13. They come from the authors' development repository and file system, so they are
self-reported: no third party timestamped the protocol, and it was not deposited in an
external registry. The protocol was written, frozen and tested on the same day by the same
analyst.

## Commits in the development repository

| Time | Commit | Content |
| :-- | :-- | :-- |
| 05:56:01 | `6805710` | Analysis code for the context-matched controls and the confirmatory analysis |
| 05:56:07 | `df53860` | Protocol (`confirmatory.md`) and ordered list of confirmatory tiles frozen |
| 05:56:20 | `1f65c35` | Addendum A: SHA-256 of the frozen thresholds and software versions |
| 06:04:33 | `4d1ff84` | Guards before the confirmatory evaluation (AI-assisted code review) |
| 06:38:16 | `dda2502` | Statistical robustness fixes (AI-assisted code review) |
| 06:38:34 | `6732d77` | Re-evaluation of the original scope with the frozen thresholds |
| 07:54:27 | `cb8a363` | Post hoc checks and Addendum B (deviations) |

The development repository is private. The public repository holds a single snapshot of it,
so its history does not include these commits. The hashes above identify the
development commits; the frozen files can be checked against these SHA-256 digests of their
content at `df53860`:

| File | SHA-256 at `df53860` |
| :-- | :-- |
| `confirmatory.md` (protocol body, before Addenda A and B) | `bbc2daa9cf52514cc71a358efc11d59bfaccc86dc43489616e471e3bf5ceb78f` |
| `confirm_tiles.csv` | `c5630b7bf14543739d1c251339ac26263af56064bbec93e25bc0dd23c2fcc3f4` |
| `confirm_tiles.geojson` | `c67723d6f85c060b3764fcbfdf60e0626305cba35cb8a647d2280063f5bd0d17` |

The tile list files are unchanged since `df53860`. The protocol file later received Addenda A
and B; its body above the addenda is unchanged.

## File times

| Time | Event |
| :-- | :-- |
| 05:35–05:54 | Thresholds selected on the 13 original validation tiles (`thresholds_*.json`) |
| 05:56:20 | Thresholds frozen by checksum in Addendum A |
| 05:58:15 | First confirmatory tile raster written (`c01`, `c02`) |
| 06:57–07:02 | Confirmatory probabilities, decoders and hypothesis tests (`09`–`12` logs) |

Thresholds and the original-tile probabilities were produced before the protocol commit,
because they are inputs that the protocol freezes; no confirmatory tile existed at that time.
All confirmatory downloads, labels, predictions and metrics followed the protocol commit.
After the freeze and before any confirmatory metric, the implementation was corrected
following an AI-assisted code review (Addendum B). After the confirmatory results were known,
the tests were rerun on the same per-tile metrics with the statistics module as frozen; the
decisions were unchanged (`FROZEN-STATS-CHECK.md`).
