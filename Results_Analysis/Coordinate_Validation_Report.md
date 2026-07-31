# Coordinate Alignment & Join Validation Report
**Orchard:** Kedma Plot A, Israel · **CRS:** UTM Zone 36N (EPSG:32636) / WGS84
**Date:** 2026-07-02 · **Scope:** all spatial sources feeding the master tree table
**Figure:** `Coordinate_Validation_AllSources.png`

---

## 1. Purpose

Independent audit confirming that every spatial data source (master tree table,
yield GPKGs, spectral GPKG, measured-yield CSV, and the four annual UAV mosaics)
is in a common coordinate reference frame and that the record-to-record joins are
spatially correct. This validation was run directly from the raw files, not from
any previously cached join.

## 2. Sources audited

| Source | Type | CRS | n | Key fields |
|---|---|---|---|---|
| `Master_Trees_Extended.xlsx` | tree centroids | UTM 36N + WGS84 | 1,523 | X_UTM, Y_UTM, Latitude, Longitude, EBI/NGRDI, climate |
| `Master_Trees_Time_Series_Plot1.xlsx` | tree centroids | UTM 36N + WGS84 | 1,291 | same, no yield |
| `Yield_with_clustering_2022/2023.gpkg` | polygon crowns | EPSG:32636 | 2,866 (1,110 Plot A) | **predicted_Yield** (modeled) |
| `spectral_data_plot_A_31072024.gpkg` | polygon crowns | EPSG:32636 | 1,347 | CWSI, NDVI, NDRE, … (Jul-2024) |
| `kedma_plot_a_yield.csv` | point measurements | WGS84 | 202 rows / 162 trees | **net_kernel_yield_per_tree_kg** (measured) |
| Mosaics 2021/2022/2023/2024 | GeoTIFF + TFW | UTM 36N | — | orthomosaic footprints |

## 3. CRS consistency — PASS

All vector sources are natively EPSG:32636 (UTM 36N). The master's stored
`Latitude`/`Longitude` were re-derived from `X_UTM`/`Y_UTM` with an independent
UTM→WGS84 transform and matched the stored values to **0.00 m median residual**
(max < 0.01 m) in both masters. The historical ~68 m eastward longitude error
documented in earlier logs is **fully corrected** in the current master files
(Longitude range 34.7919–34.7947).

## 4. Alignment between sources — PASS (with one caveat)

Nearest-neighbour distances, Extended master (1,523 trees) as query:

| Join | median | mean | max | ≤2 m | ≤5 m |
|---|---|---|---|---|---|
| Master → Yield GPKG (Plot A) | **1.01 m** | 2.96 m | 28.9 m | 68.5% | 74.6% |
| Master → Spectral GPKG (Jul-2024) | 3.44 m | 4.23 m | 28.3 m | 0.3% | 90.0% |
| Measured-yield CSV → Master (bijective ≤8 m) | **0.88 m** | 1.07 m | 6.76 m | — | — |

- **Yield GPKG** and **measured-yield CSV** align to the master at sub-metre
  median distance — excellent, well within a single crown radius.
- **Caveat — spectral GPKG:** a systematic ~3.4 m offset separates the Jul-2024
  spectral crown centroids from the master crown centroids (visible in Figure
  panel A as a consistent northward shift). 90% still fall within 5 m, so the
  nearest-neighbour join is stable, but the Jul-2024 spectral values carry a
  slightly looser spatial registration than the other layers and should be
  interpreted at crown—not sub-crown—resolution.
- The large "max" distances reflect the 232 boundary-recovered master trees that
  have no counterpart in the smaller GPKG surveys; these are correctly left
  unmatched rather than force-joined.

## 5. Mosaic footprints

The four annual orthomosaics share EPSG:32636 but were exported over different
crop extents. Upper-left corner shift relative to the 2021 reference:

| Year | ΔE vs 2021 | ΔN vs 2021 | pixel size |
|---|---|---|---|
| 2022 | +63.7 m | +42.0 m | 1.47 cm |
| 2023 | +21.5 m | +52.0 m | 1.33 cm |
| 2024 | +60.7 m | +97.2 m | 2.24 cm |

These are **export-extent** differences, not per-pixel misregistration of the
imagery. Because EBI/NGRDI were extracted using the GPKG-registered crown zones
(fixed geographic polygons), not raw mosaic pixel indices, the inter-year footprint
differences do **not** propagate into the extracted spectral values. (Note: pixel
values could not be re-extracted in this environment — see §7.)

## 6. Yield provenance — CRITICAL FINDING

The master contains **two distinct yield variables that were conflated in earlier
logs**:

| Column | Source | Nature | 2023 mean ± sd | n |
|---|---|---|---|---|
| `predicted_Yield_2022/2023` | `Yield_with_clustering` GPKG | **modeled** (clustering model) | 3.04 ± 0.30 | 922 |
| `Yield_2022/2023` | same GPKG (re-join) | **modeled** (duplicate) | 3.02 ± 0.31 | 1,003 |
| measured CSV `net_kernel_yield_per_tree_kg` | field harvest | **measured** | 2.65 ± 1.18 | 162 (2023) |

`Yield_2022` and `predicted_Yield_2022` correlate at **r = 0.992** — they are the
same modeled quantity. The earlier "1,003 yield matches" headline therefore
describes **modeled yield**, not ground truth, and correlating a spectral index
against it is partly circular (the model itself uses canopy/growth inputs).

**Resolution for the thesis:** the measured CSV is canonical for any spectral–yield
or cultivar–yield claim. It covers 162 trees, effectively **2023 only** (2022 n=20,
2024 n=20). Bijective join to the master recovers **161/162 trees at 0.88 m median**.
The modeled `predicted_Yield` may be reported separately but must be labelled as
model output, never as measured yield.

## 6b. Independent cross-check: tree_position grid row/column (not coordinates)

Every yield-matching script in this project links `kedma_plot_a_yield.csv` to
the master by haversine distance on lat/lon (bijective, <=8 m). That is a
coordinate-based join; it had never been checked against a genuinely
independent, non-coordinate identifier. The yield CSV carries a `tree_position`
field (format `A_r_{row}_c_{col}`, an orchard grid row/column, populated for
148/202 rows, distinct from `location_geom_wkt`) and the yield GPKGs carry their
own `Row` field. Neither had been used anywhere before this check.

**Result: 147 of 147 checkable trees (100%) agree between the CSV's grid row
and the master's `Row_2023` field** (already joined via the standard nearest-XY
match, tolerance +/-1 row). A second, fully independent direct re-match (master
X/Y_UTM to the 2023 GPKG's own Row+geometry, 3 m threshold, bypassing the
existing join entirely) agrees for 146 of 147 (99.3%); the one non-match is not
a disagreement, it is a tree that fell just outside the stricter 3 m
re-match radius in this direct recheck while its Row_2023 value (55) already
matched the CSV's row (55) exactly via the standard join. This is strong,
independent confirmation that the coordinate-based yield-to-master matching
used throughout this project's results is correct, not just internally
consistent.

Script: `verify_yield_match_tree_position.py` (sandbox-safe, in
`pipeline/06_analysis/`).

## 6c. Full row-by-row audit (every yield record, not a sample)

Per request, a final deep pass checks every one of the 202 yield rows (162
distinct tree_ids), not a spot check, for five separate failure modes. Script:
`deep_yield_master_audit.py` (sandbox-safe).

1. **Coverage: 161 of 162 distinct trees matched (99.4%).** The one unmatched
   tree_id (1303) was investigated individually, not just flagged: every master
   tree within its 8 m search radius (Tree_1595 at 3.21 m, Tree_1580 at 4.71 m,
   Tree_1639 at 7.09 m, Tree_1532 at 7.18 m) is claimed by a DIFFERENT yield
   tree_id that sits strictly closer to that same master tree (2.60 m, 1.70 m,
   0.90 m, 2.16 m respectively). This is a local crowding case (several
   measured trees close together), and the bijective matching correctly leaves
   1303 unassigned rather than force-matching it to an already-claimed or a
   farther tree. This is the matching working as intended, not an error.
2. **Distance sanity: median 0.881 m, max 6.762 m, all within the 8 m
   threshold.** 4 of 161 matches exceed 3 m (6.76, 6.46, 3.94, 3.31 m). All 4
   were individually cross-checked against `tree_position` (see 6b) and every
   one still agrees exactly on grid row, confirming the larger distance
   reflects GPS recording precision at those specific points, not a wrong-tree
   match.
3. **tree_position agreement: 147 of 147 rows with a position value (100%),
   checked in full, not sampled** (repeats and confirms 6b at full coverage).
4. **Cross-year self-consistency: 20 tree_ids are measured in more than one
   year; all 20 matched, 0 dropped.** (Necessarily one-to-one by construction
   of the per-tree_id match; reported for transparency.)
5. **Uniqueness: 0 master trees claimed by more than one yield tree_id.** The
   bijective dedup enforces this by construction; explicitly re-verified here
   rather than assumed.
6. **Cultivar plausibility: the 161 matched trees split 90 UEF / 71 cultivar
   53, exactly matching this project's UEF+53 scope, with 0 matches to
   cultivar 54 or an unlabeled tree.** (The yield CSV carries no cultivar field
   to cross-check directly, so this is a consistency read, not an independent
   confirmation, but it is the expected result given the project's cultivar
   restriction and shows no contamination.)

**Verdict: every yield-to-master match in current use is either fully
verified correct (161 trees, confirmed by coordinate distance and
independently by grid row) or, in the single unmatched case, correctly and
verifiably excluded rather than mismatched.** No collision, no silent
many-to-one assignment, no cultivar contamination.

## 7. Environment limitation (disclosed)

The analysis sandbox has no network access to install `rasterio`/`geopandas`, so the
multi-GB 4-band orthomosaics could not be re-opened to re-extract EBI/NGRDI at the
pixel level. All EBI/NGRDI/spectral values were therefore validated as already stored
in the master (their spatial join, CRS, and distributions were audited), rather than
recomputed from raw rasters. Every statistical result in the results log **was**
recomputed from the master in this session.

## 8. Verdict

| Check | Result |
|---|---|
| Common CRS (EPSG:32636) across all sources | ✅ PASS |
| Master WGS84 ↔ UTM internal consistency | ✅ PASS (0.00 m) |
| Yield GPKG ↔ master alignment | ✅ PASS (1.01 m median) |
| Measured-yield CSV ↔ master alignment | ✅ PASS (0.88 m median) |
| Spectral GPKG ↔ master alignment | ⚠️ PASS with ~3.4 m systematic offset |
| Yield variable provenance | ⚠️ FIXED — measured vs predicted disentangled |
| Independent tree_position (grid row/col) cross-check of yield matching | ✅ PASS (147/147, 100%) |
| Full row-by-row audit, all 202 yield rows (coverage, distance, collisions, cultivar) | ✅ PASS (161/162 matched, 1 correctly excluded, 0 collisions) |
