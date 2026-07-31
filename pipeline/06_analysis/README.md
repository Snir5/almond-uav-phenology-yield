# pipeline/06_analysis — Analysis Scripts

## Canonical scripts (run these)

| Script | Purpose | Output |
|--------|---------|--------|
| `thesis_analysis.py` | Master analysis: recomputes every headline statistic from `Master_Trees_Extended.xlsx` in thesis order (EBI dynamics, cultivar ANOVA, split-plot, Moran's I, climate, measured-yield). | `Results_Analysis/thesis_results.json` |
| `climate_yield_ebi.py` | Climate x EBI x yield over years + measured-yield ANCOVA (EBI x cultivar). | `Results_Analysis/climate_yield_ebi.json` |
| `coord_audit.py` | Full coordinate/join/provenance audit across all sources. | `Results_Analysis/Coordinate_Validation_Report.md` |
| `fix_boundary_maps.py` | Regenerates spatial maps with corrected field boundary (derived from trees, never hard-coded). | Spatial PNGs in `Results_Analysis/02_Spatial_Maps/` |
| `geo_guardrails.py` | **Shared utility - import this.** `load_master()` validates lat/lon vs UTM; `field_boundary()` derives the plot hull from trees. Never bypass these. |  |
| `stats_utils.py` | **Shared utility - import this.** From-scratch F/t/chi-square/Pearson, PCA via SVD, split-plot ANOVA. (scipy/statsmodels not available in the sandbox.) |  |
| `check_pipeline.py` | Pre-flight check: flags stale paths, hard-coded boundaries, coordinate drift. **Run before regenerating any figures.** |  |

### Pre-flight command
```
python3 pipeline/06_analysis/check_pipeline.py
```

### Full re-run order
```
python3 pipeline/06_analysis/thesis_analysis.py
python3 pipeline/06_analysis/climate_yield_ebi.py
python3 pipeline/06_analysis/coord_audit.py
python3 pipeline/06_analysis/fix_boundary_maps.py
```

---

## archive/ — Exploratory and superseded scripts

These were used during earlier analysis sessions and are kept for traceability.
They are **not** part of the reproducible pipeline and may reference old paths.

| Script | What it was for |
|--------|----------------|
| `analysis12-28_*.py` | Exploratory phenology, PCA, MANOVA, Moran's I (now in thesis_analysis.py) |
| `analysis_yield_ebi.py` | Early yield-EBI work (superseded by climate_yield_ebi.py) |
| `anova_cultivar_climate.py` | Cultivar x climate ANOVA (superseded by thesis_analysis.py) |
| `generate_analysis.py`, `generate_analysis_part2.py` | Old figure generation scripts |
| `run_analyses.py` | Old batch runner |
| `yield_join_and_figs.py` | Old yield join (superseded by geo_guardrails + thesis_analysis) |
| `sd_vs_cultivar_confound.py` | Diagnostic for SD-cultivar confound |
| `interp.py` | Interpolation utility (experimental) |

---

## Key conventions (see also GEOSPATIAL_GUARDRAILS.md at Thesis root)

- **Always load the master via `geo_guardrails.load_master()`** - raises if
  coordinates are inconsistent (68 m GPS shift in old data).
- **Never hard-code the plot boundary** - use `geo_guardrails.field_boundary()`.
- **Measured vs modelled yield**: use `kedma_plot_a_yield.csv` (162 trees) for
  any EBI-yield or cultivar-yield claim. The `predicted_Yield` columns in the
  master are a clustering model output (r=0.992 with each other) and correlating
  EBI against them is circular.
- **Chilling = Chill Portions (Dynamic Model)** via `Chill_Portions_*` columns.
  Do not use `Chill_Hrs` or the Utah/Weinberger models.
- **CRS = EPSG:32636** everywhere. No internet in the sandbox; use `sqlite3` for
  GeoPackage reads; `rasterio`/`geopandas` are not available.
