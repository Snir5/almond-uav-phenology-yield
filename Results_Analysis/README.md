# Results_Analysis — Figure Index

All figures are organized into thematic subfolders. The HTML presentations
(`Analysis_Presentation.html` and `Analysis_Presentation_Hebrew.html`) at the
Thesis root reference figures by their full subfolder path.

---

## Subfolders

### 01_EBI_Bloom/
EBI temporal dynamics, bloom distributions, and cultivar ANOVA (2021-2024).
Key figures: `EBI_Distributions_by_Year.png`, `EBI_Cultivar_ANOVA.png`,
`ANOVA_PostHoc_Heatmap.png`, `EBI_Trend_Summary.png`.
Finding: mean EBI is flat (~0.55) but SD halves (0.105 -> 0.048); cultivar
effect converges over years (F=1385 in 2021, F=4.1 in 2024).

### 02_Spatial_Maps/
Orchard layout maps, spatial EBI maps per year, cultivar crown maps, and
Moran's I spatial autocorrelation. CRS: EPSG:32636 (UTM Zone 36N).
Key figures: `Cultivar_Crown_Maps_AllYears.png`, `EBI_Spatial_Maps.png`,
`Morans_I_Spatial_Full.png`, `Orchard_Map_Y0_Corrected.png`.
Finding: bloom is spatially clustered every year (Moran's I 0.12-0.36, p<0.01).

### 03_Climate/
Climate vs EBI correlations, per-cultivar sensitivity, PCA of agro-climate
features, and ANCOVA. Climate variables: Chill Portions (Dynamic Model),
GDD, frost days, heat stress, rainfall, wind.
Key figures: `Climate_EBI_Scatter_ByCultivar.png`,
`Climate_Sensitivity_ByCultivar.png`, `Analysis15_Climate_PCA.png`.
Note: climate is station-level (one value/year, n=4) so year-level links
are directional only; tree-level slopes carry the inference.

### 04_Yield/
Measured yield analyses (n=162 trees, 2023 season ground truth from
`Trees Data and Survay/kedma_plot_a_yield.csv`). Yield-EBI correlation,
cultivar ANOVA, and ANCOVA with EBI x cultivar interaction.
Key figures: `Measured_Yield_EBI_Cultivar.png`, `ANOVA_Yield_Cultivar_Full.png`,
`Yield_Climate_EBI_Combined.png`.
Finding: cultivar governs yield (F=13.46***; UEF 2.94 kg > cv-53 2.28 kg);
EBI x cultivar interaction F=11.58 (p=0.0009) - UEF r=+0.29, cv-53 r=-0.26.
WARNING: `predicted_Yield` columns in the master are a clustering-model output
(NOT ground truth) and must not be used for EBI-yield inference.

### 05_Phenology/
Field phenology analyses cross-referencing 2021 ground survey bloom timing
with UAV EBI values (analysis series 20-28).
Key figures: `Analysis21_Phenology_vs_EBI.png`,
`Analysis24_Cultivar_BloomTiming_Permutation.png`,
`Analysis28_Synthesis_Hypothesis.png`.

### 06_Methods/
Pipeline and methodology diagrams, summer spectral ANOVA (July 2024 NDVI/NDRE),
NGRDI analysis, MANOVA.
Key figures: `Methods_Pipeline_Diagram.png`, `Analysis12_SummerSpectral_ANOVA.png`,
`Analysis13_NGRDI_ANOVA.png`.

### 07_Coordinate_Validation/
Spatial alignment audits across all data sources. Not shown in the main
presentation deck. See also `Coordinate_Validation_Report.md` at this level.
All sources confirmed EPSG:32636; yield-CSV join ~0.88 m; spectral-GPKG
join ~3.4 m systematic offset (crown-level registration only).

### _Redundant/
Superseded or draft figures: early cultivar map versions (v2, v5),
diagnostic row-check figure, and an uncorrected crown map draft. These are
retained for traceability but are not used in analysis or the presentation.

---

## Data files (at this level)

| File | Contents |
|------|----------|
| `thesis_results.json` | All headline statistics from `thesis_analysis.py` |
| `climate_yield_ebi.json` | Climate x EBI x yield year-level results |
| `measured_yield_results.json` | Measured-yield ANOVA and ANCOVA output |
| `Scientific_Summary.txt` | Plain-language results summary (English) |
| `Scientific_Summary_Part2.txt` | Results summary continuation |
| `ANOVA_cultivar_climate_summary.txt` | Cultivar x climate ANOVA table |
| `Analysis_Yield_EBI_Summary.txt` | Yield-EBI correlation summary |
| `Coordinate_Validation_Report.md` | Full coordinate alignment audit |
| `Full_Analysis_Audit_Report.md` | Superseded audit report (see THESIS_PROGRESS_LOG.md) |
| `progress_report.pdf` | Interim progress report PDF |
| `coordinate_audit_report.txt` | Coordinate audit text output |
| `Analysis12_SummerSpectral_ANOVA_summary.txt` | Summer spectral ANOVA table |
| `Analysis25_2021_TreeLevel_Phenology_vs_EBI.csv` | Tree-level phenology-EBI data (2021) |
