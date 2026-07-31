# Kedma Almond Orchard: UAV Bloom Phenology, Cultivar Structure and Yield

MSc thesis code and results (Ben-Gurion University). The work links per-tree bloom
phenology measured from UAV orthomosaics to cultivar, winter climate and measured
yield across four seasons (2021-2024) in Plot A of the Kedma orchard, for 1,523 trees.

## What is in this repository

| Path | Contents |
|---|---|
| `models/` | Trained segmentation weights (`best_model.pth`) |
| `code/` | Detection and segmentation model: training, inference, post-processing, GeoTIFF export. Start with [`code/MODEL_USAGE.md`](code/MODEL_USAGE.md) |
| `notebooks/` | Training and experiment notebooks behind the model and the empirical search of Chapter 3 (outputs stripped) |
| `pipeline/01_mosaic_build` … `06_analysis` | Ordered processing stages, from mosaic assembly to the statistical analysis |
| `Results_Analysis/` | Result figures (PNG) and the JSON files holding every reported statistic |
| `Thesis_Writing/` | The thesis document |

## Model weights

The trained weights are included at `models/best_model.pth` (about 51 MB, optimizer state
stripped). See [`code/MODEL_USAGE.md`](code/MODEL_USAGE.md) for how to load them and run
inference on new imagery.

## Method in brief

1. **Detection.** UNet++ with an EfficientNet-B3 encoder, trained with Focal Tversky
   loss (alpha 0.3, beta 0.7, gamma 0.75); watershed instance segmentation separates
   touching crowns.
2. **Georeferencing.** EXIF GPS to UTM Zone 36N (EPSG:32636), exported as 4-band
   GeoTIFF (RGB plus tree mask), assembled into per-season orthomosaics in ArcGIS Pro.
3. **Per-tree indices.** Enhanced Bloom Index (EBI) and NGRDI computed for every tree
   in every season from fixed tree zones.
4. **Analysis.** Cultivar ANOVA and split-plot ANOVA, Moran's I, bloom timing against
   agro-climate, and yield models validated against the growers' harvest survey.

## Headline results

- Cultivar dominates bloom variation and the cultivars converge: the between-cultivar
  share of bloom variance falls from 69 % to 1 % across the record.
- Bloom is spatially clustered within every cultivar in every season.
- Bloom **timing** tracks winter climate (later bloom in warmer winters, r = +0.69,
  p = 0.03, n = 10 cultivar-years) whereas bloom **intensity** does not.
- Measured yield is governed by cultivar, and the bloom-to-yield relationship
  **reverses in sign** between cultivars (interaction F = 11.58, p = 0.0009).
- A compact remote-sensing model predicts the measured harvest better than the
  existing modelled product (46 % versus 29 % of variance under cross-validation).

## Reproducing the analysis

The analysis layer is dependency-light: `numpy`, `pandas`, `openpyxl`, `matplotlib`.
Statistical routines (F, t and chi-square p-values, PCA, split-plot ANOVA, Moran's I)
are implemented from first principles in `pipeline/06_analysis/stats_utils.py`.

```bash
python3 pipeline/06_analysis/check_pipeline.py     # pre-flight checks
python3 pipeline/06_analysis/thesis_analysis.py    # recompute headline statistics
```

## Data not included

Raw and mosaicked imagery (tens of GB), the growers' yield and phenology records, the
climate station files and all third-party publications are **not** distributed here.
The imagery and field records belong to the orchard operator and the collaborating
laboratory; published papers are cited in the thesis rather than redistributed.

## Credentials

No API keys are stored in this repository. Notebooks that download the annotated
dataset read the key from the environment:

```bash
export ROBOFLOW_API_KEY="your-key-here"
```

## Related work

Yield modelling from UAV multi-sensor data for the same orchard is described in
Efrat et al. (2025), *Agricultural Water Management* 320, 109868; canopy nitrogen
retrieval in Woldenberg et al. (2025), *Smart Agricultural Technology* 12, 101355.
