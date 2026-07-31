# Almond tree segmentation: standalone model

**If you only want to detect almond trees in drone imagery, this folder is all you need.**
It contains the trained weights and a single self-contained script. Nothing here imports
from the rest of the repository, so you can copy the folder out and use it on its own.

The rest of the repository is the thesis analysis: orthomosaic assembly, bloom indices,
climate joins and the statistics. None of it is required to run the model.

---

## Contents

| File | What it is |
|---|---|
| `best_model.pth` | Trained UNet++ / EfficientNet-B3 weights, 706 tensors, 52.6 MB |
| `predict.py` | Preprocessing, inference, instance splitting and GeoTIFF export in one file |
| `requirements.txt` | Python packages needed |

## Install

Python 3.11 (3.10 also works).

```bash
pip install -r requirements.txt
```

Use the pinned versions. `segmentation-models-pytorch` must stay on 0.3.x, because the
checkpoint stores the UNet++ decoder keys in that layout and later releases renamed them,
and `albumentations` must stay on 1.x, because 2.0 changed the transform behaviour the
model was trained with. `predict.py` reports the mismatch clearly if either drifts.

A GPU is optional. The script picks CUDA if present, then Apple MPS, then CPU. Expect
roughly 1 to 3 seconds per image on GPU and 20 to 40 seconds on CPU.

## Run

```bash
python predict.py --images /path/to/drone_photos --output /path/to/results
```

That is the whole workflow. The weights are found automatically next to the script.

### Output

```
results/
├── photos/     # original-resolution JPG, each tree shaded a different colour
├── masks/      # binary PNG mask, white is canopy
└── rasters/    # 4-band GeoTIFF: R, G, B, tree mask (needs GPS EXIF)
```

Instance labels are resized back to the input resolution, so masks and overlays line up
with the source photos pixel for pixel.

### Options

| Flag | Default | When to change it |
|---|---|---|
| `--thresh` | 0.5 | Lower towards 0.35 if canopies are missed, raise towards 0.6 if soil is picked up |
| `--min-distance` | 25 | Raise if one tree is split in two, lower if neighbouring trees merge |
| `--min-area` | 200 | Smallest accepted object, in pixels |
| `--device` | auto | Force `cuda`, `mps` or `cpu` |
| `--deterministic` | off | Repeatable masks, see the note below |
| `--weights` | `best_model.pth` | Point at a different checkpoint |

`--min-distance` is the one to reach for first. The default is tuned for full bloom, when
bright white canopies produce several distance peaks within a single tree.

---

## What the model expects

| Property | Value |
|---|---|
| Input | 3-band RGB drone photo, any size |
| Internal resize | 1024 x 1024 |
| Normalization | ImageNet |
| Preprocessing | Grey-world white balance, adaptive gamma, adaptive CLAHE, applied automatically |
| Architecture | UNet++, EfficientNet-B3 encoder, 1 output class |

Do not enhance the images before feeding them in. The script already applies colour
correction, and stacking two corrections degrades the mask.

## Georeferencing

GeoTIFFs are written from the drone GPS in the image EXIF. The GPS point is treated as the
image centre and converted to the top-left corner assuming a ground sample distance of 2 cm
per pixel, in EPSG:32636 (UTM Zone 36N).

Two things to adjust for a different site or flight:

- `PIXEL_SIZE_M` in `predict.py`, currently 0.02. It scales the footprint linearly, so a
  different flight altitude needs a different value.
- `EPSG_CODE`, currently 32636. Set it to the UTM zone of your own site.

Images without GPS EXIF still produce overlays and masks; only the GeoTIFF is skipped.
If `rasterio`, `pyproj` and `piexif` are not installed, the script runs without them and
skips GeoTIFF export entirely.

## A note on repeatability

The preprocessing used for the thesis includes three steps that fire probabilistically
(Equalize, CLAHE and Sharpen, at p = 0.5, 0.5 and 0.4). The same image can therefore give
slightly different masks on repeated runs. The default reproduces the thesis behaviour.
Pass `--deterministic` to drop those three steps and get identical output every time, which
is worth doing for any comparison between images or dates.

## Retraining

Training code, the COCO data loader and the dataset preparation notebooks are in
[`../code/`](../code/) and [`../notebooks/`](../notebooks/). The model was trained with
Focal Tversky loss (alpha 0.3, beta 0.7, gamma 0.75), which weights recall over precision
to limit missed canopies.

`best_model.pth` holds `model_state_dict` plus the `epoch`, `val_dice` and `loss` records.
The Adam optimizer state was removed to cut the file from 152 MB to 52.6 MB. That state is
only needed to resume training from the exact epoch it stopped; it has no effect on
inference or on fine-tuning from these weights. Use `../code/strip_checkpoint.py` to do the
same to a checkpoint of your own.
