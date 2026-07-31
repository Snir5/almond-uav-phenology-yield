# Using the Segmentation Model

Quick start for running the trained almond tree segmentation model on new drone imagery.
For the full architecture, ArcGIS mosaic workflow and production notes, see
[`PRODUCTION_GUIDE.md`](PRODUCTION_GUIDE.md).

---

## 1. Get the weights

The trained checkpoint is not stored in this repository because of its size.
Download `best_model.pth` from the link in the repository description and place it
anywhere on disk. The path is passed explicitly, so it does not need to sit next to the code.

The checkpoint is a dictionary containing `model_state_dict`, `optimizer_state_dict`,
`epoch`, `val_dice` and `loss`. Inference reads `model_state_dict` only.

---

## 2. Set up the environment

Python 3.10 or 3.11. Install:

```bash
pip install torch torchvision segmentation-models-pytorch albumentations \
            opencv-python numpy scikit-image rasterio pyproj piexif
```

A GPU is optional. The scripts select CUDA if present, then Apple MPS, then CPU.
Expect roughly 1 to 3 seconds per image on GPU and 20 to 40 seconds on CPU.

---

## 3. Run inference

Run from inside the `code/` directory, since the modules import each other by flat name.

```bash
cd code

python main.py --mode predict \
  --model-path /path/to/best_model.pth \
  --predict-dir /path/to/drone_images \
  --output-dir /path/to/results \
  --thresh 0.5
```

`--predict-dir` is scanned for `.jpg`, `.jpeg` and `.png` files. Add `--device cpu`,
`--device mps` or `--device cuda` to override automatic device selection.

### What the model expects

| Property | Value |
|---|---|
| Input | 3-band RGB drone photo, any size |
| Internal resize | 1024 x 1024 |
| Normalization | ImageNet, mean (0.485, 0.456, 0.406), std (0.229, 0.224, 0.225) |
| Preprocessing | adaptive colour correction, then histogram equalization and CLAHE |
| Architecture | UNet++ with an EfficientNet-B3 encoder, 1 output class |

Preprocessing is applied automatically by `get_val_test_transform()`. Do not pre-enhance
the images yourself, since that stacks two corrections and degrades the mask.

---

## 4. What you get

```
results/
├── photos/     # original-resolution JPG with coloured instance overlays
└── rasters/    # 4-band GeoTIFF per image: R, G, B, binary tree mask
```

Instance labels are resized back to the original image resolution before saving, so the
GeoTIFFs match the input photos pixel for pixel.

Georeferencing is written from the drone GPS in the image EXIF. The GPS point is treated
as the image centre and converted to the top-left corner using a ground sample distance of
2 cm per pixel. Output CRS is EPSG:32636 (UTM Zone 36N). Images without GPS EXIF are still
processed, but their GeoTIFF is written at coordinate (0, 0) and will not place correctly
in GIS.

---

## 5. Parameters worth tuning

`--thresh` (default 0.5) is the probability cut on the sigmoid output. Lower it towards 0.35
if canopies are being missed, raise it towards 0.6 if bare soil is being picked up.

The instance splitting parameters are in `postprocess.py`, in `adaptive_instance_split()`:

| Parameter | Default | Effect |
|---|---|---|
| `min_area` | 200 px | Smallest accepted object |
| `min_dist_between_trees` | 25 px | Minimum spacing between watershed seeds |
| `relative_area_factor` | 0.2 | Discards objects below 20 percent of the median tree area |
| solidity filter | 0.70 | Rejects non-compact shapes |

`min_dist_between_trees` is the one to adjust first. Raise it when one tree is split into
several instances, lower it when touching canopies are merged into one. The default is
tuned for full bloom, when bright white canopies produce several distance peaks per tree.

---

## 6. Retraining

```bash
python main.py --mode train \
  --train-img-dir /path/to/train/images \
  --train-ann-path /path/to/train/_annotations.coco.json \
  --val-img-dir /path/to/valid/images \
  --val-ann-path /path/to/valid/_annotations.coco.json \
  --model-path best_model.pth
```

Annotations are COCO instance segmentation format, as exported by Roboflow. Training runs
20 epochs at batch size 4 with Focal Tversky loss (alpha 0.3, beta 0.7, gamma 0.75), which
weights recall over precision to limit missed canopies. The dataset preparation notebooks
are in [`../notebooks/`](../notebooks/) and expect a Roboflow key in the environment:

```bash
export ROBOFLOW_API_KEY="your-key"
```

---

## 7. Troubleshooting

**`KeyError: 'model_state_dict'`**
The file is a raw state dict rather than a training checkpoint. Load it directly with
`model.load_state_dict(torch.load(path))` instead of going through `load_trained_model()`.

**Rasters land in the wrong place in ArcGIS**
Check that GPS EXIF survived your file transfer, and that the 2 cm ground sample distance
matches your flight. `PIXEL_SIZE_M` in `predict_pipeline.py` is fixed at 0.02 and scales the
footprint linearly, so a different flight altitude needs a different value.

**Out of memory**
Inference is one image at a time, so this is usually the 1024 x 1024 tensor on a small GPU.
Use `--device cpu`, or reduce `IMAGE_SIZE` in `data.py` and retrain, since the encoder is
resolution sensitive.

**Masks look noisy or over-segmented**
Raise `--thresh` first, then `min_dist_between_trees`. Check the overlay JPGs in `photos/`
before trusting the rasters.
