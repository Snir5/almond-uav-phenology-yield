# Working with the model source

> **Just want to detect trees?** Use [`../segmentation_model/`](../segmentation_model/).
> It holds the weights and a single self-contained script, and needs nothing from this
> folder. This page covers the development code: training, the original batch pipeline,
> and the route into ArcGIS.

For the full architecture and production notes, see [`PRODUCTION_GUIDE.md`](PRODUCTION_GUIDE.md).

---

## Layout

| File | Role |
|---|---|
| `main.py` | CLI entry point, `--mode train` or `--mode predict` |
| `model.py` | UNet++ / EfficientNet-B3 definition, Focal Tversky loss, Dice and IoU |
| `data.py` | COCO dataset loader, preprocessing, train and validation transforms |
| `postprocess.py` | Watershed instance splitting and the overlay renderer |
| `predict_pipeline.py` | Batch inference, EXIF to UTM georeferencing, GeoTIFF export |
| `strip_checkpoint.py` | Drops optimizer state from a checkpoint before sharing it |

The modules import each other by flat name, so run everything from inside `code/`.

## Environment

Python 3.10 or 3.11. Beyond the inference requirements in
[`../segmentation_model/requirements.txt`](../segmentation_model/requirements.txt),
the training and batch code also needs:

```bash
pip install pycocotools xattr
```

`pycocotools` reads the annotation files; `xattr` preserves macOS file attributes and is
not needed on Linux or Windows.

## Training

```bash
cd code

python main.py --mode train \
  --train-img-dir /path/to/train/images \
  --train-ann-path /path/to/train/_annotations.coco.json \
  --val-img-dir /path/to/valid/images \
  --val-ann-path /path/to/valid/_annotations.coco.json \
  --model-path best_model.pth
```

Annotations are COCO instance segmentation format, as exported by Roboflow. Training runs
20 epochs at batch size 4 with Focal Tversky loss (alpha 0.3, beta 0.7, gamma 0.75), which
weights recall over precision to limit missed canopies. Dataset preparation notebooks are
in [`../notebooks/`](../notebooks/) and expect a Roboflow key in the environment:

```bash
export ROBOFLOW_API_KEY="your-key"
```

Before sharing a checkpoint, drop the optimizer state. It roughly thirds the file size and
has no effect on inference or on fine-tuning:

```bash
python strip_checkpoint.py best_model.pth ../segmentation_model/best_model.pth
```

## Batch inference

```bash
cd code

python main.py --mode predict \
  --model-path ../segmentation_model/best_model.pth \
  --predict-dir /path/to/drone_images \
  --output-dir /path/to/results \
  --thresh 0.5
```

This is the pipeline that produced the thesis results. It writes `photos/` overlays and
`rasters/` GeoTIFFs, which then feed the ArcGIS Pro mosaic step described in
`PRODUCTION_GUIDE.md`. The standalone script in `../segmentation_model/` runs the same
detection logic and additionally writes PNG masks.

Two behaviours to know about:

- Images without GPS EXIF are still processed, but their GeoTIFF is written at coordinate
  (0, 0) and will not place correctly in GIS.
- `PIXEL_SIZE_M` in `predict_pipeline.py` is fixed at 0.02, a 2 cm ground sample distance.
  It scales the footprint linearly, so a different flight altitude needs a different value.

## Tuning detection

`--thresh` (default 0.5) is the probability cut on the sigmoid output. Lower it towards
0.35 if canopies are missed, raise it towards 0.6 if bare soil is picked up.

Instance splitting is controlled inside `postprocess.py`, in `adaptive_instance_split()`:

| Parameter | Default | Effect |
|---|---|---|
| `min_area` | 200 px | Smallest accepted object |
| `min_dist_between_trees` | 25 px | Minimum spacing between watershed seeds |
| `relative_area_factor` | 0.2 | Discards objects below 20 percent of the median tree area |
| solidity filter | 0.70 | Rejects non-compact shapes |

`min_dist_between_trees` is the one to adjust first. Raise it when one tree is split into
several instances, lower it when touching canopies merge into one. The default is tuned for
full bloom, when bright white canopies produce several distance peaks per tree.

## Repeatability

`get_val_test_transform()` in `data.py` includes Equalize, CLAHE and Sharpen at
probabilities of 0.5, 0.5 and 0.4, so repeated runs on the same image can differ slightly.
This is the behaviour behind the thesis results. The standalone script accepts
`--deterministic` to drop those three steps when repeatable output matters.

## Troubleshooting

**`KeyError: 'model_state_dict'`**
The file is a raw state dict rather than a training checkpoint. Load it with
`model.load_state_dict(torch.load(path))` instead of `load_trained_model()`.

**Out of memory**
Inference runs one image at a time, so this is usually the 1024 x 1024 tensor on a small
GPU. Use `--device cpu`, or reduce `IMAGE_SIZE` in `data.py` and retrain, since the encoder
is resolution sensitive.

**Masks look noisy or over-segmented**
Raise `--thresh` first, then `min_dist_between_trees`. Check the overlay JPGs in `photos/`
before trusting the rasters.
