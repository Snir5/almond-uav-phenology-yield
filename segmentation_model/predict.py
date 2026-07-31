"""Standalone almond tree segmentation.

Everything needed to run the trained model on new drone imagery lives in this one
file, so this folder can be copied out of the repository and used on its own.

    python predict.py --images /path/to/drone_photos --output /path/to/results

The detection logic is identical to the pipeline used for the thesis results
(`code/predict_pipeline.py`, `code/postprocess.py`, `code/data.py`).
"""

import argparse
import os
import shutil
import sys

import cv2
import numpy as np
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp
from scipy import ndimage as ndi
from skimage import color, measure, morphology, segmentation as _seg
from skimage.feature import peak_local_max
from skimage.morphology import binary_dilation, disk
from skimage.segmentation import find_boundaries

# ---------------------------------------------------------------- configuration

IMAGE_SIZE = 1024          # the resolution the encoder was trained at
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
PIXEL_SIZE_M = 0.02        # ground sample distance, 2 cm per pixel
EPSG_CODE = "32636"        # UTM Zone 36N
DEFAULT_WEIGHTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "best_model.pth")

# Georeferencing is optional. Without these packages the masks and overlays are
# still written, only the GeoTIFF export is skipped.
try:
    from PIL import Image
    import piexif
    from pyproj import Transformer
    import rasterio
    from rasterio.transform import from_origin
    GEO_AVAILABLE = True
except ImportError:
    GEO_AVAILABLE = False

# --------------------------------------------------------------- preprocessing

def gray_world_wb(img):
    imgf = img.astype(np.float32)
    mean = imgf.reshape(-1, 3).mean(axis=0) + 1e-6
    scale = mean.mean() / mean
    return np.clip(imgf * scale, 0, 255).astype(np.uint8)


def adaptive_gamma(img, target_v=0.5, clip=(0.8, 1.3)):
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    v = hsv[..., 2].astype(np.float32) / 255.0
    v_mean = float(np.clip(v.mean(), 0.05, 0.95))
    gamma = float(np.clip(np.log(v_mean) / np.log(max(target_v, 1e-6)), clip[0], clip[1]))
    x = (img.astype(np.float32) / 255.0) ** (1.0 / gamma)
    return np.clip(x * 255.0, 0, 255).astype(np.uint8)


def adaptive_clahe(img, base_clip=2.0, tile=(8, 8)):
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    v = hsv[..., 2]
    v_std = float(v.std()) / 255.0
    clip_limit = float(np.clip(base_clip + (0.8 - v_std) * 1.0, 1.5, 3.0))
    hsv[..., 2] = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile).apply(v)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


def adaptive_color(img, **kwargs):
    """White balance, then gamma, then contrast. Compensates for varying light."""
    return adaptive_clahe(adaptive_gamma(gray_world_wb(img)))


def build_transform(deterministic=False):
    """The validation and test transform the model was evaluated with.

    Equalize, CLAHE and Sharpen carry probabilities below 1, so by default the
    same image can give slightly different masks on repeat runs. Pass
    deterministic=True to drop them and get repeatable output.
    """
    steps = [A.Lambda(image=adaptive_color), A.Resize(IMAGE_SIZE, IMAGE_SIZE)]
    if not deterministic:
        steps += [
            A.Equalize(mode="cv", p=0.5),
            A.CLAHE(clip_limit=3.0, tile_grid_size=(8, 8), p=0.5),
            A.Sharpen(alpha=(0.1, 0.3), lightness=(0.7, 1.0), p=0.4),
        ]
    steps += [A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD), ToTensorV2()]
    return A.Compose(steps)

# ---------------------------------------------------------------------- model

def load_model(weights_path, device):
    model = smp.UnetPlusPlus(
        encoder_name="efficientnet-b3",
        encoder_weights=None,     # the checkpoint supplies every weight
        in_channels=3,
        classes=1,
        activation=None,
    ).to(device)

    ckpt = torch.load(weights_path, map_location=device, weights_only=True)
    state = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state)
    model.eval()
    return model

# ------------------------------------------------------- instance segmentation

def split_instances(pred_mask, min_area=200, min_dist_between_trees=25,
                    separation_width=2, relative_area_factor=0.2):
    """Separate touching canopies with a distance transform and watershed."""
    mask = pred_mask.astype(bool)
    if not mask.any():
        return np.zeros_like(pred_mask, dtype=np.uint32)

    mask = morphology.remove_small_objects(mask, min_size=int(min_area))
    distance = ndi.distance_transform_edt(mask)

    coords = peak_local_max(distance, min_distance=min_dist_between_trees, labels=mask)
    peaks = np.zeros_like(distance, dtype=bool)
    if coords.size > 0:
        peaks[tuple(coords.T)] = True
    markers, _ = ndi.label(peaks)

    labels = _seg.watershed(-distance, markers, mask=mask)

    regions = measure.regionprops(labels)
    if not regions:
        return np.zeros_like(labels, dtype=np.uint32)

    # Reject fragments: too small relative to the typical tree, or not compact.
    median_area = np.median([r.area for r in regions])
    dynamic_min_area = max(min_area, median_area * relative_area_factor)

    out = np.zeros_like(labels, dtype=np.uint32)
    counter = 1
    for region in regions:
        if region.area >= dynamic_min_area and region.solidity >= 0.70:
            out[labels == region.label] = counter
            counter += 1

    # Open a physical gap between neighbours so GIS treats them as separate.
    if separation_width > 0:
        boundaries = find_boundaries(out, mode="outer")
        if separation_width > 1:
            boundaries = binary_dilation(boundaries, disk(separation_width - 1))
        out[boundaries] = 0

    return out


def overlay_instances(image_rgb, labels, alpha=0.35):
    img_f = np.clip(image_rgb.astype(np.float32) / 255.0, 0, 1)
    vis = color.label2rgb(labels, image=img_f, bg_label=0, alpha=float(alpha))
    return (np.clip(vis, 0, 1) * 255).astype(np.uint8)

# --------------------------------------------------------------- georeferencing

def read_gps(image_path):
    """Latitude and longitude from the drone GPS in the image EXIF."""
    if not GEO_AVAILABLE:
        return None, None
    try:
        im = Image.open(image_path)
        if "exif" not in im.info:
            return None, None
        gps = piexif.load(im.info["exif"]).get("GPS", {})
        if 2 not in gps or 4 not in gps:
            return None, None

        def to_deg(val):
            d = float(val[0][0]) / float(val[0][1])
            m = float(val[1][0]) / float(val[1][1])
            s = float(val[2][0]) / float(val[2][1])
            return d + m / 60.0 + s / 3600.0

        lat, lon = to_deg(gps[2]), to_deg(gps[4])
        if gps.get(1) == b"S":
            lat = -lat
        if gps.get(3) == b"W":
            lon = -lon
        return lat, lon
    except Exception:
        return None, None


def save_geotiff(layers, path, utm_e, utm_n):
    """Write a 4-band GeoTIFF: R, G, B, tree mask."""
    data = np.moveaxis(layers, 2, 0)
    bands, height, width = data.shape

    # The drone records the image centre; a GeoTIFF needs the top-left corner.
    west = utm_e - (width * PIXEL_SIZE_M / 2.0)
    north = utm_n + (height * PIXEL_SIZE_M / 2.0)

    with rasterio.open(
        path, "w", driver="GTiff", height=height, width=width, count=bands,
        dtype=np.uint8, crs=f"EPSG:{EPSG_CODE}",
        transform=from_origin(west, north, PIXEL_SIZE_M, PIXEL_SIZE_M),
        nodata=0, compress="lzw",
    ) as dst:
        dst.write(data)
        for i, name in enumerate(["Red", "Green", "Blue", "Tree_Mask"], start=1):
            dst.update_tags(i, BAND_NAME=name)

# ------------------------------------------------------------------ prediction

def predict_image(img_path, model, device, transform, thresh=0.5, **split_kwargs):
    image = cv2.imread(img_path)
    if image is None:
        return None, None
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    tensor = transform(image=img_rgb)["image"].unsqueeze(0).to(device)
    with torch.no_grad():
        prob = torch.sigmoid(model(tensor))[0, 0].cpu().numpy()
    labels = split_instances((prob > thresh).astype("uint8"), **split_kwargs)

    # Back to the original resolution. cv2 will not resize int32 directly.
    if labels.shape != img_rgb.shape[:2]:
        labels = cv2.resize(
            labels.astype(np.float32),
            (img_rgb.shape[1], img_rgb.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        ).astype(np.uint32)

    return img_rgb, labels


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", required=True, help="folder of drone photos")
    ap.add_argument("--output", required=True, help="folder for results")
    ap.add_argument("--weights", default=DEFAULT_WEIGHTS, help="checkpoint path")
    ap.add_argument("--device", default=None, help="cuda, mps or cpu")
    ap.add_argument("--thresh", type=float, default=0.5,
                    help="probability cut on the mask, 0.35 catches more, 0.6 fewer")
    ap.add_argument("--min-distance", type=int, default=25,
                    help="minimum pixels between tree centres; raise if one tree "
                         "splits in two, lower if neighbours merge")
    ap.add_argument("--min-area", type=int, default=200,
                    help="smallest accepted object in pixels")
    ap.add_argument("--deterministic", action="store_true",
                    help="drop the random preprocessing steps for repeatable masks")
    args = ap.parse_args()

    if args.device:
        device_name = args.device
    elif torch.cuda.is_available():
        device_name = "cuda"
    elif torch.backends.mps.is_available():
        device_name = "mps"
    else:
        device_name = "cpu"
    device = torch.device(device_name)

    if not os.path.isfile(args.weights):
        sys.exit(f"weights not found: {args.weights}")

    photos_dir = os.path.join(args.output, "photos")
    masks_dir = os.path.join(args.output, "masks")
    rasters_dir = os.path.join(args.output, "rasters")
    for d in (photos_dir, masks_dir, rasters_dir):
        os.makedirs(d, exist_ok=True)

    print(f"device: {device_name}")
    print(f"weights: {args.weights}")
    if not GEO_AVAILABLE:
        print("note: rasterio, pyproj or piexif missing, skipping GeoTIFF export")

    files = sorted(f for f in os.listdir(args.images)
                   if f.lower().endswith((".jpg", ".jpeg", ".png")))
    if not files:
        sys.exit(f"no images found in {args.images}")

    model = load_model(args.weights, device)
    transform = build_transform(args.deterministic)

    total_trees, no_gps = 0, 0
    for i, fname in enumerate(files, start=1):
        img_path = os.path.join(args.images, fname)
        try:
            img_rgb, labels = predict_image(
                img_path, model, device, transform, args.thresh,
                min_area=args.min_area, min_dist_between_trees=args.min_distance,
            )
            if img_rgb is None:
                print(f"[{i}/{len(files)}] {fname}: unreadable, skipped")
                continue

            n_trees = int(labels.max())
            total_trees += n_trees

            stem = os.path.splitext(fname)[0]
            cv2.imwrite(os.path.join(photos_dir, fname),
                        cv2.cvtColor(overlay_instances(img_rgb, labels), cv2.COLOR_RGB2BGR))
            cv2.imwrite(os.path.join(masks_dir, stem + "_mask.png"),
                        ((labels > 0).astype(np.uint8) * 255))

            geo = ""
            if GEO_AVAILABLE:
                lat, lon = read_gps(img_path)
                if lat is None:
                    no_gps += 1
                    geo = "  (no GPS, GeoTIFF skipped)"
                else:
                    e, n = Transformer.from_crs(
                        "epsg:4326", f"epsg:{EPSG_CODE}", always_xy=True
                    ).transform(lon, lat)
                    layers = np.dstack([img_rgb, (labels > 0).astype(np.uint8)])
                    save_geotiff(layers, os.path.join(rasters_dir, stem + ".tif"), e, n)

            shutil.copystat(img_path, os.path.join(photos_dir, fname))
            print(f"[{i}/{len(files)}] {fname}: {n_trees} trees{geo}")

        except Exception as exc:
            print(f"[{i}/{len(files)}] {fname}: failed, {exc}")

    print(f"\n{total_trees} trees across {len(files)} images -> {args.output}")
    if no_gps:
        print(f"{no_gps} image(s) had no GPS EXIF and produced no GeoTIFF")


if __name__ == "__main__":
    main()
