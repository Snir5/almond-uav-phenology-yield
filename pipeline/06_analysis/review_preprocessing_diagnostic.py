#!/usr/bin/env python3
"""
review_preprocessing_diagnostic.py

Two jobs.

1. Diagnose the DSC failure. On the held-out test set the model labels close to
   100 percent of every DSC image as canopy, while DJI and IMG images behave
   normally. This script checks whether that collapse is caused by the
   preprocessing chain or is a property of the model itself.

2. Answer Tarin Paz-Kagan's comment [275]: "Gray World, gamma correction, and
   CLAHE are conceptually justified, but their individual contributions should be
   quantified. I suggest comparing: raw images, white balance only, gamma only,
   CLAHE only, full pipeline."

For each variant it reports the fraction of pixels predicted as canopy and the
semantic Dice against the annotations, grouped by camera.

Writes Results_Analysis/review_preprocessing_diagnostic.json
"""
import json
import os
import sys
from collections import defaultdict

import cv2
import numpy as np
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from geo_guardrails import find_thesis_root

BASE = find_thesis_root()
OUT = os.path.join(BASE, "Results_Analysis")
WEIGHTS = os.path.join(BASE, "segmentation_model", "best_model.pth")
TEST_DIR = os.path.join(BASE, "Thesis_Writing", "References", "Training_and_Reports",
                        "Training", "Notebooks", "Almons-Trees-8", "test")
IMAGE_SIZE = 1024
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def gray_world_wb(img):
    f = img.astype(np.float32)
    m = f.reshape(-1, 3).mean(axis=0) + 1e-6
    return np.clip(f * (m.mean() / m), 0, 255).astype(np.uint8)


def adaptive_gamma(img, target_v=0.5, clip=(0.8, 1.3)):
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    v = hsv[..., 2].astype(np.float32) / 255.0
    vm = float(np.clip(v.mean(), 0.05, 0.95))
    g = float(np.clip(np.log(vm) / np.log(max(target_v, 1e-6)), *clip))
    return np.clip((img.astype(np.float32) / 255.0) ** (1.0 / g) * 255, 0, 255).astype(np.uint8)


def adaptive_clahe(img, base_clip=2.0, tile=(8, 8)):
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    v = hsv[..., 2]
    cl = float(np.clip(base_clip + (0.8 - float(v.std()) / 255.0), 1.5, 3.0))
    hsv[..., 2] = cv2.createCLAHE(clipLimit=cl, tileGridSize=tile).apply(v)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


# the individual contributions Tarin asked to see separated
VARIANTS = {
    "raw":            lambda im: im,
    "white_balance":  lambda im: gray_world_wb(im),
    "gamma":          lambda im: adaptive_gamma(im),
    "clahe":          lambda im: adaptive_clahe(im),
    "full_pipeline":  lambda im: adaptive_clahe(adaptive_gamma(gray_world_wb(im))),
}

TAIL = A.Compose([A.Resize(IMAGE_SIZE, IMAGE_SIZE),
                  A.Normalize(mean=MEAN, std=STD), ToTensorV2()])

# the production validation transform, whose Equalize / CLAHE / Sharpen steps
# fire with probability below one
PROD = A.Compose([
    A.Lambda(image=lambda x, **k: adaptive_clahe(adaptive_gamma(gray_world_wb(x)))),
    A.Resize(IMAGE_SIZE, IMAGE_SIZE),
    A.Equalize(mode="cv", p=0.5),
    A.CLAHE(clip_limit=3.0, tile_grid_size=(8, 8), p=0.5),
    A.Sharpen(alpha=(0.1, 0.3), lightness=(0.7, 1.0), p=0.4),
    A.Normalize(mean=MEAN, std=STD), ToTensorV2(),
])


def camera(name):
    return "DJI" if name.startswith("DJI") else ("DSC" if name.startswith("DSC") else "IMG")


def main():
    device = torch.device("cuda" if torch.cuda.is_available()
                          else "mps" if torch.backends.mps.is_available() else "cpu")
    model = smp.UnetPlusPlus(encoder_name="efficientnet-b3", encoder_weights=None,
                             in_channels=3, classes=1, activation=None).to(device)
    ck = torch.load(WEIGHTS, map_location=device, weights_only=True)
    model.load_state_dict(ck.get("model_state_dict", ck))
    model.eval()

    ann = json.load(open(os.path.join(TEST_DIR, "_annotations.coco.json")))
    by_img = defaultdict(list)
    for a in ann["annotations"]:
        by_img[a["image_id"]].append(a)

    results = defaultdict(lambda: defaultdict(list))
    per_image = []

    for im in ann["images"]:
        path = os.path.join(TEST_DIR, im["file_name"])
        if not os.path.exists(path):
            continue
        rgb = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
        h, w = im["height"], im["width"]

        gt = np.zeros((h, w), np.uint8)
        for a in by_img[im["id"]]:
            for poly in a.get("segmentation", []):
                if len(poly) >= 6:
                    cv2.fillPoly(gt, [np.array(poly, np.float32).reshape(-1, 2).astype(np.int32)], 1)
        gt_small = cv2.resize(gt, (IMAGE_SIZE, IMAGE_SIZE),
                              interpolation=cv2.INTER_NEAREST).astype(bool)

        row = dict(name=im["file_name"], camera=camera(im["file_name"]),
                   gt_fraction=round(float(gt_small.mean()), 4))

        for label, fn in list(VARIANTS.items()) + [("production_stochastic", None)]:
            if fn is None:
                torch.manual_seed(0)
                np.random.seed(0)
                t = PROD(image=rgb)["image"]
            else:
                t = TAIL(image=fn(rgb))["image"]
            with torch.no_grad():
                p = torch.sigmoid(model(t.unsqueeze(0).to(device)))[0, 0].cpu().numpy()
            pm = p > 0.5
            inter = np.logical_and(pm, gt_small).sum()
            dice = 2 * inter / max(pm.sum() + gt_small.sum(), 1)
            row[label] = dict(predicted_fraction=round(float(pm.mean()), 4),
                              dice=round(float(dice), 4))
            results[label][row["camera"]].append((float(pm.mean()), float(dice)))

        per_image.append(row)
        print(f"  {im['file_name'][:34]:34s} " + "  ".join(
            f"{k[:9]}:{row[k]['predicted_fraction']:.2f}/{row[k]['dice']:.2f}"
            for k in list(VARIANTS) + ["production_stochastic"]))

    summary = {}
    for label, cams in results.items():
        summary[label] = {c: dict(
            mean_predicted_fraction=round(float(np.mean([x[0] for x in v])), 4),
            mean_dice=round(float(np.mean([x[1] for x in v])), 4),
            n_images=len(v)) for c, v in cams.items()}

    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "review_preprocessing_diagnostic.json"), "w") as f:
        json.dump(dict(per_image=per_image, summary=summary), f, indent=2)

    print("\n=== mean predicted canopy fraction / mean Dice, by camera ===")
    cams = sorted({c for v in summary.values() for c in v})
    print(f"  {'variant':22s} " + "  ".join(f"{c:>16s}" for c in cams))
    for label in list(VARIANTS) + ["production_stochastic"]:
        cells = []
        for c in cams:
            d = summary[label].get(c)
            cells.append(f"{d['mean_predicted_fraction']:.2f} / {d['mean_dice']:.2f}".rjust(16)
                         if d else " " * 16)
        print(f"  {label:22s} " + "  ".join(cells))
    print("\n  a predicted fraction near 1.00 means the whole image was called canopy")


if __name__ == "__main__":
    main()
