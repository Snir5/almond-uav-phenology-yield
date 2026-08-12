#!/usr/bin/env python3
"""
review_test_set_evaluation.py

Scores the trained model on the 18 held-out test images of dataset v8 and
produces the numbers Tarin Paz-Kagan asked for in her review:

  [371] "The thesis should clearly define: IoU threshold for a true-positive
         instance, one-to-one matching procedure, treatment of splits and
         merges, minimum crown area, whether edge-truncated crowns are
         included, confidence intervals, and resampling unit. Report
         instance-level AP at multiple IoU thresholds."

  [547] "Confidence intervals may use the wrong independent unit ... clarify
         that the bootstrap is by source image, flight strip, or orchard block
         rather than by pixels."
        -> every confidence interval here resamples SOURCE IMAGES (n = 18),
           which is the independent unit. Pixel-level resampling would give
           intervals that are far too narrow.

  [293] "The binary threshold of 0.5 should be justified using validation data."
        -> sweeps the probability threshold and reports the operating curve.

  [347] "Include a parameter-sensitivity analysis."
        -> sweeps the watershed minimum-distance parameter.

  [544] "Provide results by ... image year"
        -> per-image results are written out so they can be grouped.

Run this on a machine with PyTorch installed:

    python3 pipeline/06_analysis/review_test_set_evaluation.py

Writes Results_Analysis/review_test_set_evaluation.json
"""
import json
import os
import re
import sys
from collections import defaultdict

import cv2
import numpy as np
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp
from scipy import ndimage as ndi
from skimage import measure, morphology, segmentation as _seg
from skimage.feature import peak_local_max

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from geo_guardrails import find_thesis_root

BASE = find_thesis_root()
OUT = os.path.join(BASE, "Results_Analysis")
WEIGHTS = os.path.join(BASE, "segmentation_model", "best_model.pth")
TEST_DIR = os.path.join(BASE, "Thesis_Writing", "References", "Training_and_Reports",
                        "Training", "Notebooks", "Almons-Trees-8", "test")

IMAGE_SIZE = 1024
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# ---- evaluation protocol, stated explicitly so the thesis can quote it -------
IOU_TP = 0.50                     # a predicted crown counts as a true positive
                                  # if its IoU with an unused ground-truth crown
                                  # is at least this value
MIN_CROWN_AREA = 200              # predicted objects below this are discarded
EDGE_CROWNS = "included"          # crowns truncated by the image border are kept
MATCHING = "greedy one-to-one, highest IoU first"
SPLIT_MERGE_COVER = 0.25          # a prediction counts as covering a crown when it
                                  # takes at least this fraction of the crown's area;
                                  # a crown covered by two or more such predictions is
                                  # a split, a prediction covering two or more crowns
                                  # is a merge
N_BOOT = 2000
BOOT_UNIT = "source image"

THRESHOLDS = [0.3, 0.4, 0.5, 0.6, 0.7]
MIN_DISTANCES = [15, 20, 25, 30, 35]
AP_IOUS = [round(x, 2) for x in np.arange(0.5, 0.96, 0.05)]


# --------------------------------------------------------------- preprocessing
def gray_world_wb(img):
    f = img.astype(np.float32)
    mean = f.reshape(-1, 3).mean(axis=0) + 1e-6
    return np.clip(f * (mean.mean() / mean), 0, 255).astype(np.uint8)


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


def adaptive_color(img, **kw):
    return adaptive_clahe(adaptive_gamma(gray_world_wb(img)))


# Deterministic: the probabilistic Equalize/CLAHE/Sharpen steps are dropped so
# that the reported metrics are reproducible.
TRANSFORM = A.Compose([
    A.Lambda(image=adaptive_color),
    A.Resize(IMAGE_SIZE, IMAGE_SIZE),
    A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ToTensorV2(),
])


def split_instances(mask, min_area=MIN_CROWN_AREA, min_dist=25,
                    separation_width=2, relative_area_factor=0.2):
    m = mask.astype(bool)
    if not m.any():
        return np.zeros_like(mask, np.uint32)
    m = morphology.remove_small_objects(m, min_size=int(min_area))
    dist = ndi.distance_transform_edt(m)
    coords = peak_local_max(dist, min_distance=min_dist, labels=m)
    peaks = np.zeros_like(dist, bool)
    if coords.size:
        peaks[tuple(coords.T)] = True
    markers, _ = ndi.label(peaks)
    lab = _seg.watershed(-dist, markers, mask=m)
    regions = measure.regionprops(lab)
    if not regions:
        return np.zeros_like(lab, np.uint32)
    dyn = max(min_area, float(np.median([r.area for r in regions])) * relative_area_factor)
    out = np.zeros_like(lab, np.uint32)
    k = 1
    for r in regions:
        if r.area >= dyn and r.solidity >= 0.70:
            out[lab == r.label] = k
            k += 1
    return out


# ------------------------------------------------------------ ground truth
def load_ground_truth():
    ann = json.load(open(os.path.join(TEST_DIR, "_annotations.coco.json")))
    by_img = defaultdict(list)
    for a in ann["annotations"]:
        by_img[a["image_id"]].append(a)
    items = []
    for im in ann["images"]:
        path = os.path.join(TEST_DIR, im["file_name"])
        if not os.path.exists(path):
            for ext in (".jpg", ".JPG", ".png"):
                if os.path.exists(path + ext):
                    path += ext
                    break
        if not os.path.exists(path):
            print(f"  missing image file, skipped: {im['file_name'][:60]}")
            continue
        h, w = im["height"], im["width"]
        # every ground-truth crown gets its own integer label in one image
        gt_lab = np.zeros((h, w), np.int32)
        k = 0
        for a in by_img[im["id"]]:
            m = np.zeros((h, w), np.uint8)
            for poly in a.get("segmentation", []):
                if len(poly) >= 6:
                    pts = np.array(poly, np.float32).reshape(-1, 2).astype(np.int32)
                    cv2.fillPoly(m, [pts], 1)
            if m.any():
                k += 1
                gt_lab[m.astype(bool)] = k
        items.append(dict(name=im["file_name"], path=path, h=h, w=w,
                          gt_lab=gt_lab, n_gt=k, gt_union=gt_lab > 0))
    return items


def overlap_table(pred_lab, gt_lab, n_pred, n_gt):
    """Pairwise intersection areas via a single pass over the pixels.

    Comparing every predicted crown against every ground-truth crown as full
    image masks is O(n_pred * n_gt * pixels) and takes hours at 4000 x 3000.
    Encoding both label images into one index and counting gives the same
    contingency table in a single pass.
    """
    idx = pred_lab.astype(np.int64) * (n_gt + 1) + gt_lab.astype(np.int64)
    counts = np.bincount(idx.ravel(), minlength=(n_pred + 1) * (n_gt + 1))
    inter = counts.reshape(n_pred + 1, n_gt + 1)[1:, 1:]      # drop background
    area_p = np.bincount(pred_lab.ravel(), minlength=n_pred + 1)[1:]
    area_g = np.bincount(gt_lab.ravel(), minlength=n_gt + 1)[1:]
    return inter, area_p, area_g


def match_instances(inter, area_p, area_g, iou_thr):
    """Greedy one-to-one matching, highest IoU first. Returns tp, fp, fn, splits, merges."""
    n_pred, n_gt = inter.shape
    if n_pred == 0 or n_gt == 0:
        return 0, n_pred, n_gt, 0, 0
    union = area_p[:, None] + area_g[None, :] - inter
    iou = np.where(union > 0, inter / np.maximum(union, 1), 0.0)

    pi, gi = np.nonzero(iou >= iou_thr)
    order = np.argsort(-iou[pi, gi])
    used_p, used_g = set(), set()
    tp = 0
    for k in order:
        p, g = int(pi[k]), int(gi[k])
        if p in used_p or g in used_g:
            continue
        used_p.add(p)
        used_g.add(g)
        tp += 1

    # Splits and merges are both judged on how much of each ground-truth crown a
    # prediction covers. A crown divided evenly in two leaves each piece covering
    # under half of it, so the coverage threshold has to sit below 0.5.
    cover_g = inter / np.maximum(area_g[None, :], 1)
    substantial = cover_g > SPLIT_MERGE_COVER
    splits = int((substantial.sum(axis=0) >= 2).sum())   # one crown, several predictions
    merges = int((substantial.sum(axis=1) >= 2).sum())   # one prediction, several crowns
    return tp, n_pred - tp, n_gt - tp, splits, merges


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def boot_ci(per_image, fn_pool, rng, n=N_BOOT):
    """Bootstrap over SOURCE IMAGES, the independent unit."""
    k = len(per_image)
    if k == 0:
        return (float("nan"), float("nan"))
    vals = []
    for _ in range(n):
        idx = rng.integers(0, k, k)
        vals.append(fn_pool([per_image[i] for i in idx]))
    return (round(float(np.percentile(vals, 2.5)), 4),
            round(float(np.percentile(vals, 97.5)), 4))


def main():
    if not os.path.exists(WEIGHTS):
        sys.exit(f"weights not found: {WEIGHTS}")
    device = torch.device("cuda" if torch.cuda.is_available()
                          else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"device: {device}")

    model = smp.UnetPlusPlus(encoder_name="efficientnet-b3", encoder_weights=None,
                             in_channels=3, classes=1, activation=None).to(device)
    ck = torch.load(WEIGHTS, map_location=device, weights_only=True)
    model.load_state_dict(ck.get("model_state_dict", ck))
    model.eval()

    items = load_ground_truth()
    print(f"test images: {len(items)}  ground-truth crowns: {sum(i['n_gt'] for i in items)}")

    # One inference pass per image. Probability maps are reused for every
    # threshold and watershed setting, and cached to disk so that a re-run does
    # not repeat the forward passes.
    # Probability maps are kept at the network's own 1024 x 1024 resolution,
    # because the watershed parameters (min_area, min_distance) are calibrated in
    # that space. Production thresholds and separates crowns at 1024 and only then
    # upscales the label image, and this evaluation must do the same.
    cache = os.path.join(OUT, "_test_prob_cache_1024.npz")
    probs = []
    if os.path.exists(cache):
        z = np.load(cache)
        if len(z.files) == len(items):
            probs = [z[f"p{i}"] for i in range(len(items))]
            print(f"reusing cached probability maps from {cache}")
    if not probs:
        for i, it in enumerate(items, 1):
            img = cv2.cvtColor(cv2.imread(it["path"]), cv2.COLOR_BGR2RGB)
            t = TRANSFORM(image=img)["image"].unsqueeze(0).to(device)
            with torch.no_grad():
                p = torch.sigmoid(model(t))[0, 0].cpu().numpy()
            probs.append(p)                       # native 1024 x 1024
            print(f"  [{i}/{len(items)}] {it['name'][:48]}")
        os.makedirs(OUT, exist_ok=True)
        np.savez_compressed(cache, **{f"p{i}": p for i, p in enumerate(probs)})

    gt_union = [it["gt_union"] for it in items]

    def score(prob, it, thr=0.5, min_dist=25):
        """Reproduce the production pipeline, then build the overlap table.

        Threshold and separate crowns at 1024, exactly as predict_pipeline.py
        does, and only then upscale the label image to the annotation
        resolution with nearest-neighbour interpolation.
        """
        lab = split_instances(prob > thr, min_dist=min_dist)
        if lab.shape != (it["h"], it["w"]):
            lab = cv2.resize(lab.astype(np.float32), (it["w"], it["h"]),
                             interpolation=cv2.INTER_NEAREST).astype(np.int32)
        pm = lab > 0                       # the mask production actually writes
        n_pred = int(lab.max())
        inter, ap_, ag_ = overlap_table(lab, it["gt_lab"], n_pred, it["n_gt"])
        return pm, inter, ap_, ag_

    rng = np.random.default_rng(42)
    res = dict(
        protocol=dict(iou_true_positive=IOU_TP, matching=MATCHING,
                      min_crown_area_px=MIN_CROWN_AREA,
                      edge_truncated_crowns=EDGE_CROWNS,
                      bootstrap_unit=BOOT_UNIT, bootstrap_iterations=N_BOOT,
                      preprocessing="deterministic (probabilistic steps disabled)"),
        n_test_images=len(items),
        n_ground_truth_crowns=int(sum(i["n_gt"] for i in items)),
    )

    # ---------------- headline: semantic and instance metrics at the defaults
    per_img = []
    base_tables = []                      # reused for the IoU sweep
    for pr, u, it in zip(probs, gt_union, items):
        pm, inter, ap_, ag_ = score(pr, it)
        base_tables.append((inter, ap_, ag_))
        tp, fp, fn, sp, mg = match_instances(inter, ap_, ag_, IOU_TP)
        per_img.append(dict(name=it["name"], inter=int(np.logical_and(pm, u).sum()),
                            psum=int(pm.sum()), gsum=int(u.sum()), tp=tp, fp=fp, fn=fn,
                            splits=sp, merges=mg, n_gt=it["n_gt"]))

    def pool_dice(rows):
        i = sum(r["inter"] for r in rows)
        return 2 * i / max(sum(r["psum"] + r["gsum"] for r in rows), 1)

    def pool_iou(rows):
        i = sum(r["inter"] for r in rows)
        return i / max(sum(r["psum"] + r["gsum"] for r in rows) - i, 1)

    def pool_p(rows):
        return prf(sum(r["tp"] for r in rows), sum(r["fp"] for r in rows), 0)[0]

    def pool_r(rows):
        return prf(sum(r["tp"] for r in rows), 0, sum(r["fn"] for r in rows))[1]

    def pool_f(rows):
        return prf(sum(r["tp"] for r in rows), sum(r["fp"] for r in rows),
                   sum(r["fn"] for r in rows))[2]

    head = {}
    for label, fn_pool in [("semantic_dice", pool_dice), ("semantic_iou", pool_iou),
                           ("instance_precision", pool_p), ("instance_recall", pool_r),
                           ("instance_f1", pool_f)]:
        head[label] = dict(value=round(fn_pool(per_img), 4),
                           ci95=boot_ci(per_img, fn_pool, rng))
    head["splits"] = int(sum(r["splits"] for r in per_img))
    head["merges"] = int(sum(r["merges"] for r in per_img))
    res["headline"] = head
    res["per_image"] = per_img

    # ---------------- AP across IoU thresholds
    ap = {}
    for thr in AP_IOUS:
        tp = fp = fn = 0
        for inter, ap_, ag_ in base_tables:          # segmentation reused
            a, b, c, _, _ = match_instances(inter, ap_, ag_, thr)
            tp += a; fp += b; fn += c
        p, r, f = prf(tp, fp, fn)
        ap[str(thr)] = dict(precision=round(p, 4), recall=round(r, 4), f1=round(f, 4))
        print(f"  IoU {thr:.2f}: P {p:.3f}  R {r:.3f}  F1 {f:.3f}")
    res["by_iou_threshold"] = ap
    res["mean_f1_50_95"] = round(float(np.mean([v["f1"] for v in ap.values()])), 4)

    # ---------------- probability threshold sweep  [293]
    sweep = {}
    for t in THRESHOLDS:
        rows = []
        for pr, u, it in zip(probs, gt_union, items):
            pm, inter, ap_, ag_ = score(pr, it, thr=t)
            a, b, c, _, _ = match_instances(inter, ap_, ag_, IOU_TP)
            rows.append(dict(inter=int(np.logical_and(pm, u).sum()),
                             psum=int(pm.sum()), gsum=int(u.sum()), tp=a, fp=b, fn=c))
        sweep[str(t)] = dict(dice=round(pool_dice(rows), 4), f1=round(pool_f(rows), 4),
                             precision=round(pool_p(rows), 4), recall=round(pool_r(rows), 4))
        print(f"  threshold {t}: Dice {sweep[str(t)]['dice']:.3f}  F1 {sweep[str(t)]['f1']:.3f}")
    res["threshold_sweep"] = sweep

    # ---------------- watershed sensitivity  [347]
    sens = {}
    for md in MIN_DISTANCES:
        tp = fp = fn = sp = mg = 0
        for pr, it in zip(probs, items):
            _, inter, ap_, ag_ = score(pr, it, min_dist=md)
            a, b, c, s, m = match_instances(inter, ap_, ag_, IOU_TP)
            tp += a; fp += b; fn += c; sp += s; mg += m
        p, r, f = prf(tp, fp, fn)
        sens[str(md)] = dict(precision=round(p, 4), recall=round(r, 4),
                             f1=round(f, 4), splits=sp, merges=mg)
        print(f"  min_distance {md}: P {p:.3f}  R {r:.3f}  F1 {f:.3f}  splits {sp}  merges {mg}")
    res["watershed_sensitivity"] = sens

    # ---------------- stratified recall  [544]
    # Detection is resolved per ground-truth crown so that recall can be broken
    # down by the factors Tarin listed: acquisition, canopy condition, crown size
    # and whether the crown is cut by the image border.
    strat = defaultdict(lambda: [0, 0])          # stratum -> [detected, total]
    for (inter, ap_, ag_), it, pr in zip(base_tables, items, probs):
        name = it["name"]
        cam = ("DJI" if name.startswith("DJI") else
               "IMG" if name.startswith("IMG") else "DSC")
        condition = "dormant" if cam == "DSC" else "leaf-on"
        # acquisition year: DJI frames carry it in the filename, the DSC survey is
        # 2021 and the IMG survey 2022
        m_year = re.search(r"_(20\d{2})\d{4}", name)
        season = (m_year.group(1) if m_year else
                  "2021" if cam == "DSC" else "2022" if cam == "IMG" else "undated")
        if inter.size == 0:
            continue
        union = ap_[:, None] + ag_[None, :] - inter
        iou = np.where(union > 0, inter / np.maximum(union, 1), 0.0)
        detected = (iou.max(axis=0) >= IOU_TP) if iou.shape[0] else np.zeros(len(ag_), bool)

        gt_lab = it["gt_lab"]
        med = float(np.median(ag_)) if len(ag_) else 0.0
        # crowns touching the frame edge
        border = set(np.unique(np.concatenate([gt_lab[0, :], gt_lab[-1, :],
                                               gt_lab[:, 0], gt_lab[:, -1]])))
        border.discard(0)
        for gi, area in enumerate(ag_, start=1):
            hit = bool(detected[gi - 1])
            # canopy condition and camera compare the whole test set
            keys = [f"condition={condition}", f"camera={cam}"]
            # the remaining strata are reported within the leaf-on scope only,
            # since pooling in the dormant images, where detection fails entirely,
            # would put a near-zero recall into every denominator and make the
            # comparisons between strata uninterpretable
            if condition == "leaf-on":
                keys += [f"season={season}",
                         "size=small (leaf-on)" if area < med else "size=large (leaf-on)",
                         "position=edge (leaf-on)" if gi in border
                         else "position=interior (leaf-on)"]
            for key in keys:
                strat[key][0] += int(hit)
                strat[key][1] += 1
    res["stratified_recall"] = {k: dict(detected=v[0], total=v[1],
                                        recall=round(v[0] / v[1], 4) if v[1] else None)
                                for k, v in sorted(strat.items())}
    print("\n=== recall by stratum (IoU 0.50) ===")
    for k, v in res["stratified_recall"].items():
        print(f"  {k:24s} {v['recall']:.3f}   ({v['detected']}/{v['total']})")

    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "review_test_set_evaluation.json"), "w") as f:
        json.dump(res, f, indent=2)

    print("\n=== HEADLINE, held-out test set, CIs bootstrapped by source image ===")
    for k in ("semantic_dice", "semantic_iou", "instance_precision",
              "instance_recall", "instance_f1"):
        v = head[k]
        print(f"  {k:20s} {v['value']:.4f}   95% CI [{v['ci95'][0]:.3f}, {v['ci95'][1]:.3f}]")
    print(f"  splits {head['splits']}   merges {head['merges']}")
    print(f"\nwrote {os.path.join(OUT, 'review_test_set_evaluation.json')}")


if __name__ == "__main__":
    main()
