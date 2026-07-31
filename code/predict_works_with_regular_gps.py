import torch
import os
import cv2
import numpy as np
import json
from PIL import Image
import piexif
import shutil
import tifffile
import xattr

from model import build_model
from data import get_val_test_transform
from postprocess import adaptive_instance_split, overlay_instances_on_image

def extract_exif_dict(image_path):
    # Extract structured EXIF (all camera, GPS, etc)
    try:
        im = Image.open(image_path)
        if "exif" not in im.info:
            return None
        return piexif.load(im.info["exif"])
    except Exception as e:
        print(f"Warning: Failed to extract EXIF dict from {image_path}: {e}")
        return None

def copy_extended_attributes(src_path, dst_path):
    try:
        attrs = xattr.listxattr(src_path)
        for attr in attrs:
            value = xattr.getxattr(src_path, attr)
            xattr.setxattr(dst_path, attr, value)
    except Exception as e:
        print(f"Warning copying extended attributes from {src_path} to {dst_path}: {e}")

def augment_with_layers(rgb_img, labels):
    rgb = rgb_img.astype(np.uint8)
    R, G, B = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    labels_layer = labels.astype(np.int16)
    ebi = (2*B - (R+G)).astype(np.int16)
    extra_layers = np.dstack([R, G, B, labels_layer, ebi])
    return extra_layers

def get_polygons_for_object(labels, obj_id):
    mask = (labels == obj_id).astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygons = []
    for cnt in contours:
        epsilon = 0.01 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        polygon = approx.reshape(-1, 2).tolist()
        polygons.append(polygon)
    return polygons

def save_raster_tiff_with_exif(raster_layers, raster_path, exif_dict):
    # Save TIFF, then embed EXIF using piexif (camera/GPS/settings)
    # Step 1: Write TIFF data with tifffile
    if raster_layers.dtype not in [np.uint8, np.uint16, np.int16]:
        raster_layers = raster_layers.astype(np.int16)
    tifffile.imwrite(raster_path, raster_layers, photometric='minisblack')
    # Step 2: Write EXIF using PIL + piexif
    if exif_dict is not None:
        try:
            img = Image.open(raster_path)
            exif_bytes = piexif.dump(exif_dict)
            img.save(raster_path, exif=exif_bytes)
        except Exception as e:
            print(f"Warning: Could not embed EXIF info in TIFF: {e}")

def save_annotation_json(labels, image_name, output_json_path):
    object_ids = np.unique(labels)
    object_ids = [int(id_) for id_ in object_ids if id_ != 0]
    annotation_objects = []
    for obj_id in object_ids:
        mask = (labels == obj_id).astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        polygons = []
        for cnt in contours:
            epsilon = 0.01 * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)
            polygon = approx.reshape(-1, 2).tolist()
            polygons.append(polygon)
        annotation_objects.append({
            "object_id": obj_id,
            "polygons": polygons,
            "image": image_name
        })
    data = {"image": image_name, "objects": annotation_objects}
    with open(output_json_path, "w") as f:
        json.dump(data, f, indent=2)

def load_trained_model(model_path, device):
    model = build_model(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model

def predict_on_image(img_path, model, device, transform, thresh=0.5):
    image = cv2.imread(img_path)
    if image is None:
        print(f"ERROR: Could not load image: {img_path}. Skipping.")
        return None, None, None, None
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    try:
        augmented = transform(image=image_rgb)
        aug_image = augmented.get('image', None)
        if aug_image is None:
            print(f"ERROR: Augmentation returned None image for {img_path}. Skipping.")
            return None, None, None, None
    except Exception as e:
        print(f"ERROR during augmentation for {img_path}: {e}")
        return None, None, None, None
    image_t = aug_image.unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(image_t)
        prob = torch.sigmoid(logits)[0, 0].cpu().numpy()
        pred_mask = (prob > thresh).astype('uint8')
    if pred_mask is None or pred_mask.size == 0:
        print(f"WARNING: Prediction mask empty for {img_path}. Skipping.")
        return None, None, None, None
    labels = adaptive_instance_split(pred_mask)
    if labels.shape != image_rgb.shape[:2]:
        labels = cv2.resize(labels.astype('uint8'), (image_rgb.shape[1], image_rgb.shape[0]), interpolation=cv2.INTER_NEAREST)
    overlay_img = overlay_instances_on_image(image_rgb, labels)
    extra_layers = augment_with_layers(image_rgb, labels)
    return overlay_img, labels, image_rgb, extra_layers

def predict_folder(folder, model_path, device, output_root, transform, thresh=0.5):
    photos_dir = os.path.join(output_root, 'photos')
    rasters_dir = os.path.join(output_root, 'rasters')
    os.makedirs(photos_dir, exist_ok=True)
    os.makedirs(rasters_dir, exist_ok=True)
    model = load_trained_model(model_path, device)

    all_annotations = []
    for fname in os.listdir(folder):
        if fname.lower().endswith(".jpg"):
            img_path = os.path.join(folder, fname)
            overlay_img, labels, orig_img, extra_layers = predict_on_image(img_path, model, device, transform, thresh)
            if overlay_img is not None:
                base_name, ext = os.path.splitext(fname)
                out_img_path = os.path.join(photos_dir, fname)
                # Save overlay image
                cv2.imwrite(out_img_path, cv2.cvtColor(overlay_img, cv2.COLOR_RGB2BGR))
                exif_dict = extract_exif_dict(img_path)
                if exif_dict is not None:
                    try:
                        from PIL import Image as PILImage
                        img_overlay = PILImage.open(out_img_path)
                        exif_bytes = piexif.dump(exif_dict)
                        img_overlay.save(out_img_path, exif=exif_bytes)
                    except Exception as e:
                        print(f"Warning: Could not copy EXIF to overlay photo {fname}: {e}")
                try:
                    shutil.copystat(img_path, out_img_path)
                except Exception as e:
                    print(f"Warning: Failed to copy timestamps for {fname}: {e}")
                try:
                    copy_extended_attributes(img_path, out_img_path)
                except Exception as e:
                    print(f"Warning: Failed to copy extended attributes for {fname}: {e}")
                # Save 5-layer TIFF with EXIF from original image
                raster_path = os.path.join(rasters_dir, f"{base_name}_full_layers.tif")
                save_raster_tiff_with_exif(extra_layers, raster_path, exif_dict)
                try:
                    shutil.copystat(img_path, raster_path)
                except Exception as e:
                    print(f"Warning: Failed to copy timestamps for TIFF {base_name}: {e}")
                try:
                    copy_extended_attributes(img_path, raster_path)
                except Exception as e:
                    print(f"Warning: Failed to copy extended attributes for TIFF {base_name}: {e}")
                # Save per-image JSON annotations
                json_path = os.path.join(rasters_dir, f"{base_name}.json")
                save_annotation_json(labels, fname, json_path)
                # Collect objects/polygons for summary
                object_ids = np.unique(labels)
                object_ids = [int(id_) for id_ in object_ids if id_ != 0]
                object_entries = []
                for obj_id in object_ids:
                    polygons = get_polygons_for_object(labels, obj_id)
                    object_entries.append({
                        "object_id": obj_id,
                        "polygons": polygons
                    })
                all_annotations.append({
                    "image": fname,
                    "objects": object_entries
                })
            else:
                print(f"Skipping file {fname} due to previous error.")

    summary_path = os.path.join(output_root, "all_annotations_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(all_annotations, f, indent=2)
    print(f"Saved full annotation summary to {summary_path}")


    summary_path = os.path.join(output_root, "all_annotations_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(all_annotations, f, indent=2)
    print(f"Saved full annotation summary to {summary_path}")
