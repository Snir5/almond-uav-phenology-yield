import os
import shutil
import cv2
import numpy as np
import torch
from PIL import Image
import piexif
import xattr
from pyproj import Transformer
import rasterio
from rasterio.transform import from_origin

from model import build_model
from postprocess import adaptive_instance_split, overlay_instances_on_image

# ---------- CONFIGURATION ----------
PIXEL_SIZE_M = 0.02  # GSD: 2 cm per pixel (Standard for low flight)
EPSG_CODE = "32636"  # Target CRS: UTM Zone 36N (Israel/Middle East)

# ---------- EXIF & GPS UTILITIES ----------

def extract_exif_dict(image_path):
    """Loads EXIF data safely."""
    try:
        im = Image.open(image_path)
        if "exif" in im.info:
            return piexif.load(im.info["exif"])
        return None
    except Exception:
        return None

def gps_exif_to_latlonalt(exif_dict):
    """
    Extracts Latitude, Longitude, AND Altitude from EXIF.
    Crucial for correct 3D positioning in ArcGIS.
    """
    gps = exif_dict.get("GPS", {})
    if not gps: return None, None, None

    # Check for required tags: 2 (Lat), 4 (Lon)
    if 2 not in gps or 4 not in gps: return None, None, None

    def to_deg(val):
        # Converts rational tuple (numerator, denominator) to float degrees
        d = float(val[0][0]) / float(val[0][1])
        m = float(val[1][0]) / float(val[1][1])
        s = float(val[2][0]) / float(val[2][1])
        return d + (m / 60.0) + (s / 3600.0)

    try:
        lat = to_deg(gps[2])
        lon = to_deg(gps[4])
        
        # Tag 6 is Altitude. Tag 5 is AltitudeRef (0=Above Sea Level)
        alt = 0.0
        if 6 in gps:
            alt = float(gps[6][0]) / float(gps[6][1])
            # If AltitudeRef is 1, it means below sea level (rare for almonds, but good to know)
            if 5 in gps and gps[5] == 1: 
                alt = -alt

        # Check references for South/West
        if gps.get(1) == b"S": lat = -lat
        if gps.get(3) == b"W": lon = -lon
        
        return lat, lon, alt
    except Exception:
        return None, None, None

def latlon_to_utm(lat, lon):
    """Converts WGS84 coordinates to the project's UTM Zone."""
    transformer = Transformer.from_crs("epsg:4326", f"epsg:{EPSG_CODE}", always_xy=True)
    easting, northing = transformer.transform(lon, lat)
    return easting, northing

def copy_extended_attributes(src_path, dst_path):
    """Preserves Mac/Linux file attributes."""
    try:
        attrs = xattr.listxattr(src_path)
        for attr in attrs:
            value = xattr.getxattr(src_path, attr)
            xattr.setxattr(dst_path, attr, value)
    except Exception: pass

# ---------- RASTER GENERATION ----------

def save_geotiff_optimized(raster_layers, raster_path, utm_e, utm_n):
    """
    Saves a 4-band GeoTIFF optimized for ArcGIS Mosaic Datasets.
    
    Args:
        raster_layers: Numpy array (H, W, 4) -> RGB + Mask
        raster_path: Output filename
        utm_e, utm_n: CENTER coordinates of the image (from Drone GPS)
    """
    # Move channels to first dimension: (H, W, C) -> (C, H, W)
    data = np.moveaxis(raster_layers, 2, 0)
    bands, height, width = data.shape

    # --- CRITICAL CALCULATION ---
    # Drone GPS is at the CENTER of the image. 
    # GeoTIFF Transform needs the TOP-LEFT corner coordinates.
    west = utm_e - (width * PIXEL_SIZE_M / 2.0)
    north = utm_n + (height * PIXEL_SIZE_M / 2.0)

    transform = from_origin(west, north, PIXEL_SIZE_M, PIXEL_SIZE_M)

    with rasterio.open(
        raster_path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=bands,
        dtype=np.uint8,
        crs=f"EPSG:{EPSG_CODE}",
        transform=transform,
        nodata=0,              # Defines transparency for background
        compress='lzw'         # Lossless compression to save disk space
    ) as dst:
        dst.write(data)
        
        # --- METADATA INJECTION ---
        # Naming bands helps ArcGIS auto-configure the RGB composition
        dst.update_tags(1, BAND_NAME='Red')
        dst.update_tags(2, BAND_NAME='Green')
        dst.update_tags(3, BAND_NAME='Blue')
        dst.update_tags(4, BAND_NAME='Tree_Mask')

# ---------- PREDICTION LOGIC ----------

def load_trained_model(model_path, device):
    model = build_model(device)
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model

def predict_on_image(img_path, model, device, transform, thresh=0.5):
    # Load and convert to RGB
    image = cv2.imread(img_path)
    if image is None: return None, None, None, None
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Preprocess for Model
    aug = transform(image=img_rgb)
    image_t = aug["image"].unsqueeze(0).to(device)

    # Inference
    with torch.no_grad():
        logits = model(image_t)
        prob = torch.sigmoid(logits)[0, 0].cpu().numpy()
        pred_mask = (prob > thresh).astype("uint8")

    # Instance Segmentation (Post-Process)
    labels = adaptive_instance_split(pred_mask)
    
    # Resize labels back to original image size safely
    # (OpenCV resize doesn't handle int32 well, so we use float trick)
    if labels.shape != img_rgb.shape[:2]:
        labels_float = labels.astype(np.float32)
        resized_float = cv2.resize(
            labels_float, 
            (img_rgb.shape[1], img_rgb.shape[0]), 
            interpolation=cv2.INTER_NEAREST
        )
        labels = resized_float.astype(np.uint32)

    # Create Outputs
    overlay = overlay_instances_on_image(img_rgb, labels)
    
    # Create 4-layer raster: RGB + Binary Mask (0/1)
    binary_mask = (labels > 0).astype(np.uint8)
    layers = np.dstack([img_rgb, binary_mask])
    
    return overlay, labels, img_rgb, layers

def predict_folder(folder, model_path, device, output_root, transform, thresh=0.5):
    # Setup directories
    photos_dir = os.path.join(output_root, "photos")
    rasters_dir = os.path.join(output_root, "rasters")
    os.makedirs(photos_dir, exist_ok=True)
    os.makedirs(rasters_dir, exist_ok=True)
    
    print(f"Loading model from: {model_path}")
    model = load_trained_model(model_path, device)

    print(f"Starting prediction on folder: {folder}")
    files = [f for f in os.listdir(folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    for i, fname in enumerate(files):
        img_path = os.path.join(folder, fname)
        print(f"[{i+1}/{len(files)}] Processing: {fname}...")
        
        try:
            # Run Prediction
            overlay, labels, _, layers = predict_on_image(img_path, model, device, transform, thresh)
            if overlay is None: 
                print(f"   Skipping {fname} (Read Error)")
                continue

            # Extract GPS Data
            exif = extract_exif_dict(img_path)
            utm_e, utm_n = 0.0, 0.0
            
            if exif:
                lat, lon, alt = gps_exif_to_latlonalt(exif)
                if lat and lon:
                    utm_e, utm_n = latlon_to_utm(lat, lon)
                    # Note: We are currently not using 'alt' in the filename/transform 
                    # but it is available here if you want to add it to metadata later.
            
            # Save Visual Photo (JPG)
            cv2.imwrite(os.path.join(photos_dir, fname), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
            
            # Save GeoTIFF (TIF)
            tif_name = os.path.splitext(fname)[0] + ".tif"
            save_geotiff_optimized(layers, os.path.join(rasters_dir, tif_name), utm_e, utm_n)

            # Metadata Sync (Preserve creation dates)
            shutil.copystat(img_path, os.path.join(photos_dir, fname))
            
        except Exception as e:
            print(f"   Error processing {fname}: {e}")

    print(f"\nProcessing complete. Outputs in: {output_root}")