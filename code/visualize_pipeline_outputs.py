import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def visualize_instance_mask(raster_path):
    labels = np.load(raster_path)
    if labels.ndim != 2:
        print(f"Label raster {raster_path} is not 2D. Skipping.")
        return None

    unique_labels = np.unique(labels)
    print(f"Raster '{os.path.basename(raster_path)}' unique labels: {unique_labels}")

    # Mask background as black
    masked_labels = np.ma.masked_where(labels == 0, labels)
    plt.imshow(masked_labels, cmap='tab20', interpolation='nearest')
    plt.colorbar(label='Object ID')
    plt.title('Instance Segmentation Mask')
    plt.axis('off')
    return plt

def visualize_extra_layers(extra_layers_path):
    layers = np.load(extra_layers_path)
    if layers.ndim != 3 or layers.shape[2] != 5:
        print(f"Extra layers file {extra_layers_path} unexpected shape {layers.shape}, skipping.")
        return None

    R, G, B = layers[:,:,0], layers[:,:,1], layers[:,:,2]
    grayscale = layers[:,:,3]
    ebi = layers[:,:,4]

    rgb_img = np.dstack([R,G,B]).astype(np.uint8)
    ebi_norm = (ebi - ebi.min()) / (ebi.max() - ebi.min() + 1e-9)

    fig, axs = plt.subplots(1,3, figsize=(15,5))
    axs[0].imshow(rgb_img)
    axs[0].set_title('RGB')
    axs[0].axis('off')

    axs[1].imshow(grayscale, cmap='gray')
    axs[1].set_title('Grayscale')
    axs[1].axis('off')

    im = axs[2].imshow(ebi_norm, cmap='jet')
    axs[2].set_title('EBI Index')
    axs[2].axis('off')
    fig.colorbar(im, ax=axs[2], fraction=0.046, pad=0.04)

    return plt

def print_annotation_json(json_path):
    if not os.path.exists(json_path):
        print(f"Annotation JSON {json_path} not found.")
        return
    with open(json_path, 'r') as f:
        data = json.load(f)
    print(f"Annotations for {os.path.basename(json_path)}:")
    for obj in data.get('objects', []):
        print(f"  Object ID: {obj['object_id']} - Image: {obj['image']}")

def visualize_one(image_path, raster_path, extra_layers_path, json_path):
    print(f"\nVisualizing image {os.path.basename(image_path)}")

    # Show annotated RGB image
    img = Image.open(image_path)
    plt.figure(figsize=(8,8))
    plt.imshow(img)
    plt.title('Annotated RGB Image')
    plt.axis('off')
    plt.show()

    # Visualize instance mask
    mask_plot = visualize_instance_mask(raster_path)
    if mask_plot:
        mask_plot.show()

    # Visualize extra layers
    extra_plot = visualize_extra_layers(extra_layers_path)
    if extra_plot:
        extra_plot.show()

    # Print annotation metadata
    print_annotation_json(json_path)

def main(output_root):
    photos_dir = os.path.join(output_root, 'photos')
    rasters_dir = os.path.join(output_root, 'rasters')

    photos = {f for f in os.listdir(photos_dir) if f.lower().endswith('.jpg')}
    rasters = {f for f in os.listdir(rasters_dir) if f.endswith('_raster.npy')}

    # Match files by base name ignoring suffixes
    for photo_name in sorted(photos):
        basename = os.path.splitext(photo_name)[0]
        raster_name = f"{basename}_raster.npy"
        extra_name = f"{basename}_extra_layers.npy"
        json_name = f"{basename}.json"

        raster_path = os.path.join(rasters_dir, raster_name)
        extra_layers_path = os.path.join(rasters_dir, extra_name)
        json_path = os.path.join(rasters_dir, json_name)
        photo_path = os.path.join(photos_dir, photo_name)

        if not (os.path.exists(raster_path) and os.path.exists(extra_layers_path)):
            print(f"Skipping {photo_name} - missing raster or extra layers.")
            continue

        visualize_one(photo_path, raster_path, extra_layers_path, json_path)

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python visualize_pipeline_outputs.py /path/to/output_root")
    else:
        root = sys.argv[1]
        main(root)
