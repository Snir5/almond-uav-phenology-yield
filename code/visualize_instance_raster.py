import numpy as np
import matplotlib.pyplot as plt
import os
import sys

def visualize_instance_raster(raster_path):
    labels = np.load(raster_path)
    if labels.ndim != 2:
        print(f"Unexpected raster dimensionality: {labels.shape}, expected 2D array.")
        return

    print(f"Raster shape: {labels.shape}, unique labels: {np.unique(labels)}")

    plt.figure(figsize=(8, 8))
    # Use a categorical colormap; background=0 will be masked as black
    cmap = plt.get_cmap('tab20')

    masked_labels = np.ma.masked_where(labels == 0, labels)
    plt.imshow(masked_labels, cmap=cmap, interpolation='nearest')
    plt.colorbar(label='Object ID')
    plt.title(f"Instance Raster: {os.path.basename(raster_path)}\n(black=background, colors=object IDs)")
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python visualize_instance_raster.py path/to/raster.npy")
    else:
        raster_path = sys.argv[1]
        visualize_instance_raster(raster_path)
