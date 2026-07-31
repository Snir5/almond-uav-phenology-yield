import numpy as np
from scipy import ndimage as ndi
from skimage import morphology, measure, segmentation as _seg
from skimage.feature import peak_local_max
from skimage.morphology import disk, binary_dilation
from skimage.segmentation import find_boundaries
from skimage import color

def adaptive_instance_split(
    pred_mask,
    min_area=200,                # Minimum area for a tree (in pixels)
    min_dist_between_trees=25,   # INCREASED for Blooming: Centers are further apart in white canopy
    separation_width=2,          # Width of the gap between trees
    relative_area_factor=0.2
):
    """
    Advanced instance segmentation using Distance Transform and Watershed.
    Optimized for overlapping almond tree canopies with over-splitting prevention.
    """
    # Ensure input is boolean
    mask = pred_mask.astype(bool)
    if not mask.any():
        return np.zeros_like(pred_mask, dtype=np.uint32)

    # Step 1: Clean small noise and get distance map
    # Opening removes small white noise typical in blooming orchards
    mask = morphology.remove_small_objects(mask, min_size=int(min_area))
    
    # Calculate distance from the center of the tree to the edge
    distance = ndi.distance_transform_edt(mask)
    
    # Step 2: Find peaks (tree centers)
    # Using a larger min_distance prevents finding multiple peaks in one large white tree
    coords = peak_local_max(distance, min_distance=min_dist_between_trees, labels=mask)
    
    local_maxi = np.zeros_like(distance, dtype=bool)
    if coords.size > 0:
        local_maxi[tuple(coords.T)] = True
    
    markers, _ = ndi.label(local_maxi)

    # Step 3: Watershed Algorithm
    # Floods the topology from the markers to find boundaries
    labels = _seg.watershed(-distance, markers, mask=mask)

    # Step 4: Dynamic Filtering to prevent over-splitting
    regions = measure.regionprops(labels)
    if not regions:
        return np.zeros_like(labels, dtype=np.uint32)

    all_areas = [r.area for r in regions]
    median_area = np.median(all_areas) if all_areas else 0
    dynamic_min_area = max(min_area, median_area * relative_area_factor)
    
    final_labels = np.zeros_like(labels, dtype=np.uint32)
    label_counter = 1
    
    for region in regions:
        # Check against dynamic area threshold and solidity (how "round" it is)
        if region.area >= dynamic_min_area and region.solidity >= 0.70:
            final_labels[labels == region.label] = label_counter
            label_counter += 1

    # Step 5: Create Physical Gaps for GIS visualization
    if separation_width > 0:
        boundaries = find_boundaries(final_labels, mode='outer')
        if separation_width > 1:
            boundaries = binary_dilation(boundaries, disk(separation_width - 1))
        final_labels[boundaries] = 0
    
    return final_labels

def overlay_instances_on_image(image_rgb_uint8, labels, alpha=0.35):
    """
    Creates a visualization of the detected trees over the original image.
    """
    img_f = np.clip(image_rgb_uint8.astype(np.float32) / 255.0, 0, 1)
    vis = color.label2rgb(labels, image=img_f, bg_label=0, alpha=float(alpha))
    return (np.clip(vis, 0, 1) * 255).astype(np.uint8)