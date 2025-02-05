import cv2
import os
import numpy as np
from glob import glob

# Paths to input folders
mask_folder = "/home/rendong/3DGS/ExAvatar_RELEASE/fitting/data/Custom/data/xhuman0016Take4/masks/"
depth_folder = "/home/rendong/3DGS/ExAvatar_RELEASE/fitting/data/Custom/data/xhuman0016Take4/depthmaps/"
output_folder = "/home/rendong/3DGS/ExAvatar_RELEASE/fitting/data/Custom/data/xhuman0016Take4/optimized_masks/"
os.makedirs(output_folder, exist_ok=True)
# Iterate through all mask images
for mask_path in glob(os.path.join(mask_folder, '*.png')):
    # Load the mask and depth images
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    depth_path = os.path.join(depth_folder, os.path.basename(mask_path))
    depth = cv2.imread(depth_path, cv2.IMREAD_GRAYSCALE)
    
    if depth is None:
        print(f"Depth file not found for {mask_path}")
        continue

    # 1. Normalize depth to [0, 1]
    depth_normalized = depth / 255.0

    # 2. Identify the depth-based foreground using Otsu's thresholding
    _, depth_foreground = cv2.threshold(depth, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 3. Refine the mask by combining with the depth-defined foreground
    combined_mask = cv2.bitwise_and(depth_foreground, mask)

    # 4. Morphological operations to clean up
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    refined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)  # Fill small holes
    refined_mask = cv2.morphologyEx(refined_mask, cv2.MORPH_OPEN, kernel)    # Remove noise

    # 5. Finalize: Set foreground to white and background to black
    final_mask = np.zeros_like(refined_mask)
    final_mask[refined_mask > 0] = 255

    # Save the final refined mask
    output_path = os.path.join(output_folder, os.path.basename(mask_path))
    cv2.imwrite(output_path, final_mask)
    print(f"Saved optimized mask to: {output_path}")