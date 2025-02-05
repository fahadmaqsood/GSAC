import os
import shutil
import json
import numpy as np
import glob
import cv2

# Base path where CustomTake4 should be created
base_path = "/home/rendong/3DGS/ExAvatar_RELEASE/fitting/data/Custom/data/rendong_FP/"
base_folder = os.path.join(base_path, "CustomDataset")

# Define the structure
folders_to_create = [
 
    os.path.join(base_folder, "render", "image"),
    os.path.join(base_folder, "render", "mask"),
    os.path.join(base_folder, "SMPLX")
]

# Create the folders
for folder in folders_to_create:
    os.makedirs(folder, exist_ok=True)
    print(f"Created folder: {folder}")

print("Folder structure created successfully!")
####################################################### Depth ###################################

# # Process depthmaps
# depthmaps_path = os.path.join(base_path, "depthmaps")
# output_depth_path = os.path.join(base_folder, "render", "depth")


# # Ensure the output folder exists
# os.makedirs(output_depth_path, exist_ok=True)

# # Get the depth images in their original order
# depth_images = sorted(os.listdir(depthmaps_path), key=lambda x: int(os.path.splitext(x)[0]))

# # Loop through each file, rename, and save as TIFF
# for idx, file_name in enumerate(depth_images, start=1):
#     # Load the image
#     img = cv2.imread(os.path.join(depthmaps_path, file_name), cv2.IMREAD_UNCHANGED)
    
#     # Convert to grayscale if the image has 3 channels
#     if img.ndim == 3 and img.shape[2] == 3:
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
#     # Define new file name in the specified format
#     new_file_name = f"depth_{idx:06d}.tiff"
#     output_path = os.path.join(output_depth_path, new_file_name)
    
#     # Save the image as TIFF
#     cv2.imwrite(output_path, img)
#     print(f"Saved {output_path}")
####################################################### frames ###################################
# Process frames
frames_path = os.path.join(base_path, "frames")
output_image_path = os.path.join(base_folder, "render", "image")

# Ensure the output folder exists
os.makedirs(output_image_path, exist_ok=True)

# Get the frame images in their original order
frame_images = sorted(os.listdir(frames_path), key=lambda x: int(os.path.splitext(x)[0]))

# Copy and rename the frame images
for idx, file_name in enumerate(frame_images, start=1):
    fn = file_name.split('.')[0]
    input_file = os.path.join(frames_path, file_name)
    output_file = os.path.join(output_image_path, f"color_{int(fn):06d}.png")
    
    # Check the shape of the frame
    frame = cv2.imread(input_file)
    if frame is None:
        print(f"Warning: Unable to read {input_file}")
        continue
    
    print(f"Frame {file_name}: shape = {frame.shape}")
    
    # Copy and rename the file
    shutil.copy(input_file, output_file)
    print(f"Copied and renamed: {input_file} -> {output_file}")

print("Frame images copied and renamed successfully!")

###################################################### Mask ###################################

# # Process masks
# masks_path = os.path.join(base_path, "masks")
# output_mask_path = os.path.join(base_folder, "render", "mask")

# # Get a sorted list of PNG files in the input folder
# png_files = sorted([f for f in os.listdir(masks_path) if f.endswith('.png')], key=lambda x: int(os.path.splitext(x)[0]))

# # Loop through each file, rename, and save as PNG with a single layer (grayscale)
# for idx, file_name in enumerate(png_files, start=1):
#     # Load the image in grayscale mode
#     img = cv2.imread(os.path.join(masks_path, file_name), cv2.IMREAD_GRAYSCALE)
    
#     print(img.shape)
#     # Ensure the image is single-channel
#     if len(img.shape) == 3:
#         img = img[:, :, 0]  # Take the first channel if it has multiple channels
    
#     # Define new file name in the specified format
#     new_file_name = f"{idx}.png"
#     output_path = os.path.join(output_mask_path, new_file_name)
    
#     # Save the image as PNG (single-channel)
#     cv2.imwrite(output_path, img)
#     print(f"Saved {output_path}")



######################### Convert Camera  as xhuman format ###################################



# Define paths
output_path = os.path.join(base_folder+'/render/', "cameras.npz")
camera_folder = os.path.join(base_path, "cam_params")


# Create the subject folder if it does not exist

# Initialize lists for extrinsic matrices
extrinsic_matrices = []

# Get a sorted list of JSON files in the folder
json_files = sorted(glob.glob(os.path.join(camera_folder, "*.json")), key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))

# Loop through each JSON file
for json_file in json_files:
    with open(json_file, "r") as f:
        camera_params = json.load(f)
        
        # Extract rotation matrix R and translation vector t
        R = np.array(camera_params['R'])
        t = np.array(camera_params['t']).reshape(3, 1)
        
        # Create a 4x4 extrinsic matrix
        extrinsic = np.eye(4)
        extrinsic[:3, :3] = R
        extrinsic[:3, 3] = t.flatten()
        extrinsic_matrices.append(extrinsic)

    # Set up intrinsic matrix (assuming it is the same across frames)
    if json_file == json_files[0]:  # Only set intrinsic once
        fx, fy = camera_params['focal']
        cx, cy = camera_params['princpt']
        intrinsic_matrix = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0,  0,  1]
        ])

# Convert lists to numpy arrays
extrinsic_matrices = np.array(extrinsic_matrices)

# Save to .npz file
np.savez(output_path, extrinsic=extrinsic_matrices, intrinsic=intrinsic_matrix)
print(f"Camera parameters saved to {output_path}")






######################### SMPLX ##################################################
import pickle

# Define paths
# input_folder = "data/Custom/data/rendong/smplx_optimized1/smplx_params_smoothed"
# betas_file = "data/Custom/data/rendong/smplx_optimized1/shape_param.json"
# output_folder = "data/Custom/data/Take99/SMPLX"


# Define paths
input_folder = base_path + "/smplx_optimized/smplx_params"
betas_file = base_path + "/smplx_optimized/shape_param.json"
output_folder = base_folder + "/SMPLX/"

# Load the shared betas parameter and select only the first 10 values
with open(betas_file, "r") as f:
    betas = json.load(f)[:10]

# Map old keys to new keys and define reshaping logic
key_mapping = {
    "root_pose": "global_orient",
    "body_pose": "body_pose",
    "trans": "transl",
    "lhand_pose": "left_hand_pose",
    "rhand_pose": "right_hand_pose",
    "jaw_pose": "jaw_pose",
    "leye_pose": "leye_pose",
    "reye_pose": "reye_pose",
    "expr": "expression",
}

# Define reshaping rules for each key
reshape_rules = {
    "body_pose": (63,),         # Flatten (21, 3) to (63,)
    "lhand_pose": (45,),        # Flatten (15, 3) to (45,)
    "rhand_pose": (45,),        # Flatten (15, 3) to (45,)
}

# Get a sorted list of JSON files in the input folder
json_files = sorted([f for f in os.listdir(input_folder) if f.endswith('.json')], key=lambda x: int(os.path.splitext(x)[0]))

# Loop through each JSON file
for idx, file_name in enumerate(json_files, start=1):
    file_path = os.path.join(input_folder, file_name)
    fn = file_name.split('.')[0]
    with open(file_path, "r") as f:
        smpl_params = json.load(f)
    
    # Create a new dictionary with the modified structure
    new_smpl_params = {}
    for old_key, new_key in key_mapping.items():
        param_array = np.array(smpl_params[old_key])
        
        # Apply reshape rule if needed
        if old_key in reshape_rules:
            new_shape = reshape_rules[old_key]
            param_array = param_array.reshape(new_shape)
        
        # If this is the expression key, take only the first 10 elements
        if old_key == "expr":
            param_array = param_array[:10]
        
        # Add the reshaped/sliced array to the new dictionary
        new_smpl_params[new_key] = param_array.tolist()  # Convert back to list for JSON compatibility

    # Add the shared `betas` parameter (first 10 elements only)
    new_smpl_params["betas"] = betas

    # Define the output file name in the required format
    output_file_name = f"mesh-f{int(fn):05d}_smplx.pkl"
    output_file_path = os.path.join(output_folder, output_file_name)

    # Save the new dictionary as a .pkl file
    with open(output_file_path, "wb") as f:
        pickle.dump(new_smpl_params, f)
    
    print(f"Converted and saved {output_file_path}")
