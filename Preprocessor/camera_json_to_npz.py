import numpy as np
import os
import json
import glob

xhuman_subject_name = 'Take99'
# Define paths
camera_folder = "data/Custom/data/rendong/cam_params"
output_folder = f"data/Custom/data/{xhuman_subject_name}/render/"
output_path = os.path.join(output_folder, "cameras.npz")


######################### Convert Camera  as xhuman format ###################################
# Create the subject folder if it does not exist
os.makedirs(output_folder, exist_ok=True)

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







# change name of depth 

import os
import cv2

# Define paths
xhuman_subject_name = 'Take99'
input_folder = "data/Custom/data/rendong/depthmaps"
output_folder = f"data/Custom/data/{xhuman_subject_name}/render/depth"

# Create the output folder if it does not exist
os.makedirs(output_folder, exist_ok=True)

# Get a sorted list of PNG files in the input folder
png_files = sorted([f for f in os.listdir(input_folder) if f.endswith('.png')], key=lambda x: int(os.path.splitext(x)[0]))

# Loop through each file, rename, and save as TIFF
for idx, file_name in enumerate(png_files, start=1):
    # Load the image
    img = cv2.imread(os.path.join(input_folder, file_name), cv2.IMREAD_UNCHANGED)
    
    # Define new file name in the specified format
    new_file_name = f"depth_{idx:06d}.tiff"
    output_path = os.path.join(output_folder, new_file_name)
    
    # Save the image as TIFF
    cv2.imwrite(output_path, img)
    print(f"Saved {output_path}")





########################## Ground truth image ##########################################


import os
import cv2

# Define paths
xhuman_subject_name = 'Take99'
input_folder = "data/Custom/data/rendong/frames"
output_folder = f"data/Custom/data/{xhuman_subject_name}/render/image"

# Create the output folder if it does not exist
os.makedirs(output_folder, exist_ok=True)

# Get a sorted list of PNG files in the input folder
png_files = sorted([f for f in os.listdir(input_folder) if f.endswith('.png')], key=lambda x: int(os.path.splitext(x)[0]))

# Loop through each file, rename, and save as PNG with a single layer (grayscale)
for idx, file_name in enumerate(png_files, start=1):
    # Load the image in grayscale mode
    img = cv2.imread(os.path.join(input_folder, file_name), cv2.IMREAD_GRAYSCALE)
    
    # Define new file name in the specified format
    new_file_name = f"depth_{idx:06d}.png"
    output_path = os.path.join(output_folder, new_file_name)
    
    # Save the image as PNG
    cv2.imwrite(output_path, img)
    print(f"Saved {output_path}")




######################### SMPLX ##################################################
import pickle

# Define paths
input_folder = "data/Custom/data/rendong/smplx_optimized1/smplx_params_smoothed"
betas_file = "data/Custom/data/rendong/smplx_optimized1/shape_param.json"
output_folder = "data/Custom/data/Take99/SMPLX"
os.makedirs(output_folder, exist_ok=True)

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
    output_file_name = f"mesh-f{idx:05d}_smplx.pkl"
    output_file_path = os.path.join(output_folder, output_file_name)

    # Save the new dictionary as a .pkl file
    with open(output_file_path, "wb") as f:
        pickle.dump(new_smpl_params, f)
    
    print(f"Converted and saved {output_file_path}")




    ############################### handling mask

    
# Define paths
xhuman_subject_name = 'Take99'
input_folder = "data/Custom/data/rendong/masks"
output_folder = f"data/Custom/data/{xhuman_subject_name}/render/mask"

# Create the output folder if it does not exist
os.makedirs(output_folder, exist_ok=True)

# Get a sorted list of PNG files in the input folder
png_files = sorted([f for f in os.listdir(input_folder) if f.endswith('.png')], key=lambda x: int(os.path.splitext(x)[0]))

# Loop through each file, rename, and save as PNG with a single layer (grayscale)
for idx, file_name in enumerate(png_files, start=1):
    # Load the image in grayscale mode
    img = cv2.imread(os.path.join(input_folder, file_name), cv2.IMREAD_GRAYSCALE)
    

    # Ensure the image is single-channel
    if len(img.shape) == 3:
        img = img[:, :, 0]  # Take the first channel if it has multiple channels
    
    # Define new file name in the specified format
    new_file_name = f"{idx}.png"
    output_path = os.path.join(output_folder, new_file_name)
    
    # Save the image as PNG (single-channel)
    cv2.imwrite(output_path, img)
    print(f"Saved {output_path}")