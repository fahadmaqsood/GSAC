import os
import json
import pickle
import numpy as np
import math

object = 'male4/'
root = 'data/Custom/data/' + object

smplx_folder = os.path.join(root, "smplx_optimized", "smplx_params")
camera_folder = os.path.join(root, "cam_params")
results_folder = os.path.join(root, "SMPLX", "results")

betas_file = root + "/smplx_optimized/shape_param.json"
with open(betas_file, "r") as f:
    # Load the first 10 betas as float32
    betas = np.array(json.load(f)[:10], dtype=np.float32)


def focal2fov(focal, pixels):
    """Converts focal length -> FoV in radians."""
    return 2 * math.atan(pixels / (2 * focal))


def getProjectionMatrix(znear, zfar, fovX, fovY):
    """Builds a 4x4 perspective matrix using near/far planes and FoV in X,Y."""
    tanHalfFovX = math.tan(fovX / 2)
    tanHalfFovY = math.tan(fovY / 2)

    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    P = np.zeros((4, 4), dtype=np.float32)

    z_sign = 1.0  # typical OpenGL style

    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P


def build_simple_camera_matrices(camera_dict):
    """
    Builds your current 'camera_transform' & 'camera_matrix' 
    (just placing fx, fy, cx, cy in a 4x4).
    """
    R = np.array(camera_dict["R"], dtype=np.float32)  # shape (3,3)
    t = np.array(camera_dict["t"], dtype=np.float32)  # shape (3,)
    focal = np.array(camera_dict["focal"], dtype=np.float32)    # [fx, fy]
    princpt = np.array(camera_dict["princpt"], dtype=np.float32) # [cx, cy]

    # 1) camera_transform (extrinsic)
    camera_transform = np.eye(4, dtype=np.float32)
    camera_transform[:3, :3] = R
    camera_transform[:3, 3]  = t

    # 2) camera_matrix (simple pinhole in a 4x4)
    fx, fy = focal
    cx, cy = princpt
    camera_matrix = np.eye(4, dtype=np.float32)
    camera_matrix[0, 0] = fx
    camera_matrix[1, 1] = fy
    camera_matrix[0, 2] = cx
    camera_matrix[1, 2] = cy

    return camera_transform, camera_matrix


def build_target_perspective_matrix(focal, princpt, width, height, near=0.01, far=1.0):
    """
    Builds a 4x4 perspective matrix for some real-time 3D pipeline.
    Adjust near/far to replicate your specific matrix if needed.
    """
    fx, fy = focal
    cx, cy = princpt

    # We'll build a perspective matrix that maps [0..width], [0..height] to clip space.
    proj = np.zeros((4, 4), dtype=np.float32)

    # Scale X, Y by focal / image size
    proj[0, 0] = 2.0 * fx / width
    proj[1, 1] = 2.0 * fy / height

    # Shift center based on principal point
    proj[0, 2] = 1.0 - (2.0 * cx / width)
    proj[1, 2] = (2.0 * cy / height) - 1.0

    # Depth range transform
    proj[2, 2] = (far + near) / (far - near)
    proj[2, 3] = -(2.0 * far * near) / (far - near)

    # The perspective term
    proj[3, 2] = 1.0
    print(proj)
    return proj


def transform_smplx_dict(
    old_dict, 
    betas, 
    camera_transform=None, 
    camera_matrix=None, 
    camera_matrix_target=None
):
    """
    Convert old SMPL-X keys into your new format, 
    storing camera_transform, camera_matrix, AND a 'camera_matrix_target'.
    """
    new_dict = {}

    # Store betas as (1,10) float32
    new_dict["betas"] = betas.reshape(1, -1).astype(np.float32)

    # global_orient from root_pose => shape (1,3)
    if "root_pose" in old_dict:
        arr = np.array(old_dict["root_pose"], dtype=np.float32)
        new_dict["global_orient"] = arr.reshape(1, 3)

    # body_pose => from (21,3) => flatten => (1,63)
    if "body_pose" in old_dict:
        arr = np.array(old_dict["body_pose"], dtype=np.float32)
        arr = arr.reshape(-1)  # 63
        new_dict["body_pose"] = arr.reshape(1, 63)

    # transl => from trans => (1,3)
    if "trans" in old_dict:
        arr = np.array(old_dict["trans"], dtype=np.float32)
        new_dict["transl"] = arr.reshape(1, 3)

    # left_hand_pose => from (15,3) => flatten => slice 12 => (1,12)
    if "lhand_pose" in old_dict:
        arr = np.array(old_dict["lhand_pose"], dtype=np.float32)
        arr = arr.reshape(-1)[:12]
        new_dict["left_hand_pose"] = arr.reshape(1, 12)

    # right_hand_pose => from (15,3)
    if "rhand_pose" in old_dict:
        arr = np.array(old_dict["rhand_pose"], dtype=np.float32)
        arr = arr.reshape(-1)[:12]
        new_dict["right_hand_pose"] = arr.reshape(1, 12)

    # jaw_pose => (1,3)
    if "jaw_pose" in old_dict:
        arr = np.array(old_dict["jaw_pose"], dtype=np.float32)
        new_dict["jaw_pose"] = arr.reshape(1, 3)

    # leye_pose => (1,3)
    if "leye_pose" in old_dict:
        arr = np.array(old_dict["leye_pose"], dtype=np.float32)
        new_dict["leye_pose"] = arr.reshape(1, 3)

    # reye_pose => (1,3)
    if "reye_pose" in old_dict:
        arr = np.array(old_dict["reye_pose"], dtype=np.float32)
        new_dict["reye_pose"] = arr.reshape(1, 3)

    # expression => from expr(50,) => keep first 10 => (1,10)
    if "expr" in old_dict:
        arr = np.array(old_dict["expr"], dtype=np.float32)
        arr = arr[:10]
        new_dict["expression"] = arr.reshape(1, 10)

    # Now store the camera stuff
    if camera_transform is not None:
        new_dict["camera_transform"] = camera_transform.astype(np.float32)
    if camera_matrix is not None:
        new_dict["camera_matrix"] = camera_matrix.astype(np.float32)

    return new_dict


# Ensure results folder exists
os.makedirs(results_folder, exist_ok=True)

# Suppose you know your image size (for perspective):
image_width = 1080
image_height = 1080

# near/far for perspective
near = 0.01
far = 1.0

# Iterate over JSON files in smplx_folder, transform, and save .pkl
for filename in os.listdir(smplx_folder):
    if not filename.endswith(".json"):
        continue

    # e.g. "0.json" => base_name = "0" => file_index = 0
    base_name = os.path.splitext(filename)[0]
    file_index = int(base_name)

    # Build the subfolder name (e.g. "00000")
    subfolder_name = f"{file_index:05d}"
    subfolder_path = os.path.join(results_folder, subfolder_name)
    os.makedirs(subfolder_path, exist_ok=True)

    # 1) Load SMPL-X JSON
    smplx_path = os.path.join(smplx_folder, filename)
    with open(smplx_path, "r") as f:
        old_data = json.load(f)

    # 2) Load camera JSON (R, t, focal, princpt)
    camera_file = os.path.join(camera_folder, filename)
    cam_transform, cam_matrix = None, None
    cam_matrix_target = None

    if os.path.exists(camera_file):
        with open(camera_file, "r") as cf:
            cam_data = json.load(cf)

        # Build the original transform & pinhole matrix
        cam_transform, cam_matrix = build_simple_camera_matrices(cam_data)

        # Also build a "target" perspective matrix
        fx, fy = cam_data["focal"]
        cx, cy = cam_data["princpt"]
        cam_matrix_target = build_target_perspective_matrix(
            focal=(fx, fy),
            princpt=(cx, cy),
            width=image_width,
            height=image_height,
            near=near,
            far=far
        )

    # 3) Transform old SMPL-X to new format & embed camera data + perspective
    new_data = transform_smplx_dict(
        old_dict=old_data,
        betas=betas,
        camera_transform=cam_transform,
        camera_matrix=cam_matrix_target,
        camera_matrix_target=cam_matrix_target
    )

    # 4) Save as "000.pkl" in the subfolder
    pkl_filename = "000.pkl"
    pkl_path = os.path.join(subfolder_path, pkl_filename)
    with open(pkl_path, "wb") as f:
        pickle.dump(new_data, f)

    print(f"Saved {smplx_path} (+camera) -> {pkl_path}")
