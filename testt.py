import os
import os.path as osp
from glob import glob
import sys
import argparse
import shutil
import subprocess
import os
from pathlib import Path
import cv2

def get_video(directory):
    # Common video file extensions
    video_extensions = {'.mp4', '.mov', '.avi'}

    # Get all video files in the directory
    video_files = [file for file in os.listdir(directory) 
                   if os.path.isfile(os.path.join(directory, file)) 
                   and os.path.splitext(file)[1].lower() in video_extensions]

    return video_files

def resize_with_human_focus(input_path, output_path):
    """Detects a human, crops around them, and resizes to 1080x1080."""
    # Load the image
    image = cv2.imread(input_path)
    height, width, _ = image.shape

    # Load pre-trained OpenCV human detector
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    # Detect people in the image
    boxes, _ = hog.detectMultiScale(image, winStride=(8,8), padding=(8,8), scale=1.05)

    if len(boxes) == 0:
        print(f"No human detected in {input_path}, cropping from center.")
        x, y, w, h = width//4, height//4, width//2, height//2  # Center crop
    else:
        # Get the largest detected human bounding box
        x, y, w, h = max(boxes, key=lambda box: box[2] * box[3])

    # Calculate a square crop around the detected person
    cx, cy = x + w // 2, y + h // 2  # Center of the detected person
    crop_size = max(w, h, 1080)  # Ensure crop is at least 1080x1080

    # Ensure the crop doesn't go outside the image
    x1 = max(cx - crop_size // 2, 0)
    y1 = max(cy - crop_size // 2, 0)
    x2 = min(cx + crop_size // 2, width)
    y2 = min(cy + crop_size // 2, height)

    # Crop and resize
    cropped = image[y1:y2, x1:x2]
    resized = cv2.resize(cropped, (1080, 1080), interpolation=cv2.INTER_AREA)

    # Save the result
    cv2.imwrite(output_path, resized)
    print(f"Processed: {input_path} -> {output_path}")

def process_entire_folder(input_folder, output_folder):
    """Processes all images in a folder and saves results."""

    # Process all image files in the input folder
    for img_path in Path(input_folder).glob("*.jpg"):  # Modify for other formats (e.g., "*.png")
        output_path = os.path.join(output_folder, img_path.name)
        resize_with_human_focus(str(img_path), output_path)
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--subject', type=str)
    parser.add_argument('--gender', type=str)
    parser.add_argument('--stage', type =str, default='all')
    args = parser.parse_args()
    assert args.subject, "Please set subject."
    assert args.gender, "Please set gender."
    return args

args = parse_args()

subject_id = args.subject
gender = args.gender

############################################ Video to images #################################
subject_folder = './Data/'+subject_id
video_name = get_video(subject_folder)
if len(video_name) > 1 :
    print('MORE THAN ONE VIDEO DETECTED.')
else:
    print(video_name)
    if not os.path.exists(subject_folder + '/frames/') :
        os.makedirs(subject_folder+'/frames/')  # Create the folder if it does not exist

    # if only raw video is provided.
        cmd = 'ffmpeg -i ' + subject_folder+'/'+video_name[0] + " -vf scale=720:1280,fps=5 "+ subject_folder+'/frames/%d.png  ' 
        result = os.system(cmd)
        print(' Frame extracted from '+ video_name[0])
    elif  not os.listdir(subject_folder+'/frames/'):
        cmd = 'ffmpeg -i ' + subject_folder+'/'+video_name[0] + " -vf scale=720:1280,fps=5 "+ subject_folder+'/frames/%d.png  ' 
        result = os.system(cmd)
        print(' Frame extracted from '+ video_name[0])


import os
import cv2
import numpy as np

# Input and output directories
input_folder = subject_folder + '/frames/'
output_folder = subject_folder + '/results/'

# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Target dimensions
target_pad_size = (1280, 1280)  # Zero padding to this size
final_size = (1080, 1080)       # Final resizing

# Process each image
for filename in os.listdir(input_folder):
    if filename.lower().endswith((".png", ".jpg", ".jpeg")):
        img_path = os.path.join(input_folder, filename)
        img = cv2.imread(img_path)

        if img is None:
            print(f"Skipping {filename}, couldn't read the image.")
            continue

        h, w = img.shape[:2]  # Original dimensions: 1280x720

        # Calculate padding (even padding on both left and right)
        pad_left = (target_pad_size[1] - w) // 2  # (1280 - 720) / 2 = 280
        pad_right = target_pad_size[1] - w - pad_left  # Ensure correct width

        # Apply zero padding
        padded_img = cv2.copyMakeBorder(img, 0, 0, pad_left, pad_right, cv2.BORDER_CONSTANT, value=(0, 0, 0))

        # Resize to 1080x1080
        resized_img = cv2.resize(padded_img, final_size, interpolation=cv2.INTER_CUBIC)

        # Save the processed image
        save_path = os.path.join(output_folder, filename)
        cv2.imwrite(save_path, resized_img)

        print(f"Processed and saved: {save_path}")

print("Batch processing completed!")
