import os
import os.path as osp
from glob import glob
import sys
import argparse
import shutil
import subprocess
from main.config import cfg
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', type=str, dest='root_path')
    parser.add_argument('--use_colmap', dest='use_colmap', action='store_true')
    parser.add_argument('--gender',type=str, default='male')
    args = parser.parse_args()
    assert args.root_path, "Please set root_path."
    return args

args = parse_args()
subject = args.root_path
subject_id = subject.split('/')[-1]
if subject_id == '':
    subject_id = subject.split('/')[-2]

os.chdir('..')
gsac_path = os.getcwd()
subject_path = os.getcwd()+'/Data/'+subject_id + '/frames/'
print(subject_path)
os.chdir('Preprocessor')
root_path = os.getcwd() + '/data/Custom/data/'+subject_id +'/'
os.makedirs(root_path, exist_ok=True)  # Ensure destination exists

cmd = 'cp -r ' + subject_path +" " +root_path
print(cmd)
result = os.system(cmd)
print(root_path)
# print(abc)

gender = args.gender



# Get the absolute path of config.py
CONFIG_FILE = os.path.abspath(os.path.join(os.path.dirname(__file__),  'main', 'config.py'))

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', type=str, dest='root_path', required=True)
    parser.add_argument('--use_colmap', dest='use_colmap', action='store_true')
    parser.add_argument('--gender', type=str, default='male')
    args = parser.parse_args()
    return args

args = parse_args()

# Overwrite the gender value in config.py
def update_gender_in_config(new_gender):
    with open(CONFIG_FILE, 'r') as file:
        lines = file.readlines()

    # Modify the gender line
    with open(CONFIG_FILE, 'w') as file:
        for line in lines:
            if line.strip().startswith("gender ="):
                file.write(f"    gender = '{new_gender}'\n")  # Maintain indentation
            else:
                file.write(line)

    print(f"Updated gender to '{new_gender}' in {CONFIG_FILE}")

# Update the config file with new gender
update_gender_in_config(args.gender)





os.chdir('./tools')

cmd = 'python run.py --root_path ' + root_path
print(cmd)
result = os.system(cmd)
if (result != 0):
    print('EOOER.Terminate the script.')
    sys.exit()


# sapiens mask
import numpy as np
from PIL import Image

input_path= root_path + "/frames"
output_path = root_path + "/seg"
pretrained = gsac_path + '/Preprocessor/tools/sapiens/pretrain/checkpoints/seg/sapiens_1b_goliath_best_goliath_mIoU_7994_epoch_151_torchscript.pt2'
vis_file = gsac_path +'/Preprocessor/tools/sapiens/lite/demo/vis_seg.py'
cmd = 'bash sapiens/lite/scripts/demo/torchscript/seg.sh' + ' ' + input_path + ' ' + output_path + ' ' + pretrained + ' ' + vis_file
result = os.system(cmd)
if (result != 0):
    print('EOOER.Terminate the script.')
    sys.exit()


# move masks to correct place
input_folder = root_path + "/seg/sapiens_1b/"
output_folder = root_path + "train/render/mask/"

THRESHOLD = 128  

# Loop through all .npy files in the input folder
for npy_file in sorted(os.listdir(input_folder)):
    if npy_file.endswith(".npy") and "_seg.npy" not in npy_file:
        npy_path = os.path.join(input_folder, npy_file)
        
        # Load the .npy file (image as NumPy array)
        img_array = np.load(npy_path)
        
        # Normalize values to 0-255 (if not already)
        if img_array.max() <= 1:  # If values are in range [0,1], scale them
            img_array = img_array * 255

        # Convert to binary mask: Pixels above threshold become 255, else 0
        binary_mask = (img_array >= THRESHOLD).astype(np.uint8) * 255

        # Convert NumPy array to PIL Image
        img = Image.fromarray(binary_mask, mode="L")  # "L" mode for grayscale
        
        # Define output filename (same name but with .png)
        output_filename = os.path.splitext(npy_file)[0] + ".png"
        output_path = os.path.join(output_folder, output_filename)
        
        # Save as PNG
        img.save(output_path)
        print(f"Saved binary mask: {output_path}")

print("✅ All binary masks converted successfully!")


# MV the prerprocessed folder back to Data folder

source_folder = os.path.join(root_path, "train/")
target_folder = os.path.join(gsac_path, "Data", subject_id)

target_train_folder = target_folder +'/train/'
if os.path.exists(target_train_folder ):
    shutil.rmtree(target_train_folder ) 
# Ensure the target parent directory exists
os.makedirs(os.path.dirname(target_folder), exist_ok=True)

# Run the mv command
cmd = f"mv '{source_folder}' '{target_folder}'"
print(f"Running: {cmd}")  # Debugging output
subprocess.run(cmd, shell=True, check=True)

print(f"✅ Moved {source_folder} to {target_folder} successfully!")