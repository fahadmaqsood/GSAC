import os
import os.path as osp
from glob import glob
import sys
import argparse
import shutil
import subprocess
import os
from pathlib import Path



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

# if only raw video is provided.

########################################### Preprocessor #####################################

subject_path = os.getcwd()+'/Data/'+subject_id + '/frames/'
os.chdir('Preprocessor')

cmd = ' python run.py --root_path ' + subject_id+" "  + " --gender "+ gender
print(cmd)
result = os.system(cmd)
print( "#############################################")
print( " ")
print( "          Preprocessor Job Done.")
print( " ")
print( "#############################################")

############################################# Avatar #########################################

os.chdir('../Avatar')

cmd = ' python main.py --base=./configs/GSAC_custom.yaml  --gender ' +  gender + ' ' +  '--train_subject ' + subject_id

result = os.system(cmd)


print( "#############################################")
print( " ")
print( "          Avatar Creator Job Done.")
print( " ")
print( "#############################################")

############################################# Move result to correct place #########################################

# MV the latest result to Data/Subject


log_dir = os.getcwd()+ "/logs/GSAC_custom/"

# Get all subdirectories inside the test-time directory
subdirs = [d for d in Path(log_dir).iterdir() if d.is_dir()]

# Find the latest folder based on modification time
if subdirs:
    latest_folder = max(subdirs, key=lambda d: d.stat().st_mtime)
    print("Latest folder:", latest_folder)
else:
    print("No folders found in", log_dir)
os.chdir('..')

source_path = os.path.join(os.getcwd()+'/Avatar/',log_dir,latest_folder)
target_path =os.path.join(os.getcwd() + '/Data/',subject_id)
cmd = 'mv ' + source_path + ' ' + target_path
result = os.system(cmd)


