import os
import os
import os.path as osp
from glob import glob
import sys
import argparse

import sys
import os
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
root_path = args.root_path
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