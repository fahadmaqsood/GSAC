import os
import os.path as osp
from glob import glob
import sys
import argparse
import shutil
import subprocess



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--subject', type=str)
    parser.add_argument('--gender', type=str)
    args = parser.parse_args()
    assert args.subject, "Please set subject."
    assert args.gender, "Please set gender."
    return args

args = parse_args()

subject_id = args.subject
gender = args.gender

############################################Preprocessor #####################################

subject_path = os.getcwd()+'/Data/'+subject_id + '/frames/'
os.chdir('Preprocessor')

cmd = ' python run.py --root_path ' + subject_id+" "  + " --gender "+ gender
print(cmd)
result = os.system(cmd)
# print(abc)
