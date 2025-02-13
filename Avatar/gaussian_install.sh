#!/bin/bash

cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1

pip install ./3rd_party/diff_gaussian_rasterization-alphadep
pip install git+https://github.com/facebookresearch/pytorch3d.git
