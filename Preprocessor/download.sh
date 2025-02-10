#!/bin/bash


echo "Current working directory: $(pwd)"


if [ ! -d "common/utils" ]; then
    echo "Error: /common/utils directory does not exist."
    exit 1
fi



wget -O /tmp/human_model_files.tar \
    "https://huggingface.co/RendongZhang/GSAC-Dependencies/resolve/main/human_model_files.tar"
tar -xf /tmp/human_model_files.tar -C common/utils
rm /tmp/human_model_files.tar

wget -O /tmp/tools.tar \
    "https://huggingface.co/RendongZhang/GSAC-Dependencies/resolve/main/tools.tar"
tar -xf /tmp/tools.tar -C .
rm /tmp/tools.tar

wget -O /tmp/sapiens.tar \
    "https://huggingface.co/RendongZhang/GSAC-Dependencies/resolve/main/sapiens.tar"
tar -xf /tmp/sapiens.tar -C tools/
rm /tmp/sapiens.tar


# Get the installation path of torchgeometry
TORCHGEOMETRY_PATH=$(pip show torchgeometry | grep Location | awk '{print $2}')

# Ensure torchgeometry is installed
if [ -z "$TORCHGEOMETRY_PATH" ]; then
    echo "Error: torchgeometry is not installed!"
    exit 1
fi

# Define the target directory
TARGET_DIR="${TORCHGEOMETRY_PATH}/torchgeometry/core/"

# Ensure the target directory exists
if [ ! -d "$TARGET_DIR" ]; then
    echo "Error: Target directory $TARGET_DIR does not exist!"
    exit 1
fi

# Copy file1.py to the target directory
cp -r file1.py "$TARGET_DIR"












# echo "Download and setup completed!"