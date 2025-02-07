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


# echo "Download and setup completed!"