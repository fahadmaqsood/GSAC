
FROM nvidia/cuda:12.1.0-devel-ubuntu20.04

ENV TZ=America/Chicago
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Install dependencies in a single RUN layer
RUN apt-get update && apt-get install -y \
    build-essential cmake ninja-build libglm-dev \
    libgl1-mesa-dev libegl1-mesa-dev libgles2-mesa-dev pkg-config \
    curl wget git unzip zip ffmpeg fish vim tmux && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Install Miniconda & Python dependencies
WORKDIR /opt
RUN wget -qO miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    chmod +x miniconda.sh && ./miniconda.sh -b -p /opt/miniconda && rm miniconda.sh
ENV PATH="/opt/miniconda/bin:$PATH"

COPY docker/environment.yml /tmp/environment.yml
RUN conda env create -f /tmp/environment.yml && conda clean -afy

COPY ../3rd_party/ /tmp/3rd_party
WORKDIR /tmp/3rd_party


###

###
# Install Python packages efficiently
RUN /opt/miniconda/bin/conda run -n textured-avatar pip install --no-cache-dir \
    lpips opencv-python imageio pyglet pyopengl smplx tensorboard av transforms3d plyfile\
    # package_name \
    git+https://github.com/karfly/nvdiffrast_compute-capability_6.0 \
    git+https://github.com/DmitryUlyanov/yamlenv.git \
    git+https://github.com/nghorbani/human_body_prior.git

ENV TORCH_CUDA_ARCH_LIST="Ada"
ENV FORCE_CUDA=1

RUN /opt/miniconda/bin/conda run -n textured-avatar pip install --no-cache-dir \ 
    git+https://github.com/facebookresearch/pytorch3d.git ./nvdiffrast/. 

RUN /opt/miniconda/bin/conda run -n textured-avatar pip install --no-cache-dir \ 
    ./diff_gaussian_rasterization-alphadep/. 


# Set up runtime environment variables
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility,graphics
ENV PYOPENGL_PLATFORM egl

# Set up user & fixuid
ARG USERNAME=user
RUN apt-get update && apt-get install -y sudo && \ 
    apt-get clean && rm -rf /var/lib/apt/lists/* && \
    addgroup --gid 1000 $USERNAME && \
    adduser --uid 1000 --gid 1000 --disabled-password --gecos '' $USERNAME && \
    adduser $USERNAME sudo && \
    echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers && \
    USER=$USERNAME && \
    GROUP=$USERNAME && \
    curl -SsL https://github.com/boxboat/fixuid/releases/download/v0.4/fixuid-0.4-linux-amd64.tar.gz | tar -C /usr/local/bin -xzf - && \
    chown root:root /usr/local/bin/fixuid && \
    chmod 4755 /usr/local/bin/fixuid && \
    mkdir -p /etc/fixuid && \
    printf "user: $USER\ngroup: $GROUP\n" > /etc/fixuid/config.yml
USER $USERNAME:$USERNAME

# Set Conda environment
ENV PATH="/opt/miniconda/bin:$PATH"
RUN echo "source activate textured-avatar" > ~/.bashrc

# Set working directory
WORKDIR /
ENTRYPOINT ["fixuid", "-q"]
CMD ["fixuid", "-q", "bash"]