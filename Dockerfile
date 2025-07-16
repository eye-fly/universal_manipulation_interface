FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04 AS system_deps

ENV DEBIAN_FRONTEND=noninteractive

# Install standard dependencies
RUN apt update && \
    apt install -y vim tzdata wget curl nano less git tmux build-essential software-properties-common cmake sudo iproute2 ptpd pipx libgl1 libglib2.0-0 && \
    rm /etc/localtime && \
    ln -s /usr/share/zoneinfo/Europe/Warsaw /etc/localtime && \
    curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | bash && \
    apt install -y git-lfs

# Install exiftool dependency
RUN apt install -y libimage-exiftool-perl

# Install ur_rtde dependencies
# RUN apt install -y libboost-system-dev libboost-thread-dev libboost-program-options-dev libboost-test-dev

# Install pyrealsense2 dependencies
# RUN apt install -y libusb-1.0-0

# Install ffmpeg from Conda, because it includes all the encoders
RUN wget -q -O /tmp/Miniforge3.sh "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh" && \
    bash /tmp/Miniforge3.sh -b -p /opt/conda
ENV PATH="/opt/conda/bin:$PATH"
RUN conda install -y -c conda-forge ffmpeg

WORKDIR /workspace/UMI

# # Install Lucid camera dependencies
# RUN apt install -y libibverbs1 librdmacm1 libx264-dev libx265-dev && \
#     mkdir -p /robot-interface-libs/lucid/build && \    
#     wget -O /robot-interface-libs/lucid/build/ArenaSDK_Linux.tar.gz https://storage.googleapis.com/artifacts-nomagic-ai/tar/lucid/ArenaSDK_v0.1.91_Linux_x64.tar.gz && \
#     cd /robot-interface-libs/lucid/build && \
#     tar -xvzf ArenaSDK_Linux.tar.gz && \
#     rm ArenaSDK_Linux.tar.gz && \
#     cd ArenaSDK_Linux_x64 && \
#     sh Arena_SDK_Linux_x64.conf

# Install uv
RUN pipx install uv==0.6.14
ENV PATH="/root/.local/bin/:$PATH"

# Install xauth for X11 forwarding
RUN apt-get update && \
    apt-get install -y xauth

# Install OpenGL libraries
RUN apt-get update && \
    apt-get install -y \
      libgl1-mesa-dri \
      libglx-mesa0 \
      mesa-utils \
      libosmesa6 \
      libosmesa6-dev


# install SLAM pipeline dependencies
RUN apt install -y libosmesa6-dev libglfw3 patchelf 
#missing libgl1-mesa-glx

#install miniforge
RUN curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
RUN bash Miniforge3-$(uname)-$(uname -m).sh -b

# Setup environment for SLAM pipeline 
COPY src/universal_manipulation_interface/conda_environment.yaml /tmp/
RUN mamba env create -f /tmp/conda_environment.yaml -y
      
# Setup python environment
ENV UV_LINK_MODE=copy
ENV UV_PROJECT_ENVIRONMENT=/.venv
RUN uv python install 3.11 && uv venv --python=3.11 $UV_PROJECT_ENVIRONMENT
ENV PATH="$UV_PROJECT_ENVIRONMENT/bin:$PATH"

FROM system_deps AS full_build



# Setup the project.
# Install dependencies in a separate layer.
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    GIT_LFS_SKIP_SMUDGE=1 uv sync --frozen --no-install-project
# Now copy the rest of the code and install the package.
COPY . .
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen