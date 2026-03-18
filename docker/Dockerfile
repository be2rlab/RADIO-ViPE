FROM pytorch/pytorch:2.7.0-cuda12.8-cudnn9-devel AS base


SHELL [ "/bin/bash", "-c" ]
ENV DEBIAN_FRONTEND noninteractive
ENV USER=user

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        sudo git wget nano unzip ffmpeg libsm6 libxext6 ninja-build cmake build-essential libopenblas-dev \
        xterm xauth openssh-server tmux wget mate-desktop-environment-core && \
    rm -rf /var/lib/apt/lists/*



# Install dependencies for OpenGL and Nvidia driver
RUN apt-get update \
  && apt-get install -y -qq --no-install-recommends \
  libglvnd0 \
  libgl1 \
  libglx0 \
  libegl1 \
  libxext6 \
  libx11-6

# Set Nvidia-related environment variables
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES graphics,utility,compute
ENV QT_X11_NO_MITSHM 1

RUN apt-get update && apt-get install -y \
  libxkbcommon-x11-0 libxkbcommon0 \
  libx11-xcb1 libxcb1 libxcb-render0 libxcb-shape0 libxcb-xfixes0 \
  libgl1 libvulkan1

RUN pip install ultralytics \
                segment-anything-py \
                open-clip-torch \
                transformers==5.3.0
# RUN git clone https://github.com/facebookresearch/dinov3.git /home/${USER}/dinov3

COPY ../envs/ envs/

RUN pip install -r envs/requirements.txt


RUN apt update && apt install -y libeigen3-dev zlib1g-dev

RUN pip install scikit-learn
RUN pip install ftfy regex tqdm
RUN pip install transformers==4.52
RUN pip install segment-anything-py

WORKDIR /home/${USER}/km-vipe

CMD ["/bin/bash"]
