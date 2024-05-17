ARG PYTORCH="2.2.1"
ARG CUDA="12.1"
ARG CUDNN="8"

FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel

RUN apt-get update \
    && apt-get install -y ffmpeg libsm6 libxext6 git ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --upgrade pip

WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN python -m pip install -r /app/requirements.txt
ENV PYTHONPATH=/app:$PYTHONPATH

#docker run --gpus all --rm -it --ipc host \
#  --mount type=bind,source=.,target=/app/ \
#  --mount type=bind,source=$(pwd)/data/coco,target=/app/data/coco \
#  --mount type=bind,source=/dev/shm,target=/dev/shm \
#  segdet bash

# in lowAltitude_segmentation/Mask2Former
#python train_net.py   --config-file configs/Drone_regrowth/semantic-segmentation/swin/maskformer2_swin_large_IN21k_384_bs16_160k_res640.yaml   --eval-only MODEL.WEIGHTS /home/kamyar/PycharmProjects/droneSegmentation/lowAltitude_segmentation/Mask2Former/output/model_0104999.pth