#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=10-00:00
#SBATCH --job-name=classifier
#SBATCH --output=%x-%j.out

cd ~/droneSegmentation || exit
buildah build -t droneseg_cls --layers -f DockerfileClassif .

echo "Training on GPU $CUDA_VISIBLE_DEVICES"

podman run --gpus all --rm --ipc host \
    -e CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES \
    -v .:/app \
    -v ./data \
    -v ./data/iNaturalist_split: \
    -v output: \
    -v /dev/shm/:/dev/shm/ \
    droneseg_cls bash -c "python lowAltitude_classification/Dinov2_iNaturalist_classification_fine-tuning.py"

















