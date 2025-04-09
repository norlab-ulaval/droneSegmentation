#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=4-00:00
#SBATCH --job-name=filtered-inat
#SBATCH --output=%x-%j.out

cd ~/droneSegmentation || exit
buildah build -t droneseg_cls --layers -f DockerfileClassif .

echo "Training on GPU $CUDA_VISIBLE_DEVICES"

container_id=$(
    podman run --gpus all --ipc host \
        -e CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES \
        -v .:/app/ \
        -v /data/iNat_Classifier_filtered:/app/data/iNat_Classifier_filtered \
        -v /dev/shm/:/dev/shm/ \
        -d droneseg_cls bash -c 'python lowAltitude_classification/filtered_iNat_classifier/iNat_Classifier_filtered.py'
)

stop_container() {
  podman container stop $container_id
  podman logs $container_id
  podman container rm $container_id
}

trap stop_container EXIT
echo "Container ID: $container_id"
podman wait $container_id
