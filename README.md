# droneSegmentation
```shell
# Classif
buildah build -t droneseg_cls --layers -f DockerfileClassif .

export CUDA_VISIBLE_DEVICES=3
podman run --gpus all --rm --ipc host -it \
  -e CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES \
  -v .:/app \
  -v ./data \
  -v ./data/iNaturalist_split:/home/kamyar/Documents/filtered_inat_split/ \
  -v output:/home/kamyar/PycharmProjects/droneSegmentation/lowAltitude_classification \
  -v /dev/shm/:/dev/shm/ \
  droneseg_cls bash


podman run --device nvidia.com/gpu=all --rm --ipc host -it \
  -v .:/app \
  -v ~/Datasets/Drone_Unlabeled_Dataset_Patch_split:/data/Unlabeled_Drone_Dataset/Drone_Unlabeled_Dataset_Patch_split \
  -v ~/Datasets/Best_classifier_Weight:/data/Best_classifier_Weight \
  -v ~/Datasets/droneOut:/data/droneSegResults/ \
  -v output:/home/kamyar/PycharmProjects/droneSegmentation/lowAltitude_classification \
  -v /dev/shm/:/dev/shm/ \
  droneseg_cls bash

docker run --gpus=all --rm --ipc host -it \
  -v .:/app \
  -v ~/Datasets/Drone_Unlabeled_Dataset_Patch_split:/data/Unlabeled_Drone_Dataset/Drone_Unlabeled_Dataset_Patch_split \
  -v ~/Datasets/Best_classifier_Weight:/data/Best_classifier_Weight \
  -v ~/Datasets/droneOut:/data/droneSegResults/ \
  -v output:/home/kamyar/PycharmProjects/droneSegmentation/lowAltitude_classification \
  -v /dev/shm/:/dev/shm/ \
  droneseg_cls bash

SPLIT="Fifth batch" python lowAltitude_classification/Pseudo_dataset_LA_Classification_Fast.py
SPLIT="Third batch" python lowAltitude_classification/Pseudo_dataset_LA_Classification_Fast.py

# Seg
buildah build -t droneseg_seg --layers -f DockerfileSeg .

export CUDA_VISIBLE_DEVICES=3
podman run --gpus all --devices nvidia.com/gpu=all --rm --ipc host -it \
  -e CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES \
  -v .:/app \
  -v ./data \
  -v ./data/iNaturalist_split:/home/kamyar/Documents/filtered_inat_split/ \
  -v output:/home/kamyar/PycharmProjects/droneSegmentation/lowAltitude_classification \
  -v /dev/shm/:/dev/shm/ \
  droneseg_seg bash

sftp mamba-server
pwd , lpwd
put -r 'iNaturalist_split/'

cd ~/droneSegmentation
ln -s ~/Datasets/iNaturalist_split/ data/iNaturalist_split

# iNat
python lowAltitude_classification/Dinov2_iNaturalist_classification_fine-tuning.py

# Mask2Former
# Need to compile pixel_decoder in the container
cd lowAltitude_segmentation/Mask2Former/mask2former/modeling/pixel_decoder/ops/ && sh make.sh && cd -

python lowAltitude_segmentation/Mask2Former/train_net.py --config-file configs/Drone_regrowth/semantic-segmentation/swin/maskformer2_swin_large_IN21k_384_bs16_160k_res640.yaml --eval-only MODEL.WEIGHTS data/weights/model_0104999.pth
python lowAltitude_segmentation/Mask2Former/train_net.py   --config-file configs/Drone_regrowth/semantic-segmentation/swin/maskformer2_swin_large_IN21k_384_bs16_160k_res640.yaml   --eval-only MODEL.WEIGHTS /home/kamyar/PycharmProjects/droneSegmentation/lowAltitude_segmentation/Mask2Former/output/model_0104999.pth
```

TODO
1. Transfer data
2. Add SSH Key
3. git clone the project
4. make slurm script
5. sbatch + squeue
