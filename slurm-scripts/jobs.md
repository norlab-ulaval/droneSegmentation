| **Computer** | **Model**     | **ETA**       |
|--------------|---------------|---------------|
| WGM          | Swin base     | 1d 4h         |
| DAD          | Resnet base   | 17h           |
| Mamba 1      | Resnet SSD    | Just finished |
| Mamba 2      | Swin SSD      | 21h           |
| Valeria 1    | Swin base     | 1d 15h        |
| Valeria 2    | Swin dice     | 1d 15h        |
| Valeria 3    | Swin colaug   | 1d 15h        |
| Valeria 4    | Swin crop     | 1d 15h        |
| Valeria 5    | Resnet base   | 1d 15h        |
| Valeria 6    | Resnet dice   | 1d 15h        |
| Valeria 7    | Resnet colaug | 1d 15h        |
| Valeria 8    | Resnet crop   | 1d 15h        |

| **Model**     | **Computer** | **Status** | **ETA** | **Config**                          |
|---------------|--------------|------------|---------|-------------------------------------|
| Swin base     | Valeria      | Running    | 1d 4h   | M2F_Swin_Large_base                 |
| Swin dice     | Valeria      | Running    | 1d 15h  | M2F_Swin_Large_ClassMaskDice_Weight |
| Swin colaug   | Valeria      | Running    | 1d 15h  | M2F_Swin_Large_colorAugs            |
| Swin crop     | Valeria      | Running    | 1d 15h  | M2F_Swin_Large_Crop_512             |
| Swin train    | WGM          | Done       | ---     | M2F_Swin_Large_MaxTrainSize_1024    |
| Swin SSD      | Mamba        | Running    | 21h     | M2F_Swin_Large_SSD_default          |
| Resnet base   | Valeria      | Running    | 16h     | M2F_ResNet50_base                   |
| Resnet dice   | Valeria      | Running    | 1d 15h  | M2F_ResNet50_ClassMaskDice_Weight   |
| Resnet colaug | Valeria      | Running    | 1d 15h  | M2F_ResNet50_colorAugs              |
| Resnet crop   | Valeria      | Running    | 1d 15h  | M2F_ResNet50_Crop_512               |
| Resnet train  | DAD          | Done       | ---     | M2F_ResNet50_MaxTrainSize_1024      |
| Resnet SSD    | Mamba        | Done       | ---     | M2F_ResNet50_SSD_default            |
| Swin LR       | WGM          | Running    | 1d 20h  | M2F_Swin_Large_LR                   |
| Swin small    | DAD          | Running    | 1d 16h  | M2F_Swin_Large_smallSize            |

# Fixed the bug

| **Model**    | **Computer** | **Status** | **Config**                          |
|--------------|--------------|------------|-------------------------------------|
| Swin base    | DAD          | Done       | M2F_Swin_Large_base                 |
| Swin dice    | WGM          | Done       | M2F_Swin_Large_ClassMaskDice_Weight |
| Swin colaug  | Mamba        | Running    | M2F_Swin_Large_colorAugs            |
| Swin SSD     | Mamba        | Running    | M2F_Swin_Large_SSD_default          |
| Swin crop    | Mamba        | Running    | M2F_Swin_Large_Crop_512             |
| Swin train   | Mamba        | Running    | M2F_Swin_Large_MaxTrainSize_1024    |
| Swin lr      | Valeria      | Running    | M2F_Swin_Large_LR                   |
| Swin Crop256 | WGM          | Done       | M2F_Swin_Large_Crop256              |

TODO:

- Fine-tune the best model
- New base config
- Supervised training
- Generate pseudolabels

| **Computer** | **Task**                           |
|--------------|------------------------------------|
| KN           |                                    |
| WGM          | Finetuning new -> new gen pl train |
| DAD          | Supervised new                     |
| Titan X      | PL 5                               |
| Mamba 0      | New base                           |
| Mamba 1      | PL 1                               |
| Mamba 2      | PL 3                               |
| Mamba 3      | PL 4                               |
| Valeria 1    |                                    |
| Valeria 2    |                                    |

# Scratch for finetuning

```shell
docker run --gpus=all --rm --ipc host -it \
  -v .:/app \
  -v ~/Datasets/drone_dataset:/data/drone_dataset \
  -v ~/Datasets/M2F_Train_Val_split/:/data/drone_annotated \
  -v /dev/shm/:/dev/shm/ \
  droneseg bash
  
pip install -U pip
pip install -r lowAltitude_segmentation/Mask2Former/requirements.txt
cd /app/lowAltitude_segmentation/Mask2Former/mask2former/modeling/pixel_decoder/ops
export MAX_JOBS=16
sh make.sh

cd /data/drone_annotated/train/images
for f in `find * -type f | grep .jpg`; do mv -- "$f" "${f%.jpg}.JPG"; done
cd /data/drone_annotated/val/images
for f in `find * -type f | grep .jpg`; do mv -- "$f" "${f%.jpg}.JPG"; done

cd /app
export SLURM_TMPDIR=/data/
export SPLIT='DL'
python lowAltitude_segmentation/Mask2Former/mask2former/data/datasets/register_drone_semantic.py

PYTHONPATH=$PYTHONPATH:. python lowAltitude_segmentation/Mask2Former/train_net.py --num-gpus 1 \
  --config-file lowAltitude_segmentation/Mask2Former/configs/Drone_regrowth/semantic-segmentation/swin/M2F_Swin_Large_base_finetuning.yaml
```

# Scratch for supervised training

```shell
docker run --gpus=all --rm --ipc host -it \
  -v .:/app \
  -v ~/Datasets/drone_dataset:/data/drone_dataset \
  -v ~/Datasets/M2F_Train_Val_split/:/data/drone_annotated \
  -v /dev/shm/:/dev/shm/ \
  droneseg bash
  
pip install -U pip
pip install -r lowAltitude_segmentation/Mask2Former/requirements.txt
cd /app/lowAltitude_segmentation/Mask2Former/mask2former/modeling/pixel_decoder/ops
export MAX_JOBS=16
sh make.sh

cd /data/drone_annotated/train/images
for f in `find * -type f | grep .jpg`; do mv -- "$f" "${f%.jpg}.JPG"; done
cd /data/drone_annotated/val/images
for f in `find * -type f | grep .jpg`; do mv -- "$f" "${f%.jpg}.JPG"; done

cd /app
export SLURM_TMPDIR=/data/
export SPLIT='DL'
python lowAltitude_segmentation/Mask2Former/mask2former/data/datasets/register_drone_semantic.py

PYTHONPATH=$PYTHONPATH:. python lowAltitude_segmentation/Mask2Former/train_net.py --num-gpus 1 \
  --config-file lowAltitude_segmentation/Mask2Former/configs/Drone_regrowth/semantic-segmentation/swin/M2F_Swin_Large_base_supervised.yaml
```

# Scratch new config

```shell
docker run --gpus=all --rm --ipc host -it \
  -v .:/app \
  -v ~/Datasets/drone_dataset_3.4:/data/drone_dataset_3.4 \
  -v ~/Datasets/M2F_Train_Val_split/:/data/drone_annotated \
  -v ./output_3.4:/app/output \
  -v /dev/shm/:/dev/shm/ \
  droneseg bash
  
# ---
export CUDA_VISIBLE_DEVICES=3
export SPLIT='1.8'
docker run --gpus=all --rm --ipc host -it \
  -e CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES \
  -e SPLIT=$SPLIT \
  -v .:/app \
  -v /data/drone_dataset_"$SPLIT":/data/drone_dataset_"$SPLIT" \
  -v /data/M2F_Train_Val_split/:/data/drone_annotated \
  -v "./output_$SPLIT":/app/output \
  -v /dev/shm/:/dev/shm/ \
  droneseg bash

pip install -U pip
pip install -r lowAltitude_segmentation/Mask2Former/requirements.txt
cd /app/lowAltitude_segmentation/Mask2Former/mask2former/modeling/pixel_decoder/ops
export MAX_JOBS=16
sh make.sh

cd /app
export SLURM_TMPDIR=/data/
python lowAltitude_segmentation/Mask2Former/mask2former/data/datasets/register_drone_semantic.py

PYTHONPATH=$PYTHONPATH:. python lowAltitude_segmentation/Mask2Former/train_net.py --num-gpus 1 --config-file lowAltitude_segmentation/Mask2Former/configs/Drone_regrowth/semantic-segmentation/swin/M2F_Swin_Large_base.yaml
  
PYTHONPATH=$PYTHONPATH:. python lowAltitude_segmentation/Mask2Former/train_net.py --num-gpus 1 --config-file lowAltitude_segmentation/Mask2Former/configs/Drone_regrowth/semantic-segmentation/swin/M2F_Swin_Large_Crop640.yaml
```

# Scratch pad for PL generation

```shell
docker build -t droneseg_cls -f DockerfileClassif .

docker run --gpus=all --rm --ipc host -it \
  -e CUDA_VISIBLE_DEVICES=2 \
  -v .:/app \
  -v /data/:/data/ \
  -v output:/home/kamyar/PycharmProjects/droneSegmentation/lowAltitude_classification \
  -v /dev/shm/:/dev/shm/ \
  droneseg_cls bash

docker run --gpus=all --rm --ipc host -it \
  -v .:/app \
  -v ~/Datasets/droneOut/:/data/droneSegResults \
  -v ~/Datasets/Best_classifier_Weight/:/data/Best_classifier_Weight \
  -v ~/Datasets/Drone_Unlabeled_Dataset_Patch_split:/data/Unlabeled_Drone_Dataset/Drone_Unlabeled_Dataset_Patch_split \
  -v output:/home/kamyar/PycharmProjects/droneSegmentation/lowAltitude_classification \
  -v /dev/shm/:/dev/shm/ \
  droneseg_cls bash
  
export SPLIT=Fifth
export SUBSPLIT=1
export CUDA_VISIBLE_DEVICES=0
export SUBSPLIT=2
export CUDA_VISIBLE_DEVICES=1
export SUBSPLIT=3
export CUDA_VISIBLE_DEVICES=2
export SUBSPLIT=4
export CUDA_VISIBLE_DEVICES=2

python lowAltitude_classification/Pseudo_dataset_CENTER_Padded_184_PL_generation.py
```

# Classifier

```shell
docker run --gpus=all --rm --ipc host -it \
  -v .:/app \
  -v ~/Datasets/iNat_Classifier_Non_filtered:/home/kamyar/Documents/iNat_Classifier_Non_filtered \
  -v ./lowAltitude_classification/:/home/kamyar/PycharmProjects/droneSegmentation/lowAltitude_classification/ \
  -v classif_output:/home/kamyar/PycharmProjects/droneSegmentation/lowAltitude_classification/Base_iNat_classifier/ \
  -v /dev/shm/:/dev/shm/ \
  droneseg_cls bash
  
# data on /data/iNat_Classifier_Non_filtered
python lowAltitude_classification/Base_iNat_classifier/iNat_Classifier_Base.py
```
