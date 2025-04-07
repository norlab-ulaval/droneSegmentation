.PHONY: cls-build cls-podbuild seg-build seg-podbuild cls-run cls-train cls-server seg-run seg-train

cls-build:
	docker build -t droneseg_cls -f DockerfileClassif .

cls-podbuild:
	buildah build -t droneseg_cls --layers -f DockerfileClassif .


cls-run: cls-build
	podman run --gpus all --rm --ipc host \
	  -e CUDA_VISIBLE_DEVICES=$(CUDA_VISIBLE_DEVICES) \
	  -v .:/app \
	  -v ./data \
	  -v ./data/iNaturalist_split: \
	  -v output: \
	  -v /dev/shm/:/dev/shm/ \
	  droneseg_cls bash -c "make cls-train"

cls-train:
	python lowAltitude_classification/Dinov2_iNaturalist_classification_fine-tuning.py

# Server
cls-server: cls-build
	podman run --gpus all --rm --ipc host -it \
	-e CUDA_VISIBLE_DEVICES=$(CUDA_VISIBLE_DEVICES) \
	-v output: \
	-v .:/app/ \
	-v /dev/shm/:/dev/shm/ \
	droneseg_cls bash

# Segmentation

seg-build:
	docker build -t droneseg_seg -f DockerfileSeg .

seg-podbuild:
	buildah build -t droneseg_seg --layers -f DockerfileSeg .

seg-run: seg-build
	podman run --gpus all --rm --ipc host -it \
	  -e CUDA_VISIBLE_DEVICES=$(CUDA_VISIBLE_DEVICES) \
	  -v .:/app \
	  -v ./data \
	  -v ./data/iNaturalist_split: \
	  -v output: \
	  -v /dev/shm/:/dev/shm/ \
	  droneseg_seg bash

build-pixel-decoder:
	cd lowAltitude_segmentation/Mask2Former/mask2former/modeling/pixel_decoder/ops/ && sh make.sh && cd -

seg-train:
	python lowAltitude_segmentation/Mask2Former/train_net.py   --config-file configs/Drone_regrowth/semantic-segmentation/swin/maskformer2_swin_large_IN21k_384_bs16_160k_res640.yaml   --eval-only MODEL.WEIGHTS
