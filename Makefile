.PHONY: build-cls run-cls build-seg run-seg

build-cls:
	buildah build -t droneseg_cls --layers -f DockerfileClassif .

run-cls: build-cls
	podman run --gpus all --rm --ipc host -it \
	  -e CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES \
	  -v .:/app \
	  -v ./data \
	  -v ./data/iNaturalist_split:/home/kamyar/Documents/filtered_inat_split/ \
	  -v output:/home/kamyar/PycharmProjects/droneSegmentation/lowAltitude_classification \
	  -v /dev/shm/:/dev/shm/ \
	  droneseg_cls bash

build-seg:
	buildah build -t droneseg_seg --layers -f DockerfileSeg .

run-cls: build-seg
	podman run --gpus all --rm --ipc host -it \
	  -e CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES \
	  -v .:/app \
	  -v ./data \
	  -v ./data/iNaturalist_split:/home/kamyar/Documents/filtered_inat_split/ \
	  -v output:/home/kamyar/PycharmProjects/droneSegmentation/lowAltitude_classification \
	  -v /dev/shm/:/dev/shm/ \
	  droneseg_seg bash
