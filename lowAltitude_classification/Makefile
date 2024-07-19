.PHONY: build run-server runsh

build:
	buildah build --layers -t lowclass .

run-server: build
	podman run --gpus all --rm --ipc host -it \
	-v output:/home/kamyar/PycharmProjects/droneSegmentation/lowAltitude_classification \
	-v .:/app/ \
	-v ../data:/data \
	-v /dev/shm/:/dev/shm/ \
	lowclass bash

runsh: build
	podman run --gpus all --rm --ipc host -it \
	-v output:/home/kamyar/PycharmProjects/droneSegmentation/lowAltitude_classification \
	-v .:/app/ \
	-v /dev/shm/:/dev/shm/ \
	lowclass bash

