.PHONY: build runsh

build:
	buildah build --layers -t lowclass .

runsh: build
	podman run --gpus all --rm --ipc host -it \
	-v /home/kamyar/Documents/iNaturalist_split/:/home/kamyar/Documents/iNaturalist_split/ \
	-v output:/home/kamyar/PycharmProjects/droneSegmentation/lowAltitude_classification \
	-v .:/app/ \
	-v /dev/shm/:/dev/shm/ \
	lowclass bash