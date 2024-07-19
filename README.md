# droneSegmentation
```shell
buildah build -t droneseg --layers .
podman run --gpus all --rm --ipc host -it \
  -v .:/app/ \
  -v ./data/
  -v ./data/iNaturalist_split:/home/kamyar/Documents/iNaturalist_split/ \
  -v output:/home/kamyar/PycharmProjects/droneSegmentation/lowAltitude_classification \
  -v /dev/shm/:/dev/shm/ \ 
  droneseg bash

sftp kanas@132.203.26.231
pwd , lpwd
put -r 'iNaturalist_split/'
```

TODO
1. Transfer data
2. Add SSH Key
3. git clone the project
4. make slurm script
5. sbatch + squeue
