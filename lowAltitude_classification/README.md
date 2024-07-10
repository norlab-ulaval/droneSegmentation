```shell


buildah build -t lowclass .
podman run --gpus all --rm --ipc host -it \
  -v /home/kamyar/Documents/iNaturalist_split/:/home/kamyar/Documents/iNaturalist_split/ \
  -v output:/home/kamyar/PycharmProjects/droneSegmentation/lowAltitude_classification \
  -v .:/app/ \
  -v /dev/shm/:/dev/shm/ \ 
  lowclass bash


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
