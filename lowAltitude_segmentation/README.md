# lowAltitude_segmentation

```shell
docker build -t droneseg -f DockerfileSeg .
docker run --gpus=all --rm --ipc host -it \
  -v .:/app \
  -v ~/Datasets/drone_dataset:/data/drone_dataset \
  -v /dev/shm/:/dev/shm/ \
  droneseg bash
  
docker run --gpus=all --rm --ipc host -it \
  -e CUDA_VISIBLE_DEVICES=0 \
  -v .:/app \
  -v /data/drone_dataset:/data/drone_dataset \
  -v /data/drone
  -v ./output_res_ssd:/app/output \
  -v /dev/shm/:/dev/shm/ \
  droneseg bash
  
docker run --gpus=all --rm --ipc host -it \
  -e CUDA_VISIBLE_DEVICES=1 \
  -v .:/app \
  -v /data/drone_dataset:/data/drone_dataset \
  -v ./output_swin_ssd:/app/output \
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

PYTHONPATH=$PYTHONPATH:. python lowAltitude_segmentation/Mask2Former/train_net.py --num-gpus 1 \
  --config-file lowAltitude_segmentation/Mask2Former/configs/Drone_regrowth/semantic-segmentation/"$CONFIG"
````

## Valeria
Get the data
```shell
git clone https://github.com/facebookresearch/detectron2

# ReservationName=wigum_9 StartTime=2024-08-15T00:00:00 EndTime=2024-08-21T00:00:00 Duration=6-00:00:00
#  Nodes=ul-val-pr-cpc[02-03] NodeCnt=2 CoreCnt=128 Features=(null) PartitionName=(null) Flags=SPEC_NODES
#  TRES=cpu=128
#  Users=wigum Groups=(null) Accounts=(null) Licenses=(null) State=INACTIVE BurstBuffer=(null)
#  MaxStartDelay=(null) 

salloc --time=60:00 --cpus-per-task=16 --mem=12G --partition=gpu --gres=gpu:a100:1 --reservation=wigum_9

cd $SLURM_TMPDIR
cp -r ~/detectron2 ./
cp -r ~/droneSegmentation ./

module load python/3.11.5
module load cuda/11.7
module load opencv/4.9.0

#s3cmd sync s3://annotated $SLURM_TMPDIR

virtualenv --no-download venv
source venv/bin/activate
pip install --no-index --upgrade pip

cd $SLURM_TMPDIR/detectron2
pip install -e . --no-index

cd $SLURM_TMPDIR/droneSegmentation/lowAltitude_segmentation/Mask2Former
pip install -r requirements.txt --no-index

cd mask2former/modeling/pixel_decoder/ops
sh make.sh

cd $SLURM_TMPDIR/droneSegmentation/
python lowAltitude_segmentation/Mask2Former/train_net.py --num-gpus 1 \
  --config-file lowAltitude_segmentation/Mask2Former/configs/Drone_regrowth/semantic-segmentation/swin/maskformer2_swin_large_IN21k_384_bs16_160k_res640.yaml
```
