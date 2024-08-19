#!/usr/bin/env bash
#SBATCH --time=3:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=12G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --mail-user=william.guimont-martin.1@ulaval.ca
#SBATCH --mail-type=FAIL,TIME_LIMIT
#SBATCH --job-name=droneSeg
#SBATCH --output=%x-%j.out
#SBATCH --reservation=wigum_9

module load python/3.10.13
module load cuda/12.2
module load qt
module load geos
module load llvm
module load gcc
module load opencv
module load scipy-stack
module load openblas

cd $SLURM_TMPDIR || exit 1
cp -r ~/detectron2 ./
cp -r ~/droneSegmentation ./

module load python/3.11.5
module load cuda/11.7
module load opencv/4.9.0

s3cmd sync s3://annotated $SLURM_TMPDIR

virtualenv --no-download venv
source venv/bin/activate
pip install --no-index --upgrade pip

cd $SLURM_TMPDIR/detectron2 || exit 1
pip install -e . --no-index

cd $SLURM_TMPDIR/droneSegmentation/lowAltitude_segmentation/Mask2Former || exit 1
pip install -r requirements.txt --no-index

cd mask2former/modeling/pixel_decoder/ops || exit 1
sh make.sh

cd $SLURM_TMPDIR/droneSegmentation/ || exit 1
python lowAltitude_segmentation/Mask2Former/train_net.py --num-gpus 1 \
  --config-file lowAltitude_segmentation/Mask2Former/configs/Drone_regrowth/semantic-segmentation/swin/maskformer2_swin_large_IN21k_384_bs16_160k_res640.yaml
