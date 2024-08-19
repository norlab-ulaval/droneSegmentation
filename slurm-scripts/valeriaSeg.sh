#!/usr/bin/env bash
#SBATCH --time=60:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=12G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --mail-user=william.guimont-martin.1@ulaval.ca
#SBATCH --mail-type=FAIL,TIME_LIMIT
#SBATCH --job-name=test_valeria
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

nvidia-smi

cd $SLURM_TMPDIR
cp -r ~/droneSegmentation ./
s3cmd sync s3://annotated $SLURM_TMPDIR

python lowAltitude_segmentation/Mask2Former/train_net.py --num-gpus 8 \
  --config-file lowAltitude_segmentation/Mask2Former/configs/Drone_regrowth/semantic-segmentation/swin/maskformer2_swin_large_IN21k_384_bs16_160k_res640.yaml

