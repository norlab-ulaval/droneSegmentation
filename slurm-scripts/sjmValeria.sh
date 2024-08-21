#!/usr/bin/env bash
#SBATCH --time=4:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=12G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --mail-user=william.guimont-martin.1@ulaval.ca
#SBATCH --mail-type=FAIL,TIME_LIMIT
#SBATCH --job-name=$NAME
#SBATCH --output=%x-%j.out
#SBATCH --reservation=wigum_9

cd "$SLURM_TMPDIR" || exit 1
cp -r ~/detectron2 ./
cp -r ~/droneSegmentation ./

module load python/3.11.5
module load cuda/11.7
module load opencv

#s3cmd sync s3://annotated $SLURM_TMPDIR

virtualenv --no-download venv
source venv/bin/activate
pip install --no-index --upgrade pip

cd "$SLURM_TMPDIR"/droneSegmentation/lowAltitude_segmentation/Mask2Former || exit 1
pip install -r requirements.txt --no-index

cd "$SLURM_TMPDIR"/detectron2 || exit 1
pip install -e . --no-index

cd "$SLURM_TMPDIR"/droneSegmentation/lowAltitude_segmentation/Mask2Former/mask2former/modeling/pixel_decoder/ops || exit 1
export MAX_JOBS=16
sh make.sh

cd "$SLURM_TMPDIR" || exit 1
mkdir drone_dataset/
cd drone_dataset || exit 1
mkdir train val

# Train
cd train || exit 1
mkdir images masks
cp ~/projects/ul-val-prj-def-phgig4/DroneSeg/images/*.tar.gz images/
cp ~/projects/ul-val-prj-def-phgig4/DroneSeg/masks/*.tar.gz masks/
cd images || exit 1
find -- *.tar.gz | parallel --progress tar -xvzf {}
find . -type f -name '*.JPG' -exec mv -t ./ {} +

cd ../masks || exit 1
find -- *.tar.gz | parallel --progress tar -xvzf {}
find . -type f -name '*.png' -exec mv -t ./ {} +

# Validation
cd ../../val || exit 1
mkdir images masks
cp ~/projects/ul-val-prj-def-phgig4/DroneSeg/m2f_train_val_split/*.tar.gz ./
find -- *.tar.gz | parallel --progress tar -xvzf {}
mv M2F_Train_Val_split/val/* ./
rm -r M2F_Train_Val_split/
cd images || exit 1
for f in `find * -type f | grep .jpg`; do mv -- "$f" "${f%.jpg}.JPG"; done

# Weights
cd "$SLURM_TMPDIR"/droneSegmentation/lowAltitude_segmentation/Mask2Former/ || exit 1
cp ~/projects/ul-val-prj-def-phgig4/DroneSeg/weights/M2F_IN21k_weight/* ./

# Register dataset
cd "$SLURM_TMPDIR"/droneSegmentation/ || exit 1
python lowAltitude_segmentation/Mask2Former/mask2former/data/datasets/register_drone_semantic.py

# Get results
get_results() {
  echo "Getting results"
  mkdir ~/results
  cp -r "$SLURM_TMPDIR/droneSegmentation/" "~/results/results$(date +%s)"
}

trap get_results EXIT

# Start training
PYTHONPATH=$PYTHONPATH:. python lowAltitude_segmentation/Mask2Former/train_net.py --num-gpus 1 \
  --config-file lowAltitude_segmentation/Mask2Former/configs/Drone_regrowth/semantic-segmentation/$CONFIG
