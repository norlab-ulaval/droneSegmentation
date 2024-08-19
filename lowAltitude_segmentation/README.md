# lowAltitude_segmentation

## Valeria
Get the data
```shell
# ReservationName=wigum_9 StartTime=2024-08-15T00:00:00 EndTime=2024-08-21T00:00:00 Duration=6-00:00:00
#  Nodes=ul-val-pr-cpc[02-03] NodeCnt=2 CoreCnt=128 Features=(null) PartitionName=(null) Flags=SPEC_NODES
#  TRES=cpu=128
#  Users=wigum Groups=(null) Accounts=(null) Licenses=(null) State=INACTIVE BurstBuffer=(null)
#  MaxStartDelay=(null) 
salloc --cpus-per-task=8 --time=2:00:00

cd $SLURM_TMPDIR
cp -r ~/droneSegmentation ./

module load python/3.10.13

s3cmd sync s3://annotated $SLURM_TMPDIR



python lowAltitude_segmentation/Mask2Former/train_net.py --num-gpus 8 \
  --config-file lowAltitude_segmentation/Mask2Former/configs/Drone_regrowth/semantic-segmentation/swin/maskformer2_swin_large_IN21k_384_bs16_160k_res640.yaml
```
