# lowAltitude_segmentation

## Valeria
Get the data
```shell
salloc --cpus-per-task=8 --time=2:00:00

s3cmd sync s3://annotated $SLURM_TMPDIR



python lowAltitude_segmentation/Mask2Former/train_net.py --num-gpus 8 \
  --config-file lowAltitude_segmentation/Mask2Former/configs/Drone_regrowth/semantic-segmentation/swin/maskformer2_swin_large_IN21k_384_bs16_160k_res640.yaml
```
