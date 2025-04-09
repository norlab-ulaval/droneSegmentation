# Using Citizen Science Data as Pre-Training for Semantic Segmentation of High-Resolution UAV Images for Natural Forests Post-Disturbance Assessment

[![DOI](https://zenodo.org/badge/DOI/10.3390/f16040616.svg)](https://doi.org/10.3390/f16040616)

![Overview](assets/overview.png "Overview")

> **Using Citizen Science Data as Pre-Training for Semantic Segmentation of High-Resolution UAV Images for Natural Forests Post-Disturbance Assessment**\
> Kamyar Nasiri, William Guimont-Martin, Damien LaRocque, Gabriel Jeanson, Hugo Bellemare-Vallières, Vincent Grondin, Philippe Bournival, Julie Lessard, Guillaume Drolet, Jean-Daniel Sylvain and Philippe Giguère
\
> Paper: https://www.mdpi.com/1999-4907/16/4/616

This repo contains the source code and the datasets used in our paper _Using Citizen Science Data as Pre-Training for Semantic Segmentation of High-Resolution UAV Images for Natural Forests Post-Disturbance Assessment_, published in the [_Classification of Forest Tree Species Using Remote Sensing Technologies: Latest Advances and Improvements_](https://www.mdpi.com/journal/forests/special_issues/S1W916IYIU) special issue of the [_Forests_](https://www.mdpi.com/journal/forests) MDPI journal.

## Repository organization

The repository is composed of two main directories:

<!-- TODO: Complete, following the cleaned version of the repo -->

* [`lowAltitude_classification`](lowAltitude_classification) contains the code for the image classifier $C_{\text{DINOv2}}$
* [`lowAltitude_segmentation`](lowAltitude_segmentation) contains the code for the segmentation model $S_{\text{M2F}}$

### Installation

To ease the installation of the dependencies and the training of the models, we provide two Dockerfiles, [`DockerfileClassif`](DockerfileClassif) and [`DockerfileSeg`](DockerfileSeg), respectively, for the image classifier and the segmentation model. We provide `make` commands to build the containers.

With `docker`:
```sh
make cls-build # Image classifier
make seg-build # Segmentation model
```

With `podman`:
```sh
make cls-podbuild # Image classifier
make seg-podbuild # Segmentation model
```

<!-- TODO: Continue text -->

<!-- We also provide a `Dockerfile` and a `DockerfileGPU` to build a Docker image with all the dependencies.

```sh
# Build the Docker image
docker build -t borealtc-gpu -f DockerfileGPU .

# Run the Docker image
docker run --gpus all -e CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES --rm --ipc host \
	  --mount type=bind,source=.,target=/code/ \
	  --mount type=bind,source=/dev/shm,target=/dev/shm \
	  borealtc-gpu python3 main.py
``` -->


## _WilDReF-Q_ (UAV imagery dataset)
### Labeled Dataset
| Dataset | Description | Link |
|----------|------|------|
| Training set  | 71 patched UAV images and masks (1024x1024) | [Download](http://norlab.s3.valeria.science/WilDReF-Q/annotated_data/train.tar.gz?AWSAccessKeyId=VCI7FLOHYPGLOOOAH0S5&Expires=2348491920&Signature=XaJuFVtVfBv1YHjQS6ex%2Bd5F%2B40%3D) |
| Validation set  | 46 patched UAV images and masks (1024x1024) | [Download](http://norlab.s3.valeria.science/WilDReF-Q/annotated_data/val.tar.gz?AWSAccessKeyId=VCI7FLOHYPGLOOOAH0S5&Expires=2348491907&Signature=OT2S3RYX4PGnSv2X0vF%2BDGFVSX8%3D) |
| Test set   | 36 patched UAV images and masks (1024x1024) | [Download](http://norlab.s3.valeria.science/WilDReF-Q/annotated_data/test.tar.gz?AWSAccessKeyId=VCI7FLOHYPGLOOOAH0S5&Expires=2348491807&Signature=boWUyLkH8DEhL50o%2Feaj%2FmOj9gY%3D) |


### Unlabeled UAV Datasets

| Dataset | Content | Description | Link |
|--------------|---------|-------------|----------|
| **Original (Non-Patched)** | Images | Raw UAV imagery without patching, over 11k images | [Download](http://norlab.s3.valeria.science/droneseg/unlabeled_data/unlabeled_drone_dataset_11k_original.tar.gz/unlabeled_drone_dataset_11k_original.tar.gz?AWSAccessKeyId=VCI7FLOHYPGLOOOAH0S5&Expires=2348849208&Signature=XpGy8xYa5rD81%2FeUgc%2FAfc5VW64%3D) |
| **Patched (for $S_{\text{M2F}}$ Pre-training)** | Images | Patched UAV images for segmentation model training, over 143k images | [Download](http://norlab.s3.valeria.science/WilDReF-Q/m2f_pretrain_data/m2f_pretrain_images.tar.gz/m2f_pretrain_images.tar.gz?AWSAccessKeyId=VCI7FLOHYPGLOOOAH0S5&Expires=2348849392&Signature=%2BcmqE1ycN25x1YezP4EaTwA9ek8%3D) |
| **Patched (for $S_{\text{M2F}}$ Pre-training)** | Masks | Generated pseudo-labels for patched UAV images by the classifier $C_{\text{DINOv2}}$ | [Download](http://norlab.s3.valeria.science/WilDReF-Q/m2f_pretrain_masks.tar.gz?AWSAccessKeyId=VCI7FLOHYPGLOOOAH0S5&Expires=2348491970&Signature=mgJB4B6WWtDD72jrrxC5MPt%2BOxw%3D) |



## Pre-trained Models  
| Model | Link |
|----------|------|
| Classification  | [Download](http://norlab.s3.valeria.science/WilDReF-Q/models/classification/Best_Classifier_iNat.pth?AWSAccessKeyId=VCI7FLOHYPGLOOOAH0S5&Expires=2348492028&Signature=KkOJPJYAix0WTPcktjeDxuLmQPY%3D) |
| Segmentation (Pre-trained `PT`)    | [Download](http://norlab.s3.valeria.science/WilDReF-Q/models/segmentation/PT.pth?AWSAccessKeyId=VCI7FLOHYPGLOOOAH0S5&Expires=2348492079&Signature=ovjoOXuy5V3qi7uKYO9d10oFkMc%3D) |
| Segmentation (Finetuned  `FT`)    | [Download](http://norlab.s3.valeria.science/WilDReF-Q/models/segmentation/FT.pth?AWSAccessKeyId=VCI7FLOHYPGLOOOAH0S5&Expires=2348492067&Signature=8EBXI%2FUyhGWXgjGJ62KV0Ce%2BlSo%3D) |

## Contributing

This project is maintained with `pre-commit`. To setup `pre-commit`, follow these commands.

```sh
pip install pre-commit
pre-commit install
```

## Citation

If you use the code or data in an academic context, please cite the following work:

```bibtex
@article{Nasiri2025,
  title     = {{Using Citizen Science Data as Pre-Training for Semantic Segmentation of High-Resolution UAV Images for Natural Forests Post-Disturbance Assessment}},
  volume    = {16},
  issn      = {1999-4907},
  url       = {http://dx.doi.org/10.3390/f16040616},
  doi       = {10.3390/f16040616},
  number    = {4},
  journal   = {Forests},
  publisher = {MDPI AG},
  author    = {Nasiri,  Kamyar and Guimont-Martin,  William and LaRocque,  Damien and Jeanson,  Gabriel and Bellemare-Vallières,  Hugo and Grondin,  Vincent and Bournival,  Philippe and Lessard,  Julie and Drolet,  Guillaume and Sylvain,  Jean-Daniel and Giguère,  Philippe},
  year      = {2025},
  month     = mar,
  pages     = {616}
}
```
