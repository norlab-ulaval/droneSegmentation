#!/usr/bin/env bash

sjm run valeria sjmValeria.sh NAME=swin_large CONFIG=swin/M2F_Swin_Large_base.yaml
sjm run valeria sjmValeria.sh NAME=swin_dice CONFIG=swin/M2F_Swin_Large_ClassMaskDice_Weight.yaml
sjm run valeria sjmValeria.sh NAME=swin_colaug CONFIG=swin/M2F_Swin_Large_colorAugs.yaml
sjm run valeria sjmValeria.sh NAME=swin_crop CONFIG=swin/M2F_Swin_Large_Crop_512.yaml

sjm run valeria sjmValeria.sh NAME=resnet50_base CONFIG=M2F_ResNet50_base.yaml
sjm run valeria sjmValeria.sh NAME=resnet50_dice CONFIG=M2F_ResNet50_ClassMaskDice_Weight.yaml
sjm run valeria sjmValeria.sh NAME=resnet50_colaug CONFIG=M2F_ResNet50_colorAugs.yaml
sjm run valeria sjmValeria.sh NAME=resnet50_crop CONFIG=M2F_ResNet50_Crop_512.yaml
