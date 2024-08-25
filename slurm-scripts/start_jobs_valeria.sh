#!/usr/bin/env bash

sjm run valeria sjmValeria.sh NAME=swin_large2 CONFIG=swin/M2F_Swin_Large_base.yaml
sjm run valeria sjmValeria.sh NAME=swin_dice2 CONFIG=swin/M2F_Swin_Large_ClassMaskDice_Weight.yaml
sjm run valeria sjmValeria.sh NAME=swin_colaug2 CONFIG=swin/M2F_Swin_Large_colorAugs.yaml
sjm run valeria sjmValeria.sh NAME=swin_crop2 CONFIG=swin/M2F_Swin_Large_Crop_512.yaml

sjm run valeria sjmValeria.sh NAME=resnet50_base2 CONFIG=M2F_ResNet50_base.yaml
sjm run valeria sjmValeria.sh NAME=resnet50_dice2 CONFIG=M2F_ResNet50_ClassMaskDice_Weight.yaml
sjm run valeria sjmValeria.sh NAME=resnet50_colaug2 CONFIG=M2F_ResNet50_colorAugs.yaml
sjm run valeria sjmValeria.sh NAME=resnet50_crop2 CONFIG=M2F_ResNet50_Crop_512.yaml


# WGM
#sjm run valeria sjmValeria.sh NAME=swin_train CONFIG=swin/M2F_Swin_Large_MaxTrainSize_1024.yaml
# TODO
#sjm run valeria sjmValeria.sh NAME=swin_ssd CONFIG=swin/M2F_Swin_Large_SSD_default.yaml
# DAD
#sjm run valeria sjmValeria.sh NAME=resnet50_train CONFIG=M2F_ResNet50_MaxTrainSize_1024.yaml
# TODO
#sjm run valeria sjmValeria.sh NAME=resnet50_ssd CONFIG=M2F_ResNet50_SSD_default.yaml
