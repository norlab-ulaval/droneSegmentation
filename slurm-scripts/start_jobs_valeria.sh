#!/usr/bin/env bash

sjm run valeria sjmValeria.sh NAME=swin_lr CONFIG=swin/M2F_Swin_Large_LR.yaml
sjm run valeria sjmValeria.sh NAME=swin_colaug CONFIG=swin/M2F_Swin_Large_colorAugs.yaml
sjm run valeria sjmValeria.sh NAME=swin_ssd CONFIG=swin/M2F_Swin_Large_SSD_default.yaml
sjm run valeria sjmValeria.sh NAME=swin_crop CONFIG=swin/M2F_Swin_Large_Crop_512.yaml
sjm run valeria sjmValeria.sh NAME=swin_train CONFIG=swin/M2F_Swin_Large_MaxTrainSize_1024.yaml

#sjm run valeria sjmValeria.sh NAME=swin_large3 CONFIG=swin/M2F_Swin_Large_base.yaml
#sjm run valeria sjmValeria.sh NAME=swin_dice3 CONFIG=swin/M2F_Swin_Large_ClassMaskDice_Weight.yaml
#sjm run valeria sjmValeria.sh NAME=swin_colaug3 CONFIG=swin/M2F_Swin_Large_colorAugs.yaml
#sjm run valeria sjmValeria.sh NAME=swin_crop3 CONFIG=swin/M2F_Swin_Large_Crop_512.yaml
#
#sjm run valeria sjmValeria.sh NAME=resnet50_base3 CONFIG=M2F_ResNet50_base.yaml
#sjm run valeria sjmValeria.sh NAME=resnet50_dice3 CONFIG=M2F_ResNet50_ClassMaskDice_Weight.yaml
#sjm run valeria sjmValeria.sh NAME=resnet50_colaug3 CONFIG=M2F_ResNet50_colorAugs.yaml
#sjm run valeria sjmValeria.sh NAME=resnet50_crop3 CONFIG=M2F_ResNet50_Crop_512.yaml


# WGM (done)
#sjm run valeria sjmValeria.sh NAME=swin_train CONFIG=swin/M2F_Swin_Large_MaxTrainSize_1024.yaml
# Mamba
#sjm run valeria sjmValeria.sh NAME=swin_ssd CONFIG=swin/M2F_Swin_Large_SSD_default.yaml
# DAD (done)
#sjm run valeria sjmValeria.sh NAME=resnet50_train CONFIG=M2F_ResNet50_MaxTrainSize_1024.yaml
# Mamba
#sjm run valeria sjmValeria.sh NAME=resnet50_ssd CONFIG=M2F_ResNet50_SSD_default.yaml

# DAD (in parallel to valeria)
# M2F_ResNet50_base.yaml
# WGM (in parallel to valeria)
# M2F_Swin_Large_base.yaml
