| **Computer** | **Model**     | **ETA**       |
|--------------|---------------|---------------|
| WGM          | Swin base     | 1d 4h         |
| DAD          | Resnet base   | 17h           |
| Mamba 1      | Resnet SSD    | Just finished |
| Mamba 2      | Swin SSD      | 21h           |
| Valeria 1    | Swin base     | 1d 15h        |
| Valeria 2    | Swin dice     | 1d 15h        |
| Valeria 3    | Swin colaug   | 1d 15h        |
| Valeria 4    | Swin crop     | 1d 15h        |
| Valeria 5    | Resnet base   | 1d 15h        |
| Valeria 6    | Resnet dice   | 1d 15h        |
| Valeria 7    | Resnet colaug | 1d 15h        |
| Valeria 8    | Resnet crop   | 1d 15h        |

| **Model**     | **Computer** | **Status** | **ETA** | **Config**                          |
|---------------|--------------|------------|---------|-------------------------------------|
| Swin base     | Valeria      | Running    | 1d 4h   | M2F_Swin_Large_base                 |
| Swin dice     | Valeria      | Running    | 1d 15h  | M2F_Swin_Large_ClassMaskDice_Weight |
| Swin colaug   | Valeria      | Running    | 1d 15h  | M2F_Swin_Large_colorAugs            |
| Swin crop     | Valeria      | Running    | 1d 15h  | M2F_Swin_Large_Crop_512             |
| Swin train    | WGM          | Done       | ---     | M2F_Swin_Large_MaxTrainSize_1024    |
| Swin SSD      | Mamba        | Running    | 21h     | M2F_Swin_Large_SSD_default          |
| Resnet base   | Valeria      | Running    | 16h     | M2F_ResNet50_base                   |
| Resnet dice   | Valeria      | Running    | 1d 15h  | M2F_ResNet50_ClassMaskDice_Weight   |
| Resnet colaug | Valeria      | Running    | 1d 15h  | M2F_ResNet50_colorAugs              |
| Resnet crop   | Valeria      | Running    | 1d 15h  | M2F_ResNet50_Crop_512               |
| Resnet train  | DAD          | Done       | ---     | M2F_ResNet50_MaxTrainSize_1024      |
| Resnet SSD    | Mamba        | Done       | ---     | M2F_ResNet50_SSD_default            |
| Swin LR       | WGM          | Running    | 1d 20h  | M2F_Swin_Large_LR                   |
| Swin small    | DAD          | Running    | 1d 16h  | M2F_Swin_Large_smallSize            |

# Fixed the bug

| **Model**   | **Computer** | **Status** | **ETA** | **Config**                          |
|-------------|--------------|------------|---------|-------------------------------------|
| Swin base   | DAD          | TODO       | 1d 4h   | M2F_Swin_Large_base                 |
| Swin dice   | WGM          | TODO       | 1d 15h  | M2F_Swin_Large_ClassMaskDice_Weight |
| Swin colaug | Mamba        | TODO       | 1d 15h  | M2F_Swin_Large_colorAugs            |
| Swin SSD    | Mamba        | TODO       | 21h     | M2F_Swin_Large_SSD_default          |
| Swin crop   | Mamba        | TODO       | 1d 15h  | M2F_Swin_Large_Crop_512             |
| Swin train  | Mamba        | TODO       | ---     | M2F_Swin_Large_MaxTrainSize_1024    |
| Swin lr     | Valeria      | TODO       | 1d 16h  | M2F_Swin_Large_LR                   |
