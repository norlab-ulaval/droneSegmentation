import numpy as np
import torch
import torch.nn.functional as F
from torchvision.transforms.v2 import Grayscale

from tqdm import tqdm
from utils import draw_translucent_seg_maps

def train(
    model,
    train_dataloader,
    device,
    optimizer,
    classes_to_train,
    processor,
    metric
):
    print('Training')
    model.train()
    train_running_loss = 0.0

    prog_bar = tqdm(
        train_dataloader,
        total=len(train_dataloader),
        bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}'
    )

    counter = 0 # to keep track of batch counter
    num_classes = len(classes_to_train)

    for i, data in enumerate(prog_bar):
        counter += 1

        pixel_values = data['pixel_values'].to(device)
        # import matplotlib.pyplot as plt
        # plt.imshow(np.transpose(pixel_values[4].cpu().numpy(), (1, 2, 0)))
        # plt.axis('off')
        # plt.show()
        # exit()
        mask_labels = [mask_label.to(device) for mask_label in data['mask_labels']]
        class_labels = [class_label.to(device) for class_label in data['class_labels']]
        pixel_mask = data['pixel_mask'].to(device)

        optimizer.zero_grad()
        outputs = model(
            pixel_values=pixel_values, 
            mask_labels=mask_labels,
            class_labels=class_labels,
            pixel_mask=pixel_mask
        )

        ##### BATCH-WISE LOSS #####
        loss = outputs.loss
        train_running_loss += loss.item()
        ###########################
 
        ##### BACKPROPAGATION AND PARAMETER UPDATION #####
        loss.backward()
        optimizer.step()
        ##################################################

        target_sizes = [(image.shape[0], image.shape[1]) for image in data['orig_image']]
        pred_maps = processor.post_process_semantic_segmentation(
            outputs, target_sizes=target_sizes
        )
        import matplotlib.pyplot as plt
        # print(pred_maps[0].cpu().detach().numpy().shape)
        # plt.imshow(data['orig_mask'][0])
        # plt.show()
        # exit()

        metric.add_batch(references=data['orig_mask'], predictions=pred_maps)
        # print("Batch is done")
        
    ##### PER EPOCH LOSS #####
    train_loss = train_running_loss / counter
    ##########################
    iou = metric.compute(num_labels=num_classes, ignore_index=255, reduce_labels=False)['mean_iou']
    return train_loss, iou

def validate(
    model,
    valid_dataloader,
    device,
    classes_to_train,
    label_colors_list,
    epoch,
    save_dir,
    processor,
    metric
):
    print('Validating')
    model.eval()
    valid_running_loss = 0.0
    num_classes = len(classes_to_train)

    with torch.no_grad():
        prog_bar = tqdm(
            valid_dataloader, 
            total=(len(valid_dataloader)),
            bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}'
        )
        counter = 0 # To keep track of batch counter.
        for i, data in enumerate(prog_bar):
            counter += 1

            pixel_values = data['pixel_values'].to(device)
            mask_labels = [mask_label.to(device) for mask_label in data['mask_labels']]
            class_labels = [class_label.to(device) for class_label in data['class_labels']]
            pixel_mask = data['pixel_mask'].to(device)

            outputs = model(
                pixel_values=pixel_values, 
                mask_labels=mask_labels,
                class_labels=class_labels,
                pixel_mask=pixel_mask
            )

            target_sizes = [(image.shape[0], image.shape[1]) for image in data['orig_image']]
            pred_maps = processor.post_process_semantic_segmentation(
                outputs, target_sizes=target_sizes
            )
                
            # Save the validation segmentation maps.
            if i == 0:
                draw_translucent_seg_maps(
                    pixel_values, 
                    pred_maps, 
                    epoch, 
                    i, 
                    save_dir, 
                    label_colors_list,
                )

            ##### BATCH-WISE LOSS #####
            loss = outputs.loss
            valid_running_loss += loss.item()
            ###########################

            metric.add_batch(references=data['orig_mask'], predictions=pred_maps)

    ##### PER EPOCH LOSS #####
    valid_loss = valid_running_loss / counter
    ##########################
    iou = metric.compute(num_labels=num_classes, ignore_index=255, reduce_labels=False)['mean_iou']
    return valid_loss, iou