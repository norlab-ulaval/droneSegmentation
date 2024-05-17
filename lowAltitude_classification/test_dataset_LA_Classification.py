"""
test the fine-tuned weights of iNaturalist, on the entire dataset of low-altitude images
export the annotations as a 2d map
"""

import os
import torch
import torch.nn as nn
import numpy as np
from collections import Counter
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForImageClassification, AutoImageProcessor
from statistics import mode
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_name = 'facebook/dinov2-large-imagenet1k-1-layer'
processor = AutoImageProcessor.from_pretrained(model_name)
model = AutoModelForImageClassification.from_pretrained(model_name, ignore_mismatched_sizes=True)
model = model.to(device)
num_classes = 25
model.classifier = nn.Linear(2048, num_classes).to(device)

model.load_state_dict(torch.load('/home/kamyar/PycharmProjects/droneSegmentation/best_classification_weights.pth'))
model.eval()

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=processor.image_mean, std=processor.image_std),
])

input_folder = '/home/kamyar/Documents/Dataset_lowAltitude_patchified'
output_folder = '/home/kamyar/Documents/Dataset_lowAltitude_patchified_labels_parallel'

class CustomDataset(Dataset):
    def __init__(self, folder_path, transform=None, patch_sizes=[256], overlaps=[0.8]):
        self.folder_path = folder_path
        self.image_files = [f for f in os.listdir(folder_path) if f.endswith('.JPG')]
        self.transform = transform
        self.patch_sizes = patch_sizes
        self.overlaps = overlaps

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = os.path.join(self.folder_path, self.image_files[idx])
        image = Image.open(image_path)
        width, height = image.size
        if self.transform:
            image = self.transform(image)

        patches = []
        positions = []

        for patch_size in self.patch_sizes:
            for overlap in self.overlaps:

                step_size = int(patch_size * (1 - overlap))

                for x in range(0, width - patch_size + 1, step_size):
                    for y in range(0, height - patch_size + 1, step_size):
                        if x + step_size > width or y + step_size > height:
                            continue
                        patch = image[:, y:y + patch_size, x:x + patch_size]
                        patches.append(patch)
                        positions.append((x, y))


        return patches, positions, (width, height), image_path

patch_sizes = [196]
overlaps = [0.85]

dataset = CustomDataset(input_folder, transform=transform, patch_sizes=patch_sizes, overlaps=overlaps)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)


def process_batch(patches):
    with torch.no_grad():
        output = model(torch.cat(patches, dim=0).to(device))
    return torch.argmax(output.logits, dim=-1)


batch_size = 512
image_counter = 1
for batch_patches, batch_positions, (width, height), image_path in dataloader:
    image_counter += 1
    pixel_predictions = {}
    num_patches = len(batch_patches)

    for i in range(0, num_patches, batch_size):
        batch = batch_patches[i:i + batch_size]
        predictions = process_batch(batch)
        for idx, (px, py) in enumerate(batch_positions[i:i + batch_size]):
            predicted_class = predictions[idx].item()
            if (px, py) not in pixel_predictions:
                pixel_predictions[(px, py)] = []
            pixel_predictions[(px, py)].append(predicted_class)

            # for k in range(px, px + patch_sizes[0]):
            #     pixel_predictions_x[k].append(predicted_class)
            # for k in range(py, py + patch_sizes[0]):
            #     pixel_predictions_y[k].append(predicted_class)

            # for y in range(patch_sizes[0]):
            #     for x in range(patch_sizes[0]):
            #         pixel_x = x + px
            #         pixel_y = y + py
            #         # print(pixel_x, pixel_y)
            # #         if (pixel_x, pixel_y) not in pixel_predictions:
            # #             pixel_predictions[(pixel_x, pixel_y)] = []
            # #
            #         # print(len(pixel_predictions[(pixel_x, pixel_y)]))


    final_predictions = {}

    for (px, py), predictions in pixel_predictions.items():
        square_coords = [(px.item() + x, py.item() + y) for x in range(patch_sizes[0]) for y in range(patch_sizes[0])]
        for coord in square_coords:
            if coord not in final_predictions:
                final_predictions[coord] = []
            final_predictions[coord].append(predictions)

    final_predictions_vote = {(i, j): 0 for i in range(width) for j in range(height)}
    for pixel, predictions in final_predictions.items():
        if len(predictions) > 0:
            final_predictions_vote[pixel] = mode(predictions[0])

    segmentation_map = np.ones((height, width), dtype=np.uint8)

    for pixel, class_value in final_predictions_vote.items():
        # print(pixel[1].item(), pixel[0].item(), class_value)
        segmentation_map[pixel[1], pixel[0]] = class_value + 1

    # plt.imshow(segmentation_map)
    # plt.axis('off')
    # plt.show()


    output_filename =  os.path.splitext(os.path.basename(image_path[0]))[0] + '.png'
    output_path = os.path.join(output_folder, output_filename)
    # print(np.unique(segmentation_map))

    import cv2
    cv2.imwrite(output_path, segmentation_map)
    print(image_counter)
    # img = Image.open(output_path).convert('L')
    # print(np.unique(img))




# for filename in os.listdir(input_folder):
#     if filename.endswith(".JPG"):
#         image_path = os.path.join(input_folder, filename)
#         image = Image.open(image_path)
#         image_tensor = transform(image)
#
#         for patch_size in patch_sizes:
#             for overlap in overlaps:
#                 width, height = image.size
#                 step_size = int(patch_size * (1 - overlap))
#                 pixel_predictions = {}
#                 patch_total_number = ((width - patch_size) // step_size + 1) * ((height - patch_size) // step_size + 1)
#                 iters = patch_total_number // batch_size
#                 iter_ = 0
#
#                 counter_batch = 0
#                 patches = []
#                 positions = []
#                 for x in range(0, width - patch_size + 1, step_size):
#                     for y in range(0, height - patch_size + 1, step_size):
#                             if x + step_size > width or y + step_size > height:
#                                 continue
#                             patch = image_tensor[:, y:y + patch_size, x:x + patch_size].to(device)
#
#                             with torch.no_grad():
#                                 output = model(patch.unsqueeze(0))
#
#                             # predicted_class = torch.argmax(output.logits, dim=1)
#                             predicted_class = torch.argmax(output.logits, dim=-1).item()
#
#                             for j in range(patch_size):
#                                 for i in range(patch_size):
#                                     pixel_x = x + i
#                                     pixel_y = y + j
#
#                                     if (pixel_x, pixel_y) not in pixel_predictions:
#                                         pixel_predictions[(pixel_x, pixel_y)] = []
#                                     pixel_predictions[(pixel_x, pixel_y)].append(predicted_class)
#








                # for x in range(0, width - patch_size + 1, step_size):
                #     for y in range(0, height - patch_size + 1, step_size):
                #
                #         if x + patch_size > width or y + patch_size > height:
                #             continue
                #         patch = image_tensor[:, y:y + patch_size, x:x + patch_size].to(device)
                #         patches.append(patch)
                #         positions.append((x, y))
                #         counter_batch += 1
                #         # print(counter_batch)
                #
                #         if counter_batch == batch_size:
                #             # print(iter_, ' of total ',iters)
                #             iter_ += 1
                #             with torch.no_grad():
                #                 output = model(torch.stack(patches))
                #                 del patches[:]
                #
                #             predicted_classes = torch.argmax(output.logits, dim=-1)
                #
                #             for idx in range(len(positions)):
                #                 x_patch = positions[idx][0]
                #                 y_patch = positions[idx][1]
                #                 for j in range(patch_size):
                #                     for i in range(patch_size):
                #                         pixel_x = x_patch + i
                #                         pixel_y = y_patch + j
                #                         if (pixel_x, pixel_y) not in pixel_predictions:
                #                             pixel_predictions[(pixel_x, pixel_y)] = []
                #                         else:
                #                             pixel_predictions[(pixel_x, pixel_y)].append(predicted_classes[idx].item())
                #
                #
                #             # print(pixel_predictions)
                #             counter_batch = 0
                #             patches = []
                #             positions = []




                # final_predictions = {}
                # for pixel, predictions in pixel_predictions.items():
                #     final_predictions[pixel] = Counter(predictions).most_common(1)[0][0]
                #
                # segmentation_map = np.zeros((height, width), dtype=np.uint8)
                #
                # for pixel, class_value in final_predictions.items():
                #     segmentation_map[pixel[1], pixel[0]] = class_value + 1
                #
                # print(c)
                # c = c + 1
                #
                # # print(np.max(segmentation_map))
                # output_filename = os.path.splitext(filename)[0] + '_segmentation_map.png'
                # output_path = os.path.join(output_folder, output_filename)
                # import cv2
                # cv2.imwrite(output_path, segmentation_map)
                # img.save(output_path)
                # print(output_path)

                # img = Image.open(output_path).convert('L')
                # print(np.max(np.array(img)))
                # exit()

                # import sys

                # image = Image.open(output_path).convert('L')








  # for x in range(0, width - patch_size + 1, step_size):
    #     for y in range(0, height - patch_size + 1, step_size):
    #         if x + step_size > width or y + step_size > height:
    #             continue
    #         patch = image_tensor[:, y:y + patch_size, x:x + patch_size]
    #
    #         with torch.no_grad():
    #             output = model(patch.unsqueeze(0))
    #
    #         # predicted_class = torch.argmax(output.logits, dim=1)
    #         predicted_class = torch.argmax(output.logits, dim=-1).item()
    #
    #         for j in range(patch_size):
    #             for i in range(patch_size):
    #                 pixel_x = x + i
    #                 pixel_y = y + j
    #
    #                 if (pixel_x, pixel_y) not in pixel_predictions:
    #                     pixel_predictions[(pixel_x, pixel_y)] = []
    #                 pixel_predictions[(pixel_x, pixel_y)].append(predicted_class)