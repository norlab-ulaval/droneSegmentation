import os
import time
import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from albumentations import Normalize, Compose
from albumentations.pytorch import ToTensorV2
from transformers import AutoImageProcessor, AutoModelForImageClassification
from pathlib import Path
import csv


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_name = 'facebook/dinov2-large-imagenet1k-1-layer'
processor = AutoImageProcessor.from_pretrained(model_name)
model = AutoModelForImageClassification.from_pretrained(model_name, ignore_mismatched_sizes=True)
model = model.to(device)
num_classes = 26
model.classifier = nn.Linear(2048, num_classes).to(device)
mean = processor.image_mean
std = processor.image_std

transform = Compose([
    Normalize(mean=mean, std=std),
    ToTensorV2()
])

patch_sizes = [184]
overlaps = [0.85]

root_folder = '/home/kamyar/PycharmProjects/droneSegmentation/lowAltitude_classification/Results/5_best'
image_folder = '/home/kamyar/Documents/Train-val_Annotated'

csv_filename = 'lowAltitude_classification/results/avg_voters/val/one_vote.csv'
file_exists = os.path.isfile(csv_filename)

with open(csv_filename, mode='a', newline='') as file:
    writer = csv.writer(file)
    if not file_exists or os.stat(csv_filename).st_size == 0:
        writer.writerow(["Central Size", "Patch Size", "Step Size", "Average Number of Voters"])

    for dirpath, dirnames, filenames in os.walk(root_folder):
        for filename in filenames:
            if filename == '52_Final_time2024-08-15_best_5e_acc94.pth':
                pth_file_path = os.path.join(dirpath, filename)

                model.load_state_dict(torch.load(pth_file_path))
                model.eval()

                output_folder_name = f"{filename.split('_')[0]}_{filename.split('_')[1]}_{filename.split('_')[4]}"
                output_folder = Path(f'/home/kamyar/Documents/Train-val_Annotated_Predictions/CENTER/{output_folder_name}')
                output_folder.mkdir(exist_ok=True, parents=True)

                for patch_size in patch_sizes:
                    central_window_size = 96
                    central_offset = (patch_size - central_window_size) // 2
                    x_offsets, y_offsets = np.meshgrid(
                        np.arange(central_offset, central_offset + central_window_size),
                        np.arange(central_offset, central_offset + central_window_size)
                    )
                    offsets = np.stack([x_offsets, y_offsets], axis=-1).reshape(-1, 2)

                    for overlap in overlaps:
                        padding = patch_size // 8
                        # step_size = int(patch_size * (1 - overlap))

                        ###################################################################
                        # STEP SIZE DEFINED FOR 1 VOTE HERE, OTHER THAN THIS, USE THE ABOVE LINE

                        step_size = 96                  ### = (patch size - central_window_size) / 2 + central_window_size - (patch size - central_window_size) / 2 = central_window_size
                        ###################################################################

                        batch_size = 256

                        total_votings = 0
                        total_pixels = 0
                        image_count = 0

                        for image_file in os.listdir(image_folder):
                            if image_file.endswith(('.jpg', '.JPG', '.png')):
                                begin_time = time.perf_counter()
                                image_path = os.path.join(image_folder, image_file)
                                image = Image.open(image_path)
                                image_np = np.array(image)
                                transformed = transform(image=image_np)
                                image_tensor = transformed['image'].to(device)

                                image_tensor_padded = torch.nn.functional.pad(
                                    image_tensor, (padding, padding, padding, padding), 'constant', 0
                                )

                                width, height = image.size
                                padded_width = width + 2 * padding
                                padded_height = height + 2 * padding

                                pixel_predictions = np.zeros((height, width, num_classes), dtype=np.longlong)
                                patches = []
                                coordinates = []

                                for x in range(0, padded_width - patch_size + 1, step_size):
                                    for y in range(0, padded_height - patch_size + 1, step_size):
                                        patch = image_tensor_padded[:, y:y + patch_size, x:x + patch_size]
                                        patches.append(patch)
                                        coordinates.append((x, y))

                                        if len(patches) == batch_size:
                                            patches_tensor = torch.stack(patches).to(device)

                                            with torch.no_grad(), torch.cuda.amp.autocast():
                                                outputs = model(patches_tensor)

                                            predicted_classes = torch.argmax(outputs.logits, dim=1)

                                            for patch_idx, (x, y) in enumerate(coordinates):
                                                predicted_class = predicted_classes[patch_idx]

                                                pixel_coords = offsets + np.array([x, y]) - padding
                                                valid_mask = ((pixel_coords[:, 0] < width)
                                                              & (pixel_coords[:, 1] < height)
                                                              & (pixel_coords[:, 0] > 0)
                                                              & (pixel_coords[:, 1] > 0))
                                                pixel_coords = pixel_coords[valid_mask]
                                                pixel_predictions[pixel_coords[:, 1], pixel_coords[:, 0], predicted_class] += 1

                                            patches = []
                                            coordinates = []
                                if patches:
                                    patches_tensor = torch.stack(patches).to(device)

                                    with torch.no_grad(), torch.cuda.amp.autocast():
                                        outputs = model(patches_tensor)

                                    predicted_classes = torch.argmax(outputs.logits, dim=1)

                                    for patch_idx, (x, y) in enumerate(coordinates):
                                        predicted_class = predicted_classes[patch_idx]

                                        pixel_coords = offsets + np.array([x, y]) - padding
                                        valid_mask = ((pixel_coords[:, 0] < width)
                                                      & (pixel_coords[:, 1] < height)
                                                      & (pixel_coords[:, 0] > 0)
                                                      & (pixel_coords[:, 1] > 0))
                                        pixel_coords = pixel_coords[valid_mask]
                                        pixel_predictions[pixel_coords[:, 1], pixel_coords[:, 0], predicted_class] += 1


                                # Calculate the average number of votings for the image
                                votings_per_pixel = pixel_predictions.sum(axis=2)
                                avg_votings_image = votings_per_pixel.mean()

                                sum_across_classes_image = pixel_predictions.sum(axis=2)
                                unique_sums = np.unique(sum_across_classes_image)
                                print(f"Unique sum values across all pixels: {unique_sums}")
                                # zero_sum_pixels = np.sum(sum_across_classes_image == 0)
                                #
                                # print(f"Number of pixels with a sum of 0 across all classes: {zero_sum_pixels}")


                                # Update dataset-level statistics
                                total_votings += votings_per_pixel.sum()
                                total_pixels += width * height
                                image_count += 1

                                segmentation_map = np.argmax(pixel_predictions, axis=2)

                                output_filename = Path(image_path).with_suffix('.png').name
                                overlap_folder = Path(output_folder) / f'{central_window_size}_{patch_size}_{int(step_size)}'
                                ###################################################################
                                # overlap_folder DEFINED FOR 1 VOTE HERE, OTHER THAN THIS, USE THE ABOVE LINE

                                # overlap_folder = Path(output_folder) / f'{central_window_size}_{patch_size}_{int(step_size)}_oneVOTE'
                                ###################################################################
                                overlap_folder.mkdir(exist_ok=True, parents=True)
                                cv2.imwrite(str(overlap_folder / output_filename), segmentation_map)
                                print(f'Time taken: {time.perf_counter() - begin_time:.2f}s')

                        # Calculate the average number of votings for the entire dataset
                        avg_votings_dataset = total_votings / total_pixels if total_pixels else 0
                        print(f'Average votings for the entire dataset: {avg_votings_dataset:.2f}')

                        # Save the results in the CSV file
                        writer.writerow([central_window_size, patch_size, step_size, avg_votings_dataset])

print("Processing complete. Results saved to", csv_filename)
