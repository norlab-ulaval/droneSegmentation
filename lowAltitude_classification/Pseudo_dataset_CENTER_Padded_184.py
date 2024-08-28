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


# SPLIT = os.environ.get("SPLIT", None)
# if SPLIT is None:
#     raise ValueError("SPLIT environment variable must be set: 'Fifth batch'  'First batch'  'Fourth batch'  'Second batch'  'Third batch'")

# Paths
# results_dir = Path("/data/droneSegResults")
# weight_file_path = Path("/data/Best_classifier_Weight/52_Final_time2024-08-15_best_5e_acc94.pth")
# image_folder = Path(f"/data/Unlabeled_Drone_Dataset/Drone_Unlabeled_Dataset_Patch_split/{SPLIT}/")
# output_dir = results_dir / 'Unlabeled_Drone_Dataset_PL_version2' / image_folder.name

results_dir = Path("/home/kamyar/Documents")
weight_file_path = Path("/home/kamyar/Documents/Best_classifier_Weight/52_Final_time2024-08-15_best_5e_acc94.pth")
image_folder = Path(f"/home/kamyar/Documents/Unlabeled_Drone_Dataset_Patch_split/Second batch/")
output_dir = results_dir / 'Unlabeled_Drone_Dataset_PL_version2' / image_folder.name

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

model.load_state_dict(torch.load(weight_file_path))
model.eval()

output_dir.mkdir(parents=True, exist_ok=True)


for patch_size in patch_sizes:
    central_window_size = 96
    central_offset = (patch_size - central_window_size) // 2
    x_offsets, y_offsets = np.meshgrid(
        np.arange(central_offset, central_offset + central_window_size),
        np.arange(central_offset, central_offset + central_window_size)
    )
    offsets = np.stack([x_offsets, y_offsets], axis=-1).reshape(-1, 2)

    for overlap in overlaps:
        padding = patch_size
        step_size = int(patch_size * (1 - overlap))
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
                    image_tensor, (padding, padding, padding, padding), 'constant', value=0
                )

                width, height = image.size
                padded_width = width + 2 * padding
                padded_height = height + 2 * padding

                pixel_predictions = np.zeros((height, width, num_classes), dtype=np.longlong)
                patches = []
                coordinates = []

                for x in range(0, padded_width - patch_size, step_size):
                    for y in range(0, padded_height - patch_size, step_size):
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
                                              & (pixel_coords[:, 0] >=0)
                                              & (pixel_coords[:, 1] >= 0))
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
                                      & (pixel_coords[:, 0] >= 0)
                                      & (pixel_coords[:, 1] >= 0))
                        pixel_coords = pixel_coords[valid_mask]
                        pixel_predictions[pixel_coords[:, 1], pixel_coords[:, 0], predicted_class] += 1

                votings_per_pixel = pixel_predictions.sum(axis=2)
                non_zero_votings = votings_per_pixel[votings_per_pixel > 0]
                avg_votings_image = non_zero_votings.mean() if len(non_zero_votings) > 0 else 0

                total_votings += non_zero_votings.sum()
                total_pixels += len(non_zero_votings)
                image_count += 1

                segmentation_map = np.argmax(pixel_predictions, axis=2)
                segmentation_map[votings_per_pixel == 0] = -1

                output_filename = Path(image_path).with_suffix('.png').name
                overlap_folder = Path(output_dir) / f'center-{central_window_size}_patch-{patch_size}_step-{int(step_size)}_pad-{int(padding)}'
                overlap_folder.mkdir(exist_ok=True, parents=True)
                cv2.imwrite(str(overlap_folder / output_filename), segmentation_map)
                print(f'Time taken: {time.perf_counter() - begin_time:.2f}s')


print("Processing complete.")







