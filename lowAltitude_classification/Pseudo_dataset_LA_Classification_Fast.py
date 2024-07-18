import os
import time

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from albumentations import (
    Normalize, Compose
)
from albumentations.pytorch import ToTensorV2
from transformers import AutoImageProcessor, AutoModelForImageClassification

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_name = 'facebook/dinov2-large-imagenet1k-1-layer'
processor = AutoImageProcessor.from_pretrained(model_name)
model = AutoModelForImageClassification.from_pretrained(model_name, ignore_mismatched_sizes=True)
model = model.to(device)
num_classes = 32
model.classifier = nn.Linear(2048, num_classes).to(device)
mean = processor.image_mean
std = processor.image_std

model.load_state_dict(torch.load('/home/kamyar/PycharmProjects/droneSegmentation/lowAltitude_classification/filtered_inat.pth'))
model.eval()

transform = Compose([
    Normalize(mean=mean, std=std),
    ToTensorV2()
])

image_folder = '/home/kamyar/Documents/data_lowaltitude_patches'

patch_sizes = [256]
overlaps = [0.85]



for patch_size in patch_sizes:
    x_offsets, y_offsets = np.meshgrid(np.arange(patch_size), np.arange(patch_size))
    offsets = np.stack([x_offsets, y_offsets], axis=-1).reshape(-1, 2)

    for overlap in overlaps:
        padding = patch_size // 8
        step_size = int(patch_size * (1 - overlap))
        batch_size = 1024

        output_folder = f'/home/kamyar/Documents/dataset_pred/fast_patch_{patch_size}_overlap_{int(overlap * 100)}'
        os.makedirs(output_folder, exist_ok=True)

        for image_file in os.listdir(image_folder):
            if image_file.endswith(('.jpg', '.JPG', '.png')):
                begin_time = time.perf_counter()
                image_path = os.path.join(image_folder, image_file)
                image = Image.open(image_path)
                image_np = np.array(image)
                transformed = transform(image=image_np)
                image_tensor = transformed['image'].to(device)

                image_tensor_padded = torch.nn.functional.pad(image_tensor, (padding, padding, padding, padding),
                                                              'constant', 0)

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

                segmentation_map = np.argmax(pixel_predictions, axis=2)

                output_filename = os.path.splitext(os.path.basename(image_path))[0] + '.png'
                output_path = os.path.join(output_folder, output_filename)
                cv2.imwrite(output_path, segmentation_map)
                print(f'Time taken: {time.perf_counter() - begin_time:.2f}s')

print("Processing complete.")

