import pathlib
import cv2
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification
import numpy as np
from collections import Counter
import os
from albumentations import (
    Normalize, Compose
)
from albumentations.pytorch import ToTensorV2
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_name = 'facebook/dinov2-large-imagenet1k-1-layer'
processor = AutoImageProcessor.from_pretrained(model_name)
model = AutoModelForImageClassification.from_pretrained(model_name, ignore_mismatched_sizes=True)
model = model.to(device)
num_classes = 32
model.classifier = nn.Linear(2048, num_classes).to(device)
mean = processor.image_mean
std = processor.image_std

model.load_state_dict(
    torch.load(''))
model.eval()

transform = Compose([
    Normalize(mean=mean, std=std),
    ToTensorV2()
])

image_folder = ''
patch_sizes = [256]
overlaps = [0.85]

for patch_size in patch_sizes:
    for overlap in overlaps:
        padding = patch_size // 8
        step_size = int(patch_size * (1 - overlap))
        batch_size = 256

        output_folder = f'/patch_{patch_size}_overlap_{int(overlap * 100)}'
        os.makedirs(output_folder, exist_ok=True)

        for image_file in os.listdir(image_folder):
            if image_file.endswith(('.jpg', '.JPG', '.png')):
                # start_time = time.time()
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

                pixel_predictions = {}
                patches = []
                coordinates = []

                for x in range(0, padded_width - patch_size + 1, step_size):
                    for y in range(0, padded_height - patch_size + 1, step_size):
                        patch = image_tensor_padded[:, y:y + patch_size, x:x + patch_size]
                        patches.append(patch)
                        coordinates.append((x, y))

                        if len(patches) == batch_size:
                            patches_tensor = torch.stack(patches).to(device)

                            with torch.no_grad():
                                outputs = model(patches_tensor)

                            predicted_classes = torch.argmax(outputs.logits, dim=1).cpu().numpy()

                            for patch_idx, (x, y) in enumerate(coordinates):
                                center_x = x + patch_size // 2
                                center_y = y + patch_size // 2

                                predicted_class = predicted_classes[patch_idx]

                                if (center_x, center_y) not in pixel_predictions:
                                    pixel_predictions[(center_x, center_y)] = []
                                pixel_predictions[(center_x, center_y)].append(predicted_class)

                            patches = []
                            coordinates = []

                if patches:
                    patches_tensor = torch.stack(patches).to(device)

                    with torch.no_grad():
                        outputs = model(patches_tensor)

                    predicted_classes = torch.argmax(outputs.logits, dim=1).cpu().numpy()

                    for patch_idx, (x, y) in enumerate(coordinates):
                        center_x = x + patch_size // 2
                        center_y = y + patch_size // 2

                        predicted_class = predicted_classes[patch_idx]

                        if (center_x, center_y) not in pixel_predictions:
                            pixel_predictions[(center_x, center_y)] = []
                        pixel_predictions[(center_x, center_y)].append(predicted_class)

                # end_time = time.time()
                # print(end_time - start_time)
                start_time = time.time()
                segmentation_map = np.empty((height, width), dtype=object)
                for i in range(height):
                    for j in range(width):
                        segmentation_map[i, j] = []

                for (center_x, center_y), class_value in pixel_predictions.items():
                    x_start = max(center_x - patch_size // 2, 0)
                    y_start = max(center_y - patch_size // 2, 0)
                    x_end = min(center_x + patch_size // 2, width)
                    y_end = min(center_y + patch_size // 2, height)

                    for x in range(x_start, x_end):
                        for y in range(y_start, y_end):
                            segmentation_map[y, x].append(class_value[0])

                for y in range(height):
                    for x in range(width):
                        if segmentation_map[y, x]:
                            most_common_class = Counter(segmentation_map[y, x]).most_common(1)[0][0]
                            segmentation_map[y, x] = most_common_class

                output_filename = os.path.splitext(os.path.basename(image_path))[0] + '.png'
                output_path = os.path.join(output_folder, output_filename)
                cv2.imwrite(output_path, segmentation_map)

                end_time = time.time()
                print(end_time - start_time)
                # exit()
print("Processing complete.")
