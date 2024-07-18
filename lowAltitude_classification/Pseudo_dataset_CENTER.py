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
    Normalize, CenterCrop, Compose, Resize, SmallestMaxSize
)
from albumentations.pytorch import ToTensorV2


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_name = 'facebook/dinov2-large-imagenet1k-1-layer'
processor = AutoImageProcessor.from_pretrained(model_name)
model = AutoModelForImageClassification.from_pretrained(model_name, ignore_mismatched_sizes=True)
model = model.to(device)
num_classes = 32
model.classifier = nn.Linear(2048, num_classes).to(device)
mean = processor.image_mean
std = processor.image_std

model.load_state_dict(torch.load('/home/kamyar/PycharmProjects/droneSegmentation/lowAltitude_classification/best_classification_weights.pth'))
model.eval()

transform = Compose([
    Normalize(mean=mean, std=std),
    ToTensorV2()
])

image_folder = '/home/kamyar/Documents/Test_data'

patch_sizes = [128]
overlaps = [0.85]

for patch_size in patch_sizes:
    for overlap in overlaps:
        padding = patch_size // 8
        step_size = int(patch_size * (1 - overlap))
        batch_size = 32

        output_folder = f'/home/kamyar/Documents/Test_data_pred/patch_{patch_size}_overlap_{int(overlap*100)}'
        os.makedirs(output_folder, exist_ok=True)

        for image_file in os.listdir(image_folder):
            if image_file.endswith(('.jpg', '.JPG', '.png')):
                image_path = os.path.join(image_folder, image_file)
                image = Image.open(image_path)
                image_np = np.array(image)
                transformed = transform(image=image_np)
                image_tensor = transformed['image'].to(device)

                image_tensor_padded = torch.nn.functional.pad(image_tensor, (padding, padding, padding, padding), 'constant', 0)

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
                                predicted_class = predicted_classes[patch_idx]

                                center_x = x + patch_size // 2
                                center_y = y + patch_size // 2

                                for j in range(center_y - 40, center_y + 40):
                                    for i in range(center_x - 40, center_x + 40):
                                        pixel_x = i - padding
                                        pixel_y = j - padding

                                        if 0 <= pixel_x < width and 0 <= pixel_y < height:
                                            if (pixel_x, pixel_y) not in pixel_predictions:
                                                pixel_predictions[(pixel_x, pixel_y)] = []
                                            pixel_predictions[(pixel_x, pixel_y)].append(predicted_class)

                            patches = []
                            coordinates = []

                if patches:
                    patches_tensor = torch.stack(patches).to(device)

                    with torch.no_grad():
                        outputs = model(patches_tensor)

                    predicted_classes = torch.argmax(outputs.logits, dim=1).cpu().numpy()

                    for patch_idx, (x, y) in enumerate(coordinates):
                        predicted_class = predicted_classes[patch_idx]

                        center_x = x + patch_size // 2
                        center_y = y + patch_size // 2

                        for j in range(center_y - 2, center_y + 3):
                            for i in range(center_x - 2, center_x + 3):
                                pixel_x = i - padding
                                pixel_y = j - padding

                                if 0 <= pixel_x < width and 0 <= pixel_y < height:
                                    if (pixel_x, pixel_y) not in pixel_predictions:
                                        pixel_predictions[(pixel_x, pixel_y)] = []
                                    pixel_predictions[(pixel_x, pixel_y)].append(predicted_class)

                final_predictions = {}
                for pixel, predictions in pixel_predictions.items():
                    final_predictions[pixel] = Counter(predictions).most_common(1)[0][0]

                segmentation_map = np.zeros((height, width), dtype=np.uint8)
                for pixel, class_value in final_predictions.items():
                    segmentation_map[pixel[1], pixel[0]] = class_value

                output_filename = os.path.splitext(os.path.basename(image_path))[0] + '.png'
                output_path = os.path.join(output_folder, output_filename)
                cv2.imwrite(output_path, segmentation_map)

print("Processing complete.")
