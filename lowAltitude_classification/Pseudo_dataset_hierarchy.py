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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_name = 'facebook/dinov2-large-imagenet1k-1-layer'
processor = AutoImageProcessor.from_pretrained(model_name)
model = AutoModelForImageClassification.from_pretrained(model_name, ignore_mismatched_sizes=True)
model = model.to(device)
num_classes = 32
model.classifier = nn.Linear(2048, num_classes).to(device)
mean = processor.image_mean
std = processor.image_std

model.load_state_dict(torch.load('lowAltitude_classification/best_classification_weights.pth'))
model.eval()

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std),
])

image_folder = '/home/kamyar/Documents/Test_data'

patch_sizes = [144, 256]
overlap = 0.9

def get_patch_predictions(image_tensor_padded, width, height, patch_size, step_size, padding, batch_size):
    pixel_predictions = {}
    patches = []
    coordinates = []

    for x in range(0, width + 2 * padding - patch_size + 1, step_size):
        for y in range(0, height + 2 * padding - patch_size + 1, step_size):
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

                    for j in range(patch_size):
                        for i in range(patch_size):
                            pixel_x = x + i - padding
                            pixel_y = y + j - padding

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

            for j in range(patch_size):
                for i in range(patch_size):
                    pixel_x = x + i - padding
                    pixel_y = y + j - padding

                    if 0 <= pixel_x < width and 0 <= pixel_y < height:
                        if (pixel_x, pixel_y) not in pixel_predictions:
                            pixel_predictions[(pixel_x, pixel_y)] = []
                        pixel_predictions[(pixel_x, pixel_y)].append(predicted_class)

    return pixel_predictions

for image_file in os.listdir(image_folder):
    if image_file.endswith(('.jpg', '.JPG')):
        image_path = os.path.join(image_folder, image_file)
        image = Image.open(image_path)
        image_tensor = transform(image).to(device)

        width, height = image.size

        all_pixel_predictions = {}

        for patch_size in patch_sizes:
            padding = patch_size // 8
            step_size = int(patch_size * (1 - overlap))
            batch_size = 32

            image_tensor_padded = torch.nn.functional.pad(image_tensor, (padding, padding, padding, padding), 'constant', 0)

            pixel_predictions = get_patch_predictions(image_tensor_padded, width, height, patch_size, step_size, padding, batch_size)

            for pixel, predictions in pixel_predictions.items():
                if pixel not in all_pixel_predictions:
                    all_pixel_predictions[pixel] = []
                all_pixel_predictions[pixel].extend(predictions)

        final_predictions = {}
        for pixel, predictions in all_pixel_predictions.items():
            final_predictions[pixel] = Counter(predictions).most_common(1)[0][0]

        segmentation_map = np.zeros((height, width), dtype=np.uint8)
        for pixel, class_value in final_predictions.items():
            segmentation_map[pixel[1], pixel[0]] = class_value

        output_folder = '/home/kamyar/Documents/Test_data_pred_with_background_hierarchy'
        os.makedirs(output_folder, exist_ok=True)
        output_filename = os.path.splitext(os.path.basename(image_path))[0] + '.png'
        output_path = os.path.join(output_folder, output_filename)
        cv2.imwrite(output_path, segmentation_map)

print("Processing complete.")
