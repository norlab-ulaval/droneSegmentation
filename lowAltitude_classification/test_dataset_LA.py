"""
test the fine-tuned weights of iNaturalist, on the entire dataset of low-altitude images
export the annotations as a 2d map
"""

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
num_classes = 24
model.classifier = nn.Linear(2048, num_classes).to(device)
mean = processor.image_mean
std = processor.image_std

model.load_state_dict(torch.load('best_model_weights.pth'))
model.eval()

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std),
])

input_folder = '/home/kamyar/Documents/Dataset_lowAltitude'
output_folder = '/home/kamyar/Documents/Dataset_lowAltitude_labels'

patch_sizes = [360]
overlaps = [0.8]


for filename in os.listdir(input_folder):
    if filename.endswith(".JPG"):
        image_path = os.path.join(input_folder, filename)
        image = Image.open(image_path)
        image_tensor = transform(image).to(device)

        for patch_size in patch_sizes:
            for overlap in overlaps:
                width, height = image.size
                # print(width, height)
                step_size = int(patch_size * (1 - overlap))
                pixel_predictions = {}

                for x in range(0, width - patch_size + 1, step_size):
                    for y in range(0, height - patch_size + 1, step_size):
                        if x + step_size > width or y + step_size > height:
                            continue
                        patch = image_tensor[:, y:y + patch_size, x:x + patch_size]

                        with torch.no_grad():
                            output = model(patch.unsqueeze(0))

                        predicted_class = torch.argmax(output.logits, dim=1)

                        for j in range(patch_size):
                            for i in range(patch_size):
                                pixel_x = x + i
                                pixel_y = y + j

                                if (pixel_x, pixel_y) not in pixel_predictions:
                                    pixel_predictions[(pixel_x, pixel_y)] = []
                                pixel_predictions[(pixel_x, pixel_y)].append(predicted_class)

                final_predictions = {}
                for pixel, predictions in pixel_predictions.items():
                    final_predictions[pixel] = Counter(predictions).most_common(1)[0][0]

                segmentation_map = np.zeros((height, width), dtype=np.uint8)

                for pixel, class_value in final_predictions.items():
                    segmentation_map[pixel[1], pixel[0]] = class_value + 1

                output_filename = os.path.splitext(filename)[0] + '_segmentation_map.jpg'
                output_path = os.path.join(output_folder, output_filename)
                Image.fromarray(segmentation_map).save(output_path)