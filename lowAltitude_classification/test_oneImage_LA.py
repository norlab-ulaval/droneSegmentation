"""
test the fine-tuned weights of iNaturalist, on one image of low-altitude images
"""

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_name = 'facebook/dinov2-large-imagenet1k-1-layer'
processor = AutoImageProcessor.from_pretrained(model_name)
model = AutoModelForImageClassification.from_pretrained(model_name, ignore_mismatched_sizes=True)
model = model.to(device)
num_classes = 21
model.classifier = nn.Linear(2048, num_classes).to(device)
mean = processor.image_mean
std = processor.image_std

model.load_state_dict(torch.load('best_model_weights.pth'))
model.eval()

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std),
])

image_path = '/home/kamyar/Desktop/DJI_0847.JPG'
image = Image.open(image_path)
image_tensor = transform(image).to(device)

patch_sizes = [256]
overlaps = [0.85]

for patch_size in patch_sizes:
    for overlap in overlaps:
        width, height = image.size
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

        segmentation_map = np.zeros((width, height), dtype=np.uint8)

        for pixel, class_value in final_predictions.items():
            segmentation_map[pixel[1], pixel[0]] = class_value + 1

        class_labels = {}
        class_labels[0] = 'background'
        with open("label_to_id.txt", 'r') as file:
            for line in file:
                label, idx = line.strip().split(": ")
                class_labels[int(idx)+1] = label

        color_dict = {
             0: 'Red',
             1: 'Orange',
             2: 'Yellow',
             3: 'Green',
             4: 'Blue',
             5: 'Indigo',
             6: 'Violet',
             7: 'Pink',
             8: 'Brown',
             9: 'Black',
             10: 'White',
             11: 'Gray',
             12: 'Crimson',
             13: 'Turquoise',
             14: 'Teal',
             15: 'Lime',
             16: 'Coral',
             17: 'Navy',
             18: 'Magenta',
             19: 'Gold',
             20: 'Silver',
             21: 'Lavender'
        }

        colors = list(color_dict.values())
        cmap = ListedColormap(colors)

        fig, ax = plt.subplots(1, 2, figsize=(12, 7))

        ax[0].text(0.5, 1.05, f'Patch Size: {patch_size}, Overlap: {overlap}', transform=ax[0].transAxes,
                   horizontalalignment='center', verticalalignment='bottom', fontsize=12)

        segmentation_image = ax[0].imshow(segmentation_map, cmap=cmap, vmin=0, vmax=num_classes+1)

        from matplotlib.patches import Patch
        legend_labels = [Patch(facecolor=color_dict[i], edgecolor=color_dict[i], label=label)
                         for i, label in enumerate(class_labels.values(), start=0)]

        ax[0].legend(handles=legend_labels, bbox_to_anchor=(1.05, 1), loc='center left', prop={'size': 10})

        ax[1].imshow(image)
        ax[1].imshow(segmentation_map, cmap=cmap, vmin=0, vmax=num_classes+1, alpha=0.5)
        plt.tight_layout(pad=2)
        plt.savefig(f"patch_{patch_size}_overlap_{overlap}.png")
        plt.close()
