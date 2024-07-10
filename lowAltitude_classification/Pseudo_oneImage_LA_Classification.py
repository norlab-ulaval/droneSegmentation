import pathlib
import cv2
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
num_classes = 30
model.classifier = nn.Linear(2048, num_classes).to(device)
mean = processor.image_mean
std = processor.image_std

model.load_state_dict(torch.load('/home/kamyar/PycharmProjects/droneSegmentation/lowAltitude_classification/best_classification_weights.pth'))
model.eval()

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std),
])

image_path = '/home/kamyar/Documents/test_2/2024-06-05-132224-5-ZecBatiscan-5280x5280-DJI-M3E-patch-11.jpg'
image = Image.open(image_path)
image_tensor = transform(image).to(device)

patch_size = 256
overlap = 0.95
padding = patch_size // 8
image_tensor_padded = torch.nn.functional.pad(image_tensor, (padding, padding, padding, padding), 'constant', 0)

width, height = image.size
padded_width = width + 2 * padding
padded_height = height + 2 * padding

step_size = int(patch_size * (1 - overlap))
pixel_predictions = {}

for x in range(0, padded_width - patch_size + 1, step_size):
    for y in range(0, padded_height - patch_size + 1, step_size):
        patch = image_tensor_padded[:, y:y + patch_size, x:x + patch_size]

        with torch.no_grad():
            output = model(patch.unsqueeze(0))

        predicted_class = torch.argmax(output.logits, dim=1).item()

        for j in range(patch_size):
            for i in range(patch_size):
                pixel_x = x + i - padding
                pixel_y = y + j - padding

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

class_labels = {}
with open("/home/kamyar/PycharmProjects/droneSegmentation/lowAltitude_classification/label_to_id.txt", 'r') as file:
    for line in file:
        label, idx = line.strip().split(": ")
        class_labels[int(idx)] = label

colors_dict = {
    0: (255, 0, 0), 1: (255, 165, 0), 2: (255, 255, 0), 3: (0, 128, 0), 4: (0, 0, 255),
    5: (75, 0, 130), 6: (238, 130, 238), 7: (255, 192, 203), 8: (165, 42, 42), 9: (0, 0, 0),
    10: (255, 255, 255), 11: (128, 128, 128), 12: (220, 20, 60), 13: (64, 224, 208),
    14: (0, 128, 128), 15: (0, 255, 0), 16: (255, 127, 80), 17: (0, 0, 128), 18: (255, 0, 255),
    19: (255, 215, 0), 20: (192, 192, 192), 21: (230, 230, 250), 22: (128, 0, 128),
    23: (0, 255, 255), 24: (189, 252, 201), 25: (124, 255, 78), 26: (255, 99, 71),
    27: (47, 79, 79), 28: (255, 20, 147), 29: (100, 149, 237), 30: (184, 134, 11), 31: (139, 69, 19)
}

colors = list(colors_dict.values())
cmap = ListedColormap(colors)

fig, ax = plt.subplots(1, 2, figsize=(12, 7))

ax[0].text(0.5, 1.05, f'Patch Size: {patch_size}, Overlap: {overlap}', transform=ax[0].transAxes,
           horizontalalignment='center', verticalalignment='bottom', fontsize=12)

segmentation_image = ax[0].imshow(segmentation_map, cmap=cmap, vmin=0, vmax=num_classes)

image_np = np.asarray(image)
gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
ax[1].imshow(gray, cmap='gray')
ax[1].imshow(segmentation_map, cmap=cmap, vmin=0, vmax=num_classes+1, alpha=0.5)
plt.tight_layout(pad=2)
plt.show()
