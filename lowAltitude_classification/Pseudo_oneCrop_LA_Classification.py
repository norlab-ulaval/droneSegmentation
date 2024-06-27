"""
test the fine-tuned weights of iNaturalist, on one crop of low-altitude images
"""

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
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

model.load_state_dict(torch.load('/home/kamyar/PycharmProjects/droneSegmentation/lowAltitude_classification/bestModel_otherclasses/best_classification_weights.pth'))
model.eval()

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

image = Image.open('/home/kamyar/Desktop/photo_2024-05-23_10-23-34.jpg')
image = transform(image).to(device)
image = image.unsqueeze(0)


with torch.no_grad():
    output = model(image)

predicted_class = torch.argmax(output.logits, dim=1).item()

label_to_id = {}
with open("/home/kamyar/PycharmProjects/droneSegmentation/lowAltitude_classification/label_to_id.txt", 'r') as file:
    for line in file:
        label, idx = line.strip().split(": ")
        label_to_id[int(idx)] = label

predicted_class_name = label_to_id.get(predicted_class, "Class not found")

print(f"Predicted class: {predicted_class_name}")