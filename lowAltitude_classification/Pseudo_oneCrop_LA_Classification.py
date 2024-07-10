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
num_classes = 30
model.classifier = nn.Linear(2048, num_classes).to(device)
mean = processor.image_mean
std = processor.image_std

model.load_state_dict(torch.load('/home/kamyar/PycharmProjects/droneSegmentation/lowAltitude_classification/best_classification_weights.pth'))
model.eval()

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

image = Image.open('/home/kamyar/Desktop/2024-06-05-132224-5-ZecBatiscan-5280x5280-DJI-M3E-patch-11.jpg')
image = transform(image).to(device)
image = image.unsqueeze(0)

with torch.no_grad():
    output = model(image)

# Apply softmax to get probabilities
probabilities = torch.nn.functional.softmax(output.logits, dim=1)

predicted_class = torch.argmax(probabilities, dim=1).item()

label_to_id = {}
with open("/home/kamyar/PycharmProjects/droneSegmentation/lowAltitude_classification/label_to_id.txt", 'r') as file:
    for line in file:
        label, idx = line.strip().split(": ")
        label_to_id[int(idx)] = label

predicted_class_name = label_to_id.get(predicted_class, "Class not found")

print(f"Predicted class: {predicted_class_name}")

for idx, prob in enumerate(probabilities[0]):
    class_name = label_to_id.get(idx, f"Class {idx}")
    print(f"{class_name}: {prob.item() * 100:.2f}%")
