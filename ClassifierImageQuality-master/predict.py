import os
import shutil

from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.datasets import ImageFolder
from torchvision.transforms import v2
from pathlib import Path
from torchvision import models
from torchvision.io import read_image
from torch.utils.data import DataLoader

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

image_dataset_directory = '/home/kamyar/Documents/iNat_Classifier_Non_filtered/Pine/images'
image_sort_directory = '/home/kamyar/Documents/iNat_Classifier_filtered/Pine/'
#species_list = os.listdir(Path.cwd() / image_dataset_directory)

#image_loader = DataLoader(image_dataset, batch_size=1, shuffle=True)

model = models.resnet50(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, 4)
model.load_state_dict(torch.load("ClassifierImageQuality-master/model.pth"))
model.eval()
model.to(device)

transforms = v2.Compose([
    v2.ToTensor(),
    v2.Resize(size=(224, 224))
])

categories = ['bad', 'good', 'too_close', 'too_far']

with torch.no_grad():
    #for species in species_list:

    if not os.path.exists(Path.cwd() / image_sort_directory):
        os.mkdir(Path.cwd() / image_sort_directory)
    for category in categories:
        os.mkdir(Path.cwd() / image_sort_directory / category)
    image_list = os.listdir(Path.cwd() / image_dataset_directory)
    for X in image_list:
        img = Image.open(Path.cwd() / image_dataset_directory / X).convert('RGB')
        img = transforms(img)
        img = img.to(device).float()
        pred = model(img.unsqueeze(0))
        predicted_class = pred.argmax(1).item()
        shutil.copy(Path.cwd() / image_dataset_directory / X, Path.cwd() / image_sort_directory / categories[predicted_class])
        print(predicted_class)
