import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
from transformers import AutoImageProcessor, AutoModelForImageClassification
from albumentations import (
    RandomResizedCrop, HorizontalFlip, ShiftScaleRotate, ColorJitter,
    RandomBrightnessContrast, Normalize, CenterCrop, Compose, Blur, SmallestMaxSize, RandomScale, Perspective
)
from albumentations.pytorch import ToTensorV2
from torchsampler import ImbalancedDatasetSampler
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_dataset_path = "/home/kamyar/Documents/filtered_inat_split/train"
valid_dataset_path = "/home/kamyar/Documents/filtered_inat_split/val"
test_dataset_path = "/home/kamyar/Documents/filtered_inat_split/test"

train_dataset = ImageFolder(root=train_dataset_path)
valid_dataset = ImageFolder(root=valid_dataset_path)
test_dataset = ImageFolder(root=test_dataset_path)
label_to_id = train_dataset.class_to_idx
output_file_path = "/home/kamyar/PycharmProjects/droneSegmentation/lowAltitude_classification/label_to_id.txt"

with open(output_file_path, 'w') as file:
    for label, idx in label_to_id.items():
        file.write(f"{label}: {idx}\n")

model_name = 'facebook/dinov2-large-imagenet1k-1-layer'
processor = AutoImageProcessor.from_pretrained(model_name)
model = AutoModelForImageClassification.from_pretrained(model_name, ignore_mismatched_sizes=True)

mean = processor.image_mean
std = processor.image_std

# Define the augmentations
augmentations = {
    # "train_transform_0": Compose([
    #     SmallestMaxSize(max_size=256),
    #     RandomResizedCrop(height=256, width=258, scale=(0.4, 1.0), ratio=(0.75, 1.3333)),
    #     HorizontalFlip(p=0.5),
    #     Normalize(mean=mean, std=std),
    #     ToTensorV2()
    # ]),
    # "train_transform_1": Compose([
    #     SmallestMaxSize(max_size=256),
    #     RandomResizedCrop(height=256, width=258, scale=(0.08, 1.0), ratio=(0.75, 1.3333)),
    #     HorizontalFlip(p=0.5),
    #     Normalize(mean=mean, std=std),
    #     ToTensorV2()
    # ]),
    # "train_transform_2": Compose([
    #     SmallestMaxSize(max_size=256),
    #     RandomResizedCrop(height=256, width=258, scale=(0.4, 1.0), ratio=(0.75, 1.3333)),
    #     HorizontalFlip(p=0.5),
    #     ColorJitter(brightness=(0.3, 0.5), contrast=(0.3, 0.5), saturation=(0.3, 0.5), hue=0.2, p=0.5),
    #     Normalize(mean=mean, std=std),
    #     ToTensorV2()
    # ]),
    # "train_transform_3": Compose([
    #     SmallestMaxSize(max_size=256),
    #     RandomResizedCrop(height=256, width=258, scale=(0.4, 1.0), ratio=(0.75, 1.3333)),
    #     HorizontalFlip(p=0.5),
    #     ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
    #     Normalize(mean=mean, std=std),
    #     ToTensorV2()
    # ]),
    # "train_transform_3.5": Compose([
    #     SmallestMaxSize(max_size=256),
    #     RandomResizedCrop(height=256, width=258, scale=(0.08, 1.0), ratio=(0.75, 1.3333)),
    #     HorizontalFlip(p=0.5),
    #     ColorJitter(brightness=(0.3, 0.5), contrast=(0.3, 0.5), saturation=(0.3, 0.5), hue=0.2, p=0.5),
    #     RandomBrightnessContrast(brightness_limit=(0.2, 0.3), contrast_limit=(0.2, 0.3), p=0.5),
    #     Normalize(mean=mean, std=std),
    #     ToTensorV2()
    # ]),
    # "train_transform_4": Compose([
    #     SmallestMaxSize(max_size=256),
    #     RandomResizedCrop(height=256, width=258, scale=(0.08, 1.0), ratio=(0.75, 1.3333)),
    #     HorizontalFlip(p=0.5),
    #     ColorJitter(brightness=(0.3, 0.5), contrast=(0.3, 0.5), saturation=(0.3, 0.5), hue=0.2, p=0.5),
    #     RandomBrightnessContrast(brightness_limit=(0.2, 0.3), contrast_limit=(0.2, 0.3), p=0.5),
    #     Blur(blur_limit=(3, 7), p=0.5),
    #     Normalize(mean=mean, std=std),
    #     ToTensorV2()
    # ]),
    # "train_transform_5": Compose([
    #     SmallestMaxSize(max_size=256),
    #     RandomResizedCrop(height=256, width=258, scale=(0.08, 1.0), ratio=(0.75, 1.3333)),
    #     HorizontalFlip(p=0.5),
    #     ShiftScaleRotate(shift_limit=(0.05, 0.15), scale_limit=(0.05, 0.15), rotate_limit=(30, 60), p=0.5),
    #     ColorJitter(brightness=(0.3, 0.5), contrast=(0.3, 0.5), saturation=(0.3, 0.5), hue=0.2, p=0.5),
    #     RandomBrightnessContrast(brightness_limit=(0.2, 0.3), contrast_limit=(0.2, 0.3), p=0.5),
    #     Blur(blur_limit=(3, 7), p=0.5),
    #     Normalize(mean=mean, std=std),
    #     ToTensorV2()
    # ]),
    # "train_transform_6": Compose([
    #     SmallestMaxSize(max_size=256),
    #     RandomResizedCrop(height=256, width=258, scale=(0.08, 1.0), ratio=(0.75, 1.3333)),
    #     HorizontalFlip(p=0.5),
    #     ShiftScaleRotate(shift_limit=(0.05, 0.15), scale_limit=(0.05, 0.15), rotate_limit=(30, 60), p=0.5),
    #     Perspective(scale=(0.3, 0.6), p=0.7),
    #     ColorJitter(brightness=(0.3, 0.5), contrast=(0.3, 0.5), saturation=(0.3, 0.5), hue=0.2, p=0.5),
    #     RandomBrightnessContrast(brightness_limit=(0.2, 0.3), contrast_limit=(0.2, 0.3), p=0.5),
    #     Blur(blur_limit=(3, 7), p=0.5),
    #     Normalize(mean=mean, std=std),
    #     ToTensorV2()
    # ]),
    "train_transform_7": Compose([
        SmallestMaxSize(max_size=256),
        # RandomScale(scale_limit=0.2, p=0.7),
        RandomResizedCrop(height=256, width=256, scale=(0.4, 1.0), ratio=(0.75, 1.3333)),
        HorizontalFlip(p=0.5),
        ShiftScaleRotate(shift_limit=(0.05, 0.15), scale_limit=(0.05, 0.15), rotate_limit=(30, 60), p=0.5),
        Perspective(scale=(0.05, 0.5), p=0.7),
        ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
        RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        Blur(blur_limit=7, p=0.5),
        Normalize(mean=mean, std=std),
        ToTensorV2()
    ])


}

test_transform = Compose([
    SmallestMaxSize(max_size=256),
    CenterCrop(height=256, width=256),
    Normalize(mean=mean, std=std),
    ToTensorV2()
])

def albumentations_transform(dataset, transform):
    dataset.transform = lambda img: transform(image=np.array(img))['image']
    return dataset

valid_dataset = albumentations_transform(valid_dataset, test_transform)
test_dataset = albumentations_transform(test_dataset, test_transform)

batch_size = 16
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=16)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=16)

model = model.to(device)
num_classes = 32
model.classifier = nn.Linear(2048, num_classes).to(device)
torch.manual_seed(1)
loss_fn = nn.CrossEntropyLoss()
lr = 0.00001
optimizer = optim.Adam(model.parameters(), lr=lr)
scheduler = StepLR(optimizer, step_size=5, gamma=0.1)
best_accuracy = 0.0
best_weights = None

def calculate_accuracy(model, data_loader):
    model.eval()
    correct_predictions = 0
    total_predictions = 0

    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            correct_predictions += (predicted == labels).sum().item()
            total_predictions += labels.size(0)

    accuracy = correct_predictions / total_predictions
    return accuracy

for aug_name, train_transform in augmentations.items():
    print(f"Training with {aug_name}")

    train_dataset_aug = albumentations_transform(train_dataset, train_transform)
    train_loader = DataLoader(train_dataset_aug, batch_size=batch_size, sampler=ImbalancedDatasetSampler(train_dataset_aug, num_samples=100000), num_workers=16)

    model.train()
    for epoch in range(1):
        for batch in train_loader:
            images, labels = batch
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_fn(outputs.logits, labels)
            loss.backward()
            optimizer.step()

    torch.save(model.state_dict(), f"weights_{aug_name}.pth")
    # accuracy = calculate_accuracy(model, valid_loader)
    # print(f"Validation accuracy for {aug_name}: {accuracy:.4f}")

