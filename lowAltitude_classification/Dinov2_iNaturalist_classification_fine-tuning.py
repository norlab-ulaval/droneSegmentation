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

train_dataset_path = "/home/kamyar/Documents/iNaturalist_data_split/train"
valid_dataset_path = "/home/kamyar/Documents/iNaturalist_data_split/val"
test_dataset_path = "/home/kamyar/Documents/iNaturalist_data_split/test"

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

train_transform = Compose([
    SmallestMaxSize(max_size=256),
    # RandomScale(scale_limit=0.2, p=0.5),
    RandomResizedCrop(height=256, width=256, scale=(0.4, 1.0), ratio=(0.75, 1.3333)),
    HorizontalFlip(p=0.5),
    # Perspective(scale=(0.05, 0.1), p=0.7),
    # ShiftScaleRotate(shift_limit=(0.05, 0.15), scale_limit=(0.05, 0.15), rotate_limit=(30, 60), p=0.5),
    # ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=45, p=0.5),
    # ColorJitter(brightness=(0.3, 0.5), contrast=(0.3, 0.5), saturation=(0.3, 0.5), hue=0.2, p=0.5),
    # RandomBrightnessContrast(brightness_limit=(0.2, 0.3), contrast_limit=(0.2, 0.3), p=0.5),
    # Blur(blur_limit=(3, 7), p=0.5),
    Normalize(mean=mean, std=std),
    ToTensorV2()
])

test_transform = Compose([
    SmallestMaxSize(max_size=256),
    CenterCrop(height=256, width=256),
    Normalize(mean=mean, std=std),
    ToTensorV2()
])

def albumentations_transform(dataset, transform):
    dataset.transform = lambda img: transform(image=np.array(img))['image']
    return dataset

train_dataset = albumentations_transform(train_dataset, train_transform)
valid_dataset = albumentations_transform(valid_dataset, test_transform)
test_dataset = albumentations_transform(test_dataset, test_transform)

batch_size = 16
train_loader = DataLoader(train_dataset, sampler=ImbalancedDatasetSampler(train_dataset, num_samples=100000), batch_size=batch_size, num_workers=16)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=16)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=16)

model = model.to(device)
num_classes = 27
model.classifier = nn.Linear(2048, num_classes).to(device)
torch.manual_seed(1)
num_epochs = 1
loss_fn = nn.CrossEntropyLoss()
lr = 0.00001
weight_decay = 1e-3
optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
step_size = 3
scheduler = StepLR(optimizer, step_size=step_size, gamma=0.0001)

loss_hist_train = []
accuracy_hist_train = []
loss_hist_valid = []
accuracy_hist_valid = []

best_accuracy = 0.0
best_epoch = -1
best_model_weights = None

for epoch in range(num_epochs):
    model.train()
    loss_accumulated_train = 0.0
    total_samples_train = 0
    correct_predictions_train = 0

    for x_batch, y_batch in train_loader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        output = model(x_batch)
        logits = output.logits
        loss = loss_fn(logits, y_batch)
        loss.backward()
        optimizer.step()

        loss_accumulated_train += loss.item() * y_batch.size(0)
        total_samples_train += y_batch.size(0)
        correct_predictions_train += (torch.argmax(logits, dim=1) == y_batch).sum().item()

    loss_hist_train.append(loss_accumulated_train / total_samples_train)
    accuracy_train = correct_predictions_train / total_samples_train
    accuracy_hist_train.append(accuracy_train)

    scheduler.step()

    model.eval()
    with torch.no_grad():
        loss_accumulated_valid = 0.0
        total_samples_valid = 0
        correct_predictions_valid = 0

        for x_batch, y_batch in valid_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            output = model(x_batch)
            logits = output.logits
            loss = loss_fn(logits, y_batch)

            loss_accumulated_valid += loss.item() * y_batch.size(0)
            total_samples_valid += y_batch.size(0)
            correct_predictions_valid += (torch.argmax(logits, dim=1) == y_batch).sum().item()

        loss_hist_valid.append(loss_accumulated_valid / total_samples_valid)
        accuracy_hist_valid.append(correct_predictions_valid / total_samples_valid)

    print(f'Epoch {epoch + 1} accuracy: {accuracy_hist_train[epoch]:.4f} val_accuracy: {accuracy_hist_valid[epoch]:.4f} '
          f'loss: {loss_hist_train[epoch]:.4f} val_loss: {loss_hist_valid[epoch]:.4f}')

    if accuracy_hist_valid[epoch] > best_accuracy:
        best_accuracy = accuracy_hist_valid[epoch]
        best_epoch = epoch
        best_model_weights = model.state_dict()
        torch.save(best_model_weights, '/home/kamyar/PycharmProjects/droneSegmentation/lowAltitude_classification/filtered_inat_2_2.pth')
        print(f"Best model weights saved at epoch {best_epoch + 1} with validation accuracy {best_accuracy:.4f}")

if best_model_weights is not None:
    model.load_state_dict(best_model_weights)

model.eval()
all_preds = []
all_labels = []
with torch.no_grad():
    correct_predictions_test = 0
    total_samples_test = 0

    for x_batch, y_batch in test_loader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        output = model(x_batch)
        logits = output.logits
        preds = torch.argmax(logits, dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y_batch.cpu().numpy())

        correct_predictions_test += (preds == y_batch).sum().item()
        total_samples_test += y_batch.size(0)

    test_accuracy = correct_predictions_test / total_samples_test
    print(f'Test accuracy: {test_accuracy:.4f}')
