import datetime
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from albumentations import (
    Blur,
    CenterCrop,
    ColorJitter,
    Compose,
    HorizontalFlip,
    Normalize,
    RandomResizedCrop,
    SmallestMaxSize,
)
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import StratifiedKFold
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import ImageFolder
from transformers import AutoImageProcessor, AutoModelForImageClassification

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data_folder = "/home/kamyar/Documents/iNat_Classifier_filtered"
output_file_path = "/home/kamyar/PycharmProjects/droneSegmentation/lowAltitude_classification/label_to_id.txt"
log_file_path = "/home/kamyar/PycharmProjects/droneSegmentation/lowAltitude_classification/Augmentation_iNat_classifier/log_aug20.txt"

dataset = ImageFolder(root=data_folder)
label_to_id = dataset.class_to_idx

with open(output_file_path, "w") as file:
    for label, idx in label_to_id.items():
        file.write(f"{label}: {idx}\n")

json_path = Path(output_file_path).with_suffix(".json")
with open(json_path, "w") as f:
    json.dump(label_to_id, f, indent=2, sort_keys=True)

checkpoint_dir = json_path.parent / "checkpoints"
checkpoint_dir.mkdir(parents=True, exist_ok=True)


model_name = "facebook/dinov2-large-imagenet1k-1-layer"
processor = AutoImageProcessor.from_pretrained(model_name)
model = AutoModelForImageClassification.from_pretrained(
    model_name, ignore_mismatched_sizes=True
)

mean = processor.image_mean
std = processor.image_std

train_transform = Compose(
    [
        SmallestMaxSize(max_size=256),
        RandomResizedCrop(
            height=256, width=256, scale=(0.4, 1.0), ratio=(0.75, 1.3333)
        ),
        HorizontalFlip(p=0.5),
        ColorJitter(
            brightness=(0.3, 0.5),
            contrast=(0.3, 0.5),
            saturation=(0.3, 0.5),
            hue=0.2,
            p=0.5,
        ),
        Blur(blur_limit=(3, 7), p=0.5),
        Normalize(mean=mean, std=std),
        ToTensorV2(),
    ]
)

test_transform = Compose(
    [
        SmallestMaxSize(max_size=256),
        CenterCrop(height=256, width=256),
        Normalize(mean=mean, std=std),
        ToTensorV2(),
    ]
)


def albumentations_transform(dataset, transform):
    dataset.transform = lambda img: transform(image=np.array(img))["image"]
    return dataset


dataset = albumentations_transform(dataset, train_transform)

kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
fold_accuracies = []

# Open the log file
with open(log_file_path, "w") as log_file:
    for fold, (train_idx, val_idx) in enumerate(kf.split(dataset, dataset.targets)):
        log_file.write(f"Fold {fold + 1}/{kf.get_n_splits()}\n")

        train_subset = Subset(dataset, train_idx)
        val_subset = Subset(dataset, val_idx)

        train_loader = DataLoader(
            train_subset, shuffle=True, batch_size=16, num_workers=16
        )
        val_loader = DataLoader(
            val_subset, batch_size=16, shuffle=False, num_workers=16
        )

        model = AutoModelForImageClassification.from_pretrained(
            model_name, ignore_mismatched_sizes=True
        )
        model.classifier = nn.Linear(2048, len(label_to_id)).to(device)
        model = model.to(device)

        optimizer = optim.AdamW(model.parameters(), lr=0.00001, weight_decay=1e-3)
        scheduler = StepLR(optimizer, step_size=3, gamma=0.0001)
        loss_fn = nn.CrossEntropyLoss()

        num_epochs = 5
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
                correct_predictions_train += (
                    (torch.argmax(logits, dim=1) == y_batch).sum().item()
                )

            log_file.write(
                f"Epoch {epoch + 1} - Training Loss: {loss_accumulated_train / total_samples_train:.4f} Accuracy: {correct_predictions_train / total_samples_train:.4f}\n"
            )
            scheduler.step()

            # Validation
            model.eval()
            with torch.no_grad():
                loss_accumulated_valid = 0.0
                total_samples_valid = 0
                correct_predictions_valid = 0

                for x_batch, y_batch in val_loader:
                    x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                    output = model(x_batch)
                    logits = output.logits
                    loss = loss_fn(logits, y_batch)

                    loss_accumulated_valid += loss.item() * y_batch.size(0)
                    total_samples_valid += y_batch.size(0)
                    correct_predictions_valid += (
                        (torch.argmax(logits, dim=1) == y_batch).sum().item()
                    )

                accuracy_valid = correct_predictions_valid / total_samples_valid
                log_file.write(
                    f"Validation Loss: {loss_accumulated_valid / total_samples_valid:.4f} Accuracy: {accuracy_valid:.4f}\n"
                )

                accuracy = accuracy_valid
                model_weights = model.state_dict()
                t = datetime.date.today()
                pth_name = f"20{fold + 1}_aug0_time{t}_{epoch + 1}e_acc{100 * accuracy:2.0f}.pth"
                torch.save(
                    model_weights,
                    f"/home/kamyar/PycharmProjects/droneSegmentation/lowAltitude_classification/checkpoints/{pth_name}",
                )

                if accuracy_valid > accuracy:
                    pth_name = f"20{fold + 1}_aug0_time{t}_best_{epoch + 1}e_acc{100 * accuracy:2.0f}.pth"
                    torch.save(
                        model_weights,
                        f"/home/kamyar/PycharmProjects/droneSegmentation/lowAltitude_classification/checkpoints/{pth_name}",
                    )
                    log_file.write(
                        f"Best model weights saved for fold {fold + 1} at epoch {epoch + 1} with accuracy {accuracy:.4f}\n"
                    )

        fold_accuracies.append(accuracy)

        log_file.write(
            f"Average validation accuracy across folds: {np.mean(fold_accuracies):.4f}\n"
        )
