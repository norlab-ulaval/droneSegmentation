from torchsampler import ImbalancedDatasetSampler
import datetime
import json
import logging
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
    Perspective,
    RandomResizedCrop,
    ShiftScaleRotate,
    SmallestMaxSize,
    OneOf,
    MotionBlur,
    MedianBlur,
    OpticalDistortion,
    GridDistortion,
    Defocus,
    RandomFog
)
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import StratifiedKFold
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import ImageFolder
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm
from transformers import AutoImageProcessor, AutoModelForImageClassification
import utils as u

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# previous_checkpoint_path = "/app/lowAltitude_classification/checkpoints/55_Final_time2024-08-16_best_3e_acc94.pth"

data_folder = '/home/kamyar/Documents/iNat_Classifier_filtered'
lac_dir = Path("lowAltitude_classification")
output_file_path = lac_dir / "label_to_id.txt"
checkpoint_dir = lac_dir / "checkpoints"
checkpoint_dir.mkdir(parents=True, exist_ok=True)

log_file_path = lac_dir / "Final_Classifier_VIT_NotDIno/log_vit.txt"
u.setup_logging("vit", log_file_path)
logger = logging.getLogger("vit")

dataset = ImageFolder(root=data_folder)
label_to_id = dataset.class_to_idx

with open(output_file_path, "w") as file:
    for label, idx in label_to_id.items():
        file.write(f"{label}: {idx}\n")

json_path = output_file_path.with_suffix(".json")
with json_path.open(mode="w") as f:
    json.dump(label_to_id, f, indent=2, sort_keys=True)

model_name = "google/vit-large-patch16-224"
processor = AutoImageProcessor.from_pretrained(model_name)
model = AutoModelForImageClassification.from_pretrained(
    model_name, ignore_mismatched_sizes=True
)

mean = processor.image_mean
std = processor.image_std

train_transform = Compose(
    [
        SmallestMaxSize(max_size=224),
        RandomResizedCrop(
            height=224,
            width=224,
            scale=(0.4, 1.0),
            ratio=(0.75, 1.3333),
        ),
        HorizontalFlip(p=0.5),
        Perspective(scale=(0.3, 0.6), p=0.5),
        ShiftScaleRotate(
            shift_limit=0.1,
            scale_limit=0.1,
            rotate_limit=45,
            p=0.5,
        ),
        OneOf([
            MotionBlur(p=.2),
            MedianBlur(blur_limit=3, p=0.1),
        ], p=0.3),

        Defocus(
            radius=(3, 5),
            alias_blur=(0.1, 0.2),
            p=0.5,
        ),
        RandomFog(),
        ColorJitter(
            brightness=(0.3, 0.5),
            contrast=(0.3, 0.5),
            saturation=(0.3, 0.5),
            hue=0.2,
            p=0.5,
        ),
        OneOf([
            OpticalDistortion(p=0.3),
            GridDistortion(p=.1),
        ], p=0.2),
        Blur(blur_limit=(3, 7), p=0.5),
        Normalize(mean=mean, std=std),
        ToTensorV2(),
    ]
)

test_transform = Compose(
    [
        SmallestMaxSize(max_size=224),
        CenterCrop(height=224, width=224),
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

# For each fold
for fold, (train_idx, val_idx) in enumerate(kf.split(dataset, dataset.targets)):
    logger.debug(f"Fold {fold + 1}/{kf.get_n_splits()}")

    train_subset = Subset(dataset, train_idx)
    val_subset = Subset(dataset, val_idx)

    ##### special for ImbalancedDatasetSampler
    train_labels = [dataset.targets[i] for i in train_idx]
    ######

    # average of number of classes: 11360 -> * 26 = 295,360
    train_loader = DataLoader(
        train_subset,
        sampler=ImbalancedDatasetSampler(
            train_subset,
            labels=train_labels,
            num_samples=295360,
        ),
        batch_size=16,
        num_workers=16,
    )
    val_loader = DataLoader(val_subset, batch_size=16, shuffle=False, num_workers=16)

    model = AutoModelForImageClassification.from_pretrained(
        model_name, ignore_mismatched_sizes=True
    )
    model.classifier = nn.Linear(1024, len(label_to_id)).to(device)
    model = model.to(device)

    # if previous_checkpoint_path:
    #     model.load_state_dict(torch.load(previous_checkpoint_path))

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

        with logging_redirect_tqdm():
            for x_batch, y_batch in tqdm(
                train_loader,
                desc=f"Epoch {epoch} - Train",
            ):
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

        logger.debug(
            f"Epoch {epoch + 1} - Training Loss: {loss_accumulated_train / total_samples_train:.4f} Accuracy: {correct_predictions_train / total_samples_train:.4f}"
        )
        scheduler.step()

        # Validation
        model.eval()
        with torch.no_grad():
            loss_accumulated_valid = 0.0
            total_samples_valid = 0
            correct_predictions_valid = 0

            with logging_redirect_tqdm():
                for x_batch, y_batch in tqdm(
                    val_loader,
                    desc=f"Epoch {epoch} - Val",
                ):
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
            logger.debug(
                f"Validation Loss: {loss_accumulated_valid / total_samples_valid:.4f} Accuracy: {accuracy_valid:.4f}"
            )

            accuracy = accuracy_valid
            model_weights = model.state_dict()
            t = datetime.date.today()
            pth_name = f"5{fold + 1}_vit-final_time{t}_{epoch + 1}e_acc{100 * accuracy:2.0f}.pth"
            torch.save(
                model_weights,
                checkpoint_dir / pth_name,
            )

            if accuracy_valid > best_accuracy:
                best_accuracy = accuracy_valid
                best_epoch = epoch
                best_model_weights = model_weights
                pth_name = f"5{fold + 1}_vit-final_time{t}_best_{epoch + 1}e_acc{100 * accuracy:2.0f}.pth"
                torch.save(
                    model_weights,
                    checkpoint_dir / pth_name,
                )
                logger.debug(
                    f"Best model weights saved for fold {fold + 1} at epoch {epoch + 1} with accuracy {accuracy:.4f}"
                )

    fold_accuracies.append(accuracy)

logger.debug(
    f"Average validation accuracy across folds: {np.mean(fold_accuracies):.4f}"
)