import os
import time

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from albumentations import Normalize, Compose
from albumentations.pytorch import ToTensorV2
from transformers import AutoImageProcessor, AutoModelForImageClassification
from pathlib import Path
import csv

# SPLIT = os.environ.get("SPLIT", None)
# if SPLIT is None:
#     raise ValueError("SPLIT environment variable must be set: 'group_1'  'group_2'  'group_3'  'group_4_1'  'group_4_2'  'group_5_1'  'group_5_2'")

# Paths
# results_dir = Path("data/droneseg")
# weight_file_path = Path("checkpoints/best-classifier/53_Final_time2024-12-05_best_5e_acc95.pth")
# image_folder = Path(f"data/Unlabeled_Drone_Dataset/Unlabeled_Drone_Dataset_143k_Patched_split_7_Subsets/{SPLIT}/")
# output_dir = results_dir / 'results/PseudoLabel' / image_folder.name

results_dir = Path("results/PL_Step_Size_Experiment")
weight_file_path = Path(
    "checkpoints/best-classifier/53_Final_time2024-12-05_best_5e_acc95.pth"
)
image_folder_test = Path("data/Test_Annotated")
image_folder_val = Path("data/Train-val_Annotated")
image_folders = [image_folder_test]
############

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_name = "facebook/dinov2-large-imagenet1k-1-layer"
processor = AutoImageProcessor.from_pretrained(model_name)
model = AutoModelForImageClassification.from_pretrained(
    model_name, ignore_mismatched_sizes=True
)
model = model.to(device)
num_classes = 24
model.classifier = nn.Linear(2048, num_classes).to(device)
mean = processor.image_mean
std = processor.image_std

transform = Compose([Normalize(mean=mean, std=std), ToTensorV2()])

patch_sizes = [256]
step_sizes = [32, 38, 64, 128, 256]

model.load_state_dict(torch.load(weight_file_path))
model.eval()


###################################################################### only to see the avg number of votes
with open(
    "lowAltitude_classification/PL_Step_Size_Experiment/PL_num_votes_per_step_size.csv",
    mode="w",
    newline="",
) as file:
    writer = csv.writer(file)
    writer.writerow(["Step Size", "Average Votings per Image"])
    ###############################################################################################

    for image_folder in image_folders:
        if image_folder.name == "Test_Annotated":
            Split = "test"
        else:
            Split = "val"

        for patch_size in patch_sizes:
            x_offsets, y_offsets = np.meshgrid(
                np.arange(patch_size), np.arange(patch_size)
            )
            offsets = np.stack([x_offsets, y_offsets], axis=-1).reshape(-1, 2)

            for step_size in step_sizes:
                output_dir = results_dir / f"{Split}/{step_size}"
                output_dir.mkdir(parents=True, exist_ok=True)

                padding = patch_size
                batch_size = 256

                for image_file in os.listdir(image_folder):
                    if image_file.endswith((".jpg", ".JPG", ".png")):
                        begin_time = time.perf_counter()
                        image_path = os.path.join(image_folder, image_file)
                        image = Image.open(image_path)
                        image_np = np.array(image)
                        transformed = transform(image=image_np)
                        image_tensor = transformed["image"].to(device)

                        image_tensor_padded = torch.nn.functional.pad(
                            image_tensor,
                            (padding, padding, padding, padding),
                            "constant",
                            0,
                        )

                        width, height = image.size
                        padded_width = width + 2 * padding
                        padded_height = height + 2 * padding

                        pixel_predictions = np.zeros(
                            (height, width, num_classes), dtype=np.longlong
                        )
                        patches = []
                        coordinates = []

                        for x in range(0, padded_width - patch_size + 1, step_size):
                            for y in range(
                                0, padded_height - patch_size + 1, step_size
                            ):
                                patch = image_tensor_padded[
                                    :, y : y + patch_size, x : x + patch_size
                                ]
                                patches.append(patch)
                                coordinates.append((x, y))

                                if len(patches) == batch_size:
                                    patches_tensor = torch.stack(patches).to(device)

                                    with torch.no_grad(), torch.cuda.amp.autocast():
                                        outputs = model(patches_tensor)

                                    predicted_classes = torch.argmax(
                                        outputs.logits, dim=1
                                    )

                                    for patch_idx, (x, y) in enumerate(coordinates):
                                        predicted_class = predicted_classes[patch_idx]

                                        pixel_coords = (
                                            offsets + np.array([x, y]) - padding
                                        )
                                        valid_mask = (
                                            (pixel_coords[:, 0] < width)
                                            & (pixel_coords[:, 1] < height)
                                            & (pixel_coords[:, 0] >= 0)
                                            & (pixel_coords[:, 1] >= 0)
                                        )
                                        pixel_coords = pixel_coords[valid_mask]
                                        pixel_predictions[
                                            pixel_coords[:, 1],
                                            pixel_coords[:, 0],
                                            predicted_class,
                                        ] += 1

                                    patches = []
                                    coordinates = []
                        if patches:
                            patches_tensor = torch.stack(patches).to(device)

                            with torch.no_grad(), torch.cuda.amp.autocast():
                                outputs = model(patches_tensor)

                            predicted_classes = torch.argmax(outputs.logits, dim=1)

                            for patch_idx, (x, y) in enumerate(coordinates):
                                predicted_class = predicted_classes[patch_idx]

                                pixel_coords = offsets + np.array([x, y]) - padding
                                valid_mask = (
                                    (pixel_coords[:, 0] < width)
                                    & (pixel_coords[:, 1] < height)
                                    & (pixel_coords[:, 0] >= 0)
                                    & (pixel_coords[:, 1] >= 0)
                                )
                                pixel_coords = pixel_coords[valid_mask]
                                pixel_predictions[
                                    pixel_coords[:, 1],
                                    pixel_coords[:, 0],
                                    predicted_class,
                                ] += 1

                        segmentation_map = np.argmax(pixel_predictions, axis=2)

                        ################################################### only to see the avg number of votes
                        votings_per_pixel = pixel_predictions.sum(axis=2)
                        avg_votings_image = (
                            votings_per_pixel.mean()
                            if len(votings_per_pixel) > 0
                            else 0
                        )
                        writer.writerow([step_size, avg_votings_image])
                        break
                        #####################################################################

                        output_filename = Path(image_path).with_suffix(".png").name
                        cv2.imwrite(str(output_dir / output_filename), segmentation_map)
                        print(f"Time taken: {time.perf_counter() - begin_time:.2f}s")

print("Processing complete.")
