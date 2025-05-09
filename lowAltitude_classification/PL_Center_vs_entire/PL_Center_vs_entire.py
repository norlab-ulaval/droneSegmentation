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
overlaps = [0.85]

results_dir = Path("results/PL_Center_vs_entire")
weight_file_path = Path(
    "checkpoints/best-classifier/53_Final_time2024-12-05_best_5e_acc95.pth"
)
image_folder_test = Path("data/Test_Annotated")
image_folder_val = Path("data/Train-val_Annotated")
image_folders = [image_folder_test, image_folder_val]

model.load_state_dict(torch.load(weight_file_path))
model.eval()

for image_folder in image_folders:
    if image_folder.name == "Test_Annotated":
        Split = "test"
    else:
        Split = "val"

    batch_size = 256
    for patch_size in patch_sizes:
        padding = patch_size
        for central_window_size in [patch_size // 2, patch_size]:
            output_dir = results_dir / f"{Split}/{central_window_size}"
            output_dir.mkdir(parents=True, exist_ok=True)

            central_offset = (patch_size - central_window_size) // 2
            x_offsets, y_offsets = np.meshgrid(
                np.arange(central_offset, central_offset + central_window_size),
                np.arange(central_offset, central_offset + central_window_size),
            )
            offsets = np.stack([x_offsets, y_offsets], axis=-1).reshape(-1, 2)

            for overlap in overlaps:
                step_size = int(patch_size * (1 - overlap))
                for image_file in os.listdir(image_folder):
                    if image_file.endswith((".jpg", ".JPG", ".png")):
                        begin_time = time.perf_counter()
                        image_path = os.path.join(image_folder, image_file)
                        image = Image.open(image_path)
                        # image = image.transform() # tx=0, ty=padding
                        image_np = np.array(image)
                        transformed = transform(image=image_np)
                        image_tensor = transformed["image"].to(device)

                        image_tensor_padded = torch.nn.functional.pad(
                            image_tensor,
                            (padding, padding, padding, padding),
                            "constant",
                            value=0,
                        )

                        width, height = image.size
                        padded_width = width + 2 * padding
                        padded_height = height + 2 * padding

                        pixel_predictions = np.zeros(
                            (height, width, num_classes), dtype=np.longlong
                        )
                        patches = []
                        coordinates = []

                        for x in range(0, padded_width - patch_size, step_size):
                            for y in range(0, padded_height - patch_size, step_size):
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
                        output_filename = Path(image_path).with_suffix(".png").name
                        cv2.imwrite(str(output_dir / output_filename), segmentation_map)
                        print(f"Time taken: {time.perf_counter() - begin_time:.2f}s")


print("Processing complete.")
