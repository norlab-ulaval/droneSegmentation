import argparse
import os
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from albumentations import Compose, Normalize
from albumentations.pytorch import ToTensorV2
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification

# Paths
data_path = Path("/data/Annotated_drone_split")
image_folder = data_path / "Train-val_Annotated"
annot_folder = data_path / "Train-val_Annotated_masks"

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model
model_name = "facebook/dinov2-large-imagenet1k-1-layer"
processor = AutoImageProcessor.from_pretrained(model_name)
model = AutoModelForImageClassification.from_pretrained(
    model_name,
    ignore_mismatched_sizes=True,
)
model = model.to(device)
num_classes = 26
model.classifier = nn.Linear(2048, num_classes).to(device)
mean = processor.image_mean
std = processor.image_std
results_dir = Path("/data/droneSegResults")
model.load_state_dict(
    torch.load(
        results_dir / "5_best/checkpoints/52_Final_time2024-08-15_best_5e_acc94.pth",
        map_location="cpu",
    )
)
model.eval()
transform = Compose([Normalize(mean=mean, std=std), ToTensorV2()])

patch_sizes = [128]
overlaps = [0.85]
batch_size = 1024

# GSD metrics
GSD_FACTOR = 1.5
N_GSD = 4
# GSD_FACTOR=8 and N_GSD = 4
# => SCALES = [1, 1/8, 1/64, 1/512]
SCALES = np.logspace(0, -(N_GSD - 1), num=N_GSD, base=GSD_FACTOR)


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        help="Resizing mode",
        default="resize",
        choices=["resize", "gaussian"],
    )
    return parser.parse_args()


def generate_pseudo_labels(
    image: np.ndarray,
    patch_size: int,
    step_size: int,
    offsets: np.ndarray,
) -> np.ndarray:
    height, width = image.shape[:2]
    padding = patch_size // 8
    padded_width = width + 2 * padding
    padded_height = height + 2 * padding

    transformed = transform(image=image)
    image_tensor = transformed["image"].to(device)

    image_tensor_padded = torch.nn.functional.pad(
        image_tensor,
        (
            padding,
            padding,
            padding,
            padding,
        ),
        "constant",
        0,
    )

    pixel_predictions = np.zeros((height, width, num_classes), dtype=np.longlong)
    patches = []
    coordinates = []

    for x in range(0, padded_width - patch_size + 1, step_size):
        for y in range(0, padded_height - patch_size + 1, step_size):
            patch = image_tensor_padded[:, y : y + patch_size, x : x + patch_size]
            patches.append(patch)
            coordinates.append((x, y))

            if len(patches) == batch_size:
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
                        & (pixel_coords[:, 0] > 0)
                        & (pixel_coords[:, 1] > 0)
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
                & (pixel_coords[:, 0] > 0)
                & (pixel_coords[:, 1] > 0)
            )
            pixel_coords = pixel_coords[valid_mask]
            pixel_predictions[
                pixel_coords[:, 1], pixel_coords[:, 0], predicted_class
            ] += 1

    segmentation_map = np.argmax(pixel_predictions, axis=2)
    return segmentation_map


def main():
    args = parse_arguments()
    gsddat_folder = Path("data") / "gsds" / args.mode / "val"
    for patch_size in patch_sizes:
        x_offsets, y_offsets = np.meshgrid(np.arange(patch_size), np.arange(patch_size))
        offsets = np.stack([x_offsets, y_offsets], axis=-1).reshape(-1, 2)

        for overlap in overlaps:
            step_size = int(patch_size * (1 - overlap))

            patch_overlap = f"p{patch_size:04}-o{overlap * 100:.0f}"
            gsd_po_dir = gsddat_folder / patch_overlap

            # For each GSD:
            for gsd_idx, scale in enumerate(SCALES):
                gsd_metrics = {}

                gsd_dir = gsd_po_dir / f"GSD{gsd_idx}"

                # Create directories for GSD
                gsd_plab_dir = gsd_dir / "pseudolabels"
                gsd_plab_dir.mkdir(parents=True, exist_ok=True)
                gsd_data_dir = gsd_dir / "data"
                gsd_data_dir.mkdir(parents=True, exist_ok=True)
                gsd_annot_dir = gsd_dir / "annotations"
                gsd_annot_dir.mkdir(parents=True, exist_ok=True)

                for image_path in image_folder.glob("*"):
                    if image_path.suffix not in (".jpg", ".JPG", ".png"):
                        continue

                    # Read Image
                    image = np.array(Image.open(image_path))

                    if args.mode.lower() == "resize":
                        scaled_image = cv2.resize(
                            image,
                            None,
                            fx=scale,
                            fy=scale,
                            interpolation=cv2.INTER_NEAREST_EXACT,
                        )
                    else:
                        raise NotImplementedError(
                            f"Not implemented for mode {args.mode}"
                        )
                    gsd_metrics.setdefault("SIZE", []).append(scaled_image.shape[0])

                    # Pseudo labels
                    pslab_start = time.perf_counter()
                    segmentation_map = generate_pseudo_labels(
                        image=scaled_image,
                        patch_size=patch_size,
                        step_size=step_size,
                        offsets=offsets,
                    )
                    pslab_time = time.perf_counter() - pslab_start
                    print(f"[PL] Time taken: {pslab_time:.2f}s")

                    gsd_metrics.setdefault("PSLAB_TIME", []).append(pslab_time)

                    # Open annotation
                    annot_paths = annot_folder.glob(f"{image_path.stem}*")
                    annot_path = next(annot_paths)
                    annot_img = np.array(Image.open(annot_path))

                    if args.mode.lower() == "resize":
                        scaled_annot = cv2.resize(
                            annot_img,
                            None,
                            fx=scale,
                            fy=scale,
                            interpolation=cv2.INTER_NEAREST_EXACT,
                        )
                    else:
                        raise NotImplementedError(
                            f"Not implemented for mode {args.mode}"
                        )

                    # Save images
                    out_fname = image_path.with_suffix(".png").name
                    gsd_dat_path = gsd_data_dir / out_fname
                    gsd_plab_path = gsd_plab_dir / out_fname
                    gsd_annot_path = gsd_annot_dir / out_fname

                    cv2.imwrite(gsd_dat_path, scaled_image)
                    cv2.imwrite(gsd_plab_path, segmentation_map)
                    cv2.imwrite(gsd_annot_path, scaled_annot)

    print("[PL] Processing complete.")


if __name__ == "__main__":
    main()
