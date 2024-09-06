from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm


def papermode(plt, size: int | None = None, has_latex: bool = True):
    if has_latex:
        plt.rc("font", family="serif", serif="Times")
        plt.rc("text", usetex=True)
    if size is not None:
        plt.rc("xtick", labelsize=size)
        plt.rc("ytick", labelsize=size)
        plt.rc("axes", labelsize=size)
        plt.rc("figure", labelsize=size)
        plt.rc("legend", fontsize=size)


def load_image(image_path):
    return np.array(Image.open(image_path))


def compute_metrics(pred_folder: Path | str, annot_folder: Path | str):
    all_preds = []
    all_annots = []

    pred_dir = Path(pred_folder)
    annot_dir = Path(annot_folder)

    win_name = pred_dir.parents[1].stem
    gsd_name = pred_dir.parents[0].stem

    pred_paths = [p for p in pred_dir.glob("**/*") if p.is_file()]
    for pred_path in tqdm(pred_paths, desc=f"{win_name}-{gsd_name}"):
        pred_fname = pred_path.name
        annot_path = annot_dir / pred_fname

        pred_image = load_image(pred_path)
        annot_image = load_image(annot_path)

        assert pred_image.shape == annot_image.shape, f"Shape mismatch: {pred_fname}"

        all_preds.extend(pred_image.flatten())
        all_annots.extend(annot_image.flatten())

    all_preds = np.array(all_preds)
    all_annots = np.array(all_annots)

    overall_f1_score = f1_score(all_annots, all_preds, average="macro")
    pixel_accuracy = accuracy_score(all_annots, all_preds)

    return overall_f1_score, pixel_accuracy
