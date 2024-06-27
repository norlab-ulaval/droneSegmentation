import os
import cv2
import numpy as np


def load_images_from_folder(folder):
    images = []
    for filename in sorted(os.listdir(folder)):
        img = cv2.imread(os.path.join(folder, filename), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            images.append(img)
    return images


def compute_iou(prediction, target):
    intersection = np.logical_and(prediction, target)
    union = np.logical_or(prediction, target)
    iou_score = np.sum(intersection) / np.sum(union)
    return iou_score


def compute_pixel_accuracy(prediction, target):
    correct = np.sum(prediction == target)
    total = prediction.size
    accuracy = correct / total
    return accuracy


def evaluate_segmentation(pred_folder, target_folder):
    pred_images = load_images_from_folder(pred_folder)
    target_images = load_images_from_folder(target_folder)

    assert len(pred_images) == len(target_images), "Number of prediction and target images must be the same"

    total_iou = 0
    total_accuracy = 0

    for pred, target in zip(pred_images, target_images):
        iou = compute_iou(pred, target)
        accuracy = compute_pixel_accuracy(pred, target)

        total_iou += iou
        total_accuracy += accuracy

    avg_iou = total_iou / len(pred_images)
    avg_accuracy = total_accuracy / len(pred_images)

    return avg_iou, avg_accuracy


# Folders containing the prediction and target images
pred_folder = 'path/to/predictions'
target_folder = 'path/to/targets'

avg_iou, avg_accuracy = evaluate_segmentation(pred_folder, target_folder)

print(f'Average IoU: {avg_iou:.4f}')
print(f'Average Pixel Accuracy: {avg_accuracy:.4f}')
