import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

def load_images_from_folder(folder, color_mode=cv2.IMREAD_GRAYSCALE):
    images = []
    for filename in sorted(os.listdir(folder)):
        img = cv2.imread(os.path.join(folder, filename), color_mode)
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

def compute_f1_score(prediction, target):
    tp = np.sum((prediction == 1) & (target == 1))
    fp = np.sum((prediction == 1) & (target == 0))
    fn = np.sum((prediction == 0) & (target == 1))

    precision = tp / (tp + fp) if (tp + fp) != 0 else 0
    recall = tp / (tp + fn) if (tp + fn) != 0 else 0

    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) != 0 else 0
    return f1_score

def evaluate_segmentation(pred_folder, target_folder):
    pred_images = load_images_from_folder(pred_folder)
    target_images = load_images_from_folder(target_folder)

    assert len(pred_images) == len(target_images), "Number of prediction and target images must be the same"

    total_iou = 0
    total_accuracy = 0
    total_f1_score = 0

    for pred, target in zip(pred_images, target_images):
        iou = compute_iou(pred, target)
        accuracy = compute_pixel_accuracy(pred, target)
        f1_score = compute_f1_score(pred, target)

        total_iou += iou
        total_accuracy += accuracy
        total_f1_score += f1_score

    avg_iou = total_iou / len(pred_images)
    avg_accuracy = total_accuracy / len(pred_images)
    avg_f1_score = total_f1_score / len(pred_images)

    return avg_iou, avg_accuracy, avg_f1_score

def visualize_segmentation(orig_folder, pred_folder, target_folder):
    orig_images = load_images_from_folder(orig_folder, cv2.IMREAD_COLOR)
    pred_images = load_images_from_folder(pred_folder, cv2.IMREAD_GRAYSCALE)
    target_images = load_images_from_folder(target_folder, cv2.IMREAD_GRAYSCALE)

    for orig, pred, target in zip(orig_images, pred_images, target_images):
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        axes[0].imshow(cv2.cvtColor(orig, cv2.COLOR_BGR2RGB))
        axes[0].set_title('Original Image')
        axes[0].axis('off')

        axes[1].imshow(pred, cmap='gray')
        axes[1].set_title('Prediction')
        axes[1].axis('off')

        axes[2].imshow(target, cmap='gray')
        axes[2].set_title('Annotation')
        axes[2].axis('off')

        plt.show()

pred_folder = '/home/kamyar/Documents/Test_data_pred'
target_folder = '/home/kamyar/Documents/Test_data_annotation'
orig_folder = '/home/kamyar/Documents/Test_data'

avg_iou, avg_accuracy, avg_f1_score = evaluate_segmentation(pred_folder, target_folder)

print(f'Average IoU: {avg_iou:.4f}')
print(f'Average Pixel Accuracy: {avg_accuracy:.4f}')
print(f'Average F1 Score: {avg_f1_score:.4f}')

visualize_segmentation(orig_folder, pred_folder, target_folder)
