import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def load_images_from_folder(folder, color_mode=cv2.IMREAD_GRAYSCALE):
    images = []
    for filename in sorted(os.listdir(folder)):
        img = cv2.imread(os.path.join(folder, filename), color_mode)
        if img is not None:
            images.append(img)
    return images

def read_class_mapping(file_path):
    class_mapping = {}
    with open(file_path, "r") as file:
        for line in file:
            name, idx = line.strip().split(": ")
            class_mapping[int(idx)] = name
    return class_mapping

def map_class_values(image, mapping):
    new_image = np.copy(image)
    for old_value, new_value in mapping.items():
        new_image[image == old_value] = new_value
    return new_image

def compute_iou(prediction, target, num_classes, ignored_classes, epsilon=1e-7):
    iou_scores = []
    for cls in range(num_classes):
        if cls in ignored_classes or cls not in target:
            continue
        intersection = np.logical_and(prediction == cls, target == cls)
        union = np.logical_or(prediction == cls, target == cls)
        iou = np.sum(intersection) / (np.sum(union) + epsilon)
        iou_scores.append(iou)
    return np.mean(iou_scores) if iou_scores else 0

def compute_pixel_accuracy(prediction, target, ignored_classes):
    mask = np.isin(target, ignored_classes, invert=True)
    correct = np.sum((prediction == target) & mask)
    total = np.sum(mask)
    accuracy = correct / total if total != 0 else 0
    return accuracy

def compute_f1_score(prediction, target, num_classes, ignored_classes, epsilon=1e-7):
    f1_scores = []
    for cls in range(num_classes):
        if cls in ignored_classes or cls not in target:
            continue
        tp = np.sum((prediction == cls) & (target == cls))
        fp = np.sum((prediction == cls) & (target != cls))
        fn = np.sum((prediction != cls) & (target == cls))

        precision = tp / (tp + fp + epsilon) if (tp + fp) != 0 else 0
        recall = tp / (tp + fn + epsilon) if (tp + fn) != 0 else 0

        f1 = (
            2 * precision * recall / (precision + recall + epsilon)
            if (precision + recall) != 0
            else 0
        )
        f1_scores.append(f1)
    return np.mean(f1_scores) if f1_scores else 0, precision, recall

def evaluate_segmentation(pred_folder, target_folder, ignored_classes):
    pred_images = load_images_from_folder(pred_folder)
    target_images = load_images_from_folder(target_folder)

    total_iou = 0
    total_accuracy = 0
    total_f1_score = 0
    total_precision = 0
    total_recall = 0

    for pred, target in zip(pred_images, target_images):
        iou = compute_iou(
            pred, target, num_classes=26, ignored_classes=ignored_classes
        )
        accuracy = compute_pixel_accuracy(
            pred, target, ignored_classes=ignored_classes
        )
        f1_score, precision, recall = compute_f1_score(
            pred, target, num_classes=26, ignored_classes=ignored_classes
        )

        total_iou += iou
        total_accuracy += accuracy
        total_f1_score += f1_score
        total_precision += precision
        total_recall += recall

    avg_iou = total_iou / len(pred_images)
    avg_accuracy = total_accuracy / len(pred_images)
    avg_f1_score = total_f1_score / len(pred_images)
    avg_precision = total_precision / len(pred_images)
    avg_recall = total_recall / len(pred_images)

    return avg_iou, avg_accuracy, avg_f1_score, avg_precision, avg_recall

if __name__ == "__main__":
    ignored_classes = {1}

    parent_preds_folder = "/home/kamyar/Documents/Train-val_Annotated_Predictions/CENTER/52_Final_5e"
    target_folder = "/home/kamyar/Documents/Train-val_Annotated_masks"

    results = []

    for idx, pred_folder_name in enumerate(os.listdir(parent_preds_folder)):
        pred_folder_path = os.path.join(parent_preds_folder, pred_folder_name)
        if os.path.isdir(pred_folder_path):
            avg_iou, avg_accuracy, avg_f1_score, avg_precision, avg_recall = evaluate_segmentation(
                pred_folder_path, target_folder, ignored_classes
            )
            print(f'mIoU: {avg_iou}, pAccuracy: {avg_accuracy}, f1_score: {avg_f1_score}')

    #         results.append({
    #             "Experiment": f"experiment {idx}",
    #             "Patch Size": pred_folder_name.split('_')[0],
    #             "Overlap": pred_folder_name.split('_')[1],
    #             "precision": f'{avg_precision:.4f}',
    #             "recall": f'{avg_recall:.4f}',
    #             "mIoU": f'{avg_iou:.4f}',
    #             "pAcc": f'{avg_accuracy:.4f}',
    #             "F1": f'{avg_f1_score:.4f}',
    #         })
    #
    # df = pd.DataFrame(results)
    # df = df.sort_values(by=["Patch Size", "Overlap"])
    #
    # df.to_csv("lowAltitude_classification/Result_Val_DifferentPatcheSize/Result_Val_DifferentPatcheSize.csv",
    #           index=False)
