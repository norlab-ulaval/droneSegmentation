import os
import cv2
import numpy as np
from sklearn.metrics import confusion_matrix


def load_images_from_folder(folder, color_mode=cv2.IMREAD_GRAYSCALE):
    images = []
    for filename in sorted(os.listdir(folder)):
        img = cv2.imread(os.path.join(folder, filename), color_mode)
        if img is not None:
            images.append(img)
    return images


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
    return np.mean(f1_scores) if f1_scores else 0


def evaluate_segmentation(pred_folder, target_folder, mapping, ignored_classes):
    pred_images = load_images_from_folder(pred_folder)
    target_images = load_images_from_folder(target_folder)

    total_iou = 0
    total_accuracy = 0
    total_f1_score = 0
    all_predictions = []
    all_targets = []

    for pred, target in zip(pred_images, target_images):
        mapped_pred = map_class_values(pred, mapping)
        iou = compute_iou(
            mapped_pred, target, num_classes=32, ignored_classes=ignored_classes
        )
        accuracy = compute_pixel_accuracy(
            mapped_pred, target, ignored_classes=ignored_classes
        )
        f1_score = compute_f1_score(
            mapped_pred, target, num_classes=32, ignored_classes=ignored_classes
        )

        total_iou += iou
        total_accuracy += accuracy
        total_f1_score += f1_score

        all_predictions.append(mapped_pred.flatten())
        all_targets.append(target.flatten())

    avg_iou = total_iou / len(pred_images)
    avg_accuracy = total_accuracy / len(pred_images)
    avg_f1_score = total_f1_score / len(pred_images)

    all_predictions = np.concatenate(all_predictions)
    all_targets = np.concatenate(all_targets)

    return avg_iou, avg_accuracy, avg_f1_score, all_predictions, all_targets


def plot_confusion_matrix(predictions, targets, num_classes, class_mapping):
    cm = confusion_matrix(
        targets, predictions, labels=range(num_classes), normalize="true"
    )
    class_names = [class_mapping[i] for i in range(num_classes)]
    # print(class_names)
    # exit()


IDENTICAL_MAPPING = {i: i for i in range(32)}

MAPPING = {
    0: 0,
    1: 1,
    2: 3,
    3: 4,
    4: 5,
    5: 6,
    6: 7,
    7: 8,
    8: 9,
    9: 10,
    10: 11,
    11: 12,
    12: 13,
    13: 14,
    14: 15,
    15: 16,
    16: 17,
    17: 18,
    18: 19,
    19: 20,
    20: 21,
    21: 22,
    22: 23,
    23: 24,
    24: 25,
    25: 26,
    26: 27,
    27: 28,
    28: 29,
    29: 30,
    30: 31,
}

if __name__ == "__main__":
    import json

    # mapping_path = "lowAltitude_classification/label_to_id.txt"
    # class_mapping = read_class_mapping(mapping_path)

    ignored_classes = {}

    pred_folder = "/home/kamyar/Documents/Test_data_mask2former"
    target_folder = "/home/kamyar/Documents/Test_data_annotation"
    orig_folder = "/home/kamyar/Documents/Test_data"

    avg_iou, avg_accuracy, avg_f1_score, all_predictions, all_targets = (
        evaluate_segmentation(
            pred_folder,
            target_folder,
            IDENTICAL_MAPPING,
            ignored_classes,
        )
    )

    print(f"Average IoU: {avg_iou:.4f}")
    print(f"Average Pixel Accuracy: {avg_accuracy:.4f}")
    print(f"Average F1 Score: {avg_f1_score:.4f}")

    #
    # file_path = '/home/kamyar/PycharmProjects/droneSegmentation/lowAltitude_classification/label_to_id.txt'
    # class_mapping = read_class_mapping(file_path)
    # plot_confusion_matrix(all_predictions, all_targets, num_classes=32, class_mapping=class_mapping)
