# import os
# import cv2
# import numpy as np
# import matplotlib.pyplot as plt
#
# def load_images_from_folder(folder, color_mode=cv2.IMREAD_GRAYSCALE):
#     images = []
#     for filename in sorted(os.listdir(folder)):
#         img = cv2.imread(os.path.join(folder, filename), color_mode)
#         if color_mode == cv2.IMREAD_GRAYSCALE:
#             img = img
#         if img is not None:
#             images.append(img)
#     return images
#
# def compute_iou(prediction, target, num_classes, epsilon=1e-7):
#     iou_scores = []
#     for cls in range(num_classes):
#         intersection = np.logical_and(prediction == cls, target == cls)
#         union = np.logical_or(prediction == cls, target == cls)
#         iou = np.sum(intersection) / (np.sum(union) + epsilon)
#         iou_scores.append(iou)
#     return np.mean(iou_scores)
#
# def compute_pixel_accuracy(prediction, target):
#     correct = np.sum(prediction == target)
#     total = prediction.size
#     accuracy = correct / total
#     return accuracy
#
# def compute_f1_score(prediction, target, num_classes, epsilon=1e-7):
#     f1_scores = []
#     for cls in range(num_classes):
#         tp = np.sum((prediction == cls) & (target == cls))
#         fp = np.sum((prediction == cls) & (target != cls))
#         fn = np.sum((prediction != cls) & (target == cls))
#
#         precision = tp / (tp + fp + epsilon)
#         recall = tp / (tp + fn + epsilon)
#
#         f1 = 2 * precision * recall / (precision + recall + epsilon)
#         f1_scores.append(f1)
#     return np.mean(f1_scores)
#
# def evaluate_segmentation(pred_folder, target_folder):
#     pred_images = load_images_from_folder(pred_folder)
#     target_images = load_images_from_folder(target_folder)
#
#     assert len(pred_images) == len(target_images), "Number of prediction and target images must be the same"
#
#     total_iou = 0
#     total_accuracy = 0
#     total_f1_score = 0
#
#     for pred, target in zip(pred_images, target_images):
#         iou = compute_iou(pred, target)
#         accuracy = compute_pixel_accuracy(pred, target)
#         f1_score = compute_f1_score(pred, target)
#
#         total_iou += iou
#         total_accuracy += accuracy
#         total_f1_score += f1_score
#
#     avg_iou = total_iou / len(pred_images)
#     avg_accuracy = total_accuracy / len(pred_images)
#     avg_f1_score = total_f1_score / len(pred_images)
#
#     return avg_iou, avg_accuracy, avg_f1_score
#
# def visualize_segmentation(orig_folder, pred_folder, target_folder):
#     orig_images = load_images_from_folder(orig_folder, cv2.IMREAD_COLOR)
#     pred_images = load_images_from_folder(pred_folder, cv2.IMREAD_GRAYSCALE)
#     target_images = load_images_from_folder(target_folder, cv2.IMREAD_GRAYSCALE)
#
#     for orig, pred, target in zip(orig_images, pred_images, target_images):
#         fig, axes = plt.subplots(1, 3, figsize=(15, 5))
#
#         axes[0].imshow(cv2.cvtColor(orig, cv2.COLOR_BGR2RGB))
#         axes[0].set_title('Original Image')
#         axes[0].axis('off')
#
#         axes[1].imshow(pred, cmap='gray')
#         axes[1].set_title('Prediction')
#         axes[1].axis('off')
#
#         axes[2].imshow(target, cmap='gray')
#         axes[2].set_title('Annotation')
#         axes[2].axis('off')
#
#         plt.show()
#
# pred_folder = '/home/kamyar/Documents/Test_data_pred'
# target_folder = '/home/kamyar/Documents/Test_data_annotation'
# orig_folder = '/home/kamyar/Documents/Test_data'
#
# avg_iou, avg_accuracy, avg_f1_score = evaluate_segmentation(pred_folder, target_folder)
#
# print(f'Average IoU: {avg_iou:.4f}')
# print(f'Average Pixel Accuracy: {avg_accuracy:.4f}')
# print(f'Average F1 Score: {avg_f1_score:.4f}')
#
# visualize_segmentation(orig_folder, pred_folder, target_folder)





import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

def load_images_from_folder(folder, color_mode=cv2.IMREAD_GRAYSCALE):
    images = []
    for filename in sorted(os.listdir(folder)):
        img = cv2.imread(os.path.join(folder, filename), color_mode)
        if color_mode == cv2.IMREAD_GRAYSCALE:
            img = img
        if img is not None:
            images.append(img)
    return images

def map_class_values(image, mapping):
    new_image = np.copy(image)
    for old_value, new_value in mapping.items():
        new_image[image == old_value] = new_value
    return new_image

import numpy as np

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

        f1 = 2 * precision * recall / (precision + recall + epsilon) if (precision + recall) != 0 else 0
        f1_scores.append(f1)
    return np.mean(f1_scores) if f1_scores else 0


def evaluate_segmentation(pred_folder, target_folder, mapping, ignored_classes):
    pred_images = load_images_from_folder(pred_folder)
    target_images = load_images_from_folder(target_folder)

    # assert len(pred_images) == len(target_images), "Number of prediction and target images must be the same"

    total_iou = 0
    total_accuracy = 0
    total_f1_score = 0

    for pred, target in zip(pred_images, target_images):
        mapped_pred = map_class_values(pred, mapping)
        iou = compute_iou(mapped_pred, target, num_classes=32, ignored_classes=ignored_classes)
        accuracy = compute_pixel_accuracy(mapped_pred, target, ignored_classes=ignored_classes)
        f1_score = compute_f1_score(mapped_pred, target, num_classes=32, ignored_classes=ignored_classes)

        total_iou += iou
        total_accuracy += accuracy
        total_f1_score += f1_score

    avg_iou = total_iou / len(pred_images)
    avg_accuracy = total_accuracy / len(pred_images)
    avg_f1_score = total_f1_score / len(pred_images)

    return avg_iou, avg_accuracy, avg_f1_score

def visualize_segmentation(orig_folder, pred_folder, target_folder, mapping):
    orig_images = load_images_from_folder(orig_folder, cv2.IMREAD_COLOR)
    pred_images = load_images_from_folder(pred_folder, cv2.IMREAD_GRAYSCALE)
    target_images = load_images_from_folder(target_folder, cv2.IMREAD_GRAYSCALE)

    for orig, pred, target in zip(orig_images, pred_images, target_images):
        mapped_pred = map_class_values(pred, mapping)

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        axes[0].imshow(cv2.cvtColor(orig, cv2.COLOR_BGR2RGB))
        axes[0].set_title('Original Image')
        axes[0].axis('off')

        axes[1].imshow(mapped_pred, cmap='gray')
        axes[1].set_title('Prediction')
        axes[1].axis('off')

        axes[2].imshow(target, cmap='gray')
        axes[2].set_title('Annotation')
        axes[2].axis('off')

        plt.show()

identical_mapping = {i: i for i in range(32)}

mapping = {
    0: 0,  # Alders
    1: 1,  # American Mountain-ash
    2: 3,  # Black spruce
    3: 4,  # Bog Labrador Tea
    4: 6,  # Canada Yew (Taxus canadensis)
    5: 7,  # Common Haircap Moss
    6: 8,  # Dead_trees
    7: 9,  # Feather Mosses
    8: 10,  # Ferns (Fougères)
    9: 11,  # Fir
    10: 12,  # Fire Cherry (Cerisier de pensylvanie)
    11: 13,  # Jack pine
    12: 14,  # Larch
    13: 15,  # Leatherleaf (Cassandre)
    14: 16,  # Mountain Maple (Érable à épis)
    15: 17,  # Paper birch
    16: 18,  # Pixie Cup Lichens (Cladonia)
    17: 19,  # Red Maple
    18: 20,  # Red raspberry (Framboisier)
    19: 21,  # Sedges (Carex)
    20: 22,  # Serviceberries (amélanchier)
    21: 23,  # Sheep Laurel (Kalmia)
    22: 24,  # Sphagnum Mosses
    23: 25,  # Trembling aspen
    24: 26,  # Viburnums
    25: 27,  # Willowherbs (Épilobe)
    26: 28,  # Willows
    27: 29,  # blueberry
    28: 30,  # wood
    29: 31,  # yellow birch
}

ignored_classes = {}

pred_folder = '/home/kamyar/Documents/Test_data_pred/patch_128_overlap_85'
target_folder = '/home/kamyar/Documents/Test_data_annotation'
orig_folder = '/home/kamyar/Documents/Test_data'

avg_iou, avg_accuracy, avg_f1_score = evaluate_segmentation(pred_folder, target_folder, identical_mapping, ignored_classes)

print(f'Average IoU: {avg_iou:.4f}')
print(f'Average Pixel Accuracy: {avg_accuracy:.4f}')
print(f'Average F1 Score: {avg_f1_score:.4f}')

visualize_segmentation(orig_folder, pred_folder, target_folder, identical_mapping)

