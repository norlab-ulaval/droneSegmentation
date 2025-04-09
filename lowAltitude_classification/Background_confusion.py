import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpmath.libmp import normalize
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import seaborn as sns


def load_images_from_folder(folder, color_mode=cv2.IMREAD_GRAYSCALE):
    images = []
    for filename in sorted(os.listdir(folder)):
        img = cv2.imread(os.path.join(folder, filename), color_mode)
        if img is not None:
            images.append(img)
    return images


def load_label_mapping(label_to_id_path):
    with open(label_to_id_path, 'r') as file:
        lines = file.readlines()
    label_mapping = {}
    for line in lines:
        label, idx = line.split(':')
        label_mapping[int(idx)] = label.strip()
    return label_mapping


def compute_confusion_matrix(annotations, predictions, label_mapping):
    annotations_flat = np.concatenate([img.flatten() for img in annotations])
    predictions_flat = np.concatenate([img.flatten() for img in predictions])

    labels = list(label_mapping.keys())
    cm = confusion_matrix(annotations_flat, predictions_flat, labels=labels)
    return cm


def plot_confusion_matrix(cm, label_mapping):
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(label_mapping.values()))
    disp.plot(cmap=plt.cm.Blues, xticks_rotation=90)
    plt.title('Confusion Matrix')
    plt.show()



def calculate_metrics(cm, class_idx):
    # True Positives (TP)
    tp = cm[class_idx, class_idx]

    # False Positives (FP)
    fp = np.sum(cm[:, class_idx]) - tp

    # False Negatives (FN)
    fn = np.sum(cm[class_idx, :]) - tp

    # True Negatives (TN)
    tn = np.sum(cm) - (tp + fp + fn)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return tp, fp, tn, fn, precision, recall, f1_score


if __name__ == '__main__':
    annotation_folder = 'data/Train-val_Annotated_masks'
    prediction_folder = 'data/Train-val_Annotated_Predictions/12_filtered_1e'
    label_to_id_path = 'lowAltitude_classification/label_to_id.txt'

    annotations = load_images_from_folder(annotation_folder)
    predictions = load_images_from_folder(prediction_folder)
    label_mapping = load_label_mapping(label_to_id_path)

    cm = compute_confusion_matrix(annotations, predictions, label_mapping)
    class_idx = 1
    tp, fp, tn, fn, precision, recall, f1_score = calculate_metrics(cm, class_idx)

    print(f'TP: {tp}, FP: {fp}, TN: {tn}, FN: {fn}')
    print(f'Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1_score:.4f}')

    # confusion_matrix_background = np.array([[tn, fp],
    #                              [fn, tp]])
    #
    # plt.figure(figsize=(6, 4))
    # sns.heatmap(confusion_matrix_background, annot=True, cmap='Blues',
    #             xticklabels=['True Negative', 'False Positive'],
    #             yticklabels=['False Negative', 'True Positive'])
    #
    # plt.title('Background Predictions Heatmap')
    # plt.xlabel('Predicted Labels')
    # plt.ylabel('True Labels')
    # plt.show()


    # plot_confusion_matrix(cm, label_mapping)
