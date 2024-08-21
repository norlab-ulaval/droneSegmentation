import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpmath.libmp import normalize
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


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
    cm = confusion_matrix(annotations_flat, predictions_flat, labels=labels, normalize='true')
    return cm


def plot_confusion_matrix(cm, label_mapping):
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(label_mapping.values()))
    disp.plot(cmap=plt.cm.Blues, xticks_rotation=45)
    plt.title('Confusion Matrix')
    plt.show()


if __name__ == '__main__':
    annotation_folder = '/home/kamyar/Documents/Train-val_Annotated_masks'
    prediction_folder = '/home/kamyar/Documents/Train-val_Annotated_Predictions/41_background_best'
    label_to_id_path = 'lowAltitude_classification/label_to_id.txt'

    annotations = load_images_from_folder(annotation_folder)
    predictions = load_images_from_folder(prediction_folder)
    label_mapping = load_label_mapping(label_to_id_path)

    cm = compute_confusion_matrix(annotations, predictions, label_mapping)
    plot_confusion_matrix(cm, label_mapping)
