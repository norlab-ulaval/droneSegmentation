import os

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


def load_images_from_folder(folder):
    images = {}
    for filename in os.listdir(folder):
        img = Image.open(os.path.join(folder, filename))
        if img is not None:
            images[filename] = np.array(img)
    return images


def load_label_mapping(label_to_id_path):
    with open(label_to_id_path, 'r') as file:
        lines = file.readlines()
    label_mapping = {}
    for line in lines:
        label, idx = line.split(':')
        label_mapping[int(idx)] = label.strip()
    return label_mapping


if __name__ == '__main__':
    annotation_folder = '/home/kamyar/Documents/Train-val_Annotated_masks'
    prediction_folder = '/home/kamyar/Documents/Train-val_Annotated_Predictions/41_background_best'
    label_to_id_path = 'lowAltitude_classification/label_to_id.txt'

    annotations = load_images_from_folder(annotation_folder)
    predictions = load_images_from_folder(prediction_folder)
    label_mapping = load_label_mapping(label_to_id_path)

    all_annotations = np.concatenate([annotations[key].flatten() for key in annotations.keys()])
    all_predictions = np.concatenate([predictions[key].flatten() for key in predictions.keys()])

    unique_labels = np.unique(np.concatenate([all_annotations, all_predictions]))

    filtered_label_mapping = {i: label_mapping[i] for i in unique_labels if i in label_mapping}

    cm = confusion_matrix(all_annotations, all_predictions, labels=list(filtered_label_mapping.keys()),
                          normalize='true')
    class_labels = [filtered_label_mapping[i] for i in filtered_label_mapping.keys()]

    # correct_pixels = np.diag(cm)
    # total_pixels = np.sum(cm, axis=1)
    #
    # pixel_accuracy = np.mean(correct_pixels / total_pixels)

    fig, ax = plt.subplots(figsize=(12, 12))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_labels)
    disp.plot(ax=ax, cmap=plt.cm.Blues)
    plt.xticks(rotation=90)
    plt.title("Normalized Confusion Matrix with Class Labels")
    plt.show()

    # print(f"Pixel Accuracy: {pixel_accuracy:.4f}")
