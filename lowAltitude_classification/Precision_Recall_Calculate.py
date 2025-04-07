import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score
from PIL import Image
import numpy as np

annotations_folder = ''
predictions_folder = ''

def load_image(image_path):
    return np.array(Image.open(image_path))

all_preds = []
all_annots = []

for filename in os.listdir(predictions_folder):
    pred_path = os.path.join(predictions_folder, filename)
    base_name = os.path.splitext(filename)[0]
    annotation_filename = f'{base_name}-label-ground-truth-semantic.png'
    annot_path = os.path.join(annotations_folder, annotation_filename)
    pred_image = load_image(pred_path)
    annot_image = load_image(annot_path)

    assert pred_image.shape == annot_image.shape, f"Shape mismatch: {filename}"

    all_preds.extend(pred_image.flatten())
    all_annots.extend(annot_image.flatten())

all_preds = np.array(all_preds)
all_annots = np.array(all_annots)
pr

overall_f1_score = f1_score(all_annots, all_preds, average='macro')
overall_precision = precision_score(all_annots, all_preds, average='macro')
overall_recall = recall_score(all_annots, all_preds, average='macro')

results_df = pd.DataFrame([{
    'mean_precision': overall_precision,
    'mean_recall': overall_recall,
    'mean_f1': overall_f1_score
}])


class_f1_score = f1_score(all_annots, all_preds, average=None)
class_precision = precision_score(all_annots, all_preds, average=None)
class_recall = recall_score(all_annots, all_preds, average=None)
print(class_f1_score)

results_class_df = pd.DataFrame({
    'precision': class_precision,
    'recall': class_recall,
    'f1_score': class_f1_score
})

results_class_df.index = range(len(class_f1_score))

save_directory = 'lowAltitude_classification/confusion_matrix_Precision_Recall/pt'
metrics_path = os.path.join(save_directory, 'metrics_cm.csv')
metrics_path_class = os.path.join(save_directory, 'per_class_metrics.csv')
results_df.to_csv(metrics_path, index=False)
results_class_df.to_csv(metrics_path_class, index=False)
