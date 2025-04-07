import os
import numpy as np
from sklearn.metrics import f1_score, accuracy_score
from PIL import Image
import pandas as pd
from pathlib import Path
import csv

def load_image(image_path):
    return np.array(Image.open(image_path))

def calculate_metrics(pred_folder, annot_folder):
    all_preds = []
    all_annots = []

    for filename in os.listdir(pred_folder):
        pred_path = os.path.join(pred_folder, filename)
        base_name = os.path.splitext(filename)[0]
        annotation_filename = f'{base_name}-label-ground-truth-semantic.png'
        annot_path = os.path.join(annot_folder, annotation_filename)
        pred_image = load_image(pred_path)
        annot_image = load_image(annot_path)

        assert pred_image.shape == annot_image.shape, f"Shape mismatch: {filename}"

        all_preds.extend(pred_image.flatten())
        all_annots.extend(annot_image.flatten())

    all_preds = np.array(all_preds)
    all_annots = np.array(all_annots)

    overall_f1_score = f1_score(all_annots, all_preds, average='macro')
    pixel_accuracy = accuracy_score(all_annots, all_preds)

    return overall_f1_score, pixel_accuracy


pred_folder =  Path('')
annot_folder = Path('')

csv_num_votes = Path('lowAltitude_classification/PL_Step_Size_Experiment/PL_num_votes_per_step_size.csv')
with open(csv_num_votes, mode='r') as file:
    reader = csv.reader(file)
    num_votes_dict = {row[0]: row[1] for row in reader}


results = []
for subdir in os.listdir(pred_folder):
    subdir_path = os.path.join(pred_folder, subdir)
    overall_f1, pAcc = calculate_metrics(subdir_path, annot_folder)
    results.append({
        "Dataset": pred_folder.name,
        "step size": subdir,
        "num_votes": num_votes_dict[subdir],
        "F1": f'{overall_f1:.4f}',
        "pAcc": f'{pAcc:.4f}',
    })

df = pd.DataFrame(results)
df.to_csv(f"lowAltitude_classification/PL_Step_Size_Experiment/PL_Step_Size_Experiment_{pred_folder.name}.csv",
          index=False)