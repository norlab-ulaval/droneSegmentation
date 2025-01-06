import os
import time
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from albumentations import Normalize, Compose
from albumentations.pytorch import ToTensorV2
from transformers import AutoImageProcessor, AutoModelForImageClassification
from pathlib import Path
from sklearn.metrics import f1_score
from tqdm import tqdm
import pandas as pd
import torch.nn.functional as F

# Paths
# parent_weights_folder = Path("lowAltitude_classification/results_Journal_classifier_weights/final")

val_image_folder = Path("/home/kamyar/Documents/Train-val_Annotated/")
test_image_folder = Path("/home/kamyar/Documents/Test_Annotated/")
val_annotation_folder = Path("/home/kamyar/Documents/Train-val_Annotated_masks_updated")
test_annotation_folder = Path("/home/kamyar/Documents/Test_Annotated_masks_updated")
output_csv_path = 'lowAltitude_classification/Cls_Different_Inference_sizes/Different_Inference_classifier.csv'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_name = "facebook/dinov2-large-imagenet1k-1-layer"
processor = AutoImageProcessor.from_pretrained(model_name)
mean = processor.image_mean
std = processor.image_std

transform = Compose([
    Normalize(mean=mean, std=std),
    ToTensorV2()
])

patch_sizes = [144, 184, 256, 320]
overlaps = [0.0]

results = []

def process_images(image_folder, annotation_folder, y_true, y_pred, total_pixels, correct_predictions, description, model, patch_size, overlap):
    x_offsets, y_offsets = np.meshgrid(np.arange(patch_size), np.arange(patch_size))
    offsets = np.stack([x_offsets, y_offsets], axis=-1).reshape(-1, 2)

    step_size = int(patch_size * (1 - overlap))
    batch_size = 256

    image_files = [f for f in os.listdir(image_folder) if f.endswith(('.jpg', '.JPG'))]
    for image_file in tqdm(image_files, desc=f'Processing {description} Images'):
        begin_time = time.perf_counter()
        image_path = os.path.join(image_folder, image_file)

        base_name = os.path.splitext(image_file)[0]
        annotation_filename = f'{base_name}-label-ground-truth-semantic.png'
        annotation_path = os.path.join(annotation_folder, annotation_filename)

        image = Image.open(image_path)
        annotation = Image.open(annotation_path)
        image_np = np.array(image)
        annotation_np = np.array(annotation)
        transformed = transform(image=image_np)
        image_tensor = transformed['image'].to(device)

        width, height = image.size

        patches = []
        coordinates = []
        annotation_patches = []

        for x in range(0, width - patch_size + 1, step_size):
            for y in range(0, height - patch_size + 1, step_size):
                patch = image_tensor[:, y:y + patch_size, x:x + patch_size]
                annotation_patch = annotation_np[y:y + patch_size, x:x + patch_size]

                patches.append(patch)
                coordinates.append((x, y))
                annotation_patches.append(annotation_patch)

                if len(patches) == batch_size:
                    patches_tensor = torch.stack(patches).to(device)

                    with torch.no_grad(), torch.cuda.amp.autocast():
                        outputs = model(patches_tensor)

                    predicted_classes = torch.argmax(outputs.logits, dim=1)

                    for patch_idx, (x_, y_) in enumerate(coordinates):
                        predicted_class = predicted_classes[patch_idx].item()
                        patch_annotation = annotation_patches[patch_idx]

                        for px in range(patch_size):
                            for py in range(patch_size):
                                px_global, py_global = x_ + px, y_ + py
                                if px_global < width and py_global < height:
                                    true_label = patch_annotation[py, px]
                                    y_true.append(true_label)
                                    y_pred.append(predicted_class)

                                    total_pixels += 1
                                    if true_label == predicted_class:
                                        correct_predictions += 1

                    patches = []
                    coordinates = []
                    annotation_patches = []

        if patches:
            patches_tensor = torch.stack(patches).to(device)

            with torch.no_grad(), torch.cuda.amp.autocast():
                outputs = model(patches_tensor)

            predicted_classes = torch.argmax(outputs.logits, dim=1)

            for patch_idx, (x_, y_) in enumerate(coordinates):
                predicted_class = predicted_classes[patch_idx].item()

                patch_annotation = annotation_patches[patch_idx]

                for px in range(patch_size):
                    for py in range(patch_size):
                        px_global, py_global = x_ + px, y_ + py
                        if px_global < width and py_global < height:
                            true_label = patch_annotation[py, px]
                            y_true.append(true_label)
                            y_pred.append(predicted_class)

                            total_pixels += 1
                            if true_label == predicted_class:
                                correct_predictions += 1

        print(f'Time taken for {description}: {time.perf_counter() - begin_time:.2f}s')

    return total_pixels, correct_predictions, y_true, y_pred


# for weight_folder in os.listdir(parent_weights_folder):
#     weight_folder_path = parent_weights_folder
#     if weight_folder_path.is_dir():
#         for weight_file in os.listdir(weight_folder_path):
#             if weight_file.endswith('.pth'):
#                 weight_file_path = weight_folder_path / weight_file
weight_folder_path = Path('/home/kamyar/Documents/Best_classifier_Weight_NEW/')
if weight_folder_path.is_dir():
    for weight_file in os.listdir(weight_folder_path):
        if weight_file.endswith('.pth'):
            weight_file_path = weight_folder_path / weight_file
            model = AutoModelForImageClassification.from_pretrained(model_name, ignore_mismatched_sizes=True)
            model = model.to(device)
            model.classifier = nn.Linear(2048, 24).to(device)
            model.load_state_dict(torch.load(weight_file_path))
            model.eval()

            print(f'Processing with weight file: {weight_file_path.name}')

            for patch_size in patch_sizes:
                for overlap in overlaps:

                    total_pixels_val, correct_predictions_val, y_true_val, y_pred_val = process_images(
                        val_image_folder, val_annotation_folder, [], [], 0, 0, 'Validation', model, patch_size=patch_size, overlap=overlap)

                    total_pixels_test, correct_predictions_test, y_true_test, y_pred_test = process_images(
                        test_image_folder, test_annotation_folder, [], [], 0, 0, 'Testing', model, patch_size=patch_size, overlap=overlap)


                    if total_pixels_val > 0:
                        accuracy_val = correct_predictions_val / total_pixels_val
                        f1_macro_val = f1_score(y_true_val, y_pred_val, average='macro')
                        f1_weighted_val = f1_score(y_true_val, y_pred_val, average='weighted')

                        print('Val', weight_file_path.name, overlap)

                        results.append({
                            'Dataset': 'Validation',
                            'Weight File': weight_file_path.name,
                            'Patch Size': patch_size,
                            'Overlap': overlap,
                            'Accuracy': accuracy_val,
                            'F1 Score - Macro': f1_macro_val,
                            'F1 Score - Weighted': f1_weighted_val
                        })
                    else:
                        print(f'No validation pixels were processed for weight file {weight_file_path.name} with patch size {patch_size} and overlap {overlap}.')

                    if total_pixels_test > 0:
                        accuracy_test = correct_predictions_test / total_pixels_test
                        f1_macro_test = f1_score(y_true_test, y_pred_test, average='macro')
                        f1_weighted_test = f1_score(y_true_test, y_pred_test, average='weighted')

                        print('Test', weight_file_path.name, overlap)

                        results.append({
                            'Dataset': 'Test',
                            'Weight File': weight_file_path.name,
                            'Patch Size': patch_size,
                            'Overlap': overlap,
                            'Accuracy': accuracy_test,
                            'F1 Score - Macro': f1_macro_test,
                            'F1 Score - Weighted': f1_weighted_test
                        })

results_df = pd.DataFrame(results)
results_df.to_csv(output_csv_path, index=False)

print("Processing complete. Metrics saved to CSV.")