import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score
from PIL import Image
import numpy as np

conf_matrix = pd.read_csv('/home/kamyar/Documents/ConfusionMatrix_M2F/ft_cm.csv', index_col=0)
conf_matrix_normalized = conf_matrix.div(conf_matrix.sum(axis=1), axis=0)

conf_matrix_normalized_percentage = conf_matrix_normalized * 100

plt.figure(figsize=(15, 12))
sns.heatmap(conf_matrix_normalized_percentage, annot=True, fmt='.2f', cmap='Blues', cbar=True,
            xticklabels=conf_matrix.columns, yticklabels=conf_matrix.index)

plt.title("PT + FT Confusion Matrix (Percentage)", fontsize=20)
plt.xlabel("Predicted Labels", fontsize=16)
plt.ylabel("True Labels", fontsize=16)
plt.tight_layout()
plt.show()






# save_directory = '/home/kamyar/PycharmProjects/droneSegmentation/lowAltitude_classification/confusion_matrix/ft'
# os.makedirs(save_directory, exist_ok=True)
# confusion_matrix_path = os.path.join(save_directory, 'confusion_matrix.png')
# plt.savefig(confusion_matrix_path)
#
#
# annotations_folder = '/home/kamyar/Documents/Test_Annotated_masks'
# predictions_folder = '/home/kamyar/Documents/M2F_Results/PL_V1_FINETUNE/output_test'
#
# def load_image(image_path):
#     return np.array(Image.open(image_path))
#
# all_preds = []
# all_annots = []
#
# for filename in os.listdir(predictions_folder):
#     pred_path = os.path.join(predictions_folder, filename)
#     base_name = os.path.splitext(filename)[0]
#     annotation_filename = f'{base_name}-label-ground-truth-semantic.png'
#     annot_path = os.path.join(annotations_folder, annotation_filename)
#     pred_image = load_image(pred_path)
#     annot_image = load_image(annot_path)
#
#     assert pred_image.shape == annot_image.shape, f"Shape mismatch: {filename}"
#
#     all_preds.extend(pred_image.flatten())
#     all_annots.extend(annot_image.flatten())
#
# all_preds = np.array(all_preds)
# all_annots = np.array(all_annots)
#
# overall_f1_score = f1_score(all_annots, all_preds, average='macro')
# overall_precision = precision_score(all_annots, all_preds, average='macro')
# overall_recall = recall_score(all_annots, all_preds, average='macro')
# # print(overall_f1_score)
#
# ###################### class based precision and recall
# # class_precisions = precision_score(all_annots, all_preds, average=None)
# # class_recalls = recall_score(all_annots, all_preds, average=None)
# # class_f1_scores_sklearn = f1_score(all_annots, all_preds, average=None)
# #
# # class_f1_scores_manual = []
# #
# # for precision, recall in zip(class_precisions, class_recalls):
# #     if precision == 0 or recall == 0:
# #         class_f1_scores_manual.append(0)
# #     else:
# #         f1_score = 2 * (precision * recall) / (precision + recall)
# #         class_f1_scores_manual.append(f1_score)
# #
# # # Calculate average F1 score over the individual classes
# # average_f1_score_manual = np.mean(class_f1_scores_manual)
# # print(average_f1_score_manual)
# ##############################################################################################################
#
#
# results_df = pd.DataFrame([{
#     'mean_precision': overall_precision,
#     'mean_recall': overall_recall,
#     'mean_f1': overall_f1_score
# }])
#
# metrics_path = os.path.join(save_directory, 'metrics_cm.csv')
# results_df.to_csv(metrics_path, index=False)
