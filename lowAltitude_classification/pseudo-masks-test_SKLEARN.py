import os
import numpy as np
from sklearn.metrics import f1_score, accuracy_score
from PIL import Image
import pandas as pd

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






pred_folder =  '/home/kamyar/Documents/M2F_Results'
annot_folder = '/home/kamyar/Documents/Test_Annotated_masks'


# overall_f1, pAcc = calculate_metrics(pred_folder, annot_folder)
# print(f"Overall F1 Score: {overall_f1}")
# print(f"Pixel Accuracy: {pAcc}")

results = []
for subdir in os.listdir(pred_folder):

    ################################################# M2F
    subdir_path = os.path.join(pred_folder, subdir)
    params = subdir.split('_')
    PL_Version = ''

    if len(params) == 3:
        PL_Version = params[1]
        experiment = params[2]
    else:
        experiment = params[0]

    for subsubdir in os.listdir(subdir_path):
        if subsubdir == 'output_test':
            subsubdir_path = os.path.join(subdir_path, subsubdir)

            overall_f1, pAcc = calculate_metrics(subsubdir_path, annot_folder)

            print(f"Overall F1 Score: {overall_f1}")
            print(f"Pixel Accuracy: {pAcc}")

            results.append({
                "PL_Version": PL_Version,
                "experiment": experiment,

                "F1": f'{overall_f1:.4f}',
                "pAcc": f'{pAcc:.4f}',
            })


#
#
#
#     ######################################### center
#     params = subdir.split('_')
#     center = int(params[0].split('-')[1])
#     patch = int(params[1].split('-')[1])
#     step = int(params[2].split('-')[1])
#     pad = int(params[3].split('-')[1])
#
#     if pad == 184 and step == 27:
#
#         subdir_path = os.path.join(pred_folder, subdir)
#
#         overall_f1, pAcc = calculate_metrics(subdir_path, annot_folder)
#
#         print(f"Overall F1 Score: {overall_f1}")
#         print(f"Pixel Accuracy: {pAcc}")
#
#         results.append({
#             "Central Size": subdir.split('_')[0].split('-')[1],
#             "Patch Size": subdir.split('_')[1].split('-')[1],
#             "Step Size": subdir.split('_')[2].split('-')[1],
#             "Pad Size": subdir.split('_')[3].split('-')[1],
#
#             "F1": f'{overall_f1:.4f}',
#             "pAcc": f'{pAcc:.4f}',
#         })


    ######################################## patch sizes
    # patch = int(subdir.split('_')[0])
    # overlap = int(subdir.split('_')[1])
    # if overlap == 85:
    #
    #     subdir_path = os.path.join(pred_folder, subdir)
    #
    #     overall_f1, pAcc = calculate_metrics(subdir_path, annot_folder)
    #
    #     print(f"Overall F1 Score: {overall_f1}")
    #     print(f"Pixel Accuracy: {pAcc}")
    #
    #     results.append({
    #         "Central Size": subdir.split('_')[0].split('-')[1],
    #         "Patch Size": subdir.split('_')[1].split('-')[1],
    #         "Step Size": subdir.split('_')[2].split('-')[1],
    #         "Pad Size": subdir.split('_')[3].split('-')[1],
    #
    #         "F1": f'{overall_f1:.4f}',
    #         "pAcc": f'{pAcc:.4f}',
    #     })


    ################################ STRIDE
    # subdir_path = os.path.join(pred_folder, subdir)
    #
    # overall_f1, pAcc = calculate_metrics(subdir_path, annot_folder)
    #
    # print(f"Overall F1 Score: {overall_f1}")
    # print(f"Pixel Accuracy: {pAcc}")
    #
    # results.append({
    #     "Central Size": subdir.split('_')[0].split('-')[1],
    #     "Patch Size": subdir.split('_')[1].split('-')[1],
    #     "Step Size": subdir.split('_')[2].split('-')[1],
    #     "Pad Size": subdir.split('_')[3].split('-')[1],
    #
    #     "F1": f'{overall_f1:.4f}',
    #     "pAcc": f'{pAcc:.4f}',
    # })
#
df = pd.DataFrame(results)
# df = df.sort_values(by=["Pad Size"])

df.to_csv("lowAltitude_classification/results/NEW_phase_3/test/phase3-test-M2F_METRICS.csv",
          index=False)