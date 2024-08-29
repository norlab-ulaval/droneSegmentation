import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def load_images_from_folder(folder, color_mode=cv2.IMREAD_GRAYSCALE):
    images = []
    filenames = []
    for filename in sorted(os.listdir(folder)):
        img = cv2.imread(os.path.join(folder, filename), color_mode)
        if img is not None:
            images.append(img)
            filenames.append(filename)
    return images, filenames

# def compute_iou(prediction, target, num_classes, ignored_classes, epsilon=1e-7):
#     iou_scores = []
#     for cls in range(num_classes):
#         if cls in ignored_classes or cls not in target:
#             continue
#         intersection = np.logical_and(prediction == cls, target == cls)
#         union = np.logical_or(prediction == cls, target == cls)
#         iou = np.sum(intersection) / (np.sum(union) + epsilon)
#         iou_scores.append(iou)
#     return np.mean(iou_scores) if iou_scores else 0
#
# def compute_pixel_accuracy(prediction, target, ignored_classes):
#     mask = np.isin(target, ignored_classes, invert=True)
#     correct = np.sum((prediction == target) & mask)
#     total = np.sum(mask)
#     accuracy = correct / total if total != 0 else 0
#     return accuracy
#
# def compute_f1_score(prediction, target, num_classes, ignored_classes, epsilon=1e-7):
#     f1_scores = []
#     for cls in range(num_classes):
#         if cls in ignored_classes or cls not in target:
#             continue
#         tp = np.sum((prediction == cls) & (target == cls))
#         fp = np.sum((prediction == cls) & (target != cls))
#         fn = np.sum((prediction != cls) & (target == cls))
#
#         precision = tp / (tp + fp + epsilon) if (tp + fp) != 0 else 0
#         recall = tp / (tp + fn + epsilon) if (tp + fn) != 0 else 0
#
#         f1 = (
#             2 * precision * recall / (precision + recall + epsilon)
#             if (precision + recall) != 0
#             else 0
#         )
#         f1_scores.append(f1)
#     return np.mean(f1_scores) if f1_scores else 0, precision, recall


def compute_iou(prediction, target, num_classes, ignored_classes, epsilon=1e-7):
    iou_scores = []
    for cls in range(num_classes):
        if cls in ignored_classes or cls not in target:
            continue
        valid_mask = (prediction != -1) & (target != -1)
        if not np.any(valid_mask):
            continue
        intersection = np.logical_and(prediction == cls, target == cls) & valid_mask
        union = np.logical_or(prediction == cls, target == cls) & valid_mask
        iou = np.sum(intersection) / (np.sum(union) + epsilon)
        iou_scores.append(iou)
    return np.mean(iou_scores) if iou_scores else 0

def compute_pixel_accuracy(prediction, target, ignored_classes):
    valid_mask = (prediction != -1) & (target != -1)
    if not np.any(valid_mask):
        return 0
    mask = np.isin(target, ignored_classes, invert=True) & valid_mask
    correct = np.sum((prediction == target) & mask)
    total = np.sum(mask)
    accuracy = correct / total if total != 0 else 0
    return accuracy

def compute_f1_score(prediction, target, num_classes, ignored_classes, epsilon=1e-7):
    f1_scores = []
    for cls in range(num_classes):
        if cls in ignored_classes or cls not in target:
            continue
        valid_mask = (prediction != -1) & (target != -1)
        if not np.any(valid_mask):
            continue
        tp = np.sum((prediction == cls) & (target == cls) & valid_mask)
        fp = np.sum((prediction == cls) & (target != cls) & valid_mask)
        fn = np.sum((prediction != cls) & (target == cls) & valid_mask)

        precision = tp / (tp + fp + epsilon) if (tp + fp) != 0 else 0
        recall = tp / (tp + fn + epsilon) if (tp + fn) != 0 else 0

        f1 = (
            2 * precision * recall / (precision + recall + epsilon)
            if (precision + recall) != 0
            else 0
        )
        f1_scores.append(f1)
    return np.mean(f1_scores) if f1_scores else 0, precision, recall

def evaluate_segmentation(pred_folder, target_folder, ignored_classes):
    pred_images, pred_filenames = load_images_from_folder(pred_folder)
    target_images, _ = load_images_from_folder(target_folder)

    total_iou = 0
    total_accuracy = 0
    total_f1_score = 0
    total_precision = 0
    total_recall = 0

    for pred, target, filename in zip(pred_images, target_images, pred_filenames):
        iou = compute_iou(
            pred, target, num_classes=26, ignored_classes=ignored_classes
        )
        accuracy = compute_pixel_accuracy(
            pred, target, ignored_classes=ignored_classes
        )
        f1_score, precision, recall = compute_f1_score(
            pred, target, num_classes=26, ignored_classes=ignored_classes
        )

        total_iou += iou
        total_accuracy += accuracy
        total_f1_score += f1_score
        total_precision += precision
        total_recall += recall

        # # Plot prediction and target side by side
        # plt.figure(figsize=(10, 5))
        #
        # # Plot the target image
        # plt.subplot(1, 2, 1)
        # plt.imshow(target, cmap='gray')
        # plt.title(f"Ground Truth\nIoU: {iou:.2f}, Accuracy: {accuracy:.2f}\nF1 Score: {f1_score:.2f}, Precision: {precision:.2f}, Recall: {recall:.2f}")
        # plt.axis('off')
        #
        # # Plot the prediction image
        # plt.subplot(1, 2, 2)
        # plt.imshow(pred, cmap='gray')
        # plt.title(f"Prediction\nIoU: {iou:.2f}, Accuracy: {accuracy:.2f}\nF1 Score: {f1_score:.2f}, Precision: {precision:.2f}, Recall: {recall:.2f}")
        # plt.axis('off')
        #
        # # Show the plots
        # plt.suptitle(f"Image: {filename}", fontsize=16)
        # plt.show()

    avg_iou = total_iou / len(pred_images)
    avg_accuracy = total_accuracy / len(pred_images)
    avg_f1_score = total_f1_score / len(pred_images)
    avg_precision = total_precision / len(pred_images)
    avg_recall = total_recall / len(pred_images)

    return avg_iou, avg_accuracy, avg_f1_score, avg_precision, avg_recall

if __name__ == "__main__":
    ignored_classes = [1]

    parent_preds_folder =  '/home/kamyar/Documents/Test_Annotated_Predictions/CENTER/center-96_patch-184_step-27_pad-184'
    target_folder = '/home/kamyar/Documents/Test_Annotated_masks'

    # results = []
    #
    # for subdir in os.listdir(parent_preds_folder):
    #     # if subdir == 'CENTER' or subdir == 'DifferentPatchSize':
    #     #     continue
    #
    #     # print(subdir)
    #/home/kamyar/Documents/M2F_Results/Swin_fineTune/output_test
    #     subdir_path = os.path.join(parent_preds_folder, subdir)

    if os.path.isdir(parent_preds_folder):
        avg_iou, avg_accuracy, avg_f1_score, avg_precision, avg_recall = evaluate_segmentation(
            parent_preds_folder, target_folder, ignored_classes
        )
        print(f'mIoU: {avg_iou}, pAccuracy: {avg_accuracy}, F1 Score: {avg_f1_score}, Precision: {avg_precision}, Recall: {avg_recall}')
            # results.append({
            #     "Experiment": f"experiment {subdir}",
            #     "mIoU": f'{avg_iou:.4f}',
            #     "pAcc": f'{avg_accuracy:.4f}',
            #     "F1": f'{avg_f1_score:.4f}',
            # })

            ####################### Different Patch sizes and overlaps
            # results.append({
            #     # "Experiment": f"experiment {idx}",
            #     "Patch Size": subdir.split('_')[0],
            #     "Overlap": subdir.split('_')[1],
            #     # "precision": f'{avg_precision:.4f}',
            #     # "recall": f'{avg_recall:.4f}',
            #     "mIoU": f'{avg_iou:.4f}',
            #     "pAcc": f'{avg_accuracy:.4f}',
            #     "F1": f'{avg_f1_score:.4f}',
            # })

            ######################## Different center assignment sizes
    #         results.append({
    #             "Central Size": subdir.split('_')[0].split('-')[1],
    #             "Patch Size": subdir.split('_')[1].split('-')[1],
    #             "Step Size": subdir.split('_')[2].split('-')[1],
    #             "Pad Size": subdir.split('_')[3].split('-')[1],
    #
    #             # "precision": f'{avg_precision:.4f}',
    #             # "recall": f'{avg_recall:.4f}',
    #             "mIoU": f'{avg_iou:.4f}',
    #             "pAcc": f'{avg_accuracy:.4f}',
    #             "F1": f'{avg_f1_score:.4f}',
    #         })
    #
    # df = pd.DataFrame(results)
    # df = df.sort_values(by=["Pad Size"])
    #
    # df.to_csv("lowAltitude_classification/results/phase2/center/val/phase2-val-center.csv",
    #           index=False)
