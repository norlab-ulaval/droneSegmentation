import numpy as np
import cv2
import torch
import os
import matplotlib.pyplot as plt
import torch.nn as nn


plt.style.use('ggplot')


def set_class_values(all_classes, classes_to_train):
    """
    This (`class_values`) assigns a specific class label to the each of the classes.
    For example, `animal=0`, `archway=1`, and so on.

    :param all_classes: List containing all class names.
    :param classes_to_train: List containing class names to train.
    """
    class_values = [all_classes.index(cls.lower()) for cls in classes_to_train]
    return class_values


def get_label_mask(mask, class_values, label_colors_list):
    """
    This function encodes the pixels belonging to the same class
    in the image into the same label

    :param mask: NumPy array, segmentation mask.
    :param class_values: List containing class values, e.g car=0, bus=1.
    :param label_colors_list: List containing RGB color value for each class.
    """
    label_mask = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.uint8)
    for value in class_values:
        for ii, label in enumerate(label_colors_list):
            if value == label_colors_list.index(label):
                label = np.array(label)
                label_mask[np.where(np.all(mask == label, axis=-1))[:2]] = value + 1
    label_mask = label_mask.astype(int)
    return label_mask


def denormalize(x, mean=None, std=None):
    # x should be a Numpy array of shape [H, W, C]
    x = torch.tensor(x).permute(2, 0, 1).unsqueeze(0)
    for t, m, s in zip(x, mean, std):
        t.mul_(s).add_(m)
    res = torch.clamp(t, 0, 1)
    res = res.squeeze(0).permute(1, 2, 0).numpy()
    return res


class SaveBestModel:
    """
    Class to save the best model while training. If the current epoch's
    validation loss is less than the previous least less, then save the
    model state.
    """

    def __init__(self, best_valid_loss=float('inf')):
        self.best_valid_loss = best_valid_loss

    def __call__(
            self, current_valid_loss, epoch, model, out_dir, name='model'
    ):
        if current_valid_loss < self.best_valid_loss:
            self.best_valid_loss = current_valid_loss
            print(f"\nBest validation loss: {self.best_valid_loss}")
            print(f"\nSaving best model for epoch: {epoch + 1}\n")
            model.save_pretrained(os.path.join(out_dir, name))


class SaveBestModelIOU:
    """
    Class to save the best model while training. If the current epoch's
    IoU is higher than the previous highest, then save the
    model state.
    """

    def __init__(self, best_iou=float(0)):
        self.best_iou = best_iou

    def __call__(self, current_iou, epoch, model, out_dir, name='model'):
        if current_iou > self.best_iou:
            self.best_iou = current_iou
            print(f"\nBest validation IoU: {self.best_iou}")
            print(f"\nSaving best model for epoch: {epoch + 1}\n")
            model.save_pretrained(os.path.join(out_dir, name))


def save_model(model, out_dir, name='model'):
    """
    Function to save the trained model to disk.
    """
    model.save_pretrained(os.path.join(out_dir, name))


def save_plots(
        train_loss, valid_loss,
        train_miou, valid_miou,
        out_dir
):
    """
    Function to save the loss and accuracy plots to disk.
    """

    # Loss plots.
    plt.figure(figsize=(10, 7))
    plt.plot(
        train_loss, color='tab:blue', linestyle='-',
        label='train loss'
    )
    plt.plot(
        valid_loss, color='tab:red', linestyle='-',
        label='validataion loss'
    )
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(out_dir, 'loss.png'))

    # mIOU plots.
    plt.figure(figsize=(10, 7))
    plt.plot(
        train_miou, color='tab:blue', linestyle='-',
        label='train mIoU'
    )
    plt.plot(
        valid_miou, color='tab:red', linestyle='-',
        label='validataion mIoU'
    )
    plt.xlabel('Epochs')
    plt.ylabel('mIoU')
    plt.legend()
    plt.savefig(os.path.join(out_dir, 'miou.png'))


def predict(model, extractor, image, device):
    """
    :param model: The Segformer model.
    :param extractor: The Segformer feature extractor.
    :param image: The image in RGB format.
    :param device: The compute device.

    Returns:
        labels: The final labels (classes) in h x w format.
    """

    pixel_values = extractor(image, return_tensors='pt').to(device)
    with torch.no_grad():
        outputs = model(**pixel_values)

    pred_map = extractor.post_process_semantic_segmentation(
        outputs, target_sizes=[(image.shape[0], image.shape[1])]
    )[0]

    return pred_map


def draw_segmentation_map(labels, palette):
    """
    :param labels: Label array from the model.Should be of shape
        <height x width>. No channel information required.
    :param palette: List containing color information.
        e.g. [[0, 255, 0], [255, 255, 0]]
    """
    # create Numpy arrays containing zeros
    # later to be used to fill them with respective red, green, and blue pixels
    red_map = np.zeros_like(labels).astype(np.uint8)
    green_map = np.zeros_like(labels).astype(np.uint8)
    blue_map = np.zeros_like(labels).astype(np.uint8)

    for label_num in range(0, len(palette)):
        index = labels == label_num
        red_map[index] = np.array(palette)[label_num, 0]
        green_map[index] = np.array(palette)[label_num, 1]
        blue_map[index] = np.array(palette)[label_num, 2]

    segmentation_map = np.stack([red_map, green_map, blue_map], axis=2)
    return segmentation_map


def image_overlay(image, segmented_image):
    """
    :param image: Image in RGB format.
    :param segmented_image: Segmentation map in RGB format.
    """
    alpha = 0.8  # transparency for the original image
    beta = 1.0  # transparency for the segmentation map
    gamma = 0  # scalar added to each sum

    segmented_image = cv2.cvtColor(segmented_image, cv2.COLOR_RGB2BGR)
    image = np.array(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.addWeighted(image, alpha, segmented_image, beta, gamma, image)
    return image