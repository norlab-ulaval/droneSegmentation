import glob
import albumentations as A
import cv2
import numpy as np

from utils import get_label_mask, set_class_values
from torch.utils.data import Dataset, DataLoader
from functools import partial
from config import LABEL_COLORS_LIST

# ADE data mean and standard deviation.
ADE_MEAN = np.array([0.48500001430511475, 0.4560000002384186, 0.4059999883174896])
ADE_STD = np.array([0.2290000021457672, 0.2239999920129776, 0.22499999403953552])

def get_images(root_path):
    train_images = glob.glob(f"{root_path}/train/images/*")
    train_images.sort()
    train_masks = glob.glob(f"{root_path}/train/masks/*")
    train_masks.sort()
    valid_images = glob.glob(f"{root_path}/val/images/*")
    valid_images.sort()
    valid_masks = glob.glob(f"{root_path}/val/masks/*")
    valid_masks.sort()

    return train_images, train_masks, valid_images, valid_masks

def train_transforms(img_size):
    """
    Transforms/augmentations for training images and masks.

    :param img_size: Integer, for image resize.
    """
    train_image_transform = A.Compose([
        # A.Resize(100, 100, always_apply=True),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.2),
        A.RandomBrightnessContrast(p=0.1),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        A.Rotate(limit=15),
        A.Normalize(mean=ADE_MEAN, std=ADE_STD)
    ], is_check_shapes=False)
    return train_image_transform

def valid_transforms(img_size):
    """
    Transforms/augmentations for validation images and masks.

    :param img_size: Integer, for image resize.
    """
    valid_image_transform = A.Compose([
        # A.Resize(img_size[1], img_size[0], always_apply=True),
        A.Normalize(mean=ADE_MEAN, std=ADE_STD)
    ], is_check_shapes=False)
    return valid_image_transform

def collate_fn(batch, image_processor):
    inputs = list(zip(*batch))
    images = inputs[0]
    segmentation_maps = inputs[1]

    batch = image_processor(
        images,
        segmentation_maps=segmentation_maps,
        return_tensors='pt',
        do_resize=False,
        do_rescale=False,
        do_normalize=False
    )

    batch['orig_image'] = inputs[2]
    batch['orig_mask'] = inputs[3]
    return batch

class SegmentationDataset(Dataset):
    def __init__(
        self, 
        image_paths, 
        mask_paths, 
        tfms, 
        label_colors_list,
        classes_to_train,
        all_classes,
        feature_extractor
    ):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.tfms = tfms
        self.label_colors_list = label_colors_list
        self.all_classes = all_classes
        self.classes_to_train = classes_to_train
        self.class_values = set_class_values(
            self.all_classes, self.classes_to_train
        )
        self.feature_extractor = feature_extractor

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image = cv2.imread(self.image_paths[index], cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.mask_paths[index], cv2.IMREAD_GRAYSCALE)

        color_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
        for i in range(mask.shape[0]):
            for j in range(mask.shape[1]):
                # print(mask[i, j])
                idx = mask[i, j]
                color_mask[i, j] = LABEL_COLORS_LIST[idx]


        mask = color_mask
        transformed = self.tfms(image=image, mask=mask)

        orig_image = image.copy()

        image = transformed['image']
        # image = image.transpose(2, 0, 1)
        mask = transformed['mask']
        # import matplotlib.pyplot as plt
        # plt.imshow(orig_image)
        # plt.show()
        # plt.imshow(image)
        # plt.show()
        # exit()
        
        # Get 2D label mask.
        mask = get_label_mask(mask, self.class_values, self.label_colors_list)
        orig_mask = mask.copy()

        return image, mask, orig_image, orig_mask

def get_dataset(
    train_image_paths, 
    train_mask_paths,
    valid_image_paths,
    valid_mask_paths,
    all_classes,
    classes_to_train,
    label_colors_list,
    img_size,
    feature_extractor
):
    train_tfms = train_transforms(img_size)
    valid_tfms = valid_transforms(img_size)

    train_dataset = SegmentationDataset(
        train_image_paths,
        train_mask_paths,
        train_tfms,
        label_colors_list,
        classes_to_train,
        all_classes, 
        feature_extractor
    )
    valid_dataset = SegmentationDataset(
        valid_image_paths,
        valid_mask_paths,
        valid_tfms,
        label_colors_list,
        classes_to_train,
        all_classes,
        feature_extractor
    )
    return train_dataset, valid_dataset

def get_data_loaders(train_dataset, valid_dataset, batch_size, processor):
    collate_func = partial(collate_fn, image_processor=processor)

    train_data_loader = DataLoader(
        train_dataset, 
        batch_size=8,
        drop_last=False, 
        num_workers=16,
        shuffle=True,
        collate_fn=collate_func
    )
    valid_data_loader = DataLoader(
        valid_dataset, 
        batch_size=32,
        drop_last=False, 
        num_workers=16,
        shuffle=False,
        collate_fn=collate_func
    )

    return train_data_loader, valid_data_loader