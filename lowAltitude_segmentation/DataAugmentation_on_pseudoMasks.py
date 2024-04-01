"""
Doing data augmentation on the results of classification of LA images obtained by iNaturalist
we do data augmentation here to make it similar to the HA images, so we first crop the LA images, with considering a margin, then downscale it to 256x256 size
256 can be discussed, might be other options
margin is to avoid being near borders of the image, because pseudo labels there, are very noisy and inaccurate
"""
import cv2
import os
import numpy as np

def find_largest_multiple(dim):
    return (dim // 256) * 256


# margin is to avoid being near borders of the image, because pseudo labels there, are very noisy and inaccurate
def process_image(image_path, seg_map_path, mode, margin):
    image = cv2.imread(image_path)
    seg_map = cv2.imread(seg_map_path)

    height, width = image.shape[:2]

    h = find_largest_multiple(min(width, height))

    # Ensure the start points allow for cropping within image dimensions and specified margins
    # Adjust the range for start_x to ensure it doesn't exceed the image's width boundaries considering h
    if width - h - margin * 2 > 0:
        start_x = np.random.randint(margin, width - h - margin)
    else:
        start_x = margin

    # Adjust the range for start_y to ensure it doesn't exceed the image's height boundaries considering h
    if height - h - margin * 2 > 0:
        start_y = np.random.randint(margin, height - h - margin)
    else:
        start_y = margin


    # print(start_x, start_y)
    cropped_image = image[start_y:start_y + h, start_x:start_x + h]
    cropped_seg_map = seg_map[start_y:start_y + h, start_x:start_x + h]



    resized_image = cv2.resize(cropped_image, (256, 256), interpolation=cv2.INTER_AREA)
    # For segmentation maps or any other label maps, such as semantic segmentation outputs, using cv2.INTER_NEAREST is preferred.
    # This method assigns the nearest neighbor value to each pixel in the output image, which is crucial for maintaining the discrete nature of class labels.
    resized_seg_map = cv2.resize(cropped_seg_map, (256, 256), interpolation=cv2.INTER_NEAREST)

    output_path_img = "/home/kamyar/Documents/segmentation_augmentation/" + mode + '/' + 'images/'
    os.makedirs(output_path_img, exist_ok=True)
    output_path_img = output_path_img + os.path.basename(image_path)
    cv2.imwrite(output_path_img, resized_image)

    seg_output_path = "/home/kamyar/Documents/segmentation_augmentation/" + mode + '/' + 'masks/'
    os.makedirs(seg_output_path, exist_ok=True)
    seg_output_path = seg_output_path + os.path.basename(seg_map_path)
    cv2.imwrite(seg_output_path, resized_seg_map)

train_img_folder = "/home/kamyar/Documents/segmentation/train/images"
train_mask_folder = "/home/kamyar/Documents/segmentation/train/masks"
val_img_folder = "/home/kamyar/Documents/segmentation/val/images"
val_mask_folder = "/home/kamyar/Documents/segmentation/val/masks"

for image_name in os.listdir(train_img_folder):
    image_path = os.path.join(train_img_folder, image_name)
    seg_map_path = os.path.join(train_mask_folder, os.path.splitext(image_name)[0] + '_segmentation_map' + '.jpg')
    # margin is to avoid being near borders of the image, because pseudo labels there, are very noisy and inaccurate
    process_image(image_path, seg_map_path, 'train', margin=100)

for image_name in os.listdir(val_img_folder):
    image_path = os.path.join(val_img_folder, image_name)
    seg_map_path = os.path.join(val_mask_folder, os.path.splitext(image_name)[0] + '_segmentation_map' + '.jpg')
    process_image(image_path, seg_map_path, 'val', margin=100)