import os
import random
import shutil

image_folder = '/home/kamyar/Documents/data_lowaltitude_patches'
segmentation_folder = '/home/kamyar/Documents/dataset_pred/fast_patch_256_overlap_85'

train_img_folder = '/home/kamyar/Documents/Dataset_mask2former/train/images'
train_mask_folder = '/home/kamyar/Documents/Dataset_mask2former/train/masks'
validation_img_folder = '/home/kamyar/Documents/Dataset_mask2former/val/images'
validation_mask_folder = '/home/kamyar/Documents/Dataset_mask2former/val/masks'

os.makedirs(train_img_folder, exist_ok=True)
os.makedirs(train_mask_folder, exist_ok=True)
os.makedirs(validation_img_folder, exist_ok=True)
os.makedirs(validation_mask_folder, exist_ok=True)

split_ratio = 0.8
image_files = os.listdir(image_folder)
random.shuffle(image_files)

num_train = int(len(image_files) * split_ratio)

train_images = image_files[:num_train]
validation_images = image_files[num_train:]

for image_file in train_images:
    image_name = os.path.splitext(image_file)[0]
    segmentation_file = image_name + '.png'
    if os.path.exists(os.path.join(segmentation_folder, segmentation_file)):
        shutil.copy(os.path.join(image_folder, image_file), os.path.join(train_img_folder, image_file))
        shutil.copy(os.path.join(segmentation_folder, segmentation_file), os.path.join(train_mask_folder, segmentation_file))
for image_file in validation_images:
    image_name = os.path.splitext(image_file)[0]
    segmentation_file = image_name + '.png'
    if os.path.exists(os.path.join(segmentation_folder, segmentation_file)):
        shutil.copy(os.path.join(image_folder, image_file), os.path.join(validation_img_folder, image_file))
        shutil.copy(os.path.join(segmentation_folder, segmentation_file), os.path.join(validation_mask_folder, segmentation_file))

print("Splitting completed.")
