import os
import random
import shutil

def split_data(source, out_directory, split_ratio=(0.8, 0.1, 0.1), max_per_class=10000):
    if not os.path.exists(out_directory):
        os.makedirs(out_directory)

    for class_folder in os.listdir(source):
        class_path = os.path.join(source, class_folder) + '/images'
        if os.path.isdir(class_path):
            images = os.listdir(class_path)
            random.shuffle(images)

            num_images = min(len(images), max_per_class)
            num_train = int(num_images * split_ratio[0])
            num_test = int(num_images * split_ratio[1])
            num_val = num_images - num_train - num_test

            train_images = images[:num_train]
            test_images = images[num_train:num_train + num_test]
            val_images = images[num_train + num_test:num_train + num_test + num_val]

            train_out = os.path.join(out_directory, 'train', class_folder)
            val_out = os.path.join(out_directory, 'val', class_folder)
            test_out = os.path.join(out_directory, 'test', class_folder)

            if not os.path.exists(train_out):
                os.makedirs(train_out)
            if not os.path.exists(val_out):
                os.makedirs(val_out)
            if not os.path.exists(test_out):
                os.makedirs(test_out)

            for img in train_images:
                shutil.copy(os.path.join(class_path, img), train_out)
            for img in test_images:
                shutil.copy(os.path.join(class_path, img), val_out)
            for img in val_images:
                shutil.copy(os.path.join(class_path, img), test_out)

source_directory = "/home/kamyar/Documents/filtered_inat"
out_directory = "/home/kamyar/Documents/filtered_inat_split"

split_data(source_directory, out_directory, max_per_class=20000)


# from PIL import Image
# from torchvision import transforms
# import os
# import random
# import shutil
#
# def oversample_images(class_path, target_num_images):
#     images = os.listdir(class_path)
#     num_images = len(images)
#
#     if num_images >= target_num_images:
#         return images
#
#     num_duplicates = target_num_images - num_images
#
#     # Define augmentation transformations
#     augmentation = transforms.Compose([
#         transforms.Resize((256, 256)),
#         transforms.RandomRotation(15),
#         transforms.RandomHorizontalFlip(),
#         transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
#         transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.3),
#     ])
#     print(class_path)
#     while num_images < target_num_images:
#         if num_images < 1000:
#             break
#         images = os.listdir(class_path)
#         # Duplicate random images to oversample
#         sampled_images = random.sample(images, min(num_duplicates, num_images))
#         for img in sampled_images:
#             original_img_path = os.path.join(class_path, img)
#             new_img_path = original_img_path
#             if not img.startswith("oversampled_"):
#                 new_img_path = os.path.join(class_path, f"oversampled_{img}")
#
#             # Apply data augmentation
#             image = Image.open(original_img_path)
#             augmented_image = augmentation(image)
#             augmented_image.save(new_img_path)
#
#         num_images = len(os.listdir(class_path))
#         num_duplicates = target_num_images - num_images
#
#     return os.listdir(class_path)
#
#
# def split_data(source, out_directory, split_ratio=(0.9, 0.05, 0.05), max_per_class=20000):
#     if not os.path.exists(out_directory):
#         os.makedirs(out_directory)
#
#     for class_folder in os.listdir(source):
#         class_path = os.path.join(source, class_folder, 'images')
#         if os.path.isdir(class_path):
#             # Oversample if needed
#             images = oversample_images(class_path, max_per_class)
#
#             random.shuffle(images)
#             num_images = min(len(images), max_per_class)
#             num_train = int(num_images * split_ratio[0])
#             num_test = int(num_images * split_ratio[1])
#             num_val = num_images - num_train - num_test
#
#             train_images = images[:num_train]
#             test_images = images[num_train:num_train + num_test]
#             val_images = images[num_train + num_test:num_train + num_test + num_val]
#
#             train_out = os.path.join(out_directory, 'train', class_folder)
#             val_out = os.path.join(out_directory, 'val', class_folder)
#             test_out = os.path.join(out_directory, 'test', class_folder)
#
#             if not os.path.exists(train_out):
#                 os.makedirs(train_out)
#             if not os.path.exists(val_out):
#                 os.makedirs(val_out)
#             if not os.path.exists(test_out):
#                 os.makedirs(test_out)
#
#             for img in train_images:
#                 shutil.copy(os.path.join(class_path, img), os.path.join(train_out, img))
#             for img in test_images:
#                 shutil.copy(os.path.join(class_path, img), os.path.join(val_out, img))
#             for img in val_images:
#                 shutil.copy(os.path.join(class_path, img), os.path.join(test_out, img))
#
#
# source_directory = "/home/kamyar/Documents/iNaturalist_data + Other classes"
# out_directory = "/home/kamyar/Documents/iNaturalist_split"
#
# split_data(source_directory, out_directory, max_per_class=20000)
