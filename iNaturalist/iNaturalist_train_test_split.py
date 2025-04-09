import os
import random
import shutil


def split_data(
    source, out_directory, split_ratio=(0.8, 0.1, 0.1), max_per_class=100000
):
    if not os.path.exists(out_directory):
        os.makedirs(out_directory)

    for class_folder in os.listdir(source):
        class_path = os.path.join(source, class_folder) + "/images"
        if os.path.isdir(class_path):
            images = os.listdir(class_path)
            random.shuffle(images)

            num_images = min(len(images), max_per_class)
            num_train = int(num_images * split_ratio[0])
            num_test = int(num_images * split_ratio[1])
            num_val = num_images - num_train - num_test

            train_images = images[:num_train]
            test_images = images[num_train : num_train + num_test]
            val_images = images[num_train + num_test : num_train + num_test + num_val]

            train_out = os.path.join(out_directory, "train", class_folder)
            val_out = os.path.join(out_directory, "val", class_folder)
            test_out = os.path.join(out_directory, "test", class_folder)

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


source_directory = "data/iNaturalist_data + Other classes"
out_directory = "data/iNaturalist_data_split"

split_data(source_directory, out_directory, max_per_class=100000)
