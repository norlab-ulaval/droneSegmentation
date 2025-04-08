import os
import random
import shutil


def get_base_name(file_name):
    return os.path.splitext(file_name)[0]


def sample_and_move_images(folder1, folder2, dest_folder1, dest_folder2, sample_size=20):
    # Create destination folders if they don't exist
    os.makedirs(dest_folder1, exist_ok=True)
    os.makedirs(dest_folder2, exist_ok=True)

    # Get list of files in each folder
    files1 = os.listdir(folder1)
    files2 = os.listdir(folder2)

    # Get base names of files in each folder
    base_names1 = set(get_base_name(f) for f in files1)
    base_names2 = set(get_base_name(f) for f in files2)

    # Find common base names
    common_base_names = list(base_names1 & base_names2)

    # If the sample size is larger than the number of common images, adjust sample size
    if len(common_base_names) < sample_size:
        sample_size = len(common_base_names)
        print(f"Sample size adjusted to {sample_size} due to limited common images.")

    # Randomly sample the base names
    sampled_base_names = random.sample(common_base_names, sample_size)

    # Move the selected images to the destination folders and remove from the original folders
    for base_name in sampled_base_names:
        file1 = next(f for f in files1 if get_base_name(f) == base_name)
        file2 = next(f for f in files2 if get_base_name(f) == base_name)

        src_file1 = os.path.join(folder1, file1)
        src_file2 = os.path.join(folder2, file2)
        dest_file1 = os.path.join(dest_folder1, file1)
        dest_file2 = os.path.join(dest_folder2, file2)

        shutil.move(src_file1, dest_file1)
        shutil.move(src_file2, dest_file2)

    print(f"Sampled and moved {sample_size} images to destination folders, and removed them from the original folders.")


# Define the folder paths
folder1 = 'data/Dataset_mask2former/val/images'
folder2 = 'data/Dataset_mask2former/val/masks'
dest_folder1 = 'data/Dataset_mask2former/finetune/images'
dest_folder2 = 'data/Dataset_mask2former/finetune/masks'

# Sample and move images
sample_and_move_images(folder1, folder2, dest_folder1, dest_folder2)
