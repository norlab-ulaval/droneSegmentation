import os
import random
import shutil


def split_images_into_groups(source_folder, output_folder, num_groups=2):
    """
    Splits images from a source folder with subfolders into multiple groups.

    Args:
        source_folder (str): Path to the source folder containing images.
        output_folder (str): Path to the output folder for the groups.
        num_groups (int): Number of groups to split the images into.
    """
    # Gather all image paths
    image_paths = []
    for root, _, files in os.walk(source_folder):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.JPG')):
                image_paths.append(os.path.join(root, file))

    # Shuffle the image paths for randomness
    random.shuffle(image_paths)

    # Split the images into groups
    groups = [[] for _ in range(num_groups)]
    for idx, image_path in enumerate(image_paths):
        groups[idx % num_groups].append(image_path)

    # Create group folders and copy files
    for i, group in enumerate(groups):
        group_folder = os.path.join(output_folder, f"group_{i + 1}")
        os.makedirs(group_folder, exist_ok=True)
        for image_path in group:
            relative_path = os.path.relpath(image_path, source_folder)
            dest_path = os.path.join(group_folder, relative_path)
            os.makedirs(os.path.dirname(dest_path), exist_ok=True)
            shutil.copy(image_path, dest_path)

    print(f"Images successfully split into {num_groups} groups at {output_folder}")


# Example usage
source_folder = "/home/kamyar/Documents/split/group_5"  # Replace with your source folder
output_folder = "/home/kamyar/Documents/split/group_5_1"  # Replace with your output folder
split_images_into_groups(source_folder, output_folder)
