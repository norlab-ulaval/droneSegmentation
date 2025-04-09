import os
import random
import shutil

source_dir = "data/Unlabeled_Drone_Dataset_Patch"
half_dir = "data/Unlabeled_Drone_Dataset_Patch_Half"
quarter_dir = "data/Unlabeled_Drone_Dataset_Patch_Quarter"

os.makedirs(half_dir, exist_ok=True)
os.makedirs(quarter_dir, exist_ok=True)

# Walk through all directories and subdirectories
all_images = []
for root, dirs, files in os.walk(source_dir):
    for f in files:
        all_images.append(os.path.join(root, f))

# Sample half of the images
half_images = random.sample(all_images, len(all_images) // 2)

# Copy sampled images to the half_dir
for image in half_images:
    relative_path = os.path.relpath(image, source_dir)
    dest_path = os.path.join(half_dir, relative_path)
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    shutil.copy(image, dest_path)

# Sample half of the images in half_dir
quarter_images = random.sample(half_images, len(half_images) // 2)

# Copy sampled images to the quarter_dir
for image in quarter_images:
    relative_path = os.path.relpath(image, source_dir)
    dest_path = os.path.join(quarter_dir, relative_path)
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    shutil.copy(image, dest_path)

print("Images have been successfully copied.")
