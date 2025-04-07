import os
import cv2
from tqdm import tqdm

def patchify_image(image, patch_size=(1024, 1024)):
    height, width = image.shape[:2]
    patches = []

    num_patches_y = height // patch_size[0] + (1 if height % patch_size[0] != 0 else 0)
    num_patches_x = width // patch_size[1] + (1 if width % patch_size[1] != 0 else 0)

    for j in range(num_patches_y):
        for i in range(num_patches_x):
            y_start = j * patch_size[0]
            y_end = min((j + 1) * patch_size[0], height)
            x_start = i * patch_size[1]
            x_end = min((i + 1) * patch_size[1], width)
            if y_end - y_start == patch_size[0] and x_end - x_start == patch_size[1]:
                patch = image[y_start:y_end, x_start:x_end]
                patches.append(patch)

    return patches


def process_images(parent_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    for subdir, _, files in os.walk(parent_folder):
        for filename in tqdm([f for f in files if f.lower().endswith('.jpg')], desc=f"Processing images in {subdir}"):
            image_path = os.path.join(subdir, filename)
            image = cv2.imread(image_path)
            patches = patchify_image(image)

            for i, patch in enumerate(patches):
                relative_path = os.path.relpath(subdir, parent_folder)
                sub_output_folder = os.path.join(output_folder, relative_path)
                os.makedirs(sub_output_folder, exist_ok=True)

                output_filename = os.path.join(sub_output_folder, f'{os.path.splitext(filename)[0]}_patch_{i}.JPG')
                cv2.imwrite(output_filename, patch)


input_folder = ''
output_folder = ''
process_images(input_folder, output_folder)
