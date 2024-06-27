import os
import cv2
from tqdm import tqdm

def patchify_image(image, patch_size=(1024, 1024)):
    height, width = image.shape[:2]
    patches = []

    # Calculate the number of patches in each dimension
    num_patches_y = height // patch_size[0] + (1 if height % patch_size[0] != 0 else 0)
    num_patches_x = width // patch_size[1] + (1 if width % patch_size[1] != 0 else 0)

    for j in range(num_patches_y):
        for i in range(num_patches_x):
            # Calculate patch coordinates
            y_start = j * patch_size[0]
            y_end = min((j + 1) * patch_size[0], height)
            x_start = i * patch_size[1]
            x_end = min((i + 1) * patch_size[1], width)

            # Check if patch size is exactly 1024x1024
            if y_end - y_start == patch_size[0] and x_end - x_start == patch_size[1]:
                # Extract patch
                patch = image[y_start:y_end, x_start:x_end]
                patches.append(patch)

    return patches


def process_images(input_folder, output_folder):
    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Get list of image files in input folder
    image_files = [f for f in os.listdir(input_folder) if f.endswith('.JPG')]

    for filename in tqdm(image_files, desc="Processing images"):
        # Load image
        image = cv2.imread(os.path.join(input_folder, filename))

        # Patchify the image
        patches = patchify_image(image)

        # Save patchified images
        for i, patch in enumerate(patches):
            output_filename = os.path.join(output_folder, f'{os.path.splitext(filename)[0]}_patch_{i}.JPG')
            cv2.imwrite(output_filename, patch)


input_folder = '/home/kamyar/Documents/Dataset_LowAltitude/ZecChapais_June20_indexed_annotation'
output_folder = '/home/kamyar/Documents/Dataset_LowAltitude/ZecChapais_June20_indexed_annotation_patch'
process_images(input_folder, output_folder)
