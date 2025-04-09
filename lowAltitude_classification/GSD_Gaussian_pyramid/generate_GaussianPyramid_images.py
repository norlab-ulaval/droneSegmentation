import os
import cv2
import numpy as np
from pathlib import Path

def apply_gaussian_filter(input_folder, output_folder, sigmas):
    Path(output_folder).mkdir(parents=True, exist_ok=True)

    for image_name in os.listdir(input_folder):
        image_path = os.path.join(input_folder, image_name)

        if os.path.isfile(image_path) and image_name.lower().endswith(('JPG', 'jpg')):
            image = cv2.imread(image_path)

            for sigma in sigmas:
                filtered_image = cv2.GaussianBlur(image, (2*int(sigma * 4) + 1, 2*int(sigma * 4) + 1), sigma)
                sigma_folder = os.path.join(output_folder, f'sigma_{sigma}')
                Path(sigma_folder).mkdir(parents=True, exist_ok=True)

                output_image_path = os.path.join(sigma_folder, image_name)
                cv2.imwrite(output_image_path, filtered_image)
                print(f"Processed {image_name} with sigma={sigma}")

if __name__ == "__main__":
    input_folder = "data/Test_Annotated"
    output_folder = "results/GSD_Gaussian/images_newKernel"
    sigmas = [1, 2, 4, 8, 16]

    apply_gaussian_filter(input_folder, output_folder, sigmas)
