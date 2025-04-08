import os
import cv2


def resize_image(image_path, output_path, size):
    """Resizes the image to the given size and saves it to the output path."""
    img = cv2.imread(image_path)

    if img is not None:
        resized_img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
        cv2.imwrite(output_path, resized_img)
    else:
        print(f"Error loading image: {image_path}")


def process_images(input_folder, output_folder, size):
    """Processes all images in the input folder, resizing them and saving to the output folder."""
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)

        resize_image(input_path, output_path, size)
        print(f"Processed {filename}")


if __name__ == "__main__":
    input_folder = 'data/DataSegmentation_split/val/images'
    output_folder = 'data/DataSegmentation_split/val_downsampled/images'
    size = (256, 256)

    process_images(input_folder, output_folder, size)
    print("All images have been processed and saved.")
