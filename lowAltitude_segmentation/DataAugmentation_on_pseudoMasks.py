import cv2
import os

def find_largest_multiple(dim):
    return (dim // 256) * 256

def process_image(image_path, seg_map_path):
    image = cv2.imread(image_path)
    seg_map = cv2.imread(seg_map_path)

    height, width = image.shape[:2]

    h = find_largest_multiple(min(width, height))

    left_margin = (width - h) // 2
    upper_margin = (height - h) // 2
    right_margin = left_margin + h
    lower_margin = upper_margin + h

    cropped_image = image[upper_margin:lower_margin, left_margin:right_margin]

    resized_image = cv2.resize(cropped_image, (256, 256), interpolation=cv2.INTER_AREA)

    output_path = "output/" + os.path.basename(image_path)
    cv2.imwrite(output_path, resized_image)

    seg_output_path = "output/" + os.path.basename(seg_map_path)
    cv2.imwrite(seg_output_path, seg_map)

images_folder = "images/"
seg_maps_folder = "segmentation_maps/"

for image_name in os.listdir(images_folder):
    if image_name.endswith(".jpg") or image_name.endswith(".png"):
        image_path = os.path.join(images_folder, image_name)
        seg_map_path = os.path.join(seg_maps_folder, image_name)

        if os.path.exists(seg_map_path):
            process_image(image_path, seg_map_path)
        else:
            print(f"No segmentation map found for {image_name}")
