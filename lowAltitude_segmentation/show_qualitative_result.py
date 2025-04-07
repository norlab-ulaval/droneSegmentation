import numpy as np
from PIL import Image
import os


DRONE_SEM_SEG_CATEGORIES = [
    {"color": [255, 0, 0], "id": 0, "name": "American Mountain-Ash"},
    {"color": [128, 128, 128], "id": 1, "name": "Background"},
    {"color": [0, 0, 255], "id": 2, "name": "Bog Labrador Tea"},
    {"color": [255, 255, 0], "id": 3, "name": "Boulders"},
    {"color": [0, 255, 255], "id": 4, "name": "Canada Yew"},
    {"color": [255, 0, 255], "id": 5, "name": "Dead Trees"},
    {"color": [128, 0, 0], "id": 6, "name": "Fern"},
    {"color": [0, 128, 0], "id": 7, "name": "Fir"},
    {"color": [0, 0, 128], "id": 8, "name": "Fire Cherry"},
    {"color": [128, 128, 0], "id": 9, "name": "LowBush Blueberry"},
    {"color": [0, 128, 128], "id": 10, "name": "Moss"},
    {"color": [128, 0, 128], "id": 11, "name": "Mountain Maple"},
    {"color": [192, 192, 192], "id": 12, "name": "Paper Birch"},
    {"color": [0, 255, 0], "id": 13, "name": "Pine"},
    {"color": [64, 224, 208], "id": 14, "name": "Red Maple"},
    {"color": [255, 165, 0], "id": 15, "name": "Red Raspberry"},
    {"color": [255, 215, 0], "id": 16, "name": "Sedges"},
    {"color": [135, 206, 250], "id": 17, "name": "Serviceberries"},
    {"color": [75, 0, 130], "id": 18, "name": "Sheep Laurel"},
    {"color": [199, 21, 133], "id": 19, "name": "Spruce"},
    {"color": [244, 164, 96], "id": 20, "name": "Trembling Aspen"},
    {"color": [60, 179, 113], "id": 21, "name": "Willowherbs"},
    {"color": [147, 112, 219], "id": 22, "name": "Wood"},
    {"color": [166, 60, 20], "id": 23, "name": "Yellow Birch"}
]

id_to_color_name = {category['id']: (category['color'], category['name']) for category in DRONE_SEM_SEG_CATEGORIES}

image_folder = ''
annotation_folder = ''
voting_folder = ''
PT_folder = ''
PTFT_folder = ''

def save_image_and_txt(image_array, image_color, image_name, folder, color_mapping):
    color_image = Image.fromarray(image_color)
    image_save_path = os.path.join(folder, f'{image_name}.png')
    color_image.save(image_save_path)

    txt_save_path = os.path.join(folder, f'{image_name}.txt')
    with open(txt_save_path, 'w') as txt_file:
        unique_ids = np.unique(image_array)
        for id_val in unique_ids:
            if id_val in color_mapping:
                color, class_name = color_mapping[id_val]
                txt_file.write(f"Class: {class_name} (ID: {id_val}), Color: {color}\n")

for image_file in os.listdir(image_folder):
    if image_file.endswith('.jpg'):
        image_path = os.path.join(image_folder, image_file)
        annotation_file = image_file.replace('.jpg', '.png')

        base_name = os.path.splitext(image_file)[0]
        annotation_filename = f'{base_name}-label-ground-truth-semantic.png'
        annotation_path = os.path.join(annotation_folder, annotation_filename)

        voting_path = os.path.join(voting_folder, annotation_file)
        PT_path = os.path.join(PT_folder, annotation_file)
        PTFT_path = os.path.join(PTFT_folder, annotation_file)

        if os.path.exists(annotation_path) and os.path.exists(voting_path):
            image = np.array(Image.open(image_path))
            annotation = np.array(Image.open(annotation_path))
            voting = np.array(Image.open(voting_path))
            PT = np.array(Image.open(PT_path))
            PTFT = np.array(Image.open(PTFT_path))

            color_annotation = np.zeros((annotation.shape[0], annotation.shape[1], 3), dtype=np.uint8)
            color_voting = np.zeros((voting.shape[0], voting.shape[1], 3), dtype=np.uint8)
            color_PT = np.zeros((PT.shape[0], PT.shape[1], 3), dtype=np.uint8)
            color_PTFT = np.zeros((PTFT.shape[0], PTFT.shape[1], 3), dtype=np.uint8)

            for id_val in np.unique(annotation):
                if id_val in id_to_color_name:
                    color_annotation[annotation == id_val] = id_to_color_name[id_val][0]

            for id_val in np.unique(voting):
                if id_val in id_to_color_name:
                    color_voting[voting == id_val] = id_to_color_name[id_val][0]

            for id_val in np.unique(PT):
                if id_val in id_to_color_name:
                    color_PT[PT == id_val] = id_to_color_name[id_val][0]

            for id_val in np.unique(PTFT):
                if id_val in id_to_color_name:
                    color_PTFT[PTFT == id_val] = id_to_color_name[id_val][0]

            file_name = os.path.splitext(image_file)[0]
            save_image_and_txt(annotation, color_annotation, f'{file_name}', annotation_folder, id_to_color_name)
            save_image_and_txt(voting, color_voting, f'{file_name}', voting_folder, id_to_color_name)
            save_image_and_txt(PT, color_PT, f'{file_name}', PT_folder, id_to_color_name)
            save_image_and_txt(PTFT, color_PTFT, f'{file_name}', PTFT_folder, id_to_color_name)