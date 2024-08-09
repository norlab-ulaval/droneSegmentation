import os
import cv2
import numpy as np

before_to_after_mapping = {
    0: None,  # Alders removed
    1: 0,    # American Mountain-ash -> 0
    2: 1,    # Background -> 1
    3: 2,    # Black spruce -> 2
    4: 4,    # Bog Labrador Tea -> 4
    5: 5,    # Boulders -> 5
    6: 6,    # Canada Yew -> 6
    7: 13,   # Common Haircap Moss -> combined into Moss -> 13
    8: 7,    # Dead trees -> 7
    9: 13,   # Feather Mosses -> combined into Moss -> 13
    10: 8,   # Ferns -> 8
    11: 9,   # Fir -> 9
    12: 10,  # Fire Cherry -> 10
    13: 11,  # Jack Pine -> 11
    14: None, # Larch removed
    15: 12,  # Leatherleaf -> 12
    16: 14,  # Mountain Maple -> 14
    17: 15,  # Paper Birch -> 15
    18: None, # Pixie Cup Lichens removed
    19: 16,  # Red Maple -> 16
    20: 17,  # Red Raspberry -> 17
    21: 18,  # Sedges -> 18
    22: 19,  # Serviceberries -> 19
    23: 20,  # Sheep Laurel -> 20
    24: 13,  # Sphagnum Mosses -> combined into Moss -> 13
    25: 21,  # Trembling Aspen -> 21
    26: 22,  # Viburnums -> 22
    27: 23,  # Willowherbs -> 23
    28: None, # Willows removed
    29: 3,   # Blueberry -> 3
    30: 24,  # Wood -> 24
    31: 25   # Yellow Birch -> 25
}

#
# before_to_after_mapping = {
#     2: 0,
#     3: 1,
#     4: 2,
#     5: 3,
#     6: 4,
#     7: 5,
#     8: 6,
#     9: 13,
#     10: 7,
#     11: 13,
#     12: 8,
#     13: 9,
#     14: 10,
#     15: 11,
#     16: None,
#     17: 12,
#     18: 14,
#     19: 15,
#     20: None,
#     21: 18,
#     22: 19,
#     23: 13,
#     24: 21,
#     25: 22,
#     26: 23,
#     27: None,
#     28: 24,
#     29: 25,
#     30: None,
#     31: 16,
#     32: 17,
#     33: 20
# }


source_folder_path = '/home/kamyar/Documents/Test_data_annotation'
destination_folder_path = '/home/kamyar/Documents/Test_data_annotation_new_index'


os.makedirs(destination_folder_path, exist_ok=True)

for filename in os.listdir(source_folder_path):
    if filename.endswith(".png"):
        image_path = os.path.join(source_folder_path, filename)
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        updated_image = np.zeros_like(image)

        for old_index, new_index in before_to_after_mapping.items():
            if new_index is not None:
                updated_image[image == old_index] = new_index

        updated_image_path = os.path.join(destination_folder_path, filename)
        cv2.imwrite(updated_image_path, updated_image)

print(f"Index values updated and saved to {destination_folder_path} successfully.")
