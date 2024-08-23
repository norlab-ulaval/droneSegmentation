# import os
#
# folder_path = '/home/kamyar/Documents/iNaturalist_data + Other classes'
#
# for root, dirs, files in os.walk(folder_path):
#     for file in files:
#         if file.startswith('oversampled_'):
#             print(root.replace(folder_path, '').lstrip(os.path.sep))
#             break
# for filename in os.listdir(folder_path):
#     if filename.startswith('oversampled_'):
#         os.remove(os.path.join(folder_path, filename))



#
# import os
#
# def remove_non_jpg_files(directory):
#     for root, dirs, files in os.walk(directory):
#         for file in files:
#             if not file.lower().endswith('.jpg'):
#                 file_path = os.path.join(root, file)
#                 try:
#                     os.remove(file_path)
#                     print(f"Removed: {file_path}")
#                 except Exception as e:
#                     print(f"Error removing {file_path}: {e}")
#
# if __name__ == "__main__":
#     directory = '/home/kamyar/Documents/Drone'
#     remove_non_jpg_files(directory)
#
#
# import os
#
#
# def rename_images(folder_path):
#     for filename in os.listdir(folder_path):
#         # Skip non-image files
#         if not any(filename.lower().endswith(ext) for ext in ['.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff']):
#             continue
#
#         # Replace underscores with dashes and remove colons
#         new_filename = filename.replace('_', '-').replace(':', '')
#
#         # Construct full file paths
#         old_file = os.path.join(folder_path, filename)
#         new_file = os.path.join(folder_path, new_filename)
#
#         # Rename the file
#         os.rename(old_file, new_file)
#         print(f'Renamed: {filename} -> {new_filename}')
#
#
# # Example usage
# folder_path = '/home/kamyar/Documents/Train-val_Annotated'
# rename_images(folder_path)


# import os
# import shutil
#
#
# def move_images_to_parent_folder(parent_folder):
#     # List all subdirectories in the parent folder
#     subfolders = [f.path for f in os.scandir(parent_folder) if f.is_dir()]
#
#     # Iterate over each subfolder
#     for subfolder in subfolders:
#         # List all files in the subfolder
#         for file_name in os.listdir(subfolder):
#             file_path = os.path.join(subfolder, file_name)
#             # Check if it is a file (and not a directory)
#             if os.path.isfile(file_path):
#                 # Construct the destination path in the parent folder
#                 dest_path = os.path.join(parent_folder, file_name)
#                 # Move the file to the parent folder
#                 shutil.move(file_path, dest_path)
#                 print(f"Moved {file_path} to {dest_path}")
#
#     # Remove empty subfolders
#     for subfolder in subfolders:
#         os.rmdir(subfolder)
#         print(f"Removed empty subfolder {subfolder}")
#
#
# # Example usage
# parent_folder_path = '/home/kamyar/Documents/iNaturalist_data + Other classes/wood/images'
# move_images_to_parent_folder(parent_folder_path)




# import os
#
# folder1 = '/home/kamyar/Documents/data_lowaltitude_merged'
# folder2 = '/home/kamyar/Documents/Dataset_LowAltitude/ZecBatiscan_June5_indexed_annotation'
#
# # Get the list of files in each folder
# files_in_folder1 = set(os.listdir(folder1))
# files_in_folder2 = set(os.listdir(folder2))
#
# # Find the common files
# common_files = files_in_folder1.intersection(files_in_folder2)
#
# # Remove common files from the first folder
# for file in common_files:
#     file_path = os.path.join(folder1, file)
#     if os.path.isfile(file_path):
#         os.remove(file_path)
#         print(f'Removed: {file_path}')
#
# print('Exclusion complete.')
#

#
# import os
# import shutil
#
# in_folder = '/home/kamyar/Documents/iNaturalist_data + Other classes'
# out_folder = '/home/kamyar/Documents/iNat_Classifier_Non_filtered'
#
# for subfolder_name in os.listdir(in_folder):
# #     subfolder_path = os.path.join(in_folder, subfolder_name)
# #     subout = os.path.join(out_folder, subfolder_name)
# #
# #     if os.path.isdir(subfolder_path):
# #         images_folder = os.path.join(subfolder_path, 'images')
# #
# #         if os.path.isdir(images_folder):
# #             if not os.path.exists(subout):
# #                 os.makedirs(subout)
# #
# #             for item in os.listdir(images_folder):
# #                 s = os.path.join(images_folder, item)
# #                 d = os.path.join(subout, item)
# #                 if os.path.isdir(s):
# #                     shutil.move(s, d)
# #                 else:
# #                     shutil.move(s, d)
#
# # import os
# # import shutil
# #
# # def move_images(source_folder, destination_folder, image_extensions=None):
# #     if image_extensions is None:
# #         image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff']
# #
# #     # Ensure the destination folder exists
# #     os.makedirs(destination_folder, exist_ok=True)
# #
# #     # Iterate over files in the source folder
# #     for filename in os.listdir(source_folder):
# #         # Check if the file has an image extension
# #         if any(filename.lower().endswith(ext) for ext in image_extensions):
# #             # Construct full file path
# #             src_file = os.path.join(source_folder, filename)
# #             dst_file = os.path.join(destination_folder, filename)
# #             # Move the file
# #             shutil.move(src_file, dst_file)
# #             print(f"Moved: {filename}")
# #
# # # Example usage
# # source_folder = '/home/kamyar/Documents/removed_classes/Feather Mosses/images'
# # destination_folder = '/home/kamyar/Documents/iNaturalist_data + Other classes/Moss/images'
# #
# # move_images(source_folder, destination_folder)
#
#
# import os
# import matplotlib.pyplot as plt
# import numpy as np
#
# def count_images_in_folder(folder_path, image_extensions=None):
#     if image_extensions is None:
#         image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp'}
#
#     count = 0
#     for filename in os.listdir(folder_path):
#         if any(filename.lower().endswith(ext) for ext in image_extensions):
#             count += 1
#     return count
#
# def plot_image_counts(root_folder):
#     subfolders = [f.path for f in os.scandir(root_folder) if f.is_dir()]
#     folder_names = []
#     image_counts = []
#
#     for folder in subfolders:
#         folder_name = os.path.basename(folder)
#         num_images = count_images_in_folder(folder)
#         folder_names.append(folder_name)
#         image_counts.append(num_images)
#
#     # Sort based on the number of images
#     sorted_indices = sorted(range(len(image_counts)), key=lambda i: image_counts[i], reverse=True)
#     sorted_folder_names = [folder_names[i] for i in sorted_indices]
#     sorted_image_counts = [image_counts[i] for i in sorted_indices]
#
#     img_count_arr = np.array(image_counts)
#     print(np.median(img_count_arr), np.average(img_count_arr))
#
#     # Plotting
#     plt.figure(figsize=(12, 6))
#     bars = plt.bar(sorted_folder_names, sorted_image_counts, color='skyblue')
#
#     # Add labels above bars
#     for bar in bars:
#         yval = bar.get_height()
#         plt.text(bar.get_x() + bar.get_width()/2, yval, int(yval), va='bottom', ha='center')
#
#     plt.ylabel('Number of Images')
#     plt.title('Histogram of Number of Images in Each Category')
#     plt.xticks(rotation=45, ha='right')
#     # plt.yscale('log')
#     plt.tight_layout()
#     plt.show()
#
# plot_image_counts('/home/kamyar/Documents/iNat_Classifier_filtered')


#
# import os
# from PIL import Image
# import matplotlib.pyplot as plt
#
# # Paths to the folders
# images_folder = '/home/kamyar/Documents/Test_data'
# annotations_folder = '/home/kamyar/Documents/Test_data_annotation_new_index'
#
# # List the files in the folders
# image_files = sorted(os.listdir(images_folder))
# annotation_files = sorted(os.listdir(annotations_folder))
#
# # Select the first image and annotation for demonstration
# image_file = image_files[59]
# annotation_file = annotation_files[59]
#
# # Load the image and annotation
# image_path = os.path.join(images_folder, image_file)
# annotation_path = os.path.join(annotations_folder, annotation_file)
#
# image = Image.open(image_path)
# annotation = Image.open(annotation_path)
#
# # Plot the image and annotation side by side
# fig, axes = plt.subplots(1, 2, figsize=(10, 5))
#
# axes[0].imshow(image)
# axes[0].set_title('Image')
# axes[0].axis('off')
#
# axes[1].imshow(annotation, cmap='gray')
# axes[1].set_title('Annotation')
# axes[1].axis('off')
#
# plt.show()

# import os
# import shutil
#
# # Paths to the folders
# folder1_path = "/home/kamyar/Documents/Annotated_images"
# folder2_path = "/home/kamyar/Documents/Dataset_LowAltitude"
#
# # Destination folder to store the matched images
# matched_folder_path = "/home/kamyar/Documents/Annotated_original"
# os.makedirs(matched_folder_path, exist_ok=True)
#
#
# # Function to match the images
# def match_images(folder1_path, folder2_path, matched_folder_path):
#     folder2_images = {}
#
#     # Iterate over subfolders in folder2
#     for subdir, _, files in os.walk(folder2_path):
#         for file in files:
#             folder2_images[file] = os.path.join(subdir, file)
#
#     # Iterate over images in folder1
#     for image in os.listdir(folder1_path):
#         if "-patch-" in image:
#             base_image_name = image.split("-patch-")[0] + ".JPG"
#             if base_image_name in folder2_images:
#                 # If match found, copy the image from both folders to the matched folder
#                 shutil.copy(os.path.join(folder1_path, image), matched_folder_path)
#                 shutil.copy(folder2_images[base_image_name], matched_folder_path)
#
#
# # Run the function
# match_images(folder1_path, folder2_path, matched_folder_path)
#
# print("Matching completed. Check the matched images in the specified folder.")


# import os
#
#
# def count_images_in_subfolders(folder_path):
#     subfolder_image_counts = {}
#
#     # Iterate through each subfolder in the main folder
#     for subfolder in os.listdir(folder_path):
#         subfolder_path = os.path.join(folder_path, subfolder)
#
#         # Check if the current item is a directory and not named "background"
#         if os.path.isdir(subfolder_path) and subfolder.lower() != "background"\
#                 and subfolder.lower() != "boulders" and subfolder.lower() != "wood"\
#                 and subfolder.lower() != "dead trees":
#             # Count the number of image files in the subfolder
#             image_count = len(
#                 [f for f in os.listdir(subfolder_path) if os.path.isfile(os.path.join(subfolder_path, f))])
#             subfolder_image_counts[subfolder] = image_count
#
#     return subfolder_image_counts
#
#
# # Example usage
# folder_path = '/home/kamyar/Documents/iNat_Classifier_Non_filtered'
# image_counts = count_images_in_subfolders(folder_path)
#
# sum_ = 0
# for subfolder, count in image_counts.items():
#     sum_ += count
#     print(f"{subfolder}: {count} images")
#
# print(sum_)


# import os
# import shutil
#
#
# def get_all_images(folder):
#     """Get all image file paths in the folder and its subfolders."""
#     image_paths = []
#     for root, _, files in os.walk(folder):
#         for file in files:
#             if file.endswith(('.JPG', '.jpg')):
#                 image_paths.append(os.path.join(root, file))
#     return set(image_paths)
#
#
# def get_image_names(folder):
#     """Get a set of image file names in the folder."""
#     image_names = set()
#     for root, _, files in os.walk(folder):
#         for file in files:
#             if file.endswith(('.JPG', '.jpg')):
#                 image_names.add(file)
#     return image_names
#
#
# def exclude_images(parent_folder, exclusion_folder):
#     """Exclude images in exclusion_folder from parent_folder."""
#     exclusion_images = get_image_names(exclusion_folder)
#     all_images_in_parent = get_all_images(parent_folder)
#
#     for image_path in all_images_in_parent:
#         image_name = os.path.basename(image_path)
#         if image_name in exclusion_images:
#             print(f"Removing {image_path}")
#             os.remove(image_path)
#
#
# if __name__ == "__main__":
#     parent_folder = "/home/kamyar/Documents/Drone_Unlabeled_Dataset"
#     exclusion_folder = "/home/kamyar/Documents/Test_Annotated_Original"
#
#     exclude_images(parent_folder, exclusion_folder)




# import os
# from PIL import Image
#
#
# input_folder = '/home/kamyar/Documents/Drone_Unlabeled_Dataset/ZecChapais_June20_indexed'
# output_folder = '/home/kamyar/Documents/Drone_Unlabeled_Dataset/ZecChapais_June20_indexed'
#
# os.makedirs(output_folder, exist_ok=True)
#
# for filename in os.listdir(input_folder):
#     if filename.endswith(('.JPG')):
#         # Open the image
#         image_path = os.path.join(input_folder, filename)
#         image = Image.open(image_path)
#
#         # Define the cropping box (left, upper, right, lower)
#         cropped_image = image.crop((0, 350, image.width, image.height))
#
#         cropped_image.save(os.path.join(output_folder, filename))
#
# print("Processing complete. All images have been cropped.")



# import os
# import shutil
#
# def move_images_to_parent(parent_folder):
#     # Iterate over all the subfolders in the parent folder
#     for root, dirs, files in os.walk(parent_folder):
#         # Skip the parent folder itself
#         if root == parent_folder:
#             continue
#
#         # Move each file to the parent folder
#         for file in files:
#             if file.endswith(('.jpg', '.JPG')):  # Add more extensions if needed
#                 source = os.path.join(root, file)
#                 destination = os.path.join(parent_folder, file)
#
#                 # Move the file
#                 shutil.move(source, destination)
#                 print(f"Moved {file} to {parent_folder}")
#
#         # Optionally, remove the empty subfolder after moving the images
#         if not os.listdir(root):  # Check if the subfolder is empty
#             os.rmdir(root)
#             print(f"Removed empty folder {root}")
#
# if __name__ == "__main__":
#     parent_folder = '/home/kamyar/Documents/Drone_Unlabeled_Dataset_Patch_split/Third batch'  # Replace with your parent folder path
#     move_images_to_parent(parent_folder)


# import os
# import random
# import shutil
#
#
# def split_images(folder_path, destination_folder):
#     # Create the destination folder if it doesn't exist
#     if not os.path.exists(destination_folder):
#         os.makedirs(destination_folder)
#
#     # List all the files in the folder
#     files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
#
#
#     # Determine the split point (half of the images)
#     split_point = len(files) // 2
#
#     # Move half of the images to the destination folder
#     for file in files[:split_point]:
#         shutil.move(os.path.join(folder_path, file), os.path.join(destination_folder, file))
#
#     print(f"Moved {split_point} files to {destination_folder}")
#
#
# # Example usage
# folder_path = '/home/kamyar/Documents/Drone_Unlabeled_Dataset_Patch_split/Third batch'
# destination_folder = '/home/kamyar/Documents/Drone_Unlabeled_Dataset_Patch_split/Fifth batch'
# split_images(folder_path, destination_folder)


import os
# import shutil
# import random
# from pathlib import Path
#
#
# def split_dataset(images_folder, masks_folder, output_folder, val_split=0.2):
#     # Create output directories
#     train_images_dir = Path(output_folder) / 'train' / 'images'
#     train_masks_dir = Path(output_folder) / 'train' / 'masks'
#     val_images_dir = Path(output_folder) / 'val' / 'images'
#     val_masks_dir = Path(output_folder) / 'val' / 'masks'
#
#     # Ensure output directories exist
#     for directory in [train_images_dir, train_masks_dir, val_images_dir, val_masks_dir]:
#         directory.mkdir(parents=True, exist_ok=True)
#
#     # Get a list of all image files
#     images = sorted(Path(images_folder).glob('*'))
#     masks = sorted(Path(masks_folder).glob('*'))
#
#     # Ensure that there is a corresponding mask for every image
#     assert len(images) == len(masks), "The number of images and masks must be the same"
#
#     # Combine images and masks
#     combined = list(zip(images, masks))
#
#     # Shuffle combined list to randomize the split
#     random.shuffle(combined)
#
#     # Determine the split point
#     split_point = int(len(combined) * val_split)
#
#     # Split into training and validation sets
#     val_set = combined[:split_point]
#     train_set = combined[split_point:]
#
#     # Move files to respective folders
#     for img_path, mask_path in train_set:
#         shutil.copy(img_path, train_images_dir / img_path.name)
#         shutil.copy(mask_path, train_masks_dir / mask_path.name)
#
#     for img_path, mask_path in val_set:
#         shutil.copy(img_path, val_images_dir / img_path.name)
#         shutil.copy(mask_path, val_masks_dir / mask_path.name)
#
#     print(f"Dataset split completed. Train: {len(train_set)} samples, Validation: {len(val_set)} samples.")
#
#
# # Usage example
# images_folder = '/home/kamyar/Documents/Train-val_Annotated'
# masks_folder = '/home/kamyar/Documents/Train-val_Annotated_masks'
# output_folder = '/home/kamyar/Documents/M2F_Train_Val_split'
#
# split_dataset(images_folder, masks_folder, output_folder, val_split=0.4)
#




# import os
# from PIL import Image
# import numpy as np
#
# # Define the folder containing the PNG annotations
# folder_path = '/home/kamyar/Documents/Train-val_Annotated_masks'
#
# # Define the index mappings for specific files
# index_mappings = {
#     '2024-07-25-013232-5-Mauricie-4000x4000-DJI-FC7303-patch-5-label-ground-truth-semantic.png': {19: 22},
#     '2024-06-05-132546-14.677-ZecBatiscan-5280x5280-DJI-M3E-patch-2-label-ground-truth-semantic.png': {15: 25},
#     '2024-06-05-131617-19.249-ZecBatiscan-5280x5280-DJI-M3E-patch-7-label-ground-truth-semantic.png': {9: 2},
#     '2024-06-06-023928-5-Zec-Batiscan-4000x4000-DJI-FC7303-patch-1-label-ground-truth-semantic.png': {22: 10}
# }
#
# # Function to update the indices in the images
# def update_indices(image_path, mapping):
#     # Load the image
#     img = Image.open(image_path)
#     img_array = np.array(img)
#
#     # Update the indices based on the mapping
#     for old_index, new_index in mapping.items():
#         img_array[img_array == old_index] = new_index
#
#     # Save the modified image
#     updated_img = Image.fromarray(img_array)
#     updated_img.save(image_path)
#
# # Iterate over files in the folder
# for filename in os.listdir(folder_path):
#     if filename in index_mappings:
#         file_path = os.path.join(folder_path, filename)
#         update_indices(file_path, index_mappings[filename])
#         print(f'Updated indices in {filename}')
#
# print('All applicable files have been processed.')
