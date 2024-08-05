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
# folder_path = '/home/kamyar/Documents/Dataset_LowAltitude/Lac-Saint-Jean/ANNOTATION_patch'
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


import os
import shutil

in_folder = '/home/kamyar/Documents/iNaturalist_data + Other classes'
out_folder = '/home/kamyar/Documents/iNat_Classifier_Non_filtered'

for subfolder_name in os.listdir(in_folder):
    subfolder_path = os.path.join(in_folder, subfolder_name)
    subout = os.path.join(out_folder, subfolder_name)

    if os.path.isdir(subfolder_path):
        images_folder = os.path.join(subfolder_path, 'images')

        if os.path.isdir(images_folder):
            if not os.path.exists(subout):
                os.makedirs(subout)

            for item in os.listdir(images_folder):
                s = os.path.join(images_folder, item)
                d = os.path.join(subout, item)
                if os.path.isdir(s):
                    shutil.move(s, d)
                else:
                    shutil.move(s, d)

# import os
# import shutil
#
# def move_images(source_folder, destination_folder, image_extensions=None):
#     if image_extensions is None:
#         image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff']
#
#     # Ensure the destination folder exists
#     os.makedirs(destination_folder, exist_ok=True)
#
#     # Iterate over files in the source folder
#     for filename in os.listdir(source_folder):
#         # Check if the file has an image extension
#         if any(filename.lower().endswith(ext) for ext in image_extensions):
#             # Construct full file path
#             src_file = os.path.join(source_folder, filename)
#             dst_file = os.path.join(destination_folder, filename)
#             # Move the file
#             shutil.move(src_file, dst_file)
#             print(f"Moved: {filename}")
#
# # Example usage
# source_folder = '/home/kamyar/Documents/removed_classes/Feather Mosses/images'
# destination_folder = '/home/kamyar/Documents/iNaturalist_data + Other classes/Moss/images'
#
# move_images(source_folder, destination_folder)