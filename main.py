import os

folder_path = '/home/kamyar/Documents/iNaturalist_data + Other classes'

for root, dirs, files in os.walk(folder_path):
    for file in files:
        if file.startswith('oversampled_'):
            print(root.replace(folder_path, '').lstrip(os.path.sep))
            break
# for filename in os.listdir(folder_path):
#     if filename.startswith('oversampled_'):
#         os.remove(os.path.join(folder_path, filename))


#
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
#     directory = '/home/kamyar/Documents/ZecBatiscan_June5'
#     remove_non_jpg_files(directory)


import os


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
# folder_path = '/home/kamyar/Documents/Dataset_LowAltitude/ZecBatiscan_June5_indexed_annotation_patch'
# rename_images(folder_path)
