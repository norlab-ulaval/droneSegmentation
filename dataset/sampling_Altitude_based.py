"""
Sampling the dataset, based on relative altitude of drone
For now, I just sample low-altitude drone images (altitude = 5)
"""
import os
import shutil

source_folder = "/home/kamyar/Documents/Dataset_indexed"
destination_folder = "/home/kamyar/Documents/Dataset_lowAltitude"

if not os.path.exists(destination_folder):
    os.makedirs(destination_folder)

for filename in os.listdir(source_folder):
    altitude = filename.split('_')[2]
    # low-altitude drone images, all are equal to 5
    if altitude == '5':
        source_file_path = os.path.join(source_folder, filename)
        destination_file_path = os.path.join(destination_folder, filename)
        shutil.copy(source_file_path, destination_file_path)
        print(f"File '{filename}' copied to '{destination_folder}'.")
