import os
import numpy as np
import random
from PIL import Image
import exifread
import shutil

def get_coordinates_from_metadata(image_path):
    with open(image_path, 'rb') as f:
        tags = exifread.process_file(f)

        gps_latitude = tags.get('GPS GPSLatitude')
        gps_latitude_ref = tags.get('GPS GPSLatitudeRef')
        gps_longitude = tags.get('GPS GPSLongitude')
        gps_longitude_ref = tags.get('GPS GPSLongitudeRef')

        if gps_latitude and gps_longitude and gps_latitude_ref and gps_longitude_ref:
            lat = [float(x.num) / float(x.den) for x in gps_latitude.values]
            lon = [float(x.num) / float(x.den) for x in gps_longitude.values]
            lat_ref = gps_latitude_ref.values
            lon_ref = gps_longitude_ref.values

            latitude = lat[0] + lat[1] / 60 + lat[2] / 3600
            longitude = lon[0] + lon[1] / 60 + lon[2] / 3600

            # if lat_ref != 'N':
            #     latitude = -latitude
            # if lon_ref != 'E':
            #     longitude = -longitude

            return latitude, longitude

    return None

def farthest_point_sampling(coords, n_samples):
    N = coords.shape[0]
    sampled_indices = np.zeros(n_samples, dtype=int)

    sampled_indices[0] = random.randint(0, N - 1)
    distances = np.full(N, np.inf)

    for i in range(1, n_samples):
        last_sampled_index = sampled_indices[i - 1]
        last_sampled_point = coords[last_sampled_index]

        dist_to_last_sampled = np.linalg.norm(coords - last_sampled_point, axis=1)
        distances = np.minimum(distances, dist_to_last_sampled)

        sampled_indices[i] = np.argmax(distances)

    return sampled_indices

def read_image_coordinates(image_dir):
    coords = []
    image_paths = []
    for root, _, files in os.walk(image_dir):
        for file in files:
            if file.lower().endswith(('jpg', 'jpeg', 'png')):
                image_path = os.path.join(root, file)
                coord = get_coordinates_from_metadata(image_path)
                if coord:
                    coords.append(coord)
                    image_paths.append(image_path)

    return np.array(coords), image_paths

def save_images(image_paths, dest_dir):
    os.makedirs(dest_dir, exist_ok=True)
    for image_path in image_paths:
        shutil.copy(image_path, dest_dir)

image_dir = '/home/kamyar/Documents/Dataset_LowAltitude/ZecBatiscan_June5_indexed'
dest_dir = '/home/kamyar/Documents/Dataset_LowAltitude/ZecBatiscan_June5_indexed_annotation'
n_samples = 20

coords, image_paths = read_image_coordinates(image_dir)

# for idx, coord in enumerate(coords):
#     print(f"Image: {image_paths[idx]}, Coordinates: {coord}")

sampled_indices = farthest_point_sampling(coords, n_samples)
selected_images = [image_paths[idx] for idx in sampled_indices]
selected_coords = [coords[idx] for idx in sampled_indices]

save_images(selected_images, dest_dir)

for idx, coord in zip(sampled_indices, selected_coords):
    print(f"Image: {image_paths[idx]}, Coordinates: {coord}")
