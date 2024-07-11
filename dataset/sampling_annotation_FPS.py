import os
import numpy as np
import random
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

def get_existing_images(dest_dir):
    existing_images = set()
    for root, _, files in os.walk(dest_dir):
        for file in files:
            if file.lower().endswith(('jpg', 'jpeg', 'png')):
                existing_images.add(file)
    return existing_images

image_dir = '/home/kamyar/Documents/Dataset_LowAltitude/ForetMontmorency_June28_indexed'
dest_dir_1 = '/home/kamyar/Documents/Dataset_LowAltitude/ForetMontmorency_June28_indexed_annotation'
# dest_dir_2 = '/home/kamyar/Documents/Dataset_LowAltitude/ZecChapais_June20_indexed_annotation_2'
n_samples = 50

# existing_images = get_existing_images(dest_dir_1)
coords, image_paths = read_image_coordinates(image_dir)

# filtered_coords = []
# filtered_image_paths = []

# for coord, image_path in zip(coords, image_paths):
#     if os.path.basename(image_path) not in existing_images:
#         filtered_coords.append(coord)
#         filtered_image_paths.append(image_path)

# filtered_coords = np.array(filtered_coords)

# if len(filtered_coords) < n_samples:
#     print(f"Not enough images to sample {n_samples} images. Available images: {len(filtered_coords)}")
# else:
sampled_indices = farthest_point_sampling(coords, n_samples)
selected_images = [image_paths[idx] for idx in sampled_indices]
selected_coords = [coords[idx] for idx in sampled_indices]

save_images(selected_images, dest_dir_1)

for idx, coord in zip(sampled_indices, selected_coords):
    print(f"Image: {image_paths[idx]}, Coordinates: {coord}")
