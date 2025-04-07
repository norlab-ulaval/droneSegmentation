import os
import exifread
import matplotlib.pyplot as plt

# Function to extract GPS coordinates from an image
def get_gps_coordinates(image_path):
    with open(image_path, 'rb') as f:
        tags = exifread.process_file(f)

        # Extracting GPS information
        gps_latitude = tags.get('GPS GPSLatitude')
        gps_latitude_ref = tags.get('GPS GPSLatitudeRef')
        gps_longitude = tags.get('GPS GPSLongitude')
        gps_longitude_ref = tags.get('GPS GPSLongitudeRef')

        if gps_latitude and gps_longitude and gps_latitude_ref and gps_longitude_ref:
            lat = convert_to_degrees(gps_latitude)
            lon = convert_to_degrees(gps_longitude)

            # Adjust sign based on hemisphere
            if gps_latitude_ref.values[0] != 'N':
                lat = -lat
            if gps_longitude_ref.values[0] != 'E':
                lon = -lon

            return lat, lon
        else:
            return None

# Function to convert GPS coordinates to degrees
def convert_to_degrees(value):
    d = float(value.values[0].num) / float(value.values[0].den)
    m = float(value.values[1].num) / float(value.values[1].den)
    s = float(value.values[2].num) / float(value.values[2].den)

    return d + (m / 60.0) + (s / 3600.0)

# Function to get GPS coordinates for all images in a folder
def get_coordinates_from_folder(folder_path):
    latitudes = []
    longitudes = []
    for image_name in os.listdir(folder_path):
        image_path = os.path.join(folder_path, image_name)
        coords = get_gps_coordinates(image_path)
        print(coords)
        if coords:
            latitudes.append(coords[0])
            longitudes.append(coords[1])
    return latitudes, longitudes

# Input two folders
folder1 = ''
folder2 = ''

# Get coordinates from both folders
latitudes1, longitudes1 = get_coordinates_from_folder(folder1)
latitudes2, longitudes2 = get_coordinates_from_folder(folder2)

# Plotting the GPS coordinates
plt.figure(figsize=(8, 6))
plt.scatter(longitudes1, latitudes1, c='blue', marker='o', label='Train/Val')
plt.scatter(longitudes2, latitudes2, c='red', marker='o', label='Test')

plt.title('Image GPS Coordinates')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.legend()
plt.grid(True)
plt.show()
