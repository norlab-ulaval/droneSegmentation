import shutil
import os
import piexif
import math
import subprocess
import re
from PIL import Image
import PIL


def dms_to_decimal(degrees, minutes, seconds, direction):
    decimal_degrees = degrees + minutes / 60 + seconds / 3600
    if direction in ["S", "W"]:
        decimal_degrees = -decimal_degrees
    return decimal_degrees


def haversine(
    lat1_d,
    lat1_m,
    lat1_s,
    lat1_dir,
    lon1_d,
    lon1_m,
    lon1_s,
    lon1_dir,
    lat2_d,
    lat2_m,
    lat2_s,
    lat2_dir,
    lon2_d,
    lon2_m,
    lon2_s,
    lon2_dir,
):
    lat1 = dms_to_decimal(lat1_d, lat1_m, lat1_s, lat1_dir)
    lon1 = dms_to_decimal(lon1_d, lon1_m, lon1_s, lon1_dir)
    lat2 = dms_to_decimal(lat2_d, lat2_m, lat2_s, lat2_dir)
    lon2 = dms_to_decimal(lon2_d, lon2_m, lon2_s, lon2_dir)

    lat1 = math.radians(lat1)
    lon1 = math.radians(lon1)
    lat2 = math.radians(lat2)
    lon2 = math.radians(lon2)

    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    )
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    R = 6371.0
    distance = R * c
    return distance


def extract_metadata(image_path, attributes):
    exif_dict = piexif.load(image_path)

    if "GPS" in exif_dict.keys():
        gps_info = exif_dict["GPS"]
        if (
            piexif.GPSIFD.GPSLatitudeRef
            and piexif.GPSIFD.GPSLatitude
            and piexif.GPSIFD.GPSLongitudeRef
            and piexif.GPSIFD.GPSLongitude
            and piexif.GPSIFD.GPSAltitudeRef in gps_info
        ):
            lat_dir = gps_info[piexif.GPSIFD.GPSLatitudeRef].decode("utf-8")
            lat = gps_info[piexif.GPSIFD.GPSLatitude]
            lat = tuple(
                round(num / denom, 2) if denom != 1 else num for num, denom in lat
            )
            lon_dir = gps_info[piexif.GPSIFD.GPSLongitudeRef].decode("utf-8")
            lon = gps_info[piexif.GPSIFD.GPSLongitude]
            lon = tuple(
                round(num / denom, 2) if denom != 1 else num for num, denom in lon
            )

            # print(lat, lon)

            # min_distance = float('inf')
            # closest_location_label = None
            # for ref_lat, ref_lon, label in reference_locations:
            #     distance = haversine(lat[0], lat[1], lat[2], lat_dir, lon[0], lon[1], lon[2], lon_dir,
            #                                     ref_lat[0], ref_lat[1], ref_lat[2], ref_lat[3], ref_lon[0], ref_lon[1], ref_lon[2], ref_lon[3])
            #     # print("Distance:", distance, "km")
            #     if distance < min_distance:
            #         min_distance = distance
            #         closest_location_label = label
            #
            # attributes['Location'] = closest_location_label
            # print("Closest reference location:", closest_location_label)

    if "Exif" in exif_dict.keys():
        exif_info = exif_dict["Exif"]
        if piexif.ExifIFD.DateTimeOriginal in exif_info:
            date_only = (
                exif_info[piexif.ExifIFD.DateTimeOriginal].decode("utf-8").split(" ")[0]
            )
            time_only = (
                exif_info[piexif.ExifIFD.DateTimeOriginal].decode("utf-8").split(" ")[1]
            )
            year, month, day = date_only.split(":")
            origin_data = f"{year}-{month}-{day}"
            attributes["Date Original"] = origin_data
            attributes["Time"] = time_only

        if piexif.ExifIFD.PixelXDimension and piexif.ExifIFD.PixelYDimension:
            attributes["Image Size"] = (
                f"{exif_info[piexif.ExifIFD.PixelXDimension]}x{exif_info[piexif.ExifIFD.PixelXDimension]}"
            )
            # print(exif_info[piexif.ExifIFD.PixelXDimension], exif_info[piexif.ExifIFD.PixelYDimension])

    if "0th" in exif_dict.keys():
        zeroth_info = exif_dict["0th"]
        if piexif.ImageIFD.Make and piexif.ImageIFD.Model in zeroth_info:
            make = zeroth_info[piexif.ImageIFD.Make].decode("utf-8").replace("\x00", "")
            model = (
                zeroth_info[piexif.ImageIFD.Model].decode("utf-8").replace("\x00", "")
            )
            attributes["Camera Model"] = model
            attributes["Make"] = make


def get_relative_altitude(file_path, attributes):
    result = subprocess.run(["exiftool", file_path], stdout=subprocess.PIPE, text=True)

    for line in result.stdout.split("\n"):
        if "Relative Altitude" in line:
            match = re.search(r":\s*([+-]?\d+\.\d+)", line)
            if match:
                relative_alt = float(match.group(1)[:])
                if relative_alt < 10:
                    attributes["Relative Altitude"] = 5
                else:
                    attributes["Relative Altitude"] = relative_alt

    return None


# def get_GSD(image_path, attributes):
# exif_dict = piexif.load(image_path)
# exif_info = exif_dict['Exif']
# focal_length_mm = exif_info[piexif.ExifIFD.FocalLength][0] / exif_info[piexif.ExifIFD.FocalLength][1]
#
#
# gps_info = exif_dict['GPS']
# sensor_width_mm = gps_info[piexif.ImageIFD.FocalPlaneXResolution]
# print(sensor_width_mm)
# sensor_height_mm = 24

# try:
#     img = Image.open(image_path)
#     width, height = img.size
#     img.close()
# except (IOError, OSError, AttributeError):
#     print("Error opening image or getting image dimensions.")
#     return None
#
# # Calculate GSD in centimeters per pixel
# gsd_width_cm = (sensor_width_mm / width) * (focal_length_mm / 10)  # Convert mm to cm
# gsd_height_cm = (sensor_height_mm / height) * (focal_length_mm / 10)  # Convert mm to cm
#
# # Take average of width and height GSD
# gsd_cm = (gsd_width_cm + gsd_height_cm) / 2
#
# # Convert to meters for clarity
# gsd_m = gsd_cm / 100
#
# # Update attributes dictionary
# attributes['GSD'] = round(gsd_m, 4)
#
# return None


def process_images(root_dir, temp_dir, attributes):
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith((".jpg", ".jpeg", ".png", ".JPG")):
                print()
                image_path = os.path.join(root, file)
                try:
                    Image.open(image_path)
                except PIL.UnidentifiedImageError:
                    print("Could not")
                    continue

                print(image_path)
                extract_metadata(image_path, attributes)
                get_relative_altitude(image_path, attributes)
                # get_GSD(image_path, attributes)
                # exit()
                # attributes['Reference'] = 'Ministry'
                index = "_".join(
                    [f"{value}".replace(" ", "_") for key, value in attributes.items()]
                )
                print(index)

                s = root.split("_")
                if "hiver" in s:
                    index = index + "_Winter"
                new_filename = index + ".JPG"
                new_path = os.path.join(temp_dir, new_filename)
                os.makedirs(os.path.dirname(temp_dir), exist_ok=True)
                shutil.copyfile(image_path, new_path)

        for dir in dirs:
            process_images(os.path.join(root, dir), temp_dir, attributes)


root_directory = ""
out_directory = ""
os.makedirs(out_directory, exist_ok=True)

attributes = {
    "Date Original": None,
    "Time": None,
    "Relative Altitude": None,
    "Location": None,
    "Image Size": None,
    "Make": None,
    "Camera Model": None,
}

attributes["Location"] = "Zec-Batiscan"
process_images(root_directory, out_directory, attributes)
