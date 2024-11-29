import os

def find_min_max_and_second_min_max_folders(parent_folder):
    # Dictionary to store folder names and their image count
    folder_image_count = {}

    # Loop through all subfolders
    for root, dirs, files in os.walk(parent_folder):
        # Count images in each subfolder
        image_count = sum(1 for file in files if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff')))
        if image_count > 0:  # Only include folders with images
            folder_image_count[root] = image_count

    if len(folder_image_count) < 2:
        return None  # Not enough folders with images

    # Sort the folders by image count
    sorted_folders = sorted(folder_image_count.items(), key=lambda x: x[1])

    # Get the min, second min, max, and second max
    min_folder, min_count = sorted_folders[0]
    second_min_folder, second_min_count = sorted_folders[1]
    max_folder, max_count = sorted_folders[-1]
    second_max_folder, second_max_count = sorted_folders[-2]

    return {
        "max": (max_folder, max_count),
        "second_max": (second_max_folder, second_max_count),
        "min": (min_folder, min_count),
        "second_min": (second_min_folder, second_min_count)
    }

# Example usage
parent_folder = "/home/kamyar/Documents/iNat_Classifier_filtered"  # Replace with your folder path
result = find_min_max_and_second_min_max_folders(parent_folder)

if result:
    print(f"Folder with maximum images: {result['max'][0]} ({result['max'][1]} images)")
    print(f"Folder with second maximum images: {result['second_max'][0]} ({result['second_max'][1]} images)")
    print(f"Folder with minimum images: {result['min'][0]} ({result['min'][1]} images)")
    print(f"Folder with second minimum images: {result['second_min'][0]} ({result['second_min'][1]} images)")
else:
    print("Not enough folders with images to determine second max/min.")
