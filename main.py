import os

def rename_jpg_files(folder_path):
    try:
        # Iterate over all files in the given folder
        for filename in os.listdir(folder_path):
            # Check if the file has the .JPG extension
            if filename.endswith('.JPG'):
                # Create the old and new file paths
                old_file = os.path.join(folder_path, filename)
                new_file = os.path.join(folder_path, filename.replace('.JPG', '.jpg'))

                # Rename the file
                os.rename(old_file, new_file)
                print(f"Renamed: {old_file} -> {new_file}")

        print("All .JPG files have been renamed to .jpg.")

    except Exception as e:
        print(f"An error occurred: {e}")

# Example usage
folder_path = "/home/kamyar/Documents/M2F_pretrain_data/train/images"  # Replace with your folder path
rename_jpg_files(folder_path)
