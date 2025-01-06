import os
from pathlib import Path
import shutil

def get_normalized_file_base_names(folder_path, suffix_to_remove=""):
    """Extract base file names (without extensions), removing any specified suffix."""
    base_names = set()
    for file in os.listdir(folder_path):
        if os.path.isfile(os.path.join(folder_path, file)):
            stem = Path(file).stem
            # Remove the suffix if it exists
            if suffix_to_remove and stem.endswith(suffix_to_remove):
                stem = stem[: -len(suffix_to_remove)]
            base_names.add(stem)
    return base_names

def copy_common_files(folder1, folder2, output_folder, suffix_to_remove=""):
    """Copy only files with common normalized base names to a new folder."""
    # Normalize base names from both folders
    base_names1 = get_normalized_file_base_names(folder1)
    base_names2 = get_normalized_file_base_names(folder2, suffix_to_remove=suffix_to_remove)

    # Find common base names
    common_base_names = base_names1 & base_names2

    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Process files in both folders
    for folder in [folder1, folder2]:
        for file in os.listdir(folder):
            file_path = os.path.join(folder, file)
            if os.path.isfile(file_path):
                stem = Path(file).stem
                # Normalize the stem for comparison
                if suffix_to_remove and stem.endswith(suffix_to_remove):
                    stem = stem[: -len(suffix_to_remove)]
                if stem in common_base_names:
                    # Copy the file to the output folder
                    output_path = os.path.join(output_folder, file)
                    if not os.path.exists(output_path):  # Avoid overwriting files
                        shutil.copy(file_path, output_path)

# Example usage
folder1 = "/home/kamyar/Documents/FIG_Qualitative_Journal/images"
folder2 = "/home/kamyar/Documents/FIG_Qualitative_Journal/annotations"
output_folder = "/home/kamyar/Documents/FIG_Qualitative_Journal/annotations_"

# If files in folder2 have the "-label" suffix, specify it
suffix_to_remove = "-label-ground-truth-semantic"

copy_common_files(folder1, folder2, output_folder, suffix_to_remove)

