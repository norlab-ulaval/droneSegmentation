from PIL import Image
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from gsd_utils import papermode

papermode(plt=plt, size=12)


def add_images_to_map(images_directory):
    class_counts = np.zeros(24, dtype=int)

    for filename in os.listdir(images_directory):
        image = np.array(
            Image.open(os.path.join(images_directory, filename)).convert("L")
        )

        unique, counts = np.unique(image, return_counts=True)
        class_counts[unique] += counts

    return class_counts


# test_images_directory = '/data/droneseg/Annotated_data_split/test/masks/'
# trainval_images_directory = '/data/droneseg/Annotated_data_split/train-val/masks/'
# pseudo_labels_images_directory = '/data/droneseg/M2F_pretrain_data/train/masks/'

# test_class_counts = add_images_to_map(test_images_directory)
# trainval_class_counts = add_images_to_map(trainval_images_directory)
# pseudo_labels_class_counts = add_images_to_map(pseudo_labels_images_directory)

# np.save('test_count.npy', test_class_counts)
# np.save('train_val_count.npy', trainval_class_counts)
# np.save('pseudo_count.npy', pseudo_labels_class_counts)

test_class_counts = np.load("test_count.npy")
trainval_class_counts = np.load("train_val_count.npy")
pseudo_labels_class_counts = np.load("pseudo_count.npy")

print(pseudo_labels_class_counts)


def read_class_labels(file_path):
    with open(file_path, "r") as f:
        labels = [line.split(":")[0].strip() for line in f.readlines()]
    return labels


def read_common_to_latin(file_path):
    with open(file_path, "r") as f:
        mapping = {}
        for line in f:
            common, latin = line.split(":")
            mapping[common.strip()] = latin.strip()
    return mapping


class_labels = read_class_labels("label_to_id.txt")
print(class_labels)
common_to_latin_mapping = read_common_to_latin("common_latin_map.txt")
print(common_to_latin_mapping)

test_pixel_percentage = test_class_counts / np.sum(test_class_counts)
trainval_pixel_percentage = trainval_class_counts / np.sum(trainval_class_counts)
pseudo_labels_pixel_percentage = pseudo_labels_class_counts / np.sum(
    pseudo_labels_class_counts
)

sorted_indices = np.argsort(test_pixel_percentage)[::-1]

sorted_class_labels = [class_labels[i] for i in sorted_indices]

sorted_test_pixel_percentage = test_pixel_percentage[sorted_indices]
sorted_trainval_pixel_percentage = trainval_pixel_percentage[sorted_indices]
sorted_pseudo_labels_pixel_percentage = pseudo_labels_pixel_percentage[sorted_indices]

bar_width = 0.20
indices = np.arange(len(sorted_class_labels))

sorted_latin_names = [
    (
        "\n".join(common_to_latin_mapping[name].split())
        if all(len(word) > 7 for word in common_to_latin_mapping[name].split())
        else common_to_latin_mapping[name]
    )
    if name in common_to_latin_mapping.keys()
    else name
    for name in sorted_class_labels
]
sorted_latin_names = [i if i != "Background" else "Other" for i in sorted_latin_names]

plt.figure(figsize=(16, 10), dpi=300)
plt.bar(
    indices - bar_width,
    sorted_pseudo_labels_pixel_percentage,
    width=bar_width,
    color="green",
    label="Pseudo-labels (with voting)",
)
plt.bar(
    indices,
    sorted_trainval_pixel_percentage,
    width=bar_width,
    color="blue",
    label="Train/val (annotations)",
)
plt.bar(
    indices + bar_width,
    sorted_test_pixel_percentage,
    width=bar_width,
    color="red",
    label="Test (annotations)",
)


def no_percent_formatter(x, pos):
    return f"{x * 100:.0f}"


plt.gca().yaxis.set_major_formatter(FuncFormatter(no_percent_formatter))
plt.ylabel("Percentage of Pixels", fontsize=24)
plt.xticks(indices, sorted_latin_names, rotation=90, fontsize=22)
plt.tick_params(axis="y", labelsize=20)
plt.legend(fontsize=24)
plt.tight_layout()
plt.savefig("distribution.pdf")
plt.show()
