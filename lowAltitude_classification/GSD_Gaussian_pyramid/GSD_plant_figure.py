import os
import matplotlib.pyplot as plt
from PIL import Image
from matplotlib.lines import Line2D
from gsd_utils import papermode

papermode(plt=plt, size=12)

blurred_image_folder = "blured_images"
downsampled_image_folder = "downsampled_images"

image_files = ["125", "250", "500", "1000", "2000"]
sizes_values = ["256x256", "128x128", "64x64", "32x32", "16x16"]
sigma_values = ["0", "1", "2", "4", "8"]

fig, axs = plt.subplots(2, len(image_files), figsize=(15, 7))

for i, (filename, gsd_label) in enumerate(zip(image_files, sizes_values)):
    blurred_image_path = os.path.join(blurred_image_folder, filename + ".jpg")
    blurred_image = Image.open(blurred_image_path)
    blurred_patch = blurred_image.crop((blurred_image.width - 256, 0, blurred_image.width, 256))
    axs[0, i].imshow(blurred_patch)
    axs[0, i].axis('off')

    downsampled_image_path = os.path.join(downsampled_image_folder, filename + ".jpg")
    downsampled_image = Image.open(downsampled_image_path)
    size = 256 // (2 ** i)  
    downsampled_patch = downsampled_image.crop((downsampled_image.width - size, 0, downsampled_image.width, size))
    axs[1, i].imshow(downsampled_patch)
    axs[1, i].axis('off')

for i, size_value in enumerate(sizes_values):
    axs[1, i].text(
        0.5, -0.12, size_value, fontsize=26, ha='center', transform=axs[1, i].transAxes
    )

for i, sigma in enumerate(sigma_values):
    axs[0, i].text(
        0.5, -0.12, "\(\sigma\) = " + sigma, fontsize=26, ha='center', transform=axs[0, i].transAxes
    )

axs[0, 0].text(0.5, 1.03, "Original image", fontsize=24, ha="center", transform=axs[0, 0].transAxes)

axs[0, 0].text(
    -0.13, 0.5, "Blurred", fontsize=26, va="center", rotation=90, transform=axs[0, 0].transAxes
)
axs[1, 0].text(
    -0.13, 0.5, "Downsampled", fontsize=26, va="center", rotation=90, transform=axs[1, 0].transAxes
)

plt.tight_layout()
plt.subplots_adjust(top=0.99, bottom=0.01, wspace=0.05)  

output_path = "comparison_blur_downsample.pdf"
plt.savefig(output_path, dpi=300)
plt.show()

