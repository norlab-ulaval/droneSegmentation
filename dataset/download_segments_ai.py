from segments import SegmentsClient, SegmentsDataset
from segments.utils import export_dataset

release = '/home/kamyar/Downloads/Data_Labeling_species-5th export.json'
dataset = SegmentsDataset(release, labelset='ground-truth', filter_by=['labeled', 'reviewed'])
# export_dataset(dataset, export_format='semantic-color', export_folder='/home/kamyar/PycharmProjects/droneSegmentation/segments_color')

import matplotlib.pyplot as plt
from segments.utils import get_semantic_bitmap

for sample in dataset:
    # Print the sample name and list of labeled objects
    print(sample['name'])
    print(sample['annotations'])

    # Show the image
    plt.imshow(sample['image'])
    plt.show()

    # Show the instance segmentation label
    plt.imshow(sample['segmentation_bitmap'])
    plt.show()

    # Show the semantic segmentation label
    semantic_bitmap = get_semantic_bitmap(sample['segmentation_bitmap'], sample['annotations'])
    plt.imshow(semantic_bitmap)
    plt.show()