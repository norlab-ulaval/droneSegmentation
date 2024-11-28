import os
import torch
import cv2
from detectron2.config import CfgNode as CN, CfgNode
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog
from detectron2.projects.deeplab import add_deeplab_config
from fontTools.unicodedata import script

from mask2former import add_maskformer2_config


class Predictor():
    def setup(self):
        self.metadata = MetadataCatalog.get("drone_dataset_sem_seg_val")

    def predict(self, image_path, output_path):
        im = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)

        v = Visualizer(im[:, :], self.metadata, scale=1.2, instance_mode=ColorMode.IMAGE)
        semantic_result = v.draw_sem_seg(im).get_image()
        cv2.imwrite(output_path, semantic_result)


def process_images(input_dir, output_dir):
    predictor = Predictor()
    predictor.setup()

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for image_file in os.listdir(input_dir):
        input_path = os.path.join(input_dir, image_file)
        file_name, _ = os.path.splitext(image_file)
        output_path = os.path.join(output_dir, f"{file_name}.png")
        predictor.predict(input_path, output_path)
        print(f"Processed and saved: {output_path}")


input_directory = '/home/kamyar/Documents/Train-val_Annotated_masks'
output_directory = '/home/kamyar/Documents/Train-val_annotations_colorful'

process_images(input_directory, output_directory)
print("done")
