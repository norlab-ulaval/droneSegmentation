import sys
import os
import torch

import cv2
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from detectron2.config import CfgNode as CN, CfgNode
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog
from detectron2.projects.deeplab import add_deeplab_config
from fontTools.unicodedata import script

from mask2former import add_maskformer2_config


class Predictor:
    def setup(self):
        cfg = get_cfg()
        add_deeplab_config(cfg)
        add_maskformer2_config(cfg)

        cfg["SOLVER"]["BEST_CHECKPOINTER"] = CfgNode()
        cfg["SOLVER"]["BEST_CHECKPOINTER"]["ENABLED"] = False
        cfg["SOLVER"]["BEST_CHECKPOINTER"]["METRIC"] = "yolo"

        cfg.merge_from_file(
            "lowAltitude_segmentation/Mask2Former/configs/Drone_regrowth/semantic-segmentation/swin/M2F_Swin_Large_base_ignore255.yaml",
            allow_unsafe=True,
        )

        cfg.MODEL.WEIGHTS = (
            "results/M2F_Results/Scaling/scaling_1.2_run2/model_best.pth"
        )
        cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON = True
        cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON = False
        cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON = False
        self.predictor = DefaultPredictor(cfg)
        self.metadata = MetadataCatalog.get("drone_dataset_sem_seg_ignore255_val")

    def predict(self, image_path, output_path):
        im = cv2.imread(str(image_path))
        outputs = self.predictor(im)
        cls = outputs["sem_seg"].argmax(0)

        # v = Visualizer(im[:, :, ::-1], self.metadata, scale=1.2, instance_mode=ColorMode.IMAGE)
        # semantic_result = v.draw_sem_seg(cls.to("cpu")).get_image()
        cv2.imwrite(output_path, cls.to("cpu").numpy())
        # cv2.imwrite(output_path, semantic_result)


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


input_directory = "data/Test_Annotated"
output_directory = "results/M2F_Results/Scaling/scaling_1.2_run2/output_test"

process_images(input_directory, output_directory)
print("done")
