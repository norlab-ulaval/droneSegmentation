# import sys
# sys.path.insert(0, "Mask2Former")
# import tempfile
# from pathlib import Path
# import numpy as np
# import cv2
# # import cog
#
# # import some common detectron2 utilities
# from detectron2.config import CfgNode as CN
# from detectron2.engine import DefaultPredictor
# from detectron2.config import get_cfg
# from detectron2.utils.visualizer import Visualizer, ColorMode
# from detectron2.data import MetadataCatalog
# from detectron2.projects.deeplab import add_deeplab_config
#
# # import Mask2Former project
# from mask2former import add_maskformer2_config
#
#
# class Predictor():
#     def setup(self):
#         cfg = get_cfg()
#         add_deeplab_config(cfg)
#         add_maskformer2_config(cfg)
#         cfg.merge_from_file("/home/kamyar/PycharmProjects/droneSegmentation/lowAltitude_segmentation/Mask2Former/configs/Drone_regrowth/semantic-segmentation/swin/maskformer2_swin_large_IN21k_384_bs16_160k_res640.yaml")
#         cfg.MODEL.WEIGHTS = '/home/kamyar/PycharmProjects/droneSegmentation/lowAltitude_segmentation/Mask2Former/output/model_0104999.pth'
#         cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON = True
#         cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON = False
#         cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON = False
#         self.predictor = DefaultPredictor(cfg)
#         self.metadata = MetadataCatalog.get("drone_dataset_sem_seg_val")
#
#
#     # @cog.input(
#     #     "image",
#     #     type=Path,
#     #     help="Input image for segmentation. Output will be the concatenation of Panoptic segmentation (top), "
#     #          "instance segmentation (middle), and semantic segmentation (bottom).",
#     # )
#     def predict(self, image):
#         im = cv2.imread(str(image))
#         outputs = self.predictor(im)
#         # v = Visualizer(im[:, :, ::-1], self.coco_metadata, scale=1.2, instance_mode=ColorMode.IMAGE_BW)
#         # panoptic_result = v.draw_panoptic_seg(outputs["panoptic_seg"][0].to("cpu"),
#         #                                       outputs["panoptic_seg"][1]).get_image()
#         # v = Visualizer(im[:, :, ::-1], self.coco_metadata, scale=1.2, instance_mode=ColorMode.IMAGE_BW)
#         # instance_result = v.draw_instance_predictions(outputs["instances"].to("cpu")).get_image()
#         v = Visualizer(im[:, :, ::-1], self.metadata, scale=1.2, instance_mode=ColorMode.IMAGE_BW)
#         semantic_result = v.draw_sem_seg(outputs["sem_seg"].argmax(0).to("cpu")).get_image()
#         # result = np.concatenate((semantic_result), axis=0)[:, :, ::-1]
#         out_path = "out_1_HA.png"
#         cv2.imwrite(str(out_path), semantic_result)
#         # return out_path
#
#
# predictor = Predictor()
# predictor.setup()
# predictor.predict('/home/kamyar/Desktop/50m_1x_Test.JPG')
# print("done")


import sys
import os
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
from mask2former import add_maskformer2_config


class Predictor():
    def setup(self):
        cfg = get_cfg()
        add_deeplab_config(cfg)
        add_maskformer2_config(cfg)

        cfg['SOLVER']['BEST_CHECKPOINTER'] = CfgNode()
        cfg['SOLVER']['BEST_CHECKPOINTER']['ENABLED'] = False
        cfg['SOLVER']['BEST_CHECKPOINTER']['METRIC'] = 'yolo'

        cfg.merge_from_file(
            "lowAltitude_segmentation/Mask2Former/configs/Drone_regrowth/semantic-segmentation/swin/M2F_Swin_Large_base.yaml", allow_unsafe=True)

        cfg.MODEL.WEIGHTS = '/home/kamyar/Documents/M2F_Results/Scaling/1.33/model_best.pth'
        cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON = True
        cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON = False
        cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON = False
        self.predictor = DefaultPredictor(cfg)
        self.metadata = MetadataCatalog.get("drone_dataset_sem_seg_val")

    def predict(self, image_path, output_path):
        im = cv2.imread(str(image_path))
        outputs = self.predictor(im)
        cls = outputs["sem_seg"].argmax(0)
        # plt.imshow(cls.cpu().numpy())
        # plt.show()
        # cls[cls > 0] -= 1
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


input_directory = '/home/kamyar/Documents/Test_Annotated'
output_directory = '/home/kamyar/Documents/M2F_Results/Scaling/1.33/output_test'

process_images(input_directory, output_directory)
print("done")
