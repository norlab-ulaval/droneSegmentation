from transformers import (
    Mask2FormerForUniversalSegmentation, Mask2FormerImageProcessor
)
from config import VIS_LABEL_MAP as LABEL_COLORS_LIST
from utils import (
    draw_segmentation_map, 
    image_overlay,
    predict
)

import argparse
import cv2
import os
import glob

parser = argparse.ArgumentParser()
parser.add_argument(
    '--input',
    help='path to the input image directory',
    default='input/inference_data/images'
)
parser.add_argument(
    '--device',
    default='cuda:0',
    help='compute device, cpu or cuda'
)
parser.add_argument(
    '--imgsz', 
    default=None,
    type=int,
    nargs='+',
    help='width, height'
)
parser.add_argument(
    '--model',
    default='outputs/model_iou'
)
args = parser.parse_args()

out_dir = 'outputs/inference_results_image'
os.makedirs(out_dir, exist_ok=True)

processor = Mask2FormerImageProcessor()
model = Mask2FormerForUniversalSegmentation.from_pretrained(args.model)
model.to(args.device).eval()

image_paths = glob.glob(os.path.join(args.input, '*'))
for image_path in image_paths:
    image = cv2.imread(image_path)
    if args.imgsz is not None:
        image = cv2.resize(image, (args.imgsz[0], args.imgsz[1]))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Get labels.
    labels = predict(model, processor, image, args.device)
    
    # Get segmentation map.
    seg_map = draw_segmentation_map(
        labels.cpu(), LABEL_COLORS_LIST
    )
    outputs = image_overlay(image, seg_map)
    cv2.imshow('Image', outputs)
    cv2.waitKey(1)
    
    # Save path.
    image_name = image_path.split(os.path.sep)[-1]
    save_path = os.path.join(
        out_dir, image_name
    )
    cv2.imwrite(save_path, outputs)