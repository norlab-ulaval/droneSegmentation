
# Copyright (c) Facebook, Inc. and its affiliates.
import os

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import load_sem_seg

DRONE_SEM_SEG_CATEGORIES = [
    {
        "color": [0, 128, 0],
        "instances": False,
        "readable": "American Mountain-Ash",
        "name": "American Mountain-Ash",
        "evaluate": True,
    },
    {
        "color": [128, 128, 128],
        "instances": False,
        "readable": "Background",
        "name": "Background",
        "evaluate": False,
    },
    {
        "color": [0, 0, 128],
        "instances": False,
        "readable": "Black Spruce",
        "name": "Black Spruce",
        "evaluate": True,
    },
    {
        "color": [0, 0, 255],
        "instances": False,
        "readable": "Blueberry",
        "name": "Blueberry",
        "evaluate": True,
    },
    {
        "color": [0, 255, 0],
        "instances": False,
        "readable": "Bog Labrador Tea",
        "name": "Bog Labrador Tea",
        "evaluate": True,
    },
    {
        "color": [128, 0, 0],
        "instances": False,
        "readable": "Boulders",
        "name": "Boulders",
        "evaluate": True,
    },
    {
        "color": [255, 0, 0],
        "instances": False,
        "readable": "Canada Yew",
        "name": "Canada Yew",
        "evaluate": True,
    },
    {
        "color": [128, 128, 0],
        "instances": False,
        "readable": "Dead Trees",
        "name": "Dead Trees",
        "evaluate": True,
    },
    {
        "color": [255, 255, 0],
        "instances": False,
        "readable": "Fern",
        "name": "Fern",
        "evaluate": True,
    },
    {
        "color": [0, 255, 255],
        "instances": False,
        "readable": "Fir",
        "name": "Fir",
        "evaluate": True,
    },
    {
        "color": [255, 0, 255],
        "instances": False,
        "readable": "Fire Cherry",
        "name": "Fire Cherry",
        "evaluate": True,
    },
    {
        "color": [192, 192, 192],
        "instances": False,
        "readable": "Jack Pine",
        "name": "Jack Pine",
        "evaluate": True,
    },
    {
        "color": [128, 0, 128],
        "instances": False,
        "readable": "Leatherleaf",
        "name": "Leatherleaf",
        "evaluate": True,
    },
    {
        "color": [0, 128, 128],
        "instances": False,
        "readable": "Moss",
        "name": "Moss",
        "evaluate": True,
    },
    {
        "color": [128, 128, 0],
        "instances": False,
        "readable": "Mountain Maple",
        "name": "Mountain Maple",
        "evaluate": True,
    },
    {
        "color": [255, 128, 0],
        "instances": False,
        "readable": "Paper Birch",
        "name": "Paper Birch",
        "evaluate": True,
    },
    {
        "color": [128, 255, 0],
        "instances": False,
        "readable": "Red Maple",
        "name": "Red Maple",
        "evaluate": True,
    },
    {
        "color": [128, 0, 255],
        "instances": False,
        "readable": "Red Raspberry",
        "name": "Red Raspberry",
        "evaluate": True,
    },
    {
        "color": [255, 0, 128],
        "instances": False,
        "readable": "Sedges",
        "name": "Sedges",
        "evaluate": True,
    },
    {
        "color": [255, 128, 128],
        "instances": False,
        "readable": "Serviceberries",
        "name": "Serviceberries",
        "evaluate": True,
    },
    {
        "color": [0, 128, 255],
        "instances": False,
        "readable": "Sheep Laurel",
        "name": "Sheep Laurel",
        "evaluate": True,
    },
    {
        "color": [128, 128, 255],
        "instances": False,
        "readable": "Trembling Aspen",
        "name": "Trembling Aspen",
        "evaluate": True,
    },
    {
        "color": [0, 255, 128],
        "instances": False,
        "readable": "Viburnum",
        "name": "Viburnum",
        "evaluate": True,
    },
    {
        "color": [255, 255, 128],
        "instances": False,
        "readable": "Willowherbs",
        "name": "Willowherbs",
        "evaluate": True,
    },
    {
        "color": [128, 255, 255],
        "instances": False,
        "readable": "Wood",
        "name": "Wood",
        "evaluate": True,
    },
    {
        "color": [255, 128, 255],
        "instances": False,
        "readable": "Yellow Birch",
        "name": "Yellow Birch",
        "evaluate": True,
    }
]




def _get_mapillary_vistas_meta():
    stuff_classes = [k["readable"] for k in DRONE_SEM_SEG_CATEGORIES if k["evaluate"]]
    assert len(stuff_classes) == 25

    stuff_colors = [k["color"] for k in DRONE_SEM_SEG_CATEGORIES if k["evaluate"]]
    assert len(stuff_colors) == 25

    ret = {
        "stuff_classes": stuff_classes,
        "stuff_colors": stuff_colors,
    }
    return ret


def register_all_mapillary_vistas(root):
    meta = _get_mapillary_vistas_meta()
    for name, dirname in [("train", "train"), ("val", "val")]:
        image_dir = os.path.join(root, dirname, "images")
        gt_dir = os.path.join(root, dirname, "masks")
        name = f"drone_dataset_sem_seg_{name}"
        DatasetCatalog.register(
            name, lambda x=image_dir, y=gt_dir: load_sem_seg(y, x, gt_ext="png", image_ext="JPG")
        )
        MetadataCatalog.get(name).set(
            image_root=image_dir,
            sem_seg_root=gt_dir,
            evaluator_type="sem_seg",
            ignore_label=255,
            **meta
        )


_root = "/home/kamyar/Documents/Dataset_mask2former"
register_all_mapillary_vistas(_root)
