
# Copyright (c) Facebook, Inc. and its affiliates.
import os

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import load_sem_seg

DRONE_SEM_SEG_CATEGORIES = [
    {
        "color": [0, 0, 0],
        "instances": False,
        "readable": "Background",
        "name": "Background",
        "evaluate": False,
    },
    {
        "color": [255, 0, 0],
        "instances": False,
        "readable": "Alders",
        "name": "Alders",
        "evaluate": True,
    },
    {
        "color": [255, 165, 0],
        "instances": False,
        "readable": "Black spruce",
        "name": "Black spruce",
        "evaluate": True,
    },
    {
        "color": [255, 255, 0],
        "instances": False,
        "readable": "Bog Labrador Tea",
        "name": "Bog Labrador Tea",
        "evaluate": True,
    },
    {
        "color": [0, 128, 0],
        "instances": False,
        "readable": "Common Haircap Moss",
        "name": "Common Haircap Moss",
        "evaluate": True,
    },
    {
        "color": [0, 0, 255],
        "instances": False,
        "readable": "Cornflower (bleuets)",
        "name": "Cornflower (bleuets)",
        "evaluate": True,
    },
    {
        "color": [75, 0, 130],
        "instances": False,
        "readable": "Feather Mosses",
        "name": "Feather Mosses",
        "evaluate": True,
    },
    {
        "color": [238, 130, 238],
        "instances": False,
        "readable": "Ferns (Fougères)",
        "name": "Ferns (Fougères)",
        "evaluate": True,
    },
    {
        "color": [255, 192, 203],
        "instances": False,
        "readable": "Fir",
        "name": "Fir",
        "evaluate": True,
    },
    {
        "color": [165, 42, 42],
        "instances": False,
        "readable": "Fire Cherry (Cerisier de pensylvanie)",
        "name": "Fire Cherry (Cerisier de pensylvanie)",
        "evaluate": True,
    },
    {
        "color": [0, 0, 0],
        "instances": False,
        "readable": "Jack pine",
        "name": "Jack pine",
        "evaluate": True,
    },
    {
        "color": [255, 255, 255],
        "instances": False,
        "readable": "Larch",
        "name": "Larch",
        "evaluate": True,
    },
    {
        "color": [128, 128, 128],
        "instances": False,
        "readable": "Leatherleaf (Cassandre)",
        "name": "Leatherleaf (Cassandre)",
        "evaluate": True,
    },
    {
        "color": [220, 20, 60],
        "instances": False,
        "readable": "Mountain Maple (Érable à épis)",
        "name": "Mountain Maple (Érable à épis)",
        "evaluate": True,
    },
    {
        "color": [64, 224, 208],
        "instances": False,
        "readable": "Paper birch",
        "name": "Paper birch",
        "evaluate": True,
    },
    {
        "color": [0, 128, 128],
        "instances": False,
        "readable": "Pixie Cup Lichens (Cladonia)",
        "name": "Pixie Cup Lichens (Cladonia)",
        "evaluate": True,
    },
    {
        "color": [0, 255, 0],
        "instances": False,
        "readable": "Red raspberry (Framboisier)",
        "name": "Red raspberry (Framboisier)",
        "evaluate": True,
    },
    {
        "color": [255, 127, 80],
        "instances": False,
        "readable": "Sedges (Carex)",
        "name": "Sedges (Carex)",
        "evaluate": True,
    },
    {
        "color": [0, 0, 128],
        "instances": False,
        "readable": "Serviceberries (amélanchier)",
        "name": "Serviceberries (amélanchier)",
        "evaluate": True,
    },
    {
        "color": [255, 0, 255],
        "instances": False,
        "readable": "Sheep Laurel (Kalmia)",
        "name": "Sheep Laurel (Kalmia)",
        "evaluate": True,
    },
    {
        "color": [255, 215, 0],
        "instances": False,
        "readable": "Sphagnum Mosses",
        "name": "Sphagnum Mosses",
        "evaluate": True,
    },
    {
        "color": [192, 192, 192],
        "instances": False,
        "readable": "Trembling aspen",
        "name": "Trembling aspen",
        "evaluate": True,
    },
    {
        "color": [230, 230, 250],
        "instances": False,
        "readable": "Viburnums",
        "name": "Viburnums",
        "evaluate": True,
    },
    {
        "color": [128, 0, 128],
        "instances": False,
        "readable": "Willowherbs (Épilobe)",
        "name": "Willowherbs (Épilobe)",
        "evaluate": True,
    },
    {
        "color": [0, 255, 255],
        "instances": False,
        "readable": "Willows",
        "name": "Willows",
        "evaluate": True,
    },
    {
        "color": [189, 252, 201],
        "instances": False,
        "readable": "yellow birch",
        "name": "yellow birch",
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
            ignore_label=0,
            **meta
        )


_root = "/home/kamyar/Documents/DataSegmentation_split"
register_all_mapillary_vistas(_root)
