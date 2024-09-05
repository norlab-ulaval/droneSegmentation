import functools
import math
import os
import pathlib
import multiprocessing as mp
import random

quarter_root = '/data/drone_dataset'
splits = ['eight', 'sixteenth', 'thirtysecond', 'sixtyfourth']


def copy_to(src, dst):
    os.system(f'cp {src} {dst}')


previous_split_root = pathlib.Path('/data/Unlabeled_Quarter_v1')
for split in splits:
    train_image_path = previous_split_root / 'train' / 'images'
    train_mask_path = previous_split_root / 'train' / 'masks'
    images = sorted(list(train_image_path.glob('*.JPG')))
    masks = sorted(list(train_mask_path.glob('*.png')))
    num_images = len(images)

    print(f'{previous_split_root} has {len(images)} images and {len(masks)} masks')

    chosen_idx = random.sample(range(len(images)), math.ceil(len(images) / 2))
    images = [images[i] for i in chosen_idx]
    masks = [masks[i] for i in chosen_idx]

    split_root = pathlib.Path(f'/data/Unlabeled_{split.capitalize()}_v1')
    split_root.mkdir(parents=True, exist_ok=True)
    split_images = split_root / 'train' / 'images'
    split_images.mkdir(parents=True, exist_ok=True)
    split_masks = split_root / 'train' / 'masks'
    split_masks.mkdir(parents=True, exist_ok=True)

    # Copy
    print(f'{split_root} will have {len(images)} images and {len(masks)} masks')
    print(f'{num_images / len(images)}x reduction')
    with mp.Pool(64) as p:
        p.map(functools.partial(copy_to, dst=split_images), images)
        p.map(functools.partial(copy_to, dst=split_masks), masks)

    os.system(f'cp -r {previous_split_root}/val/ {split_root}/')

    previous_split_root = split_root
