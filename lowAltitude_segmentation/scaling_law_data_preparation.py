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


full_root = pathlib.Path('/data/drone_dataset')
half_root = pathlib.Path('/data/Unlabeled_Half_v1')
new_split = pathlib.Path('/data/Unlabeled_1p5')
new_split.mkdir(parents=True, exist_ok=True)

set_full_images = set(full_root.glob('train/images/*.JPG'))
set_full_masks = set(full_root.glob('train/masks/*.png'))
set_half_images = set(half_root.glob('train/images/*.JPG'))
set_half_masks = set(half_root.glob('train/masks/*.png'))

set_images_not_in_half = set_full_images - set_half_images
set_masks_not_in_half = set_full_masks - set_half_masks

to_add_idx = random.sample(range(len(set_images_not_in_half)), math.ceil(len(set_images_not_in_half) / 2))
set_images_to_add = [list(set_images_not_in_half)[i] for i in to_add_idx]
set_masks_to_add = [list(set_masks_not_in_half)[i] for i in to_add_idx]

set_images_new_split = set_half_images.union(set_images_to_add)
set_masks_new_split = set_half_masks.union(set_masks_to_add)

with mp.Pool(64) as p:
    p.map(functools.partial(copy_to, dst=new_split / 'train' / 'images'), set_images_to_add)
    p.map(functools.partial(copy_to, dst=new_split / 'train' / 'masks'), set_masks_to_add)

os.system(f'cp -r {half_root}/val/ {new_split}/')

# previous_split_root = pathlib.Path('/data/Unlabeled_Quarter_v1')
# for split in splits:
#     train_image_path = previous_split_root / 'train' / 'images'
#     train_mask_path = previous_split_root / 'train' / 'masks'
#     images = sorted(list(train_image_path.glob('*.JPG')))
#     masks = sorted(list(train_mask_path.glob('*.png')))
#     num_images = len(images)
#
#     print(f'{previous_split_root} has {len(images)} images and {len(masks)} masks')
#
#     chosen_idx = random.sample(range(len(images)), math.ceil(len(images) / 2))
#     images = [images[i] for i in chosen_idx]
#     masks = [masks[i] for i in chosen_idx]
#
#     split_root = pathlib.Path(f'/data/Unlabeled_{split.capitalize()}_v1')
#     split_root.mkdir(parents=True, exist_ok=True)
#     split_images = split_root / 'train' / 'images'
#     split_images.mkdir(parents=True, exist_ok=True)
#     split_masks = split_root / 'train' / 'masks'
#     split_masks.mkdir(parents=True, exist_ok=True)
#
#     # Copy
#     print(f'{split_root} will have {len(images)} images and {len(masks)} masks')
#     print(f'{num_images / len(images)}x reduction')
#     with mp.Pool(64) as p:
#         p.map(functools.partial(copy_to, dst=split_images), images)
#         p.map(functools.partial(copy_to, dst=split_masks), masks)
#
#     os.system(f'cp -r {previous_split_root}/val/ {split_root}/')
#
#     previous_split_root = split_root
