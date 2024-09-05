import functools
import os
import pathlib
import multiprocessing as mp

pl1_root = '/data/drone_dataset'
splits = ['half', 'quarter']


def copy_to(src, dst):
    os.system(f'cp {src} {dst}')


for split in splits:
    root = f'/data/Unlabeled_{split.capitalize()}'

    # Content to copy
    train_images = pathlib.Path(root) / 'train' / 'images'
    train_masks = pathlib.Path(root) / 'train' / 'masks'
    images = sorted(list(train_images.glob('*.JPG')))
    masks = sorted(list(train_masks.glob('*.png')))
    print(f'Found {len(images)} images and {len(masks)} masks for {split}')
    assert len(images) == len(masks), f'Number of images and masks do not match for {split}'

    # Where to copy
    split_root = pathlib.Path(f'/data/Unlabeled_{split.capitalize()}_v1')
    split_root.mkdir(parents=True, exist_ok=True)
    split_images = split_root / 'train' / 'images'
    split_images.mkdir(parents=True, exist_ok=True)
    split_masks = split_root / 'train' / 'masks'
    split_masks.mkdir(parents=True, exist_ok=True)

    # Copy
    with mp.Pool(64) as p:
        p.map(functools.partial(copy_to, dst=split_images), images)
        p.map(functools.partial(copy_to, dst=split_masks), masks)

    os.system(f'cp -r {root}/val/ {split_root}/')
