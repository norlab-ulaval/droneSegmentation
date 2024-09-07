import argparse
import functools
import multiprocessing as mp
import os
import pathlib
import random
from collections import deque

import math
import tqdm


def read_dataset(root):
    images = sorted(list(root.glob('train/images/*.JPG')))
    masks = sorted(list(root.glob('train/masks/*.png')))
    for image, mask in zip(images, masks):
        assert image.stem == mask.stem
    return images, masks


def subset_dataset(images, masks, split):
    num_images = len(images)
    chosen_idx = random.sample(range(num_images), math.ceil(num_images * split))
    subset_images = [images[i] for i in chosen_idx]
    subset_masks = [masks[i] for i in chosen_idx]
    for image, mask in zip(subset_images, subset_masks):
        assert image.stem == mask.stem
    return subset_images, subset_masks


def copy_to(src, dst):
    os.system(f'cp {src} {dst}')


def write_dataset(images, masks, out):
    out_images = out / 'train' / 'images'
    out_images.mkdir(parents=True, exist_ok=True)
    out_masks = out / 'train' / 'masks'
    out_masks.mkdir(parents=True, exist_ok=True)

    with mp.Pool(64) as p:
        print('Copying images...')
        it = tqdm.tqdm(p.imap(functools.partial(copy_to, dst=out_images), images), total=len(images))
        deque(it, maxlen=0)
        print('Copying masks...')
        it = tqdm.tqdm(p.imap(functools.partial(copy_to, dst=out_masks), masks), total=len(masks))
        deque(it, maxlen=0)

    os.system(f'cp -r {full_root}/val/ {out}/')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--full_root', type=str, required=True, help='Path to the full dataset')
    parser.add_argument('--out', type=str, required=True, help='Path to the output directory')
    parser.add_argument('--split', type=str, required=True, help='Split factor (e.g. 1/2, 1/4, 1/8)')
    parser.add_argument('--dry_run', action='store_true', help='Dry run')
    args = parser.parse_args()

    full_root = pathlib.Path(args.full_root)
    out = pathlib.Path(args.out)
    split = eval(args.split)
    dry_run = args.dry_run

    images, masks = read_dataset(full_root)
    subset_images, subset_masks = subset_dataset(images, masks, split)

    print(f'{len(subset_images)} images and {len(subset_masks)} masks will be copied to {out}')
    print('Should reduce the dataset by', split, 'times')
    print(f'{len(subset_images) / len(images)}x reduction')
    print(f'Size of the dataset: {len(subset_images)}')

    if dry_run:
        print('Dry running...')
        exit(0)

    write_dataset(subset_images, subset_masks, out)
