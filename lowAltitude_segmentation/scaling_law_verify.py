import pathlib

import math


def read_dataset(root):
    images = sorted(list(root.glob('train/images/*.JPG')))
    masks = sorted(list(root.glob('train/masks/*.png')))
    for image, mask in zip(images, masks):
        assert image.stem == mask.stem
    return images, masks


if __name__ == '__main__':
    full_root = pathlib.Path('~/Datasets/drone_dataset').expanduser()
    data_path = pathlib.Path('~/Datasets').expanduser()

    full_images, full_masks = read_dataset(full_root)
    total_images = len(full_images)
    assert total_images == len(full_masks)

    paths = data_path.glob('drone_dataset_[0-9].[0-9]*')
    for path in paths:
        print('-' * 80)
        name = str(path).split('/')[-1]
        print('Name:', name)
        split = name.split('_')[-1].replace('.', '/')
        print('Split:', split)
        split = eval(split)
        print('Expected split:', split)
        print('Expected number of images:\t', math.ceil(total_images * split))
        images, masks = read_dataset(path)
        print('Actual number of images:\t', len(images))

        if len(images) != math.ceil(total_images * split):
            print('Mismatch!')
            raise ValueError(f'Error in {name}')
