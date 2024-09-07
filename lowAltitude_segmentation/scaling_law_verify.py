import pathlib


def read_dataset(root):
    images = sorted(list(root.glob('train/images/*.JPG')))
    masks = sorted(list(root.glob('train/masks/*.png')))
    for image, mask in zip(images, masks):
        assert image.stem == mask.stem
    return images, masks


if __name__ == '__main__':
    full_root = pathlib.Path('/data/drone_dataset')
    data_path = pathlib.Path('/data/')

    full_images, full_masks = read_dataset(full_root)
    total_images = len(full_images)
    assert total_images == len(full_masks)

    splits = data_path.glob('drone_dataset_[0-9].[0-9]*')
    for split in splits:
        print('-' * 80)
        name = str(split).split('/')[-1]
        print('Name:', name)
        split = name.split('_')[-1].replace('.', '/')
        print('Split:', split)
        split = eval(split)
        print('Expected split:', split)
        print('Expected number of images:', total_images * split)
        images, masks = read_dataset(split)
        print('Actual number of images:', len(images))
