import os

import h5py
import numpy as np
from torch.utils.data import Dataset, DataLoader


# paths to data directories
_DATA_DIR = {
    'RigidFall': './data_RigidFall/',
    'MassRope': './data_MassRope/',
}

# number of frames dropped from beginning of each video
_N_DROP_FRAMES = {
    'RigidFall': 0,
    'MassRope': 20,
}

# image mean and std for standardization of all datasets
_IMG_MEAN = 0.575
_IMG_STD = 0.375


def _load_data(path):
    """Load h5py data files from specified path."""
    hf = h5py.File(path, 'r')
    data = []
    for dn in ['images', 'positions']:
        d = np.array(hf.get(dn))
        data.append(d)
    hf.close()
    return data


def normalize(images):
    """
    Normalize an input image array.

    Input:
        images: numpy array of shape (T, H, W, C)

    Returns:
        images: numpy array of shape (T, C, H, W)
    """
    images = (images - _IMG_MEAN) / _IMG_STD

    # (T, C, H, W)
    images = np.transpose(images, (0, 3, 1, 2))

    return images


def denormalize(images):
    """
    De-normalize an input image array.

    Input:
        images: numpy array of shape (T, H, W, C)

    Output:
        images: numpy array of shape (T, C, H, W)
    """
    images = images * _IMG_STD + _IMG_MEAN

    # (T, H, W, C)
    images = np.transpose(images, (0, 2, 3, 1))

    return images


class PhyDataset(Dataset):
    """
    Dataset for physical scene observations.
    Available environments: 'RigidFall', 'MassRope'
    """

    def __init__(self, name, split):
        if name not in ['RigidFall', 'MassRope']:
            raise ValueError('Invalid dataset name {}.'.format(name))
        if split not in ['train', 'valid']:
            raise ValueError('Invalid dataset split {}.'.format(split))

        self._name = name
        self._split = split
        self._data_dir = os.path.join(
            _DATA_DIR[name], '{}_vision'.format(split))

    def __len__(self):
        if self._name == 'RigidFall':
            if self._split == 'train':
                return 4500
            else:
                return 500
        else:  # 'MassRope'
            if self._split == 'train':
                return 2700
            else:
                return 300

    def __getitem__(self, idx):
        if idx >= len(self) or idx < 0:
            raise ValueError('Invalid index {}'.format(idx))

        data_path = os.path.join(self._data_dir, str(idx) + '.h5')
        images, positions = _load_data(data_path)

        # Remove environment particles
        if self._name == 'RigidFall':
            positions = positions[:, :-1, :]
            groups = np.array([0 for _ in range(64)]
                              + [1 for _ in range(64)]
                              + [2 for _ in range(64)])
        else:  # 'MassRope'
            positions = positions[:-1, :-1, :]
            groups = np.array([0 for _ in range(81)]
                              + [1 for _ in range(14)])

        # Drop frames from beginning of video
        n_drop = _N_DROP_FRAMES[self._name]
        images = images[n_drop:, ...]
        positions = positions[n_drop:, ...]

        images = images.astype(np.float32) / 255
        images = normalize(images)

        return images, positions, groups


def get_dataloader(config, split, shuffle=None):
    name = config.dataset
    batch_size = config.batch_size
    if shuffle is None:
        shuffle = (split == 'train')

    ds = PhyDataset(name, split)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)
