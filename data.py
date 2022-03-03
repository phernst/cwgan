import json
import os
from os.path import join as pjoin

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from preprocessing import rescale


def load_h5(path: str):
    return h5py.File(path, 'r', libver='latest', swmr=True)['volume']


class CTDataset(Dataset):
    def __init__(self, level, dataset: str):
        assert dataset in ('train', 'valid', 'test')
        with open('train_valid_test.json', 'r', encoding='utf-8') as file:
            subjects = json.load(file)[f'{dataset}_subjects']
            self.all_files = sorted([
                f for f in os.listdir('data/dataset/Free')
                if f.endswith('.h5') and f.split('_', 1)[0] in subjects
            ])
        self.level = level
        self.num_slices = 200
        # self.inplane_shape = (384, 384)
        self.inplane_shape = (128, 128)

    def __len__(self):
        return len(self.all_files)*self.num_slices

    def __getitem__(self, index):
        current_file = self.all_files[index//self.num_slices]
        free_volume = load_h5(pjoin('data/dataset/Free', current_file))
        noised_volume = load_h5(pjoin(f'data/dataset/noise_{self.level}', current_file))
        prior_volume = load_h5(pjoin('data/dataset/priors', f"{current_file.split('_', 1)[0]}.h5"))

        free_img = torch.from_numpy(free_volume[..., index % self.num_slices]).float()[None, None]
        noised_img = torch.from_numpy(noised_volume[..., index % self.num_slices]).float()[None, None]
        prior_img = torch.from_numpy(prior_volume[..., index % self.num_slices]).float()[None, None]

        # free_img = F.interpolate(free_img, size=(256, 256), mode='bilinear', align_corners=True)[0]
        # noised_img = F.interpolate(noised_img, size=(256, 256), mode='bilinear', align_corners=True)[0]
        # free_img = resample_image(free_img, self.inplane_shape)[0]
        # noised_img = resample_image(noised_img, self.inplane_shape)[0]
        free_img, noised_img, prior_img = rescale(
            free_img,
            noised_img,
            prior_img,
            inplane_shape=self.inplane_shape,
        )
        free_img = free_img[0]
        noised_img = noised_img[0]
        prior_img = prior_img[0]

        return {
            "free_img": free_img,
            "noised_img": noised_img,
            "prior_img": prior_img,
        }


# def add_rice_noise(img, snr=1, mu=0.0, sigma=1):
#     level = snr * np.max(img) / 100
#     size = img.shape
#     x = level * np.random.normal(mu, sigma, size=size) + img
#     y = level * np.random.normal(mu, sigma, size=size)
#     return np.sqrt(x**2 + y**2)
#     # size = img.shape
#     # x = snr * np.random.normal(mu, sigma, size=size) + img
#     # y = snr * np.random.normal(mu, sigma, size=size)
#     # return np.sqrt(x**2 + y**2)

# MRIDataset()
# MRIValidDataset()
