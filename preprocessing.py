from enum import IntEnum
import json
import os
from os.path import join as pjoin
import random
from typing import Tuple

import h5py
import nibabel as nib
import nrrd
import numpy as np
import torch
import torch.nn.functional as F
from torch_radon import ConeBeam
from torch_radon.filtering import FourierFilters
from torch_radon.volumes import Volume3D
from tqdm import tqdm


def mu2hu(volume: torch.Tensor, mu_water: float = 0.02) -> torch.Tensor:
    return (volume - mu_water)/mu_water * 1000


def hu2mu(volume: torch.Tensor, mu_water: float = 0.02) -> torch.Tensor:
    return (volume * mu_water)/1000 + mu_water


def hu2normalized(tensor: torch.Tensor) -> torch.Tensor:
    carmh_gt_upper_99_percentile: float = 1720.43359375
    return hu2mu(tensor, .02)/hu2mu(carmh_gt_upper_99_percentile, .02)


def resample_image(image: torch.Tensor, inplane_shape: Tuple[int, int]) -> torch.Tensor:
    """
        resamples the image [N, C, x, y] to inplane_shape.
        returns resampled image [N, C, x, y]
        inplane_shape: [x, y]
    """
    theta = torch.tensor(
        [[
            [inplane_shape[-1]/image.shape[-1], 0, 0],
            [0, inplane_shape[-2]/image.shape[-2], 0],
        ]], dtype=torch.float32, device=image.device)
    sampling_grid = F.affine_grid(
        theta,
        (image.shape[0], image.shape[1], inplane_shape[-2], inplane_shape[-1]),
        align_corners=False,
    )
    resampled_t = F.grid_sample(image, sampling_grid, align_corners=False)
    return resampled_t


def rescale(*args, inplane_shape: Tuple[int, int]) -> torch.Tensor:
    # rcrop = torchvision.transforms.RandomCrop(inplane_shape)
    # free_noised = torch.stack((free, noised))
    # free_noised_crop = rcrop(free_noised)
    # return free_noised_crop[0], free_noised_crop[1]
    combined = torch.cat(args, dim=1)
    combined_re = F.interpolate(combined, size=inplane_shape)
    return torch.split(combined_re, 1, dim=1)


def filter_projections(projections: torch.Tensor, filter_name="ramp"):
    fourier_filters = FourierFilters()
    projections = projections.permute(0, 1, 3, 2, 4)
    proj_shape = projections.shape
    projections = projections.reshape(np.prod(proj_shape[:-2]), proj_shape[-2], proj_shape[-1])
    size = projections.size(-1)
    n_angles = projections.size(-2)

    # Pad projections to improve accuracy
    padded_size = max(64, int(2 ** np.ceil(np.log2(2 * size))))
    pad = padded_size - size
    padded_projections = F.pad(projections.float(), (0, pad, 0, 0))

    proj_fft = torch.fft.rfft(padded_projections, norm='ortho')

    # get filter and apply
    f = fourier_filters.get(padded_size, filter_name, projections.device)[..., 0]
    filtered_proj_fft = proj_fft * f

    # Inverse fft
    filtered_projections = torch.fft.irfft(filtered_proj_fft, norm='ortho')
    filtered_projections = filtered_projections[:, :, :-pad] * (np.pi / (2 * n_angles))

    return filtered_projections.to(dtype=projections.dtype).reshape(proj_shape).permute(0, 1, 3, 2, 4)


class DetectorBinning(IntEnum):
    BINNING1x1 = 1
    BINNING2x2 = 2
    BINNING4x4 = 4


class ArtisQSystem:
    def __init__(self, detector_binning: DetectorBinning):
        self.nb_pixels = (2480//detector_binning, 1920//detector_binning)
        self.pixel_dims = (0.154*detector_binning, 0.154*detector_binning)
        self.carm_span = 1200.0  # mm


def load_nifty(path: str):
    nii_file = nib.load(path)
    volume = nii_file.get_fdata().transpose()
    voxel_size = np.diag(nii_file.affine)[:-1]
    return volume, voxel_size


def load_nrrd(path: str):
    volume, nrrd_header = nrrd.read(path)
    voxel_size = np.diag(nrrd_header['space directions'])
    return volume, tuple(voxel_size)


def create_projection_reco(db_path: str, subject_name: str, needle_path: str,
                           needle_name: str, is_sparse: bool,
                           transpose: bool = True):
    ct_system = ArtisQSystem(DetectorBinning.BINNING4x4)
    angles = np.linspace(0, 2*np.pi, 13 if is_sparse else 360, endpoint=False, dtype=np.float32)
    src_dist = ct_system.carm_span*3/4
    det_dist = ct_system.carm_span*1/4
    src_det_dist = src_dist + det_dist
    det_spacing_v = ct_system.pixel_dims[1]
    radon = ConeBeam(
        det_count_u=ct_system.nb_pixels[0],
        angles=angles,
        src_dist=src_dist,
        det_dist=det_dist,
        det_count_v=ct_system.nb_pixels[1],
        det_spacing_u=ct_system.pixel_dims[0],
        det_spacing_v=det_spacing_v,
        pitch=0.0,
        base_z=0.0,
    )

    # create needle projections
    volume, voxel_size = load_nifty(pjoin(needle_path, f'{needle_name}.nii.gz'))
    volume_t = torch.from_numpy(volume).float().cuda()[None, None, ...]
    volume_t = hu2mu(volume_t, 0.02)
    radon.volume = Volume3D(
            depth=volume.shape[0],
            height=volume.shape[1],
            width=volume.shape[2],
            voxel_size=voxel_size)
    needle_projections_t = radon.forward(volume_t).nan_to_num()

    # create head projections
    load_fn = load_nrrd if f'{subject_name}.nrrd' in os.listdir(db_path) else load_nifty
    volume, voxel_size = load_fn(pjoin(
        db_path,
        [f for f in os.listdir(db_path) if subject_name in f][0]))
    if transpose:
        volume = volume.transpose()
    volume_t = torch.from_numpy(volume).float().cuda()[None, None, ...]
    volume_t = hu2mu(volume_t, 0.02)
    volume_t[volume_t < 0] = 0
    radon.volume = Volume3D(
            depth=volume.shape[0],
            height=volume.shape[1],
            width=volume.shape[2],
            voxel_size=voxel_size)
    head_projections_t = radon.forward(volume_t).nan_to_num()

    # combine head and needle projections
    head_needle_projections_t = head_projections_t + needle_projections_t

    # reconstruct to volume
    reco_t = radon.backprojection(filter_projections(head_needle_projections_t, 'hann'))
    reco_t = reco_t*det_spacing_v/src_det_dist*src_dist  # cone beam correction
    reco_t = mu2hu(reco_t, 0.02)
    reco = reco_t[0, 0].cpu().numpy()

    nrrd.write(
        pjoin(
            f"{'sparse_' if is_sparse else ''}interventional_recos",
            f'{subject_name}_{needle_name}.nrrd',
        ),
        reco.transpose(),
        header={
            "units": ["mm", "mm", "mm"],
            "spacings": [f'{abs(v)}' for v in voxel_size],
        }
    )


def create_head_projection_reco(subjects: str):
    with open("train_valid_test.json", "r", encoding='utf-8') as file_handle:
        test_subjects = json.load(file_handle)[subjects]

    db_path = '/mnt/nvme2/mayoclinic/Head/high_dose'
    needle_path = '/home/phernst/Documents/git/ictdl/needles'

    needle_names = [
        'Needle2_Pos1_11',
        'Needle2_Pos2_12',
        'Needle2_Pos3_13',
    ]

    random_needles = random.choices(needle_names, k=len(test_subjects))

    itrt = tqdm(zip(test_subjects, random_needles), total=len(test_subjects))
    for subject_name, needle_name in itrt:
        create_projection_reco(
            db_path,
            f'{subject_name}',
            needle_path,
            needle_name,
            is_sparse=False,
        )
        create_projection_reco(
            db_path,
            f'{subject_name}',
            needle_path,
            needle_name,
            is_sparse=True,
        )


def generate_ct_data():
    inter_gt_path = 'interventional_recos'
    inter_sparse_path = 'sparse_interventional_recos'
    prior_path = '/mnt/nvme2/mayoclinic/Head/high_dose'

    os.makedirs("./data/dataset/Free", exist_ok=True)
    os.makedirs("./data/dataset/noise_13", exist_ok=True)
    os.makedirs("./data/dataset/priors", exist_ok=True)

    num_slices = 200

    for file in tqdm(os.listdir(inter_gt_path)):
        img_data = nrrd.read(pjoin(inter_gt_path, file))[0]
        depth = img_data.shape[-1]
        img_data = img_data[..., (depth//2-num_slices//2):(depth//2+num_slices//2)]
        img_data = hu2normalized(img_data)
        img_data[img_data < 0] = 0

        h5 = h5py.File(pjoin("./data/dataset/Free", f'{file[:-5]}.h5'), "w")
        h5.create_dataset('volume', data=img_data)

        img_data = nrrd.read(pjoin(inter_sparse_path, file))[0]
        img_data = img_data[..., (depth//2-num_slices//2):(depth//2+num_slices//2)]
        img_data = hu2normalized(img_data)
        img_data[img_data < 0] = 0

        h5 = h5py.File(pjoin("./data/dataset/noise_13", f'{file[:-5]}.h5'), "w")
        h5.create_dataset('volume', data=img_data)

    subjects = {f.split('_', 1)[0] for f in os.listdir(inter_gt_path)}
    for subject in tqdm(subjects):
        img_data = nrrd.read(pjoin(prior_path, f'{subject}.nrrd'))[0]
        depth = img_data.shape[-1]
        img_data = img_data[..., (depth//2-num_slices//2):(depth//2+num_slices//2)]
        img_data = hu2normalized(img_data)
        img_data[img_data < 0] = 0

        h5 = h5py.File(pjoin("data/dataset/priors", f'{subject}.h5'), "w")
        h5.create_dataset('volume', data=img_data)


if __name__ == "__main__":
    # create_head_projection_reco('train_subjects')
    # create_head_projection_reco('valid_subjects')
    generate_ct_data()
