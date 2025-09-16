from dataclasses import dataclass
from pathlib import Path
from typing import Literal
import os

import torch
from torch import Generator, nn
import torchvision.transforms as tf
from einops import repeat
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset
import numpy as np
from einops import rearrange
import torch.nn.functional as F
import random

from torch.utils.data import DataLoader
from torchvision import transforms
from pathlib import Path

test_datasets = [{'name': 'm3d', 'dis': 0.1}, {'name': 'm3d', 'dis': 0.25}, {'name': 'm3d', 'dis': 0.5}, {'name': 'm3d', 'dis': 0.75}, {'name': 'm3d', 'dis': 1.0}, {'name': 'residential', 'dis': 0.15}, {'name': 'replica', 'dis': 0.5}]
roots = [Path('/home/qiwei/ICLR25/pano_grf')]

class DatasetMP3D(Dataset):
    def __init__(
        self,
        stage = 'train',
    ):
        super().__init__()
        self.stage = stage
        self.to_tensor = tf.ToTensor()
        # NOTE: update near & far; remember to DISABLE `apply_bounds_shim` in encoder

        self.near = 0.45
        self.far = 10.0

        # scan folders in cfg.roots[0]
        if stage == "predict":
            stage = "test"

        height = 512
        height = max(height, 512)
        resolution = (height * 2, height)
        resolution = 'x'.join(map(str, resolution))
        if stage == "test":
            self.roots = []
            for test_dataset in test_datasets:
                name = test_dataset["name"]
                dis = test_dataset["dis"]
                self.roots.append(
                    roots[0] / f"png_render_{stage}_{resolution}_seq_len_3_{name}_dist_{dis}"
                )
        else:
            self.roots = [r / f"png_render_{stage}_{resolution}_seq_len_3_m3d_dist_0.5" for r in roots]

        data = []
        for root, test_dataset in zip(self.roots, test_datasets):
            if not os.path.exists(root):
                continue
            scenes = os.listdir(root)
            scenes.sort()
            for s in scenes:
                data.append({
                    'root': root,
                    'scene_id': s,
                    'name': test_dataset["name"],
                    'dis': test_dataset["dis"],
                    'baseline': test_dataset["dis"] * 2,
                })
        self.data = data

    def __getitem__(self, idx):
        data = self.data[idx].copy()
        scene = data['scene_id']
        scene_path = data['root'] / scene
        views = os.listdir(scene_path)
        views.sort()

        # Load the images.
        rgbs_path = [str(scene_path / v / 'rgb.png') for v in views]

        return rgbs_path

    def __len__(self) -> int:
        return len(self.data)


def get_generator(seed):
    generator = Generator()
    generator.manual_seed(seed)
    return generator

def worker_init_fn(worker_id: int) -> None:
    random.seed(int(torch.utils.data.get_worker_info().seed) % (2**32 - 1))
    np.random.seed(int(torch.utils.data.get_worker_info().seed) % (2**32 - 1))

def load_MP3D_data(batch_size, stage='train'):
    """

    Args:
        batch_size: B
        area: same | cross
    """

    MP3D = DatasetMP3D(stage = stage)

    if stage == 'train':
        seed = 1234
        shuffle = True
        persistent_workers = True
    elif stage == 'val':
        seed = 3456
        shuffle = False
        persistent_workers = True
    elif stage == 'test':
        seed = 2345
        shuffle = False
        persistent_workers = False
    else:
        seed = 6789
        shuffle = False
        persistent_workers = True

    dataloader = DataLoader(
        MP3D, 
        batch_size=batch_size,
        num_workers=32,
        generator=get_generator(seed),
        worker_init_fn=worker_init_fn,
        persistent_workers=persistent_workers,
        shuffle=False
    )
    # val_dataloader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

    return dataloader