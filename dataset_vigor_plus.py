from dataclasses import dataclass
from pathlib import Path
from typing import Literal
import os

import torch
import torchvision.transforms as tf
from einops import repeat
from jaxtyping import Float
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
import numpy as np

import random
from torch import Generator, nn

from einops import rearrange
import torch.nn.functional as F
import json
from functools import cached_property
import matplotlib.pyplot as plt


class DatasetVIGORPlus(Dataset):
    def __init__(
        self,
        stage,
    ) -> None:
        super().__init__()
        self.stage = stage
        self.to_tensor = tf.ToTensor()
        # NOTE: update near & far; remember to DISABLE `apply_bounds_shim` in encoder

        if stage == "train":
            locations = ['Kansas1', 'Kansas2', 'Kansas4']
        else:
            locations = ['Kansas_mini1', 'Kansas_mini10']
        root = Path('/data/qiwei/nips25/Kansas')
        self.images = []
        for location in locations:
            seqs = [list((root / location / 'Ground').glob('*.png'))]
            seqs = sum(seqs, [])
            self.images.extend(seqs)
        


    def __getitem__(self, idx):
        # Chunks must be shuffled here (not inside __init__) for validation to show
        # random chunks.

        # When testing, the data loaders alternate chunks.
        img = torch.from_numpy(np.array(Image.open(self.images[idx]).convert('RGB').resize((1024, 512)))).permute(2, 0, 1)
        return (img, self.images[idx])

    def __len__(self) -> int:
        return len(self.images)

def get_generator(seed):
    generator = Generator()
    generator.manual_seed(seed)
    return generator

def worker_init_fn(worker_id: int) -> None:
    random.seed(int(torch.utils.data.get_worker_info().seed) % (2**32 - 1))
    np.random.seed(int(torch.utils.data.get_worker_info().seed) % (2**32 - 1))

def collate_function(data):
    transposed_data = list(zip(*data))
    imgs, paths = transposed_data[0], transposed_data[1]
    imgs = torch.stack(imgs, 0)
    return (imgs, paths)

def load_VIGOR_plus(batch_size, stage='train'):
    """

    Args:
        batch_size: B
        area: same | cross
    """

    VIGOR = DatasetVIGORPlus(stage = stage)

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
        VIGOR, 
        batch_size=batch_size,
        num_workers=8,
        generator=get_generator(seed),
        worker_init_fn=worker_init_fn,
        persistent_workers=persistent_workers,
        collate_fn=collate_function,
        shuffle=False
    )
    # val_dataloader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

    return dataloader