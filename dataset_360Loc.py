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

pano_width = 320
pano_height = 160

def two_sample(scene, extrinsics, stage="train", i=0):
    num_views, _, _ = extrinsics.shape
    # Compute the context view spacing based on the current global step.
    if stage == "val":
        # When testing, always use the full gap.
        min_gap = max_gap = 3
    else:
        min_gap = max_gap = 2
    max_gap = min(num_views - 1, min_gap)

    # Pick the gap between the context views.
    # NOTE: we keep the bug untouched to follow initial pixelsplat cfgs

    context_gap = torch.randint(
        min_gap,
        max_gap + 1,
        size=tuple(),
        device='cpu',
    ).item()

    # Pick the left and right context indices.

    if stage == "val":
        index_context_left = (num_views - context_gap - 1) * i / max((100 - 1), 1)
        index_context_left = int(index_context_left)
    else:
        index_context_left = torch.randint(
            num_views - context_gap,
            size=tuple(),
            device='cpu',
        ).item()
    index_context_right = index_context_left + context_gap

    # Pick the target view indices.
    if stage == "val":
        # When testing, pick all.
        index_target = torch.arange(
            index_context_left,
            index_context_right + 1,
            device='cpu',
        )
    else:
        # When training or validating (visualizing), pick at random.
        index_target = torch.randint(
            index_context_left,
            index_context_right + 1,
            size=(1,),
            device='cpu',
        )

    return (
        torch.tensor((index_context_left, index_context_right)),
        index_target,
    )

def one_sample(scene, extrinsics, stage="train", i=0):
    num_views, _, _ = extrinsics.shape
    context_gap = 2

    # Pick the left and right context indices.

    if stage == "val":
        index_context_left = (num_views - context_gap - 1) * i / max((100 - 1), 1)
        index_context_left = int(index_context_left)
    else:
        index_context_left = torch.randint(0, num_views, (1,)).item()

    return (
        index_context_left,
        torch.tensor((index_context_left - 1, index_context_left + 1)),
    )

class Dataset360Loc(Dataset):
    def __init__(
        self,
        stage,
    ) -> None:
        super().__init__()
        self.stage = stage
        self.to_tensor = tf.ToTensor()
        # NOTE: update near & far; remember to DISABLE `apply_bounds_shim` in encoder
        self.near = 0.45
        self.far = 50
        self.width = pano_width
        self.height = pano_height

        if stage == "train":
            locations = ['concourse', 'hall', 'piatrium', 'atrium']
        else:
            locations = ['atrium']
        root = Path('/home/qiwei/ICLR25/360Loc')
        self.data = []
        for location in locations:
            seqs = [list((root / location / folder).glob('*360*/')) for folder in ('mapping', 'query_360')]
            seqs = sum(seqs, [])
            self.data.extend(seqs)
        
        images = []
        for example_path in self.data:
            frames, extrinsics_orig = self.load_extrinsics(example_path)
            images = images + [example_path / 'image' / frame for frame in frames]
        self.images = images

    def load_extrinsics(self, example_path):
        example = example_path / 'camera_pose.json'
        with open(example) as f:
            example = json.load(f)
        frames, extrinsics_orig = list(example.keys()), list(example.values())
        extrinsics_orig = torch.tensor(extrinsics_orig)
        return frames, extrinsics_orig

    def __getitem__(self, idx):
        # Chunks must be shuffled here (not inside __init__) for validation to show
        # random chunks.

        # When testing, the data loaders alternate chunks.
        img = torch.from_numpy(np.array(Image.open(self.images[idx]).resize((1024, 512)))).permute(2, 0, 1)
        return (img, self.images[idx])

    def convert_images(
        self,
        images: list[str | Path],
    ):
        torch_images = []
        for image in images:
            image = Image.open(image)
            image = image.resize([self.width, self.height], Image.LANCZOS)
            torch_images.append(self.to_tensor(image))
        return torch.stack(torch_images)
    
    def convert_poses(
        self,
        context_extrinsics,
        target_extrinsics,
    ):  # extrinsics
        w2w = torch.tensor([  #  X -> X, Z -> Y, Y -> -Z
            [1, 0, 0, 0],
            [0, 0, -1, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1],
        ]).float()
        # w2c = context_extrinsics @ torch.inverse(target_extrinsics)
        c2w = torch.inverse(context_extrinsics) @ target_extrinsics
        c2w = w2w @ c2w

        return c2w
    
    def get_bound(
        self,
        bound: Literal["near", "far"],
        num_views: int,
    ) -> Float[Tensor, " view"]:
        value = torch.tensor(getattr(self, bound), dtype=torch.float32)
        return repeat(value, "-> v", v=num_views)

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

def load_360Loc_data(batch_size, stage='train'):
    """

    Args:
        batch_size: B
        area: same | cross
    """

    MP3D = Dataset360Loc(stage = stage)

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
        num_workers=1,
        generator=get_generator(seed),
        worker_init_fn=worker_init_fn,
        persistent_workers=persistent_workers,
        collate_fn=collate_function,
        shuffle=False
    )
    # val_dataloader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

    return dataloader

def ply_post_opencv(data):
    # 提取每一张图片的平移向量（最后一列）
    camera_positions = data[:, :, 3].numpy()

    # 提取 X 和 Y 坐标
    x = camera_positions[:, 0]
    y = camera_positions[:, 2]

    # 创建一个 2D 绘图
    plt.figure()

    # 绘制相机轨迹
    plt.plot(x, y, marker='o', color='b', label='Camera Trajectory')
    # 设置 X 和 Y 坐标轴比例一致
    plt.axis('equal')
    # 添加标签和标题
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.title('Camera Movement in XY Plane')
    plt.legend()

    # 如果需要保存图形，可以使用以下命令
    plt.savefig("camera_trajectory_xy1.png", dpi=300)
    plt.close()

def ply_post(data):
    # 提取每一张图片的平移向量（最后一列）
    camera_positions = data[:, :, 3].numpy()

    # 提取 X 和 Y 坐标
    x = camera_positions[:, 0]
    y = camera_positions[:, 1]

    # 创建一个 2D 绘图
    plt.figure()

    # 绘制相机轨迹
    plt.plot(x, y, marker='o', color='b', label='Camera Trajectory')
    # 设置 X 和 Y 坐标轴比例一致
    plt.axis('equal')
    # 添加标签和标题
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.title('Camera Movement in XY Plane')
    plt.legend()

    # 如果需要保存图形，可以使用以下命令
    plt.savefig("camera_trajectory_xy2.png", dpi=300)
    plt.close()