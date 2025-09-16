import os
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '2'

from unik3d.utils.camera import (Pinhole, OPENCV, Fisheye624, MEI, Spherical)

from unik3d.models import UniK3D
import json
import torch
import numpy as np
from PIL import Image
import plotly.graph_objs as go

def equirectangular_to_xyz(width, height, device):
    """Convert equirectangular coordinates to spherical 3D coordinates in OpenCV convention and rotate 90° around Y-axis using PyTorch."""
    # 创建 theta 和 phi 为 1D 张量
    theta = torch.linspace(0, 2 * torch.pi, width, device=device)  # 方位角 [0, 2π]
    phi = torch.linspace(0, torch.pi, height, device=device)       # 仰角 [0, π]
    
    # 生成网格，调整 indexing='ij' 确保符合 PyTorch 约定
    phi, theta = torch.meshgrid(phi, theta, indexing='ij')

    # 计算 OpenCV 形式的 X, Y, Z 坐标
    x = -torch.sin(phi) * torch.sin(theta)   # OpenCV X: 右
    y = -torch.cos(phi)                     # OpenCV Y: 下
    z = -torch.sin(phi) * torch.cos(theta)  # OpenCV Z: 前
    # 将 x, y, z 堆叠在一起，并调整维度 (height, width, 3)
    xyz = torch.stack((x, y, z), dim=-1)  # (B, V, H, W, 3)

    return xyz

# Load the model
model = UniK3D.from_pretrained("lpiccinelli/unik3d-vitl") # vitl for ViT-L backbone
model.eval()
model.to("cuda") # Move the model to GPU

# Load the RGB image and the normalization will be taken care of by the model
image_path = "grd2.png"
rgb = torch.from_numpy(np.array(Image.open(image_path))).permute(2, 0, 1) # C, H, W

# Load the camera
H, W = rgb.shape[-2:]
params = [500, 500, 320, 240, W, H, 3.14159, 1.57079] # fx, fy, cx, cy, w, h, fov_h, fov_v
params = torch.tensor(params).float() # (1, 8)
camera_name = "Spherical"
camera = eval(camera_name)(params=params)

model.resolution_level = 9
predictions = model.infer(rgb, camera=camera, normalize=True)

# depth = predictions["depth"][0] # (C, H, W)
# depth = torch.clamp(depth, min=0)
# depth  = depth * mask.to(depth.device) # (C, H, W)
# depth = depth.permute(1, 2, 0)
# xyz_coords = equirectangular_to_xyz(640, 320, depth.device) # (1, H, W, 3)
# points = xyz_coords * depth # (H, W, 3)

# points = predictions["points"][0] # (C, H, W)
# points = points.permute(1, 2, 0)  # 转换为 (H, W, 3)
# points = points.reshape(-1, 3).cpu().numpy()  # 转换为 NumPy 数组


depth = predictions["distance"][0]  # (C, H, W)

# refine depth
mask = rgb.sum(axis=0) >= 16
mask = mask.unsqueeze(0).float()  # (1, H, W) 
depth = torch.clamp(depth, min=0)  # 将深度值限制在非负范围内
depth = depth * mask.to(depth.device)  # (C, H, W)
depth[depth > 80] = 0.0

depth = depth.permute(1, 2, 0)  # 转换为 (H, W, 1)

xyz_coords = equirectangular_to_xyz(1024, 512, depth.device)
points = xyz_coords * depth  # (H, W, 3)
points = points.reshape(-1, 3).cpu().numpy() 

colors = rgb.permute(1, 2, 0).reshape(-1, 3).cpu().numpy()

# black_bg_mask = colors.sum(axis=1) >= 16
# points = points[black_bg_mask]  # 过滤掉黑色背景的点
# colors = colors[black_bg_mask]  # 过滤掉黑色背景的颜色

# far_points_mask = np.linalg.norm(points, axis=-1) < 100.0
# points = points[far_points_mask]  # 过滤掉远点
# colors = colors[far_points_mask]  # 过滤掉远点的颜色

# 计算颜色的 RGB 值
colors_rgb = ['rgb({},{},{})'.format(int(r), int(g), int(b)) for r, g, b in colors]  # 转换为字符串形式的 RGB

fig = go.Figure(data=[go.Scatter3d(
    x=points[:, 0],
    y=points[:, 1],
    z=points[:, 2],
    mode='markers',
    marker=dict(
        size=3,
        color=colors_rgb,  # 设置颜色
    )
)])
fig.update_layout(scene=dict(
    xaxis=dict(title='X', tick0=0, dtick=1),  # X 轴方向正确
    yaxis=dict(title='Y (Down)', tick0=0, dtick=-1),  # 翻转Y轴
    zaxis=dict(title='Z', tick0=0, dtick=1),  # Z 轴方向正确
    aspectmode='cube'  # 确保XYZ比例一致
))
# fig.show()

# 保存为 HTML 文件，下载后用浏览器打开
fig.write_html("point_cloud1.html")
