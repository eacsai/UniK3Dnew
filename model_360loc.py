import torch.nn as nn
import torch
from torchvision import transforms
import cv2
import torch.nn.functional as F
from PIL import Image
import numpy as np
import torch
import os
from unik3d.utils.camera import (Pinhole, OPENCV, Fisheye624, MEI, Spherical)
from unik3d.models import UniK3D


np.bool = np.bool_
np.float = np.float32
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
to_pil_image = transforms.ToPILImage()
# original_raw_Lpips_step = 50000
raw_Lpips_step = 25000
# original_L1_step = 100000
L1_step = 40000
# original_refine_Lpips_step = 100000
refine_Lpips_step = 40000
# original_discriminator_loss_active_step = 125000
discriminator_loss_active_step = 60000
import matplotlib.cm as cm

def onlyDepth(depth, save_name):
    cmap = cm.Spectral
    depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
    depth = depth.cpu().detach().numpy()
    depth = depth.astype(np.uint8)

    c_depth = (cmap(depth)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)
    cv2.imwrite(save_name, c_depth)

def tensor_to_cv2_image(tensor):
    """Convert PyTorch tensor to OpenCV image format"""
    # Denormalize if the tensor is normalized
    if tensor.max() <= 1.0:
        tensor = tensor * 255.0

    # Convert to CPU and numpy
    img = tensor.cpu().detach().numpy()

    # Handle different tensor shapes
    if img.ndim == 3:
        if img.shape[0] == 3:  # CHW format
            img = img.transpose(1, 2, 0)  # Convert to HWC
        elif img.shape[2] == 3:  # Already HWC
            pass
        else:
            raise ValueError(f"Unexpected tensor shape: {img.shape}")
    elif img.ndim == 2:  # Grayscale
        img = np.stack([img] * 3, axis=2)

    # Ensure values are in valid range
    img = np.clip(img, 0, 255).astype(np.uint8)

    # Convert RGB to BGR for OpenCV
    if img.shape[2] == 3:
        img = img[:, :, ::-1]

    return img

def showDepth(depth_tensor, rgb_image):
    """Visualize depth map alongside the original RGB image"""
    # Create depth visualization
    cmap = cm.get_cmap('Spectral')
    depth_norm = (depth_tensor - depth_tensor.min()) / (depth_tensor.max() - depth_tensor.min() + 1e-8)
    depth_vis = cmap(depth_norm.cpu().detach().numpy())[:, :, :3] * 255
    depth_vis = depth_vis.astype(np.uint8)

    # Convert depth to BGR for OpenCV
    depth_bgr = depth_vis[:, :, ::-1]

    # Ensure RGB image is in BGR format for OpenCV
    if len(rgb_image.shape) == 3 and rgb_image.shape[2] == 3:
        rgb_bgr = rgb_image
    else:
        rgb_bgr = tensor_to_cv2_image(torch.from_numpy(rgb_image))

    # Resize depth to match RGB image if needed
    if depth_bgr.shape[:2] != rgb_bgr.shape[:2]:
        depth_bgr = cv2.resize(depth_bgr, (rgb_bgr.shape[1], rgb_bgr.shape[0]))

    # Concatenate images side by side
    combined = np.hstack([rgb_bgr, depth_bgr])

    # Save visualization
    cv2.imwrite('depth_visualization.png', combined)
    print("Depth visualization saved as 'depth_visualization.png'")

class Model360Loc(nn.Module):
    def __init__(self, args, device):  # device='cuda:0',
        super(Model360Loc, self).__init__()
        self.device = device
        self.global_step = 0
        local_model_path = "/home/qiwei/.cache/torch/hub/unik3d-vitl"
        self.model = UniK3D.from_pretrained(local_model_path)   
        # self.model = UniK3D.from_pretrained("lpiccinelli/unik3d-vitl")
        # self.metric_model = torch.hub.load("/home/qiwei/.cache/torch/hub/yvanyin_metric3d_main", "metric3d_vit_large", source='local')
        # self.metric_model = torch.hub.load('yvanyin/metric3d', 'metric3d_vit_large', pretrain=True)
        self.model.eval()
        self.model.to("cuda")
        self.model.resolution_level = 9
        # Load the camera
        H = 512
        W = 1024
        params = [500, 500, 320, 240, W, H, 3.14159, 1.57079] # fx, fy, cx, cy, w, h, fov_h, fov_v
        params = torch.tensor(params).float() # (1, 8)
        camera_name = "Spherical"
        self.camera = eval(camera_name)(params=params)

    def create_data(self, grd, grd_path):
        # grd = F.interpolate(grd, (80, 160), mode='bilinear', align_corners=False)
        rgb = grd
        with torch.no_grad():
            outputs = self.model.infer(grd, camera=self.camera, normalize=True)
            # metric_depth, metric_confidence, metric_output_dict = self.metric_model.inference({'input': grd_img})
            depth = outputs["distance"]
            conf = outputs["confidence"]
            # metric_depth = F.interpolate(metric_depth, (320, 640), mode='bilinear', align_corners=True)
            # mask = (grd_img != 0).any(dim=1, keepdim=True).float()
            # metric_depth = (metric_depth * mask).squeeze(1)

            # refine depth
            mask = rgb.sum(axis=1) >= 16
            mask = mask.unsqueeze(1).float()  # (B, 1, H, W) 
            depth = torch.clamp(depth, min=0)  # 将深度值限制在非负范围内
            depth = depth * mask.to(depth.device)  # (C, H, W)
            # depth[depth > 200] = 0.0

            # Visualize the first sample in the batch
            # if len(depth) > 0:
            #     showDepth(depth[0, 0], tensor_to_cv2_image(grd[0]))

            # s_metric, d_metric_Pano = compute_scaling_factor(metric_depth, depth1)

            # test_img = to_pil_image(grd_img[0])
            # test_img.save('grd.png')
            for b in range(len(grd_path)):
                parent_path = str(grd_path[b]).split('/')
                depth_path = '/'.join(parent_path[:-2])
                depth_path = os.path.join(depth_path, 'depth_metric_2')
                if not os.path.isdir(depth_path):
                    os.makedirs(depth_path, exist_ok=True)
                # onlyDepth(depth[b, 0], 'grd_depth.png')
                d_metric_Pano_np = depth[b].cpu().numpy()
                d_conf_Pano_np = conf[b].cpu().numpy()
                np.save(os.path.join(depth_path, parent_path[-1].replace('.png', '_depth.npy')), d_metric_Pano_np)
                np.save(os.path.join(depth_path, parent_path[-1].replace('.png', '_conf.npy')), d_conf_Pano_np)
