import torch.nn as nn
import torch
from torchvision import transforms
import cv2
import torch.nn.functional as F
from PIL import Image
import numpy as np
import torch

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

class ModelMP3D(nn.Module):
    def __init__(self, args, device):  # device='cuda:0',
        super(ModelMP3D, self).__init__()
        self.device = device
        self.args = args
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

    def create_data(self, grd_path):
        # grd = F.interpolate(grd, (80, 160), mode='bilinear', align_corners=False)
        rgbs = []
        for b in range(len(grd_path)):
            rgb = torch.from_numpy(np.array(Image.open(grd_path[b]))).permute(2, 0, 1)
            rgbs.append(rgb)
        rgb = torch.stack(rgbs, dim=0).squeeze(1)
        with torch.no_grad():
            outputs = self.model.infer(rgb, camera=self.camera, normalize=True)
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
            depth[depth > 80] = 0.0

            # s_metric, d_metric_Pano = compute_scaling_factor(metric_depth, depth1)

            # test_img = to_pil_image(rgb[0])
            # test_img.save('grd.png')
            for b in range(len(grd_path)):
                # onlyDepth(depth[b, 0], 'grd_depth.png')
                d_metric_Pano_np = depth[b].cpu().numpy()
                d_conf_Pano_np = conf[b].cpu().numpy()
                np.save(grd_path[b].replace('rgb.png', 'depth_metric.npy'), d_metric_Pano_np)
                np.save(grd_path[b].replace('rgb.png', 'depth_conf.npy'), d_conf_Pano_np)
            # i = 2
            # showDepth(depth[0, i], tensor_to_cv2_image(pers_imgs[0, i]))
