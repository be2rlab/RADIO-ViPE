from ..base import DepthEstimationInput, DepthEstimationModel, DepthEstimationResult, DepthType
import os
from PIL import Image
import torch
import torchvision.transforms as T
import numpy as np
import cv2

class Datasetdepth(DepthEstimationModel):
    def __init__(self,dataset='replica',datasets_path = '/data/',scene = 'room0'):
        super().__init__()
        self.dataset = dataset
        self.datasets_path = datasets_path
        self.scene = scene

    @property
    def depth_type(self) -> DepthType:
        return DepthType.METRIC_DEPTH
    
    def estimate(self, src: DepthEstimationInput) -> DepthEstimationResult:
        if self.dataset == 'replica':
            depth_img_name = f"depth{src.index:06d}.png"
            depth_img_path = os.path.join(self.datasets_path, self.dataset, self.scene,'results',depth_img_name)
            self.scale = 6553.5
        elif self.dataset == 'tum':
            self.scale = 5000.0
            association_file_path = os.path.join(self.datasets_path,self.dataset,self.scene,'associations.txt')
            with open(association_file_path,'r') as f:
                lines = f.readlines()
            depth_img_name = lines[src.index].split()[3]
            depth_img_name = depth_img_name.strip()        
            depth_img_path = os.path.join(self.datasets_path,self.dataset,self.scene,depth_img_name)
        elif self.dataset == 'scannet':
            self.scale = 1000.0
            depth_img_name = f"{src.index}.png"
            depth_img_path = os.path.join(self.datasets_path,self.dataset,self.scene,'depth',depth_img_name)
        depth_img = cv2.imread(depth_img_path, cv2.IMREAD_UNCHANGED)
        depth_img = depth_img /self.scale
        (h1, w1), (crop_top, crop_bottom, crop_left, crop_right) = self._compute_frame_size_crop(depth_img.shape)
        depth_img = cv2.resize(depth_img, (w1, h1), interpolation=cv2.INTER_NEAREST).astype(np.float32)
        # Crop
        depth_img = depth_img[
            crop_top : h1 - crop_bottom,
            crop_left : w1 - crop_right
        ]
        depth_img = torch.from_numpy(depth_img)  # shape: (H, W)
        depth_img = depth_img.unsqueeze(0)

        return DepthEstimationResult(
            metric_depth = depth_img,
            confidence = torch.ones_like(depth_img)
        )   
    
    def _compute_frame_size_crop(self, previous_frame_size: tuple[int, int]):
        h0, w0 = previous_frame_size
        scale_factor = np.sqrt((384 * 512) / (h0 * w0))
        h1 = int(h0 * scale_factor)
        w1 = int(w0 * scale_factor)

        crop_h, crop_w = h1 % 8, w1 % 8
        crop_top, crop_bottom = crop_h // 2, crop_h - crop_h // 2
        crop_left, crop_right = crop_w // 2, crop_w - crop_w // 2

        self.fac_x, self.fac_y = w0 / w1, h0 / h1
        self.scx, self.scy = crop_left, crop_top
        return (h1, w1), (crop_top, crop_bottom, crop_left, crop_right)