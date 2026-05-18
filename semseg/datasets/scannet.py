"""ScanNet dataset loader used by semseg evaluation."""

import os
from typing import Tuple, Union

import numpy as np
import pandas as pd
import PIL
import torch
import torchvision

from semseg.datasets.base import SemSegDataset


class ScanNetDataset(SemSegDataset):
    def __init__(
        self,
        path: str,
        scene_name: str,
        rgb_resolution: Union[Tuple[int], int] = None,
        depth_resolution: Union[Tuple[int], int] = None,
        frame_skip: int = 0,
        interp_mode: str = "bilinear",
        load_semseg: bool = True,
    ):
        super().__init__(rgb_resolution=rgb_resolution, depth_resolution=depth_resolution, frame_skip=frame_skip, interp_mode=interp_mode)
        self.path = path
        self.scene_name = scene_name
        self.load_semseg = load_semseg
        self.original_h = 480
        self.original_w = 640
        self.rgb_h = self.original_h if self.rgb_h <= 0 else self.rgb_h
        self.rgb_w = self.original_w if self.rgb_w <= 0 else self.rgb_w
        self.depth_h = self.original_h if self.depth_h <= 0 else self.depth_h
        self.depth_w = self.original_w if self.depth_w <= 0 else self.depth_w

        scene_dir = os.path.join(self.path, self.scene_name)
        self.rgb_dir = os.path.join(scene_dir, "color")
        self.depth_dir = os.path.join(scene_dir, "depth")
        self.pose_dir = os.path.join(scene_dir, "pose")
        self.intrinsics_path = os.path.join(scene_dir, "intrinsic/intrinsic_depth.txt")
        self.semseg_dir = os.path.join(scene_dir, f"{scene_name}_2d-label-filt", "label-filt")
        self.intrinsics_3x3 = torch.tensor(np.loadtxt(self.intrinsics_path)[:3, :3], dtype=torch.float32)

        if self.depth_h != self.original_h or self.depth_w != self.original_w:
            h_ratio = self.depth_h / self.original_h
            w_ratio = self.depth_w / self.original_w
            self.intrinsics_3x3[0, :] *= w_ratio
            self.intrinsics_3x3[1, :] *= h_ratio

        n = len(os.listdir(self.rgb_dir))
        self.rgb_paths = [os.path.join(self.rgb_dir, f"{f}.jpg") for f in range(n)]
        self.depth_paths = [os.path.join(self.depth_dir, f"{f}.png") for f in range(n)]
        self.semseg_paths = [os.path.join(self.semseg_dir, f"{f}.png") for f in range(n)]
        self._poses_4x4 = torch.stack(
            [torch.tensor(np.loadtxt(os.path.join(self.pose_dir, f"{f}.txt")), dtype=torch.float32) for f in range(n)],
            dim=0,
        )

        if self.load_semseg:
            label_map_path = os.path.join(self.path, "scannetv2-labels.combined.tsv")
            label_map = pd.read_csv(label_map_path, sep="\t")
            self.scannet_to_nyu40 = {row["id"]: row["nyu40id"] for _, row in label_map.iterrows()}
            self.scannet_to_nyu40[0] = 0
            self._init_semseg_mappings({row["nyu40id"]: row["nyu40class"] for _, row in label_map.iterrows()})

    def __iter__(self):
        for f in range(len(self._poses_4x4)):
            if self.frame_skip > 0 and f % (self.frame_skip + 1) != 0:
                continue
            rgb_img = torchvision.io.read_image(self.rgb_paths[f]).type(torch.float32) / 255.0
            depth_img = torchvision.transforms.functional.pil_to_tensor(PIL.Image.open(self.depth_paths[f])).float() / 1e3
            if (self.rgb_h, self.rgb_w) != tuple(rgb_img.shape[-2:]):
                rgb_img = torch.nn.functional.interpolate(
                    rgb_img.unsqueeze(0), size=(self.rgb_h, self.rgb_w), mode=self.interp_mode, antialias=True
                ).squeeze(0)
            if (self.depth_h, self.depth_w) != tuple(depth_img.shape[-2:]):
                depth_img = torch.nn.functional.interpolate(
                    depth_img.unsqueeze(0), size=(self.depth_h, self.depth_w), mode="nearest-exact"
                ).squeeze(0)
            frame_data = dict(rgb_img=rgb_img, depth_img=depth_img, pose_4x4=self._poses_4x4[f])
            if self.load_semseg:
                img = np.array(PIL.Image.open(self.semseg_paths[f]))
                semseg_img = torch.from_numpy(np.vectorize(self.scannet_to_nyu40.get)(img)).long().unsqueeze(0)
                semseg_img = self._cat_id_to_index[semseg_img]
                if (self.rgb_h, self.rgb_w) != tuple(semseg_img.shape[-2:]):
                    semseg_img = torch.nn.functional.interpolate(
                        semseg_img.unsqueeze(0).float(), size=(self.rgb_h, self.rgb_w), mode="nearest-exact"
                    ).squeeze(0).long()
                frame_data["semseg_img"] = semseg_img
            yield frame_data
