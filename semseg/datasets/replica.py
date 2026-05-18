"""Replica dataset loader used by semseg evaluation."""

import json
import os
from typing import Tuple, Union

import PIL
import torch
import torchvision

from semseg.datasets.base import SemSegDataset


class NiceReplicaDataset(SemSegDataset):
    def __init__(
        self,
        path: str,
        scene_name: str,
        rgb_resolution: Union[Tuple[int], int] = None,
        depth_resolution: Union[Tuple[int], int] = None,
        frame_skip: int = 0,
        interp_mode: str = "bilinear",
    ):
        super().__init__(rgb_resolution=rgb_resolution, depth_resolution=depth_resolution, frame_skip=frame_skip, interp_mode=interp_mode)
        self.path = path
        self.scene_name = scene_name
        self.original_h = 680
        self.original_w = 1200
        self.rgb_h = self.original_h if self.rgb_h <= 0 else self.rgb_h
        self.rgb_w = self.original_w if self.rgb_w <= 0 else self.rgb_w
        self.depth_h = self.original_h if self.depth_h <= 0 else self.depth_h
        self.depth_w = self.original_w if self.depth_w <= 0 else self.depth_w

        with open(os.path.join(self.path, "cam_params.json"), "r", encoding="UTF-8") as f:
            cam_params = json.load(f)["camera"]
        self.intrinsics_3x3 = torch.tensor(
            [[cam_params["fx"], 0, cam_params["cx"]], [0, cam_params["fy"], cam_params["cy"]], [0, 0, 1]]
        )
        if self.depth_h != self.original_h or self.depth_w != self.original_w:
            h_ratio = self.depth_h / self.original_h
            w_ratio = self.depth_w / self.original_w
            self.intrinsics_3x3[0, :] = self.intrinsics_3x3[0, :] * w_ratio
            self.intrinsics_3x3[1, :] = self.intrinsics_3x3[1, :] * h_ratio

        self._depth_scale = cam_params["scale"]
        self._poses_4x4 = []
        with open(os.path.join(path, scene_name, "traj.txt"), "r", encoding="UTF-8") as traj_file:
            for traj_line in traj_file:
                self._poses_4x4.append(torch.tensor([float(x) for x in traj_line.strip().split()]).reshape(4, 4))
        self._poses_4x4 = torch.stack(self._poses_4x4, dim=0)
        n = len(self._poses_4x4)
        imgs_dir = os.path.join(self.path, self.scene_name, "results")
        self._depth_paths = [os.path.join(imgs_dir, f"depth{f:06d}.png") for f in range(n)]
        self._rgb_paths = [os.path.join(imgs_dir, f"frame{f:06d}.jpg") for f in range(n)]

        semseg_info_f = os.path.join(
            self.path,
            "semantic_info",
            self.scene_name[:-1] + "_" + self.scene_name[-1],
            "info_semantic.json",
        )
        if os.path.exists(semseg_info_f):
            with open(semseg_info_f, "r", encoding="UTF-8") as f:
                semseg_info = json.load(f)
            self._cat_id_to_name = {item["id"]: item["name"] for item in semseg_info["classes"]}
            self._init_semseg_mappings(self._cat_id_to_name)

    def __iter__(self):
        for f in range(len(self._poses_4x4)):
            if self.frame_skip > 0 and f % (self.frame_skip + 1) != 0:
                continue
            rgb_img = torchvision.io.read_image(self._rgb_paths[f]).type(torch.float) / 255
            depth_img = torchvision.transforms.functional.pil_to_tensor(PIL.Image.open(self._depth_paths[f]))
            depth_img = depth_img / self._depth_scale
            depth_img[depth_img == 0] = torch.nan
            if (self.rgb_h, self.rgb_w) != tuple(rgb_img.shape[-2:]):
                rgb_img = torch.nn.functional.interpolate(
                    rgb_img.unsqueeze(0), size=(self.rgb_h, self.rgb_w), mode=self.interp_mode, antialias=True
                ).squeeze(0)
            if (self.depth_h, self.depth_w) != tuple(depth_img.shape[-2:]):
                depth_img = torch.nn.functional.interpolate(
                    depth_img.unsqueeze(0), size=(self.depth_h, self.depth_w), mode="nearest-exact"
                ).squeeze(0)
            yield dict(rgb_img=rgb_img, depth_img=depth_img, pose_4x4=self._poses_4x4[f])
