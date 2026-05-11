"""Minimal semantic voxel map implementation."""

import torch

from semseg import geometry3d as g3d
from semseg.mapping.base import SemanticRGBDMapping


class SemanticVoxelMap(SemanticRGBDMapping):
    def __init__(
        self,
        intrinsics_3x3: torch.FloatTensor,
        device: str = None,
        clip_bbox=None,
        encoder=None,
        feat_compressor=None,
        interp_mode: str = "bilinear",
        max_pts_per_frame: int = 1000,
        vox_size: float = 1.0,
        vox_accum_period: int = 1,
    ):
        super().__init__(intrinsics_3x3, device, clip_bbox, encoder, feat_compressor, interp_mode)
        self.max_pts_per_frame = max_pts_per_frame
        self.vox_size = vox_size
        self.vox_accum_period = vox_accum_period
        self.global_vox_xyz = None
        self.global_vox_rgb_feat_conf = None
        self._vox_accum_cnt = 0
        self._tmp_pc_xyz = []
        self._tmp_pc_rgb_feat_conf = []

    @property
    def global_vox_feat(self):
        return None if self.global_vox_rgb_feat_conf is None else self.global_vox_rgb_feat_conf[:, 3:-1]

    def process_posed_rgbd(self, rgb_img, depth_img, pose_4x4, conf_map=None, feat_img=None):
        pts_xyz, selected_indices = g3d.depth_to_pointcloud(
            depth_img, pose_4x4, self.intrinsics_3x3, conf_map=conf_map, max_num_pts=self.max_pts_per_frame
        )
        pts_xyz, selected_indices = self._clip_pc(pts_xyz, selected_indices.unsqueeze(-1))
        selected_indices = selected_indices.squeeze(-1)
        b, _, dh, dw = depth_img.shape
        if rgb_img.shape[-2:] != (dh, dw):
            pts_rgb = torch.nn.functional.interpolate(rgb_img, size=(dh, dw), mode=self.interp_mode, antialias=True)
        else:
            pts_rgb = rgb_img
        pts_rgb = pts_rgb.permute(0, 2, 3, 1).reshape(-1, 3)[selected_indices]

        if self.encoder is not None:
            if feat_img is None:
                feat_img = self.encoder.encode_image_to_feat_map(rgb_img)
            feat_img = self._proj_resize_feat_map(feat_img, dh, dw)
            pts_feat = feat_img.permute(0, 2, 3, 1).reshape(-1, feat_img.shape[1])[selected_indices]
            pts_rgb_feat_conf = torch.cat((pts_rgb, pts_feat, torch.ones((pts_rgb.shape[0], 1), device=self.device)), dim=-1)
        else:
            pts_rgb_feat_conf = torch.cat((pts_rgb, torch.ones((pts_rgb.shape[0], 1), device=self.device)), dim=-1)

        self._tmp_pc_xyz.append(pts_xyz)
        self._tmp_pc_rgb_feat_conf.append(pts_rgb_feat_conf)
        self._vox_accum_cnt += b
        if self._vox_accum_cnt >= self.vox_accum_period:
            self._vox_accum_cnt = 0
            self.accum_semantic_voxels()
        return {"feat_img": feat_img} if feat_img is not None else {}

    def accum_semantic_voxels(self):
        if len(self._tmp_pc_xyz) == 0:
            return
        pts_xyz = torch.cat(self._tmp_pc_xyz)
        pts_rgb_feat_conf = torch.cat(self._tmp_pc_rgb_feat_conf, dim=0)
        self._tmp_pc_xyz.clear()
        self._tmp_pc_rgb_feat_conf.clear()
        if self.global_vox_xyz is None:
            vox_xyz, vox_rgb_feat_conf, vox_cnts = g3d.pointcloud_to_sparse_voxels(
                pts_xyz, feat_pc=pts_rgb_feat_conf, vox_size=self.vox_size, return_counts=True
            )
            vox_rgb_feat_conf[:, -1] = vox_cnts.squeeze()
            self.global_vox_xyz = vox_xyz
            self.global_vox_rgb_feat_conf = vox_rgb_feat_conf
            return
        self.global_vox_xyz, self.global_vox_rgb_feat_conf = g3d.add_weighted_sparse_voxels(
            self.global_vox_xyz, self.global_vox_rgb_feat_conf, pts_xyz, pts_rgb_feat_conf, vox_size=self.vox_size
        )
