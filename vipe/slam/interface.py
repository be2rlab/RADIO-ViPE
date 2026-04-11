# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from vipe.ext import utils_ext
from vipe.ext.lietorch import SE3
from vipe.utils.cameras import CameraType


@dataclass(kw_only=True)
class SLAMMap:
    # (M, 3) tensor of XYZ coordinates
    dense_disp_xyz: torch.Tensor
    # (M, 3) tensor of RGB colors (0-1)
    dense_disp_rgb: torch.Tensor
    # (N, V, 2) range of corresponding keyframe and view indices
    dense_disp_packinfo: torch.Tensor
    # Actual frame indices of the dense_disp_xyz (assert sorted)
    dense_disp_frame_inds: list[int]
    # Optional (M, C) tensor of embeddings aligned with dense_disp_xyz
    dense_disp_embeddings: torch.Tensor | None = None
    # Optional (M,) bool tensor signaling whether the embedding at the same index is valid
    dense_disp_embedding_valid: torch.Tensor | None = None

    def scale(self, factor: float):
        self.dense_disp_xyz *= factor

    def save(self, path: Path):
        """
        Save the SLAM map to a directory.
        """
        map_device = self.dense_disp_xyz.device
        torch.save(
            {
                "dense_disp_xyz": self.dense_disp_xyz.cpu(),
                "dense_disp_rgb": self.dense_disp_rgb.cpu(),
                "dense_disp_packinfo": self.dense_disp_packinfo.cpu(),
                "dense_disp_frame_inds": self.dense_disp_frame_inds,
                "dense_disp_embeddings": None
                if self.dense_disp_embeddings is None
                else self.dense_disp_embeddings.cpu(),
                "dense_disp_embedding_valid": None
                if self.dense_disp_embedding_valid is None
                else self.dense_disp_embedding_valid.cpu(),
                "device": map_device,
            },
            path,
        )

    @staticmethod
    def load(path: Path, device: torch.device | None = None):
        """
        Load the SLAM map from a directory.
        """
        data = torch.load(path)
        if device is None:
            device = data["device"]
        return SLAMMap(
            dense_disp_xyz=data["dense_disp_xyz"].to(device),
            dense_disp_rgb=data["dense_disp_rgb"].to(device),
            dense_disp_packinfo=data["dense_disp_packinfo"].to(device),
            dense_disp_frame_inds=data["dense_disp_frame_inds"],
            dense_disp_embeddings=None
            if data.get("dense_disp_embeddings") is None
            else data["dense_disp_embeddings"].to(device),
            dense_disp_embedding_valid=None
            if data.get("dense_disp_embedding_valid") is None
            else data["dense_disp_embedding_valid"].to(device),
        )

    @staticmethod
    def from_masked_dense_disp(
        xyz: torch.Tensor,
        rgb: torch.Tensor,
        mask: torch.Tensor,
        tstamps: torch.Tensor,
        embeddings: torch.Tensor | None = None,
        embedding_mask: torch.Tensor | None = None,
    ):
        """
        xyz: (N, V, H, W, 3)
        rgb: (N, V, H, W, 3)
        mask: (N, V, H, W)
        tstamps: (N,)
        embeddings: (N, V, H, W, C) or None
        embedding_mask: (N, V, H, W) bool mask denoting valid embeddings (defaults to mask if None)
        """
        assert torch.all(tstamps[1:] > tstamps[:-1]), "Timestamps should be sorted."
        N, V, H, W, C = xyz.shape
        xyz = xyz.reshape(-1, C)[mask.reshape(-1)]
        rgb = rgb.reshape(-1, C)[mask.reshape(-1)]
        valid_count = mask.sum([2, 3]).reshape(-1)
        packinfo = torch.stack([torch.cumsum(valid_count, 0) - valid_count, valid_count], dim=-1).reshape(N, V, 2)

        embeddings_flat = None
        embedding_valid_flat = None
        if embeddings is not None:
            assert embeddings.shape[:4] == mask.shape, "Embeddings must align with xyz/rgb grid."
            embeddings_flat = embeddings.reshape(-1, embeddings.shape[-1])[mask.reshape(-1)]
            if embedding_mask is not None:
                assert embedding_mask.shape == mask.shape, "Embedding mask must match xyz/rgb mask."
                embedding_valid_flat = embedding_mask.reshape(-1)[mask.reshape(-1)]
            else:
                embedding_valid_flat = torch.ones(
                    embeddings_flat.shape[0],
                    dtype=torch.bool,
                    device=embeddings_flat.device,
                )

        assert tstamps.shape[0] == N
        return SLAMMap(
            dense_disp_xyz=xyz,
            dense_disp_rgb=rgb,
            dense_disp_packinfo=packinfo,
            dense_disp_frame_inds=tstamps.tolist(),
            dense_disp_embeddings=embeddings_flat,
            dense_disp_embedding_valid=embedding_valid_flat,
        )

    def has_embeddings(self) -> bool:
        return self.dense_disp_embeddings is not None

    def get_dense_disp_pcd(self, keyframe_idx: int, view_idx: int = -1) -> tuple[torch.Tensor, torch.Tensor]:
        if view_idx == -1:
            xyz, color = [], []
            for v in range(self.dense_disp_packinfo.shape[1]):
                xyz_v, color_v = self.get_dense_disp_pcd(keyframe_idx, v)
                xyz.append(xyz_v)
                color.append(color_v)
            return torch.cat(xyz, dim=0), torch.cat(color, dim=0)
        else:
            start, count = self.dense_disp_packinfo[keyframe_idx, view_idx]
            return (
                self.dense_disp_xyz[start : start + count],
                self.dense_disp_rgb[start : start + count],
            )

    def get_dense_disp_pcd_with_embeddings(
        self,
        keyframe_idx: int,
        view_idx: int = -1,
        *,
        drop_invalid: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """
        Returns XYZ, RGB, embeddings, and optional validity mask for the requested keyframe/view.
        """
        if not self.has_embeddings():
            raise ValueError("This SLAM map does not contain embeddings.")

        xyz, color = self.get_dense_disp_pcd(keyframe_idx, view_idx)

        if view_idx == -1:
            emb_list, valid_list = [], []
            for v in range(self.dense_disp_packinfo.shape[1]):
                _, _, emb_v, valid_v = self.get_dense_disp_pcd_with_embeddings(keyframe_idx, v, drop_invalid=drop_invalid)
                emb_list.append(emb_v)
                if valid_v is not None:
                    valid_list.append(valid_v)
            embeddings = torch.cat(emb_list, dim=0)
            valid_mask = torch.cat(valid_list, dim=0) if valid_list else None
        else:
            start, count = self.dense_disp_packinfo[keyframe_idx, view_idx]
            embeddings = self.dense_disp_embeddings[start : start + count]
            valid_mask = (
                None
                if self.dense_disp_embedding_valid is None
                else self.dense_disp_embedding_valid[start : start + count]
            )

        if drop_invalid and valid_mask is not None:
            keep = valid_mask
            xyz = xyz[keep]
            color = color[keep]
            embeddings = embeddings[keep]
            valid_mask = valid_mask[keep]

        return xyz, color, embeddings, valid_mask

    def get_dense_disp_full_pcd(
        self,
        *,
        with_embeddings: bool = False,
        fuse_duplicates: bool = False,
        fusion_voxel_size: float = 0.01,
    ):
        """
        Returns the full point cloud of the dense disparity map.

        Args:
            with_embeddings: If True, also returns the per-point embeddings and validity mask.
            fuse_duplicates: If True, points that fall inside the same voxel (size=fusion_voxel_size)
                are merged by averaging xyz/rgb and averaging embeddings (simple mean) over valid contributors.
            fusion_voxel_size: Size of the voxel used for multiview fusion; must be > 0 when fuse_duplicates is True.
        """
        xyz_list, color_list = [], []
        embedding_list, embedding_valid_list = [], []
        n_keyframes = len(self.dense_disp_frame_inds)

        if with_embeddings and not self.has_embeddings():
            raise ValueError("Requested embeddings, but the SLAM map does not contain any.")

        for keyframe_idx in range(n_keyframes):
            if with_embeddings:
                xyz, color, emb, valid = self.get_dense_disp_pcd_with_embeddings(keyframe_idx)
                embedding_list.append(emb)
                if valid is not None:
                    embedding_valid_list.append(valid)
            else:
                xyz, color = self.get_dense_disp_pcd(keyframe_idx)

            xyz_list.append(xyz)
            color_list.append(color)

        xyz_full = torch.cat(xyz_list, dim=0)
        color_full = torch.cat(color_list, dim=0)

        embeddings_full = None
        embedding_valid_full = None
        if with_embeddings:
            embeddings_full = torch.cat(embedding_list, dim=0)
            if embedding_valid_list:
                embedding_valid_full = torch.cat(embedding_valid_list, dim=0)

        if fuse_duplicates:
            if fusion_voxel_size <= 0:
                raise ValueError("fusion_voxel_size must be > 0 when fuse_duplicates is True.")
            xyz_full, color_full, embeddings_full, embedding_valid_full = self._fuse_point_cloud(
                xyz_full,
                color_full,
                embeddings_full,
                embedding_valid_full,
                fusion_voxel_size,
            )

        if with_embeddings:
            return xyz_full, color_full, embeddings_full, embedding_valid_full
        return xyz_full, color_full

    @staticmethod
    def _fuse_point_cloud(
        xyz: torch.Tensor,
        rgb: torch.Tensor,
        embeddings: torch.Tensor | None,
        embedding_valid: torch.Tensor | None,
        voxel_size: float,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor | None,
        torch.Tensor | None,
    ]:
        """
        Merge points inside the same voxel by averaging their attributes.
        """
        quantized = torch.round(xyz / voxel_size).to(torch.int64)
        unique_coords, inverse = torch.unique(quantized, dim=0, return_inverse=True)
        counts = torch.bincount(inverse, minlength=unique_coords.shape[0]).to(xyz.dtype).unsqueeze(-1)

        fused_xyz = torch.zeros((unique_coords.shape[0], 3), dtype=xyz.dtype, device=xyz.device)
        fused_rgb = torch.zeros_like(fused_xyz)
        fused_xyz.index_add_(0, inverse, xyz)
        fused_rgb.index_add_(0, inverse, rgb)
        fused_xyz = fused_xyz / counts.clamp_min(1.0)
        fused_rgb = fused_rgb / counts.clamp_min(1.0)

        fused_embeddings = None
        fused_embedding_valid = None
        if embeddings is not None:
            emb_dim = embeddings.shape[1]
            fused_embeddings = torch.zeros((unique_coords.shape[0], emb_dim), dtype=embeddings.dtype, device=embeddings.device)
            if embedding_valid is not None:
                weights = embedding_valid.to(embeddings.dtype).unsqueeze(-1)
                fused_embeddings.index_add_(0, inverse, embeddings * weights)
                fused_counts = torch.zeros((unique_coords.shape[0], 1), dtype=embeddings.dtype, device=embeddings.device)
                fused_counts.index_add_(0, inverse, weights)
                fused_embeddings = fused_embeddings / fused_counts.clamp_min(1e-6)
                fused_embedding_valid = fused_counts.squeeze(-1) > 0.5
            else:
                fused_embeddings.index_add_(0, inverse, embeddings)
                fused_embeddings = fused_embeddings / counts.clamp_min(1e-6).to(embeddings.dtype)
                fused_embedding_valid = torch.ones(unique_coords.shape[0], dtype=torch.bool, device=embeddings.device)

        return fused_xyz, fused_rgb, fused_embeddings, fused_embedding_valid

    def project_map(
        self,
        frame_tstamp: int,
        view_idx: int,
        target_size: tuple[int, int],
        target_intrinsics: torch.Tensor,
        target_pose: SE3,
        target_camera_type: CameraType,
        infill: bool = False,
        tstamp_nn: int = 3,
    ) -> torch.Tensor:
        right_keyframe_idx = np.searchsorted(self.dense_disp_frame_inds, frame_tstamp).item()
        right_keyframe_idx = min(right_keyframe_idx + tstamp_nn, len(self.dense_disp_frame_inds) - 1)
        left_keyframe_idx = max(right_keyframe_idx - 2 * tstamp_nn, 0)

        xyz_list = []
        for keyframe_idx in range(left_keyframe_idx, right_keyframe_idx + 1):
            # If view_idx = -1 this will be all views
            xyz, _ = self.get_dense_disp_pcd(keyframe_idx, view_idx)
            xyz_list.append(xyz)
        all_xyz = torch.cat(xyz_list, dim=0)

        target_pose_mat = target_pose.inv().matrix()
        all_xyz = all_xyz @ target_pose_mat[:3, :3].T + target_pose_mat[:3, 3]

        xyz_h = torch.cat(
            [all_xyz, torch.ones(all_xyz.shape[0], device="cuda").unsqueeze(-1)],
            dim=-1,
        )
        disp = 1.0 / all_xyz[:, 2]

        camera_model = target_camera_type.build_camera_model(target_intrinsics)
        uv, _, _ = camera_model.proj_points(xyz_h, limit_min_depth=False)
        uu, vv = uv[..., 0], uv[..., 1]

        in_mask = (uu > 0) & (uu < target_size[1]) & (vv > 0) & (vv < target_size[0]) & (disp > 0)
        uu, vv, depth = uu[in_mask], vv[in_mask], disp[in_mask].reciprocal()

        if not infill:
            target_depth = torch.zeros(target_size, device="cuda")
            target_depth[vv.floor().long(), uu.floor().long()] = depth
        else:
            tree = torch.stack((uu, vv), dim=-1)
            query = torch.stack(
                torch.meshgrid(
                    torch.arange(target_size[1], device="cuda").float() + 0.5,
                    torch.arange(target_size[0], device="cuda").float() + 0.5,
                    indexing="xy",
                ),
                dim=-1,
            ).reshape(-1, 2)
            _, inds = utils_ext.nearest_neighbours(query, tree, 1)
            target_depth = depth[inds.view(-1)].reshape(target_size)
        return target_depth


@dataclass(kw_only=True)
class SLAMOutput:
    trajectory: SE3  # (N,)
    intrinsics: torch.Tensor  # (V, 4)

    rig: SE3 | None = None  # (V,)
    slam_map: SLAMMap | None = None

    # Residual of BA (unit is pixel/diagonal) -- average num of pixels/diagonal between predicted and observed flows
    # Should be of range [0, 1]
    ba_residual: float = 0.0

    @property
    def keyframe_ids(self) -> np.ndarray:
        assert self.slam_map is not None, "SLAM map not available."
        return np.array(self.slam_map.dense_disp_frame_inds)

    def get_view_trajectory(self, view_idx: int) -> SE3:
        assert self.rig is not None, "Rig not available."
        return self.trajectory * self.rig[view_idx][None]  # type: ignore
