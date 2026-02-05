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

import logging

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from einops import rearrange

from vipe.ext.lietorch import SE3
from vipe.utils.cameras import CameraType

from ..maths import geom
from ..maths.matrix import SparseBlockMatrix, SparseDenseBlockMatrix, SparseMDiagonalBlockMatrix
from ..maths.vector import SparseBlockVector
from .kernel import RobustKernel


logger = logging.getLogger(__name__)


class TermEvalReturn(ABC):
    @abstractmethod
    def jtwj(self, group_name_row: str, group_name_col: str) -> SparseBlockMatrix: ...

    @abstractmethod
    def nwjtr(self, group_name: str) -> SparseBlockVector: ...

    @abstractmethod
    def remove_jcol_inds(self, group_name: str, col_inds: torch.Tensor): ...

    @abstractmethod
    def residual(self) -> torch.Tensor: ...

    def apply_robust_kernel(self, kernel: RobustKernel):
        raise NotImplementedError


@dataclass(kw_only=True)
class ConcreteTermEvalReturn(TermEvalReturn):
    J: dict[str, SparseBlockMatrix]  # group_name -> (n_occ, res_dim, manifold_dim)
    w: torch.Tensor  # (n_terms, res_dim, )
    r: torch.Tensor  # (n_terms, res_dim, )
    alpha: torch.Tensor | None = None

    # n_occ = number of occurrences of this group_name in all the terms.
    # i.e. the number of blocks in the sparse Jacobian matrix with size n_terms x n_vars

    def jtwj(self, group_name_row: str, group_name_col: str) -> SparseBlockMatrix:
        wJ = self.J[group_name_col].scale_w_left(self.w)
        try:
            return self.J[group_name_row].tmult_mat(wJ).coalesce()
        except NotImplementedError:
            return wJ.tmult_mat(self.J[group_name_row]).transpose().coalesce()

    def nwjtr(self, group_name: str) -> SparseBlockVector:
        return self.J[group_name].tmult_vec(-self.w * self.r).coalesce()

    def remove_jcol_inds(self, group_name: str, col_inds: torch.Tensor):
        j_group = self.J[group_name]
        keep_mask = torch.isin(j_group.j_inds, col_inds, invert=True)
        self.J[group_name] = j_group.subset(keep_mask)

    def apply_robust_kernel(self, kernel: RobustKernel):
        robust_weight = kernel.apply(self.r,self.alpha)
        self.w = self.w * robust_weight

    def residual(self) -> torch.Tensor:
        return torch.sum(self.r * self.r * self.w, dim=1)


class SolverTerm(ABC):
    @abstractmethod
    def forward(self, variables: dict[str, Any], jacobian: bool = True) -> TermEvalReturn: ...

    @abstractmethod
    def group_names(self) -> set[str]: ...

    def is_active(self) -> bool:
        return True

    def update(self, solver):
        # Default implementation do nothing.
        pass


class DenseDepthFlowTerm(SolverTerm):
    """
    E(pose_pi, pose_pj, dense_disp_di, intr_qi, intr_qj) = \
        proj(rig_j.inv() * pose_j * pose_i.inv() * rig_i, dense_disp_di) - target_[ij di]

        Pose is the world2cam transform.
        Rig is the cam2world(central cam) transform.
        target_[ij di] is the target projected location.
    res_dim = H*W*2
    """

    def __init__(
        self,
        pose_i_inds: torch.Tensor,
        pose_j_inds: torch.Tensor,
        rig_i_inds: torch.Tensor,
        rig_j_inds: torch.Tensor,
        dense_disp_i_inds: torch.Tensor,
        target: torch.Tensor,
        weight: torch.Tensor,
        intrinsics: torch.Tensor | None,
        intrinsics_factor: float,
        rig: SE3 | None,
        image_size: tuple[int, int],
        camera_type: CameraType,
        dense_disp_j_inds: torch.Tensor,
        embeddings: torch.Tensor,
        embedding_weight: torch.Tensor | float,
        embedding_valid_mask: torch.Tensor | None = None,
        chunk_size: int = 4,
        residual_scale: float = 1.0,
        use_photometric_residual: bool = False,
        debug_options: dict[str, Any] | None = None,
        use_semantic_kernel: bool= False
    ) -> None:
        super().__init__()

        self.n_terms = pose_i_inds.shape[0]
        assert pose_i_inds.shape == (self.n_terms,)
        assert pose_j_inds.shape == (self.n_terms,)
        assert rig_i_inds.shape == (self.n_terms,)
        assert rig_j_inds.shape == (self.n_terms,)
        assert dense_disp_i_inds.shape == (self.n_terms,)
        assert dense_disp_j_inds.shape == (self.n_terms,)

        self.use_semantic_kernel = use_semantic_kernel
        self.pose_i_inds = pose_i_inds
        self.pose_j_inds = pose_j_inds
        self.rig_i_inds = rig_i_inds
        self.rig_j_inds = rig_j_inds
        self.dense_disp_i_inds = dense_disp_i_inds
        self.dense_disp_j_inds = dense_disp_j_inds
        self.chunk_size = chunk_size
        self.residual_scale = residual_scale
        self.use_photometric_residual = use_photometric_residual
        self.image_size = image_size
        self.camera_type = camera_type

        n_pixels = image_size[0] * image_size[1]

        self.target = target.reshape(self.n_terms, n_pixels, 2)  # (n_terms, H*W, 2)
        self.weight = weight.reshape(self.n_terms, n_pixels, 2)  # (n_terms, H*W, 2)
        self.intrinsics = intrinsics.reshape(-1, 4) if intrinsics is not None else None  # (Q, 4)
        self.intrinsics_factor = intrinsics_factor
        self.rig = rig
        self.alpha = None

        # Store raw embeddings and a pre-normalized copy for the *source* (no sampling on source)
        self.embeddings = embeddings
        if embeddings is not None:
            assert embeddings.dim() == 4, "Embeddings must be shaped (N_views, C, H, W)"
            self.embeddings = embeddings.float()
            self.embeddings_norm = F.normalize(self.embeddings, dim=1, eps=1e-8)

            self.embedding_valid_mask = embedding_valid_mask
            if embedding_valid_mask is not None:
                assert embedding_valid_mask.dim() == 3, (
                    f"embedding_valid_mask must be 3D (N_views, H, W), got {embedding_valid_mask.dim()}D"
                )
                assert embedding_valid_mask.shape[1:] == image_size, (
                    f"embedding_valid_mask spatial dims {embedding_valid_mask.shape[1:]} must match image_size {image_size}"
                )


            self.height, self.width = image_size
            assert embeddings.shape[2] == self.height and embeddings.shape[3] == self.width
            self.embedding_dim = embeddings.shape[1]
            self.embedding_weight, self._has_weight_map = self._prepare_embedding_weight(embedding_weight)

            self._debug_options = debug_options or {}
            self.debug_enabled = self._parse_bool(self._debug_options.get("enabled", False))
            self.debug_log_every = max(1, self._parse_int(self._debug_options.get("log_every", 1)))
            self.debug_log_stats = self._parse_bool(self._debug_options.get("log_stats", self.debug_enabled))
            self.debug_visualize = self.debug_enabled and self._parse_bool(self._debug_options.get("save_heatmaps", False))
            self.debug_samples_per_snapshot = max(1, self._parse_int(self._debug_options.get("samples_per_snapshot", 1)))
            self.debug_max_snapshots = max(0, self._parse_int(self._debug_options.get("max_snapshots", 0)))
            output_dir_value = self._debug_options.get("output_dir", "vipe_results/debug_embedding")
            self.debug_dir = (
                Path(output_dir_value)
                if (self.debug_visualize and self.debug_max_snapshots > 0 and output_dir_value not in (None, "null"))
                else None
            )
            if self.debug_dir is not None:
                self.debug_dir.mkdir(parents=True, exist_ok=True)
            self._forward_calls = 0
            self._debug_saved = 0
            self._active = True

    def group_names(self) -> set[str]:
        names = {"pose", "dense_disp"}
        if self.intrinsics is None:
            names.add("intrinsics")
        if self.rig is None:
            names.add("rig")
        return names

    def forward(self, variables: dict[str, Any], jacobian: bool = True) -> TermEvalReturn:
        """
        variables contain:
            - pose: (n_var, ) SE3 of poses
            - dense_disp: (n_var, H*W) tensor of disparities
            - intrinsics: (Q, 4) tensor of intrinsics (optional)

        # TODO: To accelerate, you can return a PrecomputedTermEvalReturn with kernels from Droid-SLAM.
        """
        pose, dense_disp = variables["pose"], variables["dense_disp"]
        if optimize_intrinsics := self.intrinsics is None:
            intrinsics = variables["intrinsics"]
        else:
            intrinsics = self.intrinsics
        if optimize_rig := self.rig is None:
            rig = variables["rig"]
        else:
            rig = self.rig

        assert isinstance(pose, SE3) and isinstance(dense_disp, torch.Tensor)
        assert dense_disp.shape[1] == self.image_size[0] * self.image_size[1]
        assert intrinsics.shape[0] == rig.shape[0]

        camera_model_cls = self.camera_type.camera_model_cls()

        coords, valid, (Ji, Jj, Jz), (Jfi, Jfj), (Jri, Jrj) = geom.iproj_i_proj_j_disp(
            pose,
            dense_disp.view(-1, self.image_size[0], self.image_size[1]),
            None,
            (camera_model_cls(intrinsics).scaled(1.0 / self.intrinsics_factor).intrinsics),
            self.camera_type,
            rig,
            self.pose_i_inds,
            self.pose_j_inds,
            self.rig_i_inds,
            self.rig_j_inds,
            self.dense_disp_i_inds,
            jacobian_p_d=jacobian,
            jacobian_f=jacobian and optimize_intrinsics,
            jacobian_r=jacobian and optimize_rig,
        )
        coords = rearrange(coords, "n h w c -> n (h w) c", c=2)
        weight = rearrange(valid, "n h w 1 -> n (h w) 1") * self.weight  # (n_terms, H*W, 2)
        weight = rearrange(weight, "n hw c -> n (hw c)", c=2)

        J_dict = {}
        if jacobian:
            assert Ji is not None and Jj is not None and Jz is not None
            Ji = rearrange(Ji, "n h w c d -> n (h w c) d", c=2, d=6)
            Jj = rearrange(Jj, "n h w c d -> n (h w c) d", c=2, d=6)
            Jz = rearrange(Jz, "n h w c d -> n (h w) (c d)", c=2, d=1)
            term_inds = torch.arange(self.n_terms).to(pose.device)
            J_dict = {
                "pose": SparseDenseBlockMatrix(
                    i_inds=torch.cat([term_inds, term_inds]),
                    j_inds=torch.cat([self.pose_i_inds, self.pose_j_inds]),
                    data=torch.cat([Ji, Jj], dim=0),
                ),
                "dense_disp": SparseMDiagonalBlockMatrix(
                    i_inds=term_inds,
                    j_inds=self.dense_disp_i_inds,
                    data=Jz,
                ),
            }
            if optimize_intrinsics:
                assert Jfi is not None and Jfj is not None
                Jfi = rearrange(Jfi, "n h w c d -> n (h w c) d", c=2)
                Jfj = rearrange(Jfj, "n h w c d -> n (h w c) d", c=2)
                J_dict["intrinsics"] = SparseDenseBlockMatrix(
                    i_inds=torch.cat([term_inds, term_inds]),
                    j_inds=torch.cat([self.rig_i_inds, self.rig_j_inds]),
                    data=camera_model_cls.J_scale(
                        1.0 / self.intrinsics_factor,
                        torch.cat([Jfi, Jfj], dim=0),
                    ),
                )
            if optimize_rig:
                assert Jri is not None and Jrj is not None
                Jri = rearrange(Jri, "n h w c d -> n (h w c) d", c=2, d=6)
                Jrj = rearrange(Jrj, "n h w c d -> n (h w c) d", c=2, d=6)
                J_dict["rig"] = SparseDenseBlockMatrix(
                    i_inds=torch.cat([term_inds, term_inds]),
                    j_inds=torch.cat([self.rig_i_inds, self.rig_j_inds]),
                    data=torch.cat([Jri, Jrj], dim=0),
                )
        if self.embeddings is not None and self.use_semantic_kernel:
            emedding_redisdual, embedding_residual_weights = self.compute_embedding_residuals(coords,valid,'cuda:0')
            emedding_redisdual = 1- emedding_redisdual
            self.calculate_alpha(emedding_redisdual,embedding_residual_weights)
            self.alpha = torch.repeat_interleave(self.alpha, 2, dim=1)
        return ConcreteTermEvalReturn(
            J=J_dict,
            w=weight,
            r=rearrange(coords - self.target, "n hw c -> n (hw c)", c=2),
            alpha = self.alpha
        )
    

    def compute_embedding_residuals(self, coords,valid,device) -> TermEvalReturn:

        coords = coords.view(self.n_terms, self.height, self.width, 2)
        valid = valid.view(self.n_terms, self.height, self.width, 1)

        pixel_count = self.height * self.width
        dtype = coords.dtype

        # Pre-allocate outputs
        residual = torch.zeros(self.n_terms, pixel_count, device=device, dtype=dtype)
        weight = torch.zeros(self.n_terms, pixel_count, device=device, dtype=dtype)


        # Chunk to save memory
        for chunk_start in range(0, self.n_terms, self.chunk_size):
            chunk_end = min(chunk_start + self.chunk_size, self.n_terms)
            cs = slice(chunk_start, chunk_end)
            cs_offset = slice(chunk_start + self.n_terms, chunk_end + self.n_terms)

            src_idx = self.dense_disp_i_inds[cs]
            tgt_idx = self.dense_disp_j_inds[cs]

            # Normalized source (no sampling)
            source_norm = self.embeddings_norm[src_idx]  # (N, C, H, W)

            # Target: raw features (will be normalized AFTER sampling)
            target_raw_full = self.embeddings[tgt_idx]  # (N, C, H, W)

            coords_chunk = coords[cs]  # (N, H, W, 2)
            valid_chunk = valid[cs]  # (N, H, W, 1)

            # Optional embedding validity masks (src & tgt)
            if self.embedding_valid_mask is not None:
                src_valid = self.embedding_valid_mask[src_idx].unsqueeze(-1)  # (N, H, W, 1)
                tgt_valid = self.embedding_valid_mask[tgt_idx].unsqueeze(-1)  # (N, H, W, 1)
                valid_chunk = valid_chunk * src_valid * tgt_valid

            valid_map = valid_chunk.permute(0, 3, 1, 2)  # (N, 1, H, W)

            # Per-pixel weight for this term
            if self._has_weight_map:
                base_weight = self.embedding_weight[cs]
            else:
                base_weight = self.embedding_weight
            weight_chunk = base_weight * valid_map.squeeze(1)
            weight[cs] = weight_chunk.view(-1, pixel_count)

            # Skip if fully invalid
            if not torch.any(valid_map.view(-1, pixel_count).sum(dim=1) > 0.0):
                continue
            # Bilinear sample raw target features (and spatial grads if needed)
            target_raw_sampled, _, _ = self.bilinear_sample_with_grad(
                target_raw_full, coords_chunk, return_grad=False
            )
            # Residual only path
            eps = 1e-8
            u = target_raw_sampled
            u_norm = u.norm(dim=1, keepdim=True).clamp_min(eps)
            t_norm = u / u_norm
            cos_sim = (source_norm * t_norm).sum(dim=1)
            if self.use_photometric_residual:
                residual_chunk = self.residual_scale * torch.sqrt(2.0 * (1.0 - cos_sim).clamp(min=0.0))
            else:
                residual_chunk = 1.0 - cos_sim
            residual_chunk = residual_chunk * valid_map.squeeze(1)

            residual[cs] = residual_chunk.view(-1, pixel_count)
        return residual, weight

    def calculate_alpha(self, embedding_residual, embedding_residual_weights, alpha_min=-10, alpha_max=2):
        x = embedding_residual
        
        # Create a mask for values below 0.6
        mask_below_threshold = x < 0.4
        
        # Normalize x from [0.6, 1] to [0, 1] for values >= 0.6
        x_normalized = (x - 0.4) / (1 - 0.4)  # This gives range [0, 1] for x in [0.6, 1]
        x_normalized = torch.clamp(x_normalized, 0, 1)  # Ensure values are within [0, 1]
        
        # Apply sigmoid transformation only to values >= 0.6
        sigmoid_output = (alpha_max - alpha_min) / (1 + torch.exp(-(x_normalized - 0.5) / 0.1)) + alpha_min
        
        # Set alpha to -20 for values below 0.6, otherwise use sigmoid output
        self.alpha = torch.where(mask_below_threshold, torch.tensor(alpha_min, dtype=x.dtype, device=x.device), sigmoid_output)

    # def calculate_alpha(self, embedding_residual,embedding_residual_weights,alpha_min = -20, alpha_max = 2):
    #         x = embedding_residual
    #         self.alpha = (alpha_max - alpha_min) / (1 + torch.exp(-(x - 0.5) / 0.1)) + alpha_min


    def _prepare_embedding_weight(self, weight: torch.Tensor | float) -> tuple[torch.Tensor, bool]:
        device = self.embeddings.device
        dtype = self.embeddings.dtype
        if isinstance(weight, torch.Tensor):
            weight_tensor = weight.detach().to(device=device, dtype=dtype)
            if weight_tensor.ndim == 0:
                return weight_tensor, False
            if weight_tensor.ndim == 2:
                expected = (self.n_terms, self.height * self.width)
                if weight_tensor.shape != expected:
                    raise ValueError(
                        f"Embedding weight tensor must have shape {expected} when 2D, got {tuple(weight_tensor.shape)}"
                    )
                return weight_tensor.view(self.n_terms, self.height, self.width).contiguous(), True
            if weight_tensor.ndim == 3:
                if tuple(weight_tensor.shape[1:]) != (self.height, self.width):
                    raise ValueError(
                        "Embedding weight spatial shape must match image_size; got "
                        f"{tuple(weight_tensor.shape[1:])} vs {(self.height, self.width)}"
                    )
                if weight_tensor.shape[0] == 1 and self.n_terms > 1:
                    weight_tensor = weight_tensor.expand(self.n_terms, -1, -1)
                elif weight_tensor.shape[0] != self.n_terms:
                    raise ValueError(
                        f"Embedding weight batch dimension must be 1 or {self.n_terms}; got {weight_tensor.shape[0]}"
                    )
                return weight_tensor.contiguous(), True
            raise ValueError(
                "Embedding weight tensor must be scalar, (n_terms, H*W), or (n_terms, H, W); "
                f"got ndim={weight_tensor.ndim}"
            )
        return torch.tensor(float(weight), device=device, dtype=dtype), False

    @staticmethod
    def bilinear_sample_with_grad(
        features: torch.Tensor, coords: torch.Tensor, return_grad: bool = True
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        """
        Optimized bilinear sampling with analytical gradients.

        Args:
            features: (N, C, H, W) float32
            coords:   (N, H, W, 2) in pixel (x, y)
        Returns:
            samples: (N, C, H, W)
            grad_u:  (N, C, H, W) if return_grad else None
            grad_v:  (N, C, H, W) if return_grad else None
        """
        N, C, Hf, Wf = features.shape
        _, Hc, Wc, _ = coords.shape

        device = features.device
        u = coords[..., 0]
        v = coords[..., 1]

        u_clamped = u.clamp(0.0, Wf - 1.0)
        v_clamped = v.clamp(0.0, Hf - 1.0)

        x0 = u_clamped.floor()
        y0 = v_clamped.floor()
        x0_int = x0.long()
        y0_int = y0.long()
        x1_int = (x0_int + 1).clamp(max=Wf - 1)
        y1_int = (y0_int + 1).clamp(max=Hf - 1)

        du = u_clamped - x0
        dv = v_clamped - y0

        flat_features = features.view(N, C, -1)

        idx_00 = (y0_int * Wf + x0_int).view(N, 1, -1).expand(-1, C, -1)
        idx_10 = (y0_int * Wf + x1_int).view(N, 1, -1).expand(-1, C, -1)
        idx_01 = (y1_int * Wf + x0_int).view(N, 1, -1).expand(-1, C, -1)
        idx_11 = (y1_int * Wf + x1_int).view(N, 1, -1).expand(-1, C, -1)

        F00 = torch.gather(flat_features, 2, idx_00).view(N, C, Hc, Wc)
        F10 = torch.gather(flat_features, 2, idx_10).view(N, C, Hc, Wc)
        F01 = torch.gather(flat_features, 2, idx_01).view(N, C, Hc, Wc)
        F11 = torch.gather(flat_features, 2, idx_11).view(N, C, Hc, Wc)

        du_exp = du.unsqueeze(1)
        dv_exp = dv.unsqueeze(1)

        w00 = (1 - du_exp) * (1 - dv_exp)
        w10 = du_exp * (1 - dv_exp)
        w01 = (1 - du_exp) * dv_exp
        w11 = du_exp * dv_exp

        samples = F00 * w00 + F10 * w10 + F01 * w01 + F11 * w11

        if not return_grad:
            return samples, None, None

        grad_u = (F10 - F00) * (1 - dv_exp) + (F11 - F01) * dv_exp
        grad_v = (F01 - F00) * (1 - du_exp) + (F11 - F10) * du_exp
        return samples, grad_u, grad_v
    

    @staticmethod
    def _parse_bool(value: Any, default: bool = False) -> bool:
        if value is None:
            return default
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in ("", "none", "null"):
                return default
            return lowered in ("1", "true", "yes", "on")
        return bool(value)
    
    @staticmethod
    def _parse_int(value: Any, default: int = 0) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return default



class EmbeddingSimilarityTerm(SolverTerm):
    """
    Embedding similarity term with correct photometric-style residuals and Jacobians.

    Residual (per pixel):
        r = residual_scale * sqrt( 2 * (1 - cos(s, t)) )       if use_photometric_residual
        r = 1 - cos(s, t)                                      otherwise

    where:
        s = normalized source embedding at (i)
        t = normalized target embedding at projected coords in (j), with
            normalization done *after* bilinear sampling of the raw target features.

    The Jacobian correctly accounts for the normalization of the sampled target feature:
        d cos(s, t) / d u = (s - (s·t) t) / ||u||,   u = target_raw_sampled,  t = u / ||u||
        dr/dc = -(residual_scale**2) / r   (photometric style) or -1 (plain 1 - cos)

    Notes:
    - intrinsics_factor default is 1.0. Scaling guards against division by zero.
    """

    def __init__(
        self,
        *,
        pose_i_inds: torch.Tensor,
        pose_j_inds: torch.Tensor,
        rig_i_inds: torch.Tensor,
        rig_j_inds: torch.Tensor,
        dense_disp_i_inds: torch.Tensor,
        dense_disp_j_inds: torch.Tensor,
        embeddings: torch.Tensor,
        weight: torch.Tensor | float,
        image_size: tuple[int, int],
        camera_type: CameraType,
        embedding_valid_mask: torch.Tensor | None = None,
        intrinsics: torch.Tensor | None = None,
        intrinsics_factor: float = 8.0,
        rig: SE3 | None = None,
        chunk_size: int = 4,
        residual_scale: float = 1.0,
        use_photometric_residual: bool = False,
        debug_options: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self.n_terms = pose_i_inds.shape[0]
        assert pose_i_inds.shape == (self.n_terms,)
        assert pose_j_inds.shape == (self.n_terms,)
        assert rig_i_inds.shape == (self.n_terms,)
        assert rig_j_inds.shape == (self.n_terms,)
        assert dense_disp_i_inds.shape == (self.n_terms,)
        assert dense_disp_j_inds.shape == (self.n_terms,)

        self.pose_i_inds = pose_i_inds
        self.pose_j_inds = pose_j_inds
        self.rig_i_inds = rig_i_inds
        self.rig_j_inds = rig_j_inds
        self.dense_disp_i_inds = dense_disp_i_inds
        self.dense_disp_j_inds = dense_disp_j_inds
        self.camera_type = camera_type

        # Store raw embeddings and a pre-normalized copy for the *source* (no sampling on source)
        assert embeddings.dim() == 4, "Embeddings must be shaped (N_views, C, H, W)"
        self.embeddings = embeddings.float()
        self.embeddings_norm = F.normalize(self.embeddings, dim=1, eps=1e-8)

        self.embedding_valid_mask = embedding_valid_mask
        if embedding_valid_mask is not None:
            assert embedding_valid_mask.dim() == 3, (
                f"embedding_valid_mask must be 3D (N_views, H, W), got {embedding_valid_mask.dim()}D"
            )
            assert embedding_valid_mask.shape[1:] == image_size, (
                f"embedding_valid_mask spatial dims {embedding_valid_mask.shape[1:]} must match image_size {image_size}"
            )

        self.image_size = image_size
        self.height, self.width = image_size
        assert embeddings.shape[2] == self.height and embeddings.shape[3] == self.width
        self.embedding_dim = embeddings.shape[1]
        self.embedding_weight, self._has_weight_map = self._prepare_embedding_weight(weight)
        self.chunk_size = chunk_size
        self.residual_scale = residual_scale
        self.use_photometric_residual = use_photometric_residual

        self.intrinsics = intrinsics.reshape(-1, 4) if intrinsics is not None else None
        self.intrinsics_factor = intrinsics_factor
        self.rig = rig
        self._debug_options = debug_options or {}
        self.debug_enabled = self._parse_bool(self._debug_options.get("enabled", False))
        self.debug_log_every = max(1, self._parse_int(self._debug_options.get("log_every", 1)))
        self.debug_log_stats = self._parse_bool(self._debug_options.get("log_stats", self.debug_enabled))
        self.debug_visualize = self.debug_enabled and self._parse_bool(self._debug_options.get("save_heatmaps", False))
        self.debug_samples_per_snapshot = max(1, self._parse_int(self._debug_options.get("samples_per_snapshot", 1)))
        self.debug_max_snapshots = max(0, self._parse_int(self._debug_options.get("max_snapshots", 0)))
        output_dir_value = self._debug_options.get("output_dir", "vipe_results/debug_embedding")
        self.debug_dir = (
            Path(output_dir_value)
            if (self.debug_visualize and self.debug_max_snapshots > 0 and output_dir_value not in (None, "null"))
            else None
        )
        if self.debug_dir is not None:
            self.debug_dir.mkdir(parents=True, exist_ok=True)
        self._forward_calls = 0
        self._debug_saved = 0
        self._active = True

    def group_names(self) -> set[str]:
        names = {"pose", "dense_disp"}
        if self.intrinsics is None:
            names.add("intrinsics")
        if self.rig is None:
            names.add("rig")
        return names

    def set_active(self, active: bool) -> None:
        self._active = active

    def is_active(self) -> bool:
        return self._active

    @staticmethod
    def bilinear_sample_with_grad(
        features: torch.Tensor, coords: torch.Tensor, return_grad: bool = True
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        """
        Optimized bilinear sampling with analytical gradients.

        Args:
            features: (N, C, H, W) float32
            coords:   (N, H, W, 2) in pixel (x, y)
        Returns:
            samples: (N, C, H, W)
            grad_u:  (N, C, H, W) if return_grad else None
            grad_v:  (N, C, H, W) if return_grad else None
        """
        N, C, Hf, Wf = features.shape
        _, Hc, Wc, _ = coords.shape

        device = features.device
        u = coords[..., 0]
        v = coords[..., 1]

        u_clamped = u.clamp(0.0, Wf - 1.0)
        v_clamped = v.clamp(0.0, Hf - 1.0)

        x0 = u_clamped.floor()
        y0 = v_clamped.floor()
        x0_int = x0.long()
        y0_int = y0.long()
        x1_int = (x0_int + 1).clamp(max=Wf - 1)
        y1_int = (y0_int + 1).clamp(max=Hf - 1)

        du = u_clamped - x0
        dv = v_clamped - y0

        flat_features = features.view(N, C, -1)

        idx_00 = (y0_int * Wf + x0_int).view(N, 1, -1).expand(-1, C, -1)
        idx_10 = (y0_int * Wf + x1_int).view(N, 1, -1).expand(-1, C, -1)
        idx_01 = (y1_int * Wf + x0_int).view(N, 1, -1).expand(-1, C, -1)
        idx_11 = (y1_int * Wf + x1_int).view(N, 1, -1).expand(-1, C, -1)

        F00 = torch.gather(flat_features, 2, idx_00).view(N, C, Hc, Wc)
        F10 = torch.gather(flat_features, 2, idx_10).view(N, C, Hc, Wc)
        F01 = torch.gather(flat_features, 2, idx_01).view(N, C, Hc, Wc)
        F11 = torch.gather(flat_features, 2, idx_11).view(N, C, Hc, Wc)

        du_exp = du.unsqueeze(1)
        dv_exp = dv.unsqueeze(1)

        w00 = (1 - du_exp) * (1 - dv_exp)
        w10 = du_exp * (1 - dv_exp)
        w01 = (1 - du_exp) * dv_exp
        w11 = du_exp * dv_exp

        samples = F00 * w00 + F10 * w10 + F01 * w01 + F11 * w11

        if not return_grad:
            return samples, None, None

        grad_u = (F10 - F00) * (1 - dv_exp) + (F11 - F01) * dv_exp
        grad_v = (F01 - F00) * (1 - du_exp) + (F11 - F10) * du_exp
        return samples, grad_u, grad_v

    def compute_residual_and_jacobian(
        self,
        source_norm: torch.Tensor,  # (N, C, H, W), normalized
        target_raw_sampled: torch.Tensor,  # (N, C, H, W), raw sampled
        grad_u: torch.Tensor,  # (N, C, H, W)
        grad_v: torch.Tensor,  # (N, C, H, W)
        valid_map: torch.Tensor,  # (N, 1, H, W)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute residual and gradients w.r.t. image coordinates, accounting for
        normalization after sampling of the target features.

        Returns:
            residual:   (N, H, W)
            grad_coords:(N, H, W, 2)
        """
        eps = 1e-8

        # Normalize AFTER sampling
        u = target_raw_sampled
        u_norm = u.norm(dim=1, keepdim=True).clamp_min(eps)
        t_norm = u / u_norm  # (N, C, H, W)

        # Cosine similarity and residual
        cos_sim = (source_norm * t_norm).sum(dim=1)  # (N, H, W)

        if self.use_photometric_residual:
            # r = s * sqrt(2(1 - c))
            residual = self.residual_scale * torch.sqrt(2.0 * (1.0 - cos_sim).clamp(min=0.0))
            # dr/dc = -(s^2)/r
            grad_scale = -(self.residual_scale**2) / (residual.clamp(min=1e-6) + 1e-8)
        else:
            # r = 1 - c
            residual = 1.0 - cos_sim
            grad_scale = cos_sim.new_full(cos_sim.shape, -1.0)

        # Mask residuals
        residual = residual * valid_map.squeeze(1)

        # d cos / d u = (s - (s·t) t) / ||u||
        proj = source_norm - (cos_sim.unsqueeze(1) * t_norm)  # (N, C, H, W)
        dcos_du = proj / u_norm

        # dr/du = (dr/dc) * (dc/du)
        grad_u_feat = grad_scale.unsqueeze(1) * dcos_du
        grad_u_feat = grad_u_feat * valid_map  # mask invalid

        # Chain to coords via bilinear sampling grads
        grad_u_weighted = (grad_u_feat * grad_u).sum(dim=1)  # (N, H, W)
        grad_v_weighted = (grad_u_feat * grad_v).sum(dim=1)  # (N, H, W)
        grad_coords = torch.stack([grad_u_weighted, grad_v_weighted], dim=-1)  # (N, H, W, 2)

        return residual, grad_coords

    def forward(self, variables: dict[str, Any], jacobian: bool = True) -> TermEvalReturn:
        self._forward_calls += 1
        pose, dense_disp = variables["pose"], variables["dense_disp"]

        if optimize_intrinsics := self.intrinsics is None:
            intrinsics = variables["intrinsics"]
        else:
            intrinsics = self.intrinsics

        if optimize_rig := self.rig is None:
            rig = variables["rig"]
        else:
            rig = self.rig

        assert isinstance(pose, SE3) and isinstance(dense_disp, torch.Tensor)
        assert dense_disp.shape[1] == self.height * self.width
        assert intrinsics.shape[0] == rig.shape[0]

        camera_model_cls = self.camera_type.camera_model_cls()
        # Guard scaling factor
        scale = (
            1.0 / self.intrinsics_factor
            if (self.intrinsics_factor is not None and self.intrinsics_factor != 0.0)
            else 1.0
        )

        coords, valid, (Ji, Jj, Jz), (Jfi, Jfj), (Jri, Jrj) = geom.iproj_i_proj_j_disp(
            pose,
            dense_disp.view(-1, self.height, self.width),
            None,
            (camera_model_cls(intrinsics).scaled(scale).intrinsics),
            self.camera_type,
            rig,
            self.pose_i_inds,
            self.pose_j_inds,
            self.rig_i_inds,
            self.rig_j_inds,
            self.dense_disp_i_inds,
            jacobian_p_d=jacobian,
            jacobian_f=jacobian and optimize_intrinsics,
            jacobian_r=jacobian and optimize_rig,
        )

        coords = coords.view(self.n_terms, self.height, self.width, 2)
        valid = valid.view(self.n_terms, self.height, self.width, 1)

        pixel_count = self.height * self.width
        device = pose.device
        dtype = coords.dtype

        # Pre-allocate outputs
        residual = torch.zeros(self.n_terms, pixel_count, device=device, dtype=dtype)
        weight = torch.zeros(self.n_terms, pixel_count, device=device, dtype=dtype)

        if jacobian:
            J_pose_i = torch.zeros(self.n_terms, pixel_count, 6, device=device, dtype=dtype)
            J_pose_j = torch.zeros(self.n_terms, pixel_count, 6, device=device, dtype=dtype)
            J_disp = torch.zeros(self.n_terms, pixel_count, 1, device=device, dtype=dtype)

            if optimize_intrinsics:
                intr_dim = Jfi.shape[-1]
                J_intr = torch.zeros(2 * self.n_terms, pixel_count, intr_dim, device=device, dtype=dtype)
            if optimize_rig:
                J_rig = torch.zeros(2 * self.n_terms, pixel_count, 6, device=device, dtype=dtype)

        # Chunk to save memory
        for chunk_start in range(0, self.n_terms, self.chunk_size):
            chunk_end = min(chunk_start + self.chunk_size, self.n_terms)
            cs = slice(chunk_start, chunk_end)
            cs_offset = slice(chunk_start + self.n_terms, chunk_end + self.n_terms)

            src_idx = self.dense_disp_i_inds[cs]
            tgt_idx = self.dense_disp_j_inds[cs]

            # Normalized source (no sampling)
            source_norm = self.embeddings_norm[src_idx]  # (N, C, H, W)

            # Target: raw features (will be normalized AFTER sampling)
            target_raw_full = self.embeddings[tgt_idx]  # (N, C, H, W)

            coords_chunk = coords[cs]  # (N, H, W, 2)
            valid_chunk = valid[cs]  # (N, H, W, 1)

            # Optional embedding validity masks (src & tgt)
            if self.embedding_valid_mask is not None:
                src_valid = self.embedding_valid_mask[src_idx].unsqueeze(-1)  # (N, H, W, 1)
                tgt_valid = self.embedding_valid_mask[tgt_idx].unsqueeze(-1)  # (N, H, W, 1)
                valid_chunk = valid_chunk * src_valid * tgt_valid

            valid_map = valid_chunk.permute(0, 3, 1, 2)  # (N, 1, H, W)

            # Per-pixel weight for this term
            if self._has_weight_map:
                base_weight = self.embedding_weight[cs]
            else:
                base_weight = self.embedding_weight
            weight_chunk = base_weight * valid_map.squeeze(1)
            weight[cs] = weight_chunk.view(-1, pixel_count)

            # Skip if fully invalid
            if not torch.any(valid_map.view(-1, pixel_count).sum(dim=1) > 0.0):
                continue

            # Bilinear sample raw target features (and spatial grads if needed)
            target_raw_sampled, grad_u, grad_v = self.bilinear_sample_with_grad(
                target_raw_full, coords_chunk, return_grad=jacobian
            )

            if jacobian:
                residual_chunk, grad_coords = self.compute_residual_and_jacobian(
                    source_norm, target_raw_sampled, grad_u, grad_v, valid_map
                )
            else:
                # Residual only path
                eps = 1e-8
                u = target_raw_sampled
                u_norm = u.norm(dim=1, keepdim=True).clamp_min(eps)
                t_norm = u / u_norm
                cos_sim = (source_norm * t_norm).sum(dim=1)
                if self.use_photometric_residual:
                    residual_chunk = self.residual_scale * torch.sqrt(2.0 * (1.0 - cos_sim).clamp(min=0.0))
                else:
                    residual_chunk = 1.0 - cos_sim
                residual_chunk = residual_chunk * valid_map.squeeze(1)

            residual[cs] = residual_chunk.view(-1, pixel_count)

            if self.debug_visualize:
                self._maybe_visualize_chunk(
                    chunk_start=chunk_start,
                    source_norm=source_norm,
                    target_raw_sampled=target_raw_sampled,
                    residual_chunk=residual_chunk,
                    valid_map=valid_map,
                    pose_i_inds=self.pose_i_inds[cs],
                    pose_j_inds=self.pose_j_inds[cs],
                    dense_i_inds=self.dense_disp_i_inds[cs],
                    dense_j_inds=self.dense_disp_j_inds[cs],
                )

            if jacobian:
                grad_coords_flat = grad_coords.view(-1, pixel_count, 2)

                Ji_chunk = Ji[cs].view(-1, pixel_count, 2, 6)
                Jj_chunk = Jj[cs].view(-1, pixel_count, 2, 6)
                Jz_chunk = Jz[cs].view(-1, pixel_count, 2, 1)

                J_pose_i[cs] = torch.einsum("nkd,nkdf->nkf", grad_coords_flat, Ji_chunk)
                J_pose_j[cs] = torch.einsum("nkd,nkdf->nkf", grad_coords_flat, Jj_chunk)
                J_disp[cs] = torch.einsum("nkd,nkdc->nkc", grad_coords_flat, Jz_chunk)

                if optimize_intrinsics:
                    intr_dim = Jfi.shape[-1]
                    Jfi_chunk = Jfi[cs].view(-1, pixel_count, 2, intr_dim)
                    Jfj_chunk = Jfj[cs].view(-1, pixel_count, 2, intr_dim)
                    J_intr[cs] = torch.einsum("nkd,nkdf->nkf", grad_coords_flat, Jfi_chunk)
                    J_intr[cs_offset] = torch.einsum("nkd,nkdf->nkf", grad_coords_flat, Jfj_chunk)

                if optimize_rig:
                    Jri_chunk = Jri[cs].view(-1, pixel_count, 2, 6)
                    Jrj_chunk = Jrj[cs].view(-1, pixel_count, 2, 6)
                    J_rig[cs] = torch.einsum("nkd,nkdf->nkf", grad_coords_flat, Jri_chunk)
                    J_rig[cs_offset] = torch.einsum("nkd,nkdf->nkf", grad_coords_flat, Jrj_chunk)

        # Assemble sparse Jacobians
        J_dict = {}
        if jacobian:
            term_inds = torch.arange(self.n_terms, device=device)
            J_dict = {
                "pose": SparseDenseBlockMatrix(
                    i_inds=torch.cat([term_inds, term_inds]),
                    j_inds=torch.cat([self.pose_i_inds, self.pose_j_inds]),
                    data=torch.cat([J_pose_i, J_pose_j], dim=0),
                ),
                "dense_disp": SparseMDiagonalBlockMatrix(
                    i_inds=term_inds,
                    j_inds=self.dense_disp_i_inds,
                    data=J_disp,
                ),
            }
            if optimize_intrinsics:
                J_dict["intrinsics"] = SparseDenseBlockMatrix(
                    i_inds=torch.cat([term_inds, term_inds]),
                    j_inds=torch.cat([self.rig_i_inds, self.rig_j_inds]),
                    data=camera_model_cls.J_scale(scale, J_intr),
                )
            if optimize_rig:
                J_dict["rig"] = SparseDenseBlockMatrix(
                    i_inds=torch.cat([term_inds, term_inds]),
                    j_inds=torch.cat([self.rig_i_inds, self.rig_j_inds]),
                    data=J_rig,
                )

        if self.debug_enabled and self.debug_log_stats and self._should_debug_this_call():
            self._log_debug_stats(residual=residual, weight=weight)
        return ConcreteTermEvalReturn(J=J_dict, w=weight, r=residual)

    def _prepare_embedding_weight(self, weight: torch.Tensor | float) -> tuple[torch.Tensor, bool]:
        device = self.embeddings.device
        dtype = self.embeddings.dtype
        if isinstance(weight, torch.Tensor):
            weight_tensor = weight.detach().to(device=device, dtype=dtype)
            if weight_tensor.ndim == 0:
                return weight_tensor, False
            if weight_tensor.ndim == 2:
                expected = (self.n_terms, self.height * self.width)
                if weight_tensor.shape != expected:
                    raise ValueError(
                        f"Embedding weight tensor must have shape {expected} when 2D, got {tuple(weight_tensor.shape)}"
                    )
                return weight_tensor.view(self.n_terms, self.height, self.width).contiguous(), True
            if weight_tensor.ndim == 3:
                if tuple(weight_tensor.shape[1:]) != (self.height, self.width):
                    raise ValueError(
                        "Embedding weight spatial shape must match image_size; got "
                        f"{tuple(weight_tensor.shape[1:])} vs {(self.height, self.width)}"
                    )
                if weight_tensor.shape[0] == 1 and self.n_terms > 1:
                    weight_tensor = weight_tensor.expand(self.n_terms, -1, -1)
                elif weight_tensor.shape[0] != self.n_terms:
                    raise ValueError(
                        f"Embedding weight batch dimension must be 1 or {self.n_terms}; got {weight_tensor.shape[0]}"
                    )
                return weight_tensor.contiguous(), True
            raise ValueError(
                "Embedding weight tensor must be scalar, (n_terms, H*W), or (n_terms, H, W); "
                f"got ndim={weight_tensor.ndim}"
            )
        return torch.tensor(float(weight), device=device, dtype=dtype), False

    def _should_debug_this_call(self) -> bool:
        return ((self._forward_calls - 1) % self.debug_log_every) == 0

    def _log_debug_stats(self, residual: torch.Tensor, weight: torch.Tensor) -> None:
        valid_mask = weight > 0
        total = int(valid_mask.numel())
        valid = int(valid_mask.sum().item())
        if valid == 0:
            logger.info("[EmbeddingSim] call=%d | valid pixels: 0 / %d -- skipping stats", self._forward_calls, total)
            return
        residual_valid = residual[valid_mask]
        mean = residual_valid.mean().item()
        std = residual_valid.std(unbiased=False).item()
        min_val = residual_valid.min().item()
        max_val = residual_valid.max().item()
        if self._has_weight_map:
            weight_value = float(self.embedding_weight.mean().item())
        else:
            weight_value = float(self.embedding_weight.item())
        logger.info(
            (
                "[EmbeddingSim] call=%d | valid=%d/%d (%.2f%%) "
                "residual mean=%.4f std=%.4f min=%.4f max=%.4f weight=%.4f photometric=%s"
            ),
            self._forward_calls,
            valid,
            total,
            100.0 * valid / total,
            mean,
            std,
            min_val,
            max_val,
            weight_value,
            self.use_photometric_residual,
        )

    def _maybe_visualize_chunk(
        self,
        *,
        chunk_start: int,
        source_norm: torch.Tensor,
        target_raw_sampled: torch.Tensor,
        residual_chunk: torch.Tensor,
        valid_map: torch.Tensor,
        pose_i_inds: torch.Tensor,
        pose_j_inds: torch.Tensor,
        dense_i_inds: torch.Tensor,
        dense_j_inds: torch.Tensor,
    ) -> None:
        if (
            not self.debug_visualize
            or self.debug_dir is None
            or self._debug_saved >= self.debug_max_snapshots
            or not self._should_debug_this_call()
        ):
            return

        chunk_size = source_norm.shape[0]
        captures = 0
        for local_idx in range(chunk_size):
            if captures >= self.debug_samples_per_snapshot or self._debug_saved >= self.debug_max_snapshots:
                break

            validity = valid_map[local_idx, 0] > 0.5
            if not torch.any(validity):
                continue

            global_idx = chunk_start + local_idx
            self._save_debug_snapshot(
                global_idx=global_idx,
                source_norm=source_norm[local_idx],
                target_raw_sampled=target_raw_sampled[local_idx],
                residual_map=residual_chunk[local_idx],
                valid_mask=validity,
                pose_i=int(pose_i_inds[local_idx].item()),
                pose_j=int(pose_j_inds[local_idx].item()),
                dense_i=int(dense_i_inds[local_idx].item()),
                dense_j=int(dense_j_inds[local_idx].item()),
            )
            captures += 1

    def _save_debug_snapshot(
        self,
        *,
        global_idx: int,
        source_norm: torch.Tensor,
        target_raw_sampled: torch.Tensor,
        residual_map: torch.Tensor,
        valid_mask: torch.Tensor,
        pose_i: int,
        pose_j: int,
        dense_i: int,
        dense_j: int,
    ) -> None:
        if self.debug_dir is None:
            return

        target_norm = F.normalize(target_raw_sampled.unsqueeze(0), dim=1, eps=1e-8).squeeze(0)
        cos_map = (source_norm * target_norm).sum(dim=0)

        valid_ratio = float(valid_mask.float().mean().item())
        cos_valid = cos_map[valid_mask]
        res_valid = residual_map[valid_mask]
        cos_mean = float(cos_valid.mean().item()) if cos_valid.numel() > 0 else float("nan")
        res_mean = float(res_valid.mean().item()) if res_valid.numel() > 0 else float("nan")

        panel = self._compose_debug_panel(
            source_norm=source_norm,
            target_norm=target_norm,
            cos_map=cos_map,
            residual_map=residual_map,
            valid_mask=valid_mask,
        )
        text = (
            f"term#{global_idx} pose {pose_i}->{pose_j} disp {dense_i}->{dense_j} "
            f"valid {valid_ratio * 100:.1f}% cos {cos_mean:.3f} res {res_mean:.3f}"
        )
        image = self._attach_header(panel, text)
        filename = self.debug_dir / f"call{self._forward_calls:04d}_term{global_idx:05d}.png"
        self._write_debug_image(filename, image)
        logger.info(
            "EmbeddingSimilarityTerm snapshot saved to %s (pose %d->%d, disp %d->%d, valid %.1f%%, cos %.3f, res %.3f)",
            filename,
            pose_i,
            pose_j,
            dense_i,
            dense_j,
            valid_ratio * 100.0,
            cos_mean,
            res_mean,
        )
        self._debug_saved += 1

    def _compose_debug_panel(
        self,
        *,
        source_norm: torch.Tensor,
        target_norm: torch.Tensor,
        cos_map: torch.Tensor,
        residual_map: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> np.ndarray:
        import cv2
        import numpy as np

        # --- Create the 4 base images ---
        src_rgb = self._tensor_to_rgb(source_norm)
        tgt_rgb = self._tensor_to_rgb(target_norm)
        cos_heat = self._tensor_to_heatmap(
            cos_map,
            vmin=-1.0,
            vmax=1.0,
            colormap="viridis",
            valid_mask=valid_mask,
        )
        residual_vmax = self.residual_scale * 2.0 if self.use_photometric_residual else 2.0
        res_heat = self._tensor_to_heatmap(
            residual_map,
            vmin=0.0,
            vmax=max(residual_vmax, 1e-6),
            colormap="turbo",
            valid_mask=valid_mask,
        )

        # --- Add titles to each image ---
        src_rgb_titled = self._add_title_to_image(src_rgb, "Source (Normalized)")
        tgt_rgb_titled = self._add_title_to_image(tgt_rgb, "Target (Sampled, Norm.)")
        cos_heat_titled = self._add_title_to_image(cos_heat, "Cosine Similarity")
        res_heat_titled = self._add_title_to_image(res_heat, "Residual (Error)")

        # --- Concatenate titled images ---
        top = np.concatenate([src_rgb_titled, tgt_rgb_titled], axis=1)
        bottom = np.concatenate([cos_heat_titled, res_heat_titled], axis=1)
        panel = np.concatenate([top, bottom], axis=0)

        # --- NEW RESIZING LOGIC ---
        # Define a target width for the final panel for better visibility
        target_width = 1280

        original_height, original_width, _ = panel.shape

        # Only scale up if the panel is smaller than the target width
        if original_width < target_width:
            scale_factor = target_width / original_width
            target_height = int(original_height * scale_factor)

            # Resize the entire panel using INTER_NEAREST to keep pixels sharp
            panel = cv2.resize(panel, (target_width, target_height), interpolation=cv2.INTER_NEAREST)

        return panel

    @staticmethod
    def _add_title_to_image(image: np.ndarray, title: str) -> np.ndarray:
        """Helper to put a small title on an image quadrant."""
        import cv2
        import numpy as np

        title_height = 25  # Small header for each sub-image
        h, w, c = image.shape

        # Create a black header bar
        header = np.zeros((title_height, w, c), dtype=np.uint8)

        # Put white text on it
        cv2.putText(
            header,
            title,
            (5, title_height - 7),  # Position text
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,  # Font scale
            (255, 255, 255),  # White color
            1,
            cv2.LINE_AA,
        )
        # Stack the title bar on top of the image
        return np.concatenate([header, image], axis=0)

    @staticmethod
    def _tensor_to_rgb(tensor: torch.Tensor) -> np.ndarray:
        if tensor.dim() != 3:
            raise ValueError("Expected (C, H, W) tensor for RGB conversion")
        c, h, w = tensor.shape
        if c < 3:
            pad = torch.zeros(3 - c, h, w, device=tensor.device, dtype=tensor.dtype)
            tensor = torch.cat([tensor, pad], dim=0)
        rgb = tensor[:3].detach().cpu()
        rgb = (rgb * 0.5 + 0.5).clamp(0.0, 1.0)
        rgb = rgb.permute(1, 2, 0).numpy()
        return (rgb * 255.0).astype(np.uint8)

    @staticmethod
    def _tensor_to_heatmap(
        tensor: torch.Tensor,
        *,
        vmin: float,
        vmax: float,
        colormap: str,
        valid_mask: torch.Tensor,
    ) -> np.ndarray:
        import cv2

        arr = tensor.detach().cpu().numpy()
        span = max(vmax - vmin, 1e-6)
        norm = np.clip((arr - vmin) / span, 0.0, 1.0)
        img = (norm * 255.0).astype(np.uint8)
        cmap = getattr(cv2, f"COLORMAP_{colormap.upper()}", cv2.COLORMAP_VIRIDIS)
        colored = cv2.applyColorMap(img, cmap)
        colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
        mask_np = valid_mask.detach().cpu().numpy().astype(bool)
        colored[~mask_np] = np.array([30, 30, 30], dtype=np.uint8)
        return colored

    @staticmethod
    def _attach_header(image: np.ndarray, text: str) -> np.ndarray:
        import cv2
        import numpy as np

        header_height = 40  # Increased height for clarity
        header = np.ones((header_height, image.shape[1], 3), dtype=np.uint8) * 255
        img_width = image.shape[1]

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7  # Start with a larger, clearer font
        thickness = 1

        # --- Dynamic Font Scaling ---
        # Get text size
        (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)

        # Reduce font scale until text fits within image width (with 10px padding)
        while text_width > (img_width - 10) and font_scale > 0.2:
            font_scale -= 0.05
            (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)

        # --- Centering Text ---
        # Vertically center text in the header
        text_y = (header_height + text_height) // 2
        text_x = 5  # Left-align with 5px margin

        cv2.putText(
            header,
            text,
            (text_x, text_y),  # Use calculated position
            font,
            font_scale,
            (0, 0, 0),  # Black text
            thickness,
            cv2.LINE_AA,
        )
        return np.concatenate([header, image], axis=0)

    @staticmethod
    def _write_debug_image(path: Path, image: np.ndarray) -> None:
        import imageio.v2 as imageio_v2

        path.parent.mkdir(parents=True, exist_ok=True)
        imageio_v2.imwrite(str(path), image)

    @staticmethod
    def _parse_bool(value: Any, default: bool = False) -> bool:
        if value is None:
            return default
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in ("", "none", "null"):
                return default
            return lowered in ("1", "true", "yes", "on")
        return bool(value)

    @staticmethod
    def _parse_int(value: Any, default: int = 0) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return default


class DispSensRegularizationTerm(SolverTerm):
    """
    E(dense_disp_i) = dense_disp_i - dense_disps_sens_i
    res_dim = H*W
    """

    @dataclass(kw_only=True)
    class ThisTermEvalReturn(TermEvalReturn):
        alpha: float
        i_inds: torch.Tensor
        disps_sens_res: torch.Tensor

        def jtwj(self, group_name_row: str, group_name_col: str) -> SparseBlockMatrix:
            assert group_name_row == group_name_col == "dense_disp"
            return SparseMDiagonalBlockMatrix(
                i_inds=self.i_inds,
                j_inds=self.i_inds,
                data=torch.full_like(self.disps_sens_res, self.alpha).unsqueeze(-1),
            )

        def nwjtr(self, group_name: str) -> SparseBlockVector:
            assert group_name == "dense_disp"
            return SparseBlockVector(inds=self.i_inds, data=-self.alpha * self.disps_sens_res)

        def remove_jcol_inds(self, group_name: str, col_inds: torch.Tensor):
            assert group_name == "dense_disp"
            keep_mask = torch.isin(self.i_inds, col_inds, invert=True)
            self.i_inds = self.i_inds[keep_mask]
            self.disps_sens_res = self.disps_sens_res[keep_mask]

        def residual(self) -> torch.Tensor:
            return self.alpha * (self.disps_sens_res**2).sum(dim=1)

    def __init__(self, i_inds: torch.Tensor, alpha: float, disps_sens: torch.Tensor) -> None:
        super().__init__()

        self.i_inds = i_inds
        self.alpha = alpha
        self.disps_sens = disps_sens

    def group_names(self) -> set[str]:
        return {"dense_disp"}

    def forward(self, variables: dict[str, Any], jacobian: bool = True) -> TermEvalReturn:
        """
        variables contain:
            - dense_disp: (n_var, H*W) tensor of disparities
        """
        dense_disp = variables["dense_disp"]

        assert isinstance(dense_disp, torch.Tensor)
        assert dense_disp.shape == self.disps_sens.shape

        return self.ThisTermEvalReturn(
            alpha=self.alpha,
            i_inds=self.i_inds,
            disps_sens_res=dense_disp[self.i_inds] - self.disps_sens[self.i_inds],
        )


class TracksFlowTerm(SolverTerm):
    """
    E (pose_pi, pose_pj, tracks_di, intr_qi, intr_qj) = \
        proj(rig_j.inv() * pose_j * pose_i.inv() * rig_i, tracks_di) - target_[ij di]

        Pose is the world2cam transform.
        Rig is the cam2world(central cam) transform.
        target_[ij di] is the target projected location.
    res_dim = n_tracks*2
    """

    def __init__(
        self,
        pose_i_inds: torch.Tensor,
        pose_j_inds: torch.Tensor,
        rig_i_inds: torch.Tensor,
        rig_j_inds: torch.Tensor,
        tracks_i_inds: torch.Tensor,
        target: torch.Tensor,
        weight: torch.Tensor,
        tracks_uv: torch.Tensor,
        intrinsics: torch.Tensor | None,
        rig: SE3,
        camera_type: CameraType,
    ) -> None:
        super().__init__()

        self.n_terms = pose_i_inds.shape[0]
        assert pose_i_inds.shape == (self.n_terms,)
        assert pose_j_inds.shape == (self.n_terms,)
        assert rig_i_inds.shape == (self.n_terms,)
        assert rig_j_inds.shape == (self.n_terms,)
        assert tracks_i_inds.shape == (self.n_terms,)

        self.pose_i_inds = pose_i_inds
        self.pose_j_inds = pose_j_inds
        self.rig_i_inds = rig_i_inds
        self.rig_j_inds = rig_j_inds
        self.tracks_i_inds = tracks_i_inds
        self.camera_type = camera_type

        self.target = target.reshape(self.n_terms, -1, 2)  # (n_terms, n_tracks, 2)
        self.weight = weight.reshape(self.n_terms, -1, 2)  # (n_terms, n_tracks, 2)
        self.tracks_uv = tracks_uv

        self.n_tracks = self.target.shape[1]
        assert self.target.shape[1] == self.n_tracks
        assert self.weight.shape[1] == self.n_tracks
        assert self.tracks_uv.shape[1] == self.n_tracks

        self.intrinsics = intrinsics.reshape(-1, 4) if intrinsics is not None else None
        self.rig = rig

    def group_names(self) -> set[str]:
        names = {"pose", "tracks_disp"}
        if self.intrinsics is None:
            names.add("intrinsics")
        return names

    def forward(self, variables: dict[str, Any], jacobian: bool = True) -> TermEvalReturn:
        """
        variables contain:
            - pose: (n_var, ) SE3 of poses
            - tracks_disp: (n_var, n_tracks) tensor of disparities
            - intrinsics: (Q, 4) tensor of intrinsics (optional)
        """
        pose, tracks_disp = variables["pose"], variables["tracks_disp"]
        if optimize_intrinsics := self.intrinsics is None:
            intrinsics = variables["intrinsics"]
        else:
            intrinsics = self.intrinsics

        assert isinstance(pose, SE3) and isinstance(tracks_disp, torch.Tensor)
        assert tracks_disp.shape[1] == self.n_tracks
        assert intrinsics.shape[0] == self.rig.shape[0]

        coords, valid, (Ji, Jj, Jz), (Jfi, Jfj), _ = geom.iproj_i_proj_j_disp(
            pose,
            tracks_disp,
            self.tracks_uv,
            intrinsics,
            self.camera_type,
            self.rig,
            self.pose_i_inds,
            self.pose_j_inds,
            self.rig_i_inds,
            self.rig_j_inds,
            self.tracks_i_inds,
            jacobian_p_d=jacobian,
            jacobian_f=jacobian and optimize_intrinsics,
            jacobian_r=False,
        )
        weight = self.weight * valid

        J_dict = {}
        if jacobian:
            assert Ji is not None and Jj is not None and Jz is not None
            Ji = rearrange(Ji, "n t c d -> n (t c) d", c=2, d=6)
            Jj = rearrange(Jj, "n t c d -> n (t c) d", c=2, d=6)
            Jz = rearrange(Jz, "n t c d -> n (t) (c d)", c=2, d=1)
            term_inds = torch.arange(self.n_terms).to(pose.device)
            J_dict = {
                "pose": SparseDenseBlockMatrix(
                    i_inds=torch.cat([term_inds, term_inds]),
                    j_inds=torch.cat([self.pose_i_inds, self.pose_j_inds]),
                    data=torch.cat([Ji, Jj], dim=0),
                ),
                "tracks_disp": SparseMDiagonalBlockMatrix(
                    i_inds=term_inds,
                    j_inds=self.tracks_i_inds,
                    data=Jz,
                ),
            }
            if optimize_intrinsics:
                assert Jfi is not None and Jfj is not None
                Jfi = rearrange(Jfi, "n t c d -> n (t c) d", c=2)
                Jfj = rearrange(Jfj, "n t c d -> n (t c) d", c=2)
                J_dict["intrinsics"] = SparseDenseBlockMatrix(
                    i_inds=torch.cat([term_inds, term_inds]),
                    j_inds=torch.cat([self.rig_i_inds, self.rig_j_inds]),
                    data=torch.cat([Jfi, Jfj], dim=0),
                )

        return ConcreteTermEvalReturn(
            J=J_dict,
            w=weight.view(self.n_terms, -1),
            r=rearrange(coords - self.target, "n t c -> n (t c)", c=2),
        )
