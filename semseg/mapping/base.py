"""Minimal mapping abstractions used by semseg evaluation."""

import abc
from typing import Tuple
import torch

from semseg.image_encoders import ImageSpatialEncoder
from semseg.feat_compressors import FeatCompressor


class RGBDMapping(abc.ABC):
    def __init__(self, intrinsics_3x3: torch.FloatTensor, device: str = None, clip_bbox: Tuple[Tuple] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.intrinsics_3x3 = intrinsics_3x3.to(self.device) if intrinsics_3x3 is not None else None
        self.clip_bbox = clip_bbox

    def _clip_pc(self, pc_xyz: torch.FloatTensor, *features):
        if self.clip_bbox is None:
            return [pc_xyz] + list(features)
        bbox = torch.tensor(self.clip_bbox, dtype=torch.float, device=self.device)
        mask = torch.all((pc_xyz > bbox[0]) & (pc_xyz < bbox[1]), dim=-1)
        return [pc_xyz[mask]] + [f[mask] for f in features]


class SemanticRGBDMapping(RGBDMapping):
    def __init__(
        self,
        intrinsics_3x3: torch.FloatTensor,
        device: str = None,
        clip_bbox: Tuple[Tuple] = None,
        encoder: ImageSpatialEncoder = None,
        feat_compressor: FeatCompressor = None,
        interp_mode: str = "bilinear",
    ):
        super().__init__(intrinsics_3x3, device, clip_bbox)
        self.encoder = encoder
        self.feat_compressor = feat_compressor
        self.interp_mode = interp_mode

    def _proj_resize_feat_map(self, feat_img: torch.FloatTensor, h: int, w: int) -> torch.FloatTensor:
        if self.feat_compressor is not None:
            if not self.feat_compressor.is_fitted():
                self.feat_compressor.fit(feat_img.permute(0, 2, 3, 1))
            feat_img = self.feat_compressor.compress(feat_img.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        return torch.nn.functional.interpolate(
            feat_img,
            size=(h, w),
            mode=self.interp_mode,
            antialias=self.interp_mode in ["bilinear", "bicubic"],
        )
