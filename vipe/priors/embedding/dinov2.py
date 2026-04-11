from __future__ import annotations

import contextlib
import os
import time

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import rerun as rr
import torch
import torch.nn.functional as F
import torchvision.transforms as TVT
import torchvision.transforms.functional as TVTF

from PIL import Image
from torch import Tensor, nn


class DinoV2Variant(str, Enum):
    VITS = "vits14"
    VITB = "vitb14"
    VITL = "vitl14"
    VITG = "vitg14"
    VITS_REG = "vits14_reg"
    VITB_REG = "vitb14_reg"
    VITL_REG = "vitl14_reg"
    VITG_REG = "vitg14_reg"


@dataclass(frozen=True)
class DinoV2Config:
    hub_id: str  # e.g. "dinov2_vits14"
    weights_filename: Optional[str] = None


REGISTRY: Dict[DinoV2Variant, DinoV2Config] = {
    DinoV2Variant.VITS: DinoV2Config("dinov2_vits14", "dinov2_vits14_pretrain.pth"),
    DinoV2Variant.VITB: DinoV2Config("dinov2_vitb14", "dinov2_vitb14_pretrain.pth"),
    DinoV2Variant.VITL: DinoV2Config("dinov2_vitl14", "dinov2_vitl14_pretrain.pth"),
    DinoV2Variant.VITG: DinoV2Config("dinov2_vitg14", "dinov2_vitg14_pretrain.pth"),
    DinoV2Variant.VITS_REG: DinoV2Config("dinov2_vits14_reg", "dinov2_vits14_reg4_pretrain.pth"),
    DinoV2Variant.VITB_REG: DinoV2Config("dinov2_vitb14_reg", "dinov2_vitb14_reg4_pretrain.pth"),
    DinoV2Variant.VITL_REG: DinoV2Config("dinov2_vitl14_reg", "dinov2_vitl14_reg4_pretrain.pth"),
    DinoV2Variant.VITG_REG: DinoV2Config("dinov2_vitg14_reg", "dinov2_vitg14_reg4_pretrain.pth"),
}

Alias = Union[DinoV2Variant, str]
_ALIAS_NORMALIZATION = {
    "vits": "vits14",
    "vits14": "vits14",
    "vitb": "vitb14",
    "vitb14": "vitb14",
    "vitl": "vitl14",
    "vitl14": "vitl14",
    "vitg": "vitg14",
    "vitg14": "vitg14",
    "vits_reg": "vits14_reg",
    "vits14_reg": "vits14_reg",
    "vitb_reg": "vitb14_reg",
    "vitb14_reg": "vitb14_reg",
    "vitl_reg": "vitl14_reg",
    "vitl14_reg": "vitl14_reg",
    "vitg_reg": "vitg14_reg",
    "vitg14_reg": "vitg14_reg",
}


def _normalize_alias(x: str) -> str:
    # s = x.lower().replace("_", "").replace("-", "")
    s = x.lower()
    if s.startswith("dinov2"):
        s = s.replace("dinov2", "", 1)
    s = s.strip()
    return _ALIAS_NORMALIZATION.get(s, s)


def get_config(variant: Alias) -> DinoV2Config:
    """Accepts DinoV2Variant or a string alias ('vitl', 'dinov2_vitl14', etc.)."""
    if isinstance(variant, DinoV2Variant):
        return REGISTRY[variant]
    norm = _normalize_alias(variant)
    print(f"Normalized alias: {norm}")
    for v in DinoV2Variant:
        if v.value == norm:
            return REGISTRY[v]
    raise KeyError(f"Unknown DINOv2 variant: {variant!r}")

class PyramidUpsampler:
    """Handles multi-scale feature upsampling with different blending strategies."""

    def __init__(
        self,
        scales: Optional[List[float]] = None,
        blend_mode: Literal["weighted", "average", "max"] = "weighted",
        device: str = "cuda",
    ):
        self.scales = scales or [1.0, 0.75, 0.5]
        self.blend_mode = blend_mode
        self.device = device

    def upsample_single_scale(
        self,
        features: Tensor,
        target_size: Tuple[int, int],
        mode: Literal["bilinear", "bicubic"] = "bilinear",
    ) -> Tensor:
        """Upsample features to target size using single-scale interpolation."""
        device = features.device
        h, w = features.shape[:2] if features.dim() == 3 else features.shape[2:4]

        if (h, w) == target_size:
            return features if features.dim() == 3 else features.squeeze(0).permute(1, 2, 0)

        if features.dim() == 3:
            features = features.permute(2, 0, 1).unsqueeze(0)  # [1, D, H, W]

        interp_kwargs = {"size": target_size, "mode": mode, "antialias": True}
        if mode == "bilinear":
            interp_kwargs["align_corners"] = False

        upsampled = F.interpolate(features, **interp_kwargs)
        del features

        result = upsampled.squeeze(0).permute(1, 2, 0)
        if result.is_contiguous():
            result = result.contiguous()

        del upsampled
        if device.type == "cuda":
            torch.cuda.empty_cache()

        return result

    def upsample_pyramid(self, features_pyramid: List[Tensor], target_size: Tuple[int, int]) -> Tensor:
        """
        Upsample and blend pyramid features.
        Args:
            features_pyramid: List of [H_i, W_i, D] tensors at different scales
            target_size: (H_target, W_target) final resolution
        Returns:
            Blended features: [H_target, W_target, D]
        """
        if not features_pyramid:
            raise ValueError("Empty feature pyramid")

        D = features_pyramid[0].shape[-1]

        # Use the same device as input features
        device = features_pyramid[0].device

        # Initialize accumulators
        accumulated = torch.zeros(target_size[0], target_size[1], D, device=device)
        weights = (
            torch.zeros(target_size[0], target_size[1], 1, device=device)
            if self.blend_mode in ["weighted", "average"]
            else None
        )
        for i, feats in enumerate(features_pyramid):
            upsampled = self.upsample_single_scale(feats, target_size, mode="bilinear")

            current_blend_mode = self.blend_mode

            if current_blend_mode == "weighted":
                scale_weight = self.scales[i] if i < len(self.scales) else 1.0
                accumulated.add_(upsampled, alpha=scale_weight)
                weights.add_(scale_weight)

            elif current_blend_mode == "average":
                accumulated.add_(upsampled)
                weights.add_(1.0)

            elif current_blend_mode == "max":
                if i == 0:
                    accumulated = upsampled.clone()
                else:
                    torch.maximum(accumulated, upsampled, out=accumulated)
            else:
                raise ValueError(f"Unknown blend mode: {current_blend_mode}")

            del upsampled

            if device.type == "cuda" and (i + 1) % 3 == 0:
                torch.cuda.empty_cache()

        if current_blend_mode in ["weighted", "average"]:
            result = accumulated.div_(weights + 1e-8)
            del weights
        else:
            result = accumulated

        if device.type == "cuda":
            torch.cuda.empty_cache()

        return result


class ResizeTransform(nn.Module):
    """Resize image to a fixed size."""

    def __init__(self, image_size: int = 768, patch_size: int = 16):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size

    def forward(self, img: Image.Image) -> Image.Image:
        w, h = img.size
        h_patches = int(self.image_size / self.patch_size)
        w_patches = int((w * self.image_size) / (h * self.patch_size))
        return TVTF.resize(
            img,
            (h_patches * self.patch_size, w_patches * self.patch_size),
            interpolation=TVT.InterpolationMode.BICUBIC,
        )


class DINOv2EmbeddingEngine:
    """
    A comprehensive class for embedding using DINOv2 features.

    This class handles:
    - DINOv2 model initialization and loading
    - Frame embedding and dense feature extraction
    - Mask-based feature embedding
    - Visualization of results
    """

    def __init__(
        self,
        model: Alias = DinoV2Variant.VITL,
        weights_path: Optional[Union[str, Path]] = None,
        weights_dir: Optional[Union[str, Path]] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        short_side: int = 768,  # TODO: check this number
        pyramid_scales: Optional[List[float]] = None,
        rerun_app_id: str = "dinov2_embeddings",
        enable_rerun: bool = True,
    ):
        self.device = device
        self.short_side = short_side
        self.enable_rerun = enable_rerun

        if self.enable_rerun:
            rr.init(rerun_app_id, spawn=True)

        self.cfg = get_config(model)

        if weights_path is None and self.cfg.weights_filename and weights_dir is not None:
            weights_path = Path(weights_dir) / self.cfg.weights_filename
        self.weights_path = str(weights_path) if isinstance(weights_path, Path) else weights_path
        print(f"Using weights path: {self.weights_path}")
        # Initialize model
        self.model = self._load_model(self.cfg.hub_id, self.weights_path)
        self.patch_size = getattr(self.model, "patch_size", 16)
        self.embed_dim = getattr(self.model, "embed_dim", None)
        self.transform = self._create_transform()

        # New: unified upsampler & default scales
        self.pyramid_scales = pyramid_scales or [1.0, 0.75, 0.5]
        self.upsampler = PyramidUpsampler(scales=self.pyramid_scales, device=self.device)

        self.reset_state()
        info_text = (
            f"Initialized DINOv2 engine:\n"
            f"  Variant hub_id: {self.cfg.hub_id}\n"
            f"  Patch size: {self.patch_size}\n"
            f"  Embedding dimension: {self.embed_dim}\n"
            f"  Device: {self.device}\n"
            f"  Pyramid scales: {self.pyramid_scales}\n"
            f"  Rerun enabled: {self.enable_rerun}\n"
        )
        print(info_text)

        if self.enable_rerun:
            rr.log("info", rr.TextDocument(info_text, media_type=rr.MediaType.MARKDOWN))

    def _load_model(self, hub_id: str, weights_path: Optional[str]) -> nn.Module:
        """Load DINOv2 model via torch.hub with optional local weights."""
        try:
            model = torch.hub.load(
                repo_or_dir="facebookresearch/dinov2",
                model=hub_id,
                source="github",
            )
        except Exception as e:  # pragma: no cover
            print(f"Error loading model from GitHub ({e}). If needed, provide a local repo path.")
            raise

        if weights_path and os.path.exists(weights_path):
            try:
                state = torch.load(weights_path, map_location="cpu")
                model.load_state_dict(state, strict=False)
            except Exception as e:
                print(f"Warning: failed to load provided weights: {e}. Using hub defaults.")

        model.to(self.device)
        model.eval()
        torch.set_grad_enabled(False)
        return model

    def _create_transform(self) -> TVT.Compose:
        return TVT.Compose(
            [
                ResizeTransform(image_size=self.short_side, patch_size=self.patch_size),
                TVT.ToTensor(),
                TVT.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def _patch_aligned_size(self, height: int, width: int, scale: float = 1.0) -> Tuple[int, int]:
        """Return (H, W) scaled and rounded up to the nearest patch multiple."""
        scaled_h = max(self.patch_size, int(round(height * scale)))
        scaled_w = max(self.patch_size, int(round(width * scale)))
        aligned_h = ((scaled_h + self.patch_size - 1) // self.patch_size) * self.patch_size
        aligned_w = ((scaled_w + self.patch_size - 1) // self.patch_size) * self.patch_size
        return aligned_h, aligned_w

    def _resize_tensor(self, tensor: Tensor, size: Tuple[int, int]) -> Tensor:
        """Resize CHW or BCHW tensor if it does not already match `size`."""
        if tensor.shape[-2:] == size:
            return tensor
        needs_batch_dim = tensor.dim() == 3
        data = tensor.unsqueeze(0) if needs_batch_dim else tensor
        resized = F.interpolate(data, size=size, mode="bicubic", align_corners=False)
        return resized.squeeze(0) if needs_batch_dim else resized

    def _ensure_patch_multiple(self, tensor: Tensor) -> Tensor:
        """Ensure tensor spatial dims are divisible by the patch size."""
        target_size = self._patch_aligned_size(tensor.shape[-2], tensor.shape[-1], scale=1.0)
        return self._resize_tensor(tensor, target_size)

    def reset_state(self):
        self.num_masks: int = 0
        self.frame_height: int = 0
        self.frame_width: int = 0
        self.feats_height: int = 0
        self.feats_width: int = 0

    def _amp_context(self):
        if self.device == "cuda":
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            return torch.amp.autocast(device_type="cuda", dtype=dtype)
        return contextlib.nullcontext()

    def embed_frame(self, image: Union[Image.Image, Tensor], frame_idx: Optional[int] = None) -> Tensor:
        """Extract dense pixel-wise features from a single frame → [h, w, D]."""
        if isinstance(image, Image.Image):
            img_tensor = self.transform(image).to(self.device, non_blocking=True)  # [C,H,W]
        elif isinstance(image, Tensor):
            img_tensor = image.to(self.device, non_blocking=True)
        else:
            raise ValueError("Input image must be a PIL Image or a Tensor")

        img_tensor = self._ensure_patch_multiple(img_tensor)

        # Log original image to Rerun
        if self.enable_rerun and frame_idx is not None:
            rr.set_time_sequence("frame", frame_idx)
            if isinstance(image, Image.Image):
                rr.log("input/image", rr.Image(np.array(image)))

        with torch.inference_mode(), self._amp_context():
            x = img_tensor.unsqueeze(0)  # [1,C,H,W]
            feats_list = self.model.get_intermediate_layers(x, n=1, reshape=True, norm=True)
            feats = feats_list[0]  # [1, D, h, w]
            feats = feats[0].permute(1, 2, 0).contiguous()  # [h,w,D]
        h, w, _ = feats.shape
        self.feats_height, self.feats_width = int(h), int(w)
        if isinstance(image, Image.Image):
            self.frame_width, self.frame_height = image.size
        return feats

    def embed_batch(self, batch_chw: Tensor) -> Tensor:
        """Batch embedding.

        Args:
            batch_chw: [B, C, H, W] tensor (already transformed/normalized)
        Returns:
            [B, h, w, D] features
        """
        if batch_chw.dim() != 4:
            raise ValueError("Expected batch tensor of shape [B,C,H,W]")
        batch_chw = batch_chw.to(self.device, non_blocking=True)
        batch_chw = self._ensure_patch_multiple(batch_chw)
        with torch.inference_mode(), self._amp_context():
            feats_list = self.model.get_intermediate_layers(batch_chw, n=1, reshape=True, norm=True)
            feats = feats_list[0]  # [B, D, h, w]
            feats = feats.permute(0, 2, 3, 1).contiguous()  # [B,h,w,D]
        return feats

    def _extract_dino_features_multiscale(
        self, image: Union[Image.Image, Tensor], scales: Optional[List[float]] = None
    ) -> List[Tensor]:
        """Extract features at multiple input scales. Returns list[[h_i, w_i, D], ...]."""
        scales = scales or self.pyramid_scales
        if isinstance(image, Image.Image):
            base = self.transform(image).to(self.device)
        else:
            base = image.to(self.device)
        base = self._ensure_patch_multiple(base)
        _, H, W = base.shape
        feats_pyr: List[Tensor] = []
        with torch.inference_mode(), self._amp_context():
            for s in scales:
                target_hw = self._patch_aligned_size(H, W, scale=s)
                scaled = base if target_hw == (H, W) else self._resize_tensor(base, target_hw)
                flist = self.model.get_intermediate_layers(scaled.unsqueeze(0), n=1, reshape=True, norm=True)
                f = flist[0][0].permute(1, 2, 0).contiguous()  # [h,w,D]
                feats_pyr.append(f)
        return feats_pyr

    def embed_batch_multiscale(self, batch_chw: Tensor, scales: Optional[List[float]] = None) -> List[Tensor]:
        """Multi-scale features for a batch tensor.

        Args:
            batch_chw: [B, C, H, W]
        Returns:
            List of [B, h_i, w_i, D]
        """
        scales = scales or self.pyramid_scales
        B, C, H, W = batch_chw.shape
        batch_chw = batch_chw.to(self.device)
        batch_chw = self._ensure_patch_multiple(batch_chw)
        _, _, H, W = batch_chw.shape
        outputs: List[Tensor] = []
        with torch.inference_mode(), self._amp_context():
            for s in scales:
                target_hw = self._patch_aligned_size(H, W, scale=s)
                scaled = batch_chw if target_hw == (H, W) else self._resize_tensor(batch_chw, target_hw)
                flist = self.model.get_intermediate_layers(scaled, n=1, reshape=True, norm=True)
                f = flist[0].permute(0, 2, 3, 1).contiguous()  # [B,h,w,D]
                outputs.append(f)
        return outputs

    def upsample_features(
        self,
        features: Union[Tensor, List[Tensor]],
        target_size: Tuple[int, int],
        method: Literal["bilinear", "bicubic", "pyramid_weighted", "pyramid_average", "pyramid_max"] = "bilinear",
    ) -> Tensor:
        """Upsample feature(s) to target (H,W).

        If `features` is a list, pyramid_* methods are used (with blending).
        If `features` is a tensor, single-scale interpolation is used.
        Supports both [H,W,D] and [B,H,W,D] feature tensors.
        """
        if method.startswith("pyramid"):
            if not isinstance(features, list):
                raise ValueError("Pyramid methods require a List[Tensor] from embed_*_multiscale().")
            blend = method.split("_", 1)[1]  # weighted/average/max
            self.upsampler.blend_mode = blend  # set mode dynamically
            return self.upsampler.upsample_pyramid(features, target_size)
        else:
            if not isinstance(features, Tensor):
                if len(features) == 1:
                    features = features[0]
                else:
                    raise ValueError("Single-scale methods require a Tensor.")
            if method not in ("bilinear", "bicubic"):
                raise ValueError(f"Unknown upsample method: {method}")
            return self.upsampler.upsample_single_scale(features, target_size, mode=method)

    def embed_frame_with_masks(
        self,
        image: Union[Image.Image, Tensor],
        masks: np.ndarray,
        return_mask_probs: bool = True,
        frame_idx: Optional[int] = None,
    ) -> Tuple[Tensor, Optional[Tensor], Dict[int, Tensor]]:
        """Dense features + per-mask pooled vectors.

        Returns:
            dense_features: [h, w, D]
            mask_probs: [h, w, M] (if return_mask_probs)
            mask_embeddings: {mask_id: [D]}
        """
        dense_features = self.embed_frame(image, frame_idx=frame_idx)  # [h,w,D]
        h, w, D = dense_features.shape
        self.feats_height, self.feats_width = int(h), int(w)
        if isinstance(image, Image.Image):
            self.frame_width, self.frame_height = image.size

        masks_tensor = torch.as_tensor(masks, device=self.device, dtype=torch.long)
        masks_resized = F.interpolate(masks_tensor[None, None].float(), size=(h, w), mode="nearest")[0, 0].long()
        num_masks = int(masks_resized.max().item()) + 1
        self.num_masks = num_masks

        # Log masks to Rerun
        if self.enable_rerun and frame_idx is not None:
            rr.set_time_sequence("frame", frame_idx)
            mask_rgb = self.mask_to_rgb(masks, num_masks)
            rr.log("masks/segmentation", rr.SegmentationImage(masks))
            rr.log("masks/visualization", rr.Image(mask_rgb))

        mask_probs = None
        if return_mask_probs:
            mask_probs = F.one_hot(masks_resized, num_classes=num_masks).to(torch.float32)

        labels = masks_resized.view(-1)  # [N]
        feats = dense_features.view(-1, D)  # [N,D]
        sums = torch.zeros(num_masks, D, device=self.device, dtype=feats.dtype)
        sums.index_add_(0, labels, feats)
        counts = torch.bincount(labels, minlength=num_masks).clamp_min(1).unsqueeze(1)  # [M,1]
        means = sums / counts
        mask_embeddings: Dict[int, Tensor] = {int(i): means[i] for i in range(1, num_masks) if counts[i].item() > 0}
        return dense_features, mask_probs, mask_embeddings

    @staticmethod
    def mask_to_rgb(mask: Union[np.ndarray, Tensor], num_masks: int) -> np.ndarray:
        """Convert segmentation mask to RGB visualization."""
        if isinstance(mask, Tensor):
            mask = mask.cpu().numpy().astype(int)
        else:
            mask = mask.astype(int)

        # Generate a simple, deterministic color palette
        colors = np.zeros((num_masks + 1, 3), dtype=np.uint8)
        for i in range(1, num_masks):  # Skip background
            r = i * 123457 % 255
            g = i * 345679 % 255
            b = i * 789013 % 255
            colors[i] = [r, g, b]

        mask_rgb = colors[mask]
        return mask_rgb

    def visualize_embeddings(
        self,
        features: Tensor,
        method: Literal["mean", "std", "norm", "naive_rgb"] = "naive_rgb",
        entity_path: str = "features",
        frame_idx: Optional[int] = None,
    ):
        """Visualize DINOv2 feature embeddings in Rerun.

        Args:
            features: Feature tensor of shape [h, w, D] or [B, h, w, D]
            method: Reduction method ('mean', 'std', 'norm', 'naive_rgb')
            entity_path: Base path for logging to Rerun
            frame_idx: Optional frame index for timeline
        """
        if not self.enable_rerun:
            return

        if frame_idx is not None:
            rr.set_time_sequence("frame", frame_idx)

        if features.dim() == 4:
            # Handle batched features - visualize first item
            features = features[0]

        f = features.detach().cpu()
        H, W, D = f.shape

        vis_features = None
        if method == "mean":
            vis_features = f.mean(dim=-1).numpy()
            vis_features = (vis_features - vis_features.min()) / (vis_features.max() - vis_features.min() + 1e-8)
        elif method == "std":
            vis_features = f.std(dim=-1).numpy()
            vis_features = (vis_features - vis_features.min()) / (vis_features.max() - vis_features.min() + 1e-8)
        elif method == "norm":
            vis_features = torch.linalg.norm(f, dim=-1).numpy()
            vis_features = (vis_features - vis_features.min()) / (vis_features.max() - vis_features.min() + 1e-8)
        elif method == "naive_rgb":
            if D < 3:
                print("Warning: 'naive_rgb' requires >= 3 feature dimensions. Skipping.")
                return

            vis_features_rgb = f[..., :3].numpy()
            for i in range(3):
                channel = vis_features_rgb[..., i]
                min_val, max_val = channel.min(), channel.max()
                vis_features_rgb[..., i] = (channel - min_val) / (max_val - min_val + 1e-8)

            vis_uint8 = (vis_features_rgb * 255).astype(np.uint8)
            rr.log(f"{entity_path}/naive_rgb", rr.Image(vis_uint8))
            return
        else:
            raise ValueError(f"Unknown visualization method: {method}")

        if vis_features is not None:
            rr.log(f"{entity_path}/{method}", rr.DepthImage(vis_features))

    def visualize_feature_similarity(
        self,
        features: Tensor,
        query_points: List[Tuple[int, int]],
        entity_path: str = "similarity",
        frame_idx: Optional[int] = None,
    ):
        """Visualize feature similarity maps for specific query points in Rerun.

        Args:
            features: Feature tensor of shape [h, w, D]
            query_points: List of (y, x) coordinates to use as query points
            entity_path: Base path for logging to Rerun
            frame_idx: Optional frame index for timeline
        """
        if not self.enable_rerun:
            return

        if frame_idx is not None:
            rr.set_time_sequence("frame", frame_idx)

        if features.dim() == 4:
            features = features[0]

        f = features.detach()
        h, w, D = f.shape

        # Log query points
        query_positions = np.array([[x, y] for y, x in query_points], dtype=np.float32)
        rr.log(f"{entity_path}/query_points", rr.Points2D(query_positions, radii=5, colors=[255, 0, 0]))

        features_flat = f.view(-1, D)
        for i, (qy, qx) in enumerate(query_points):
            if qy >= h or qx >= w:
                print(f"Query point ({qy}, {qx}) is out of bounds for features shape ({h}, {w})")
                continue

            query_feat = f[qy, qx].unsqueeze(0)
            similarities = F.cosine_similarity(query_feat, features_flat, dim=1)
            sim_map = similarities.view(h, w).cpu().numpy()

            # Normalize similarity map
            sim_map = (sim_map - sim_map.min()) / (sim_map.max() - sim_map.min() + 1e-8)

            rr.log(f"{entity_path}/query_{i + 1}", rr.DepthImage(sim_map))

    def get_memory_usage(self) -> Dict[str, float]:
        """Get current GPU memory usage statistics."""
        if self.device == "cuda" and torch.cuda.is_available():
            stats = {
                "allocated_gb": torch.cuda.memory_allocated() / (2**30),
                "reserved_gb": torch.cuda.memory_reserved() / (2**30),
                "max_allocated_gb": torch.cuda.max_memory_allocated() / (2**30),
            }
            return stats
        return {"message": "CUDA not available"}


# =============================================================================
# Demo
# =============================================================================
def demo_usage():
    """Demonstration of DINOv2 with Rerun visualization."""
    engine = DINOv2EmbeddingEngine(
        model=DinoV2Variant.VITS,
        weights_dir=None,
        short_side=448,
        pyramid_scales=[2.0, 1.5, 1.0, 0.75],
        rerun_app_id="dinov2_demo",
        enable_rerun=True,
    )

    # Example image paths - update these to your data
    image_dir = "/data/Replica/replica/office0/rgb"
    image_files = [f"{image_dir}/frame{str(i).zfill(6)}.jpg" for i in range(10)]

    # Demo pyramid upsampling on last frame
    if image_files:
        print(f"\n{'=' * 60}")
        print("--- Demonstrating Pyramid Upsampling ---")
        print(f"{'=' * 60}")

        for i, image in enumerate(image_files):
            last_img = Image.open(image).convert("RGB")
            target_h, target_w = last_img.height, last_img.width
            print(f"Original image size (H, W): ({target_h}, {target_w})")

            # Extract multi-scale features
            print("Extracting multi-scale features...")
            t_start = time.time()
            features_pyramid = engine._extract_dino_features_multiscale(last_img)
            t_end = time.time()
            print(f"Multi-scale extraction done in {t_end - t_start:.3f}s")
            for idx, feat in enumerate(features_pyramid):
                print(f"  Scale {engine.pyramid_scales[idx]}: {feat.shape}")

            # Blend pyramid with weighted method
            print("Blending pyramid with 'pyramid_weighted'...")
            t_start = time.time()
            blended_features = engine.upsample_features(
                features=features_pyramid, target_size=(target_h, target_w), method="pyramid_weighted"
            )
            t_end = time.time()
            print(f"Pyramid blending done in {t_end - t_start:.3f}s")
            print(f"Final Blended Feature Shape: {blended_features.shape}")

            # Visualize blended features
            engine.visualize_embeddings(
                blended_features,
                method="naive_rgb",
                entity_path="visualizations/pyramid_weighted_rgb",
                frame_idx=i,
            )

            # Compare with simple bilinear upsampling
            print("Comparing with standard 'bilinear' upsample...")
            single_scale_features = features_pyramid[0]
            bilinear_features = engine.upsample_features(
                features=single_scale_features, target_size=(target_h, target_w), method="bilinear"
            )
            engine.visualize_embeddings(
                bilinear_features,
                method="naive_rgb",
                entity_path="visualizations/bilinear_rgb",
                frame_idx=i,
            )

            print(f"\n{'=' * 60}")
            print("Demo complete! Check the Rerun viewer for visualizations.")
            print(f"{'=' * 60}")


if __name__ == "__main__":
    demo_usage()
