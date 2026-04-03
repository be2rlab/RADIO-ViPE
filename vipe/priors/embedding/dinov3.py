# --- OPTIMIZATION ---
# Removed unused imports: math, os, time, contextlib, sklearn, skimage, matplotlib
from __future__ import annotations

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


# --- DINOv3 Config (Unchanged) ---
class DinoV3Variant(str, Enum):
    VITS = "vits16"
    VITSP = "vits16plus"
    VITB = "vitb16"
    VITL = "vitl16"
    VITHP = "vith16plus"
    VIT7B = "vit7b16"


@dataclass(frozen=True)
class DinoV3Config:
    hub_id: str
    num_layers: int
    weights_filename: Optional[str] = None


REGISTRY: Dict[DinoV3Variant, DinoV3Config] = {
    DinoV3Variant.VITS: DinoV3Config("dinov3_vits16", 12, "dinov3_vits16_pretrain_lvd1689m-08c60483.pth"),
    DinoV3Variant.VITSP: DinoV3Config("dinov3_vits16plus", 12, "dinov3_vits16plus_pretrain_lvd1689m-4057cbaa.pth"),
    DinoV3Variant.VITB: DinoV3Config("dinov3_vitb16", 12, "dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth"),
    DinoV3Variant.VITL: DinoV3Config("dinov3_vitl16", 24, "dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"),
    DinoV3Variant.VITHP: DinoV3Config("dinov3_vith16plus", 32, "dinov3_vith16plus_pretrain_lvd1689m-7c1da9a5.pth"),
    DinoV3Variant.VIT7B: DinoV3Config("dinov3_vit7b16", 40, "dinov3_vit7b16_pretrain_lvd1689m-a955f4ea.pth"),
}
Alias = Union[DinoV3Variant, str]
_ALIAS_NORMALIZATION = {
    "vits": "vits16",
    "vits16": "vits16",
    "vitsp": "vits16plus",
    "vits16plus": "vits16plus",
    "vitb": "vitb16",
    "vitb16": "vitb16",
    "vitl": "vitl16",
    "vitl16": "vitl16",
    "vithp": "vith16plus",
    "vith": "vith16plus",
    "vith16plus": "vith16plus",
    "vit7b": "vit7b16",
    "vit7b16": "vit7b16",
}


def _normalize_alias(x: str) -> str:
    s = x.lower().replace("_", "").replace("-", "").strip()
    if s.startswith("dinov3"):
        s = s.replace("dinov3", "", 1)
    return _ALIAS_NORMALIZATION.get(s, s)


def get_config(variant: Alias) -> DinoV3Config:
    if isinstance(variant, DinoV3Variant):
        return REGISTRY[variant]
    norm = _normalize_alias(variant)
    for v in DinoV3Variant:
        if v.value == norm:
            return REGISTRY[v]
    raise KeyError(f"Unknown DINOv3 variant: {variant!r}")


# --- Transforms
class ResizeTransform(nn.Module):
    """Resize image to a fixed size."""

    def __init__(self, image_size: int = 768, patch_size: int = 16):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size

    def forward(self, img):
        w, h = img.size
        h_patches = self.image_size // self.patch_size
        w_patches = int((w * self.image_size) / (h * self.patch_size))
        return TVTF.resize(img, (h_patches * self.patch_size, w_patches * self.patch_size))


class DINOv3EmbeddingEngine:
    """
    A comprehensive class for embedding using DINOv3 features.
    """

    def __init__(
        self,
        model: Alias = DinoV3Variant.VITL,
        weights_path: Optional[Union[str, Path]] = None,
        weights_dir: Optional[Union[str, Path]] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        short_side: int = 768,
        pyramid_scales: Optional[List[float]] = None,  # --- OPTIMIZATION ---
        rerun_app_id: str = "dinov3_embeddings",
    ):
        self.device = device
        self.short_side = short_side
        rr.init(rerun_app_id, spawn=True)
        self.cfg = get_config(model)
        self.n_layers = self.cfg.num_layers

        if weights_path is None and self.cfg.weights_filename and weights_dir is not None:
            weights_path = Path(weights_dir) / self.cfg.weights_filename
        self.weights_path = str(weights_path) if isinstance(weights_path, Path) else weights_path

        print(f"Using weights path: {self.weights_path}")
        self.model = self._load_model(self.cfg.hub_id, self.weights_path)
        self.patch_size = getattr(self.model, "patch_size", 16)
        self.embed_dim = getattr(self.model, "embed_dim", None)
        self.transform = self._create_transform()

        # --- OPTIMIZATION ---
        # Unify upsampling logic by using PyramidUpsampler internally
        self.pyramid_scales = pyramid_scales or [1.0, 0.75, 0.5]
        self.upsampler = PyramidUpsampler(scales=self.pyramid_scales, device=self.device)

        self.reset_state()
        info_text = f"""Initialized DINOv3 engine:
  Variant hub_id: {self.cfg.hub_id}
  Num layers: {self.n_layers}
  Patch size: {self.patch_size}
  Embedding dimension: {self.embed_dim}
  Device: {self.device}
  Pyramid scales: {self.pyramid_scales}"""
        print(info_text)
        rr.log("info", rr.TextDocument(info_text, media_type=rr.MediaType.MARKDOWN))

    def _load_model(self, hub_id: str, weights_path: Optional[str]) -> nn.Module:
        """Load DINOv3 model via torch.hub with optional local weights."""
        try:
            model = torch.hub.load(
                repo_or_dir="/home/user/km-vipe/dino/dinov3",
                model=hub_id,
                source="local",
                weights=weights_path,
            )
        except Exception as e:
            # --- OPTIMIZATION ---
            # Removed hardcoded local path
            print(f"Error loading model from GitHub ({e}). Ensure model is available.")
            raise e

        if weights_path:
            try:
                if Path(weights_path).exists():
                    state = torch.load(weights_path, map_location="cpu")
                    model.load_state_dict(state, strict=False)
                else:
                    print(f"Weights path not found: {weights_path}. Using hub default.")
            except Exception as e:
                print(f"Error loading local weights: {e}. Using hub default.")

        model.to(self.device)
        model.eval()
        torch.set_grad_enabled(False)
        return model

    def _create_transform(self) -> TVT.Compose:
        """Create image preprocessing transform."""
        return TVT.Compose(
            [
                ResizeTransform(image_size=self.short_side, patch_size=self.patch_size),
                TVT.ToTensor(),
                TVT.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def reset_state(self):
        """Reset the tracking state for processing a new video."""
        self.num_masks: int = 0
        self.frame_height: int = 0
        self.frame_width: int = 0
        self.feats_height: int = 0
        self.feats_width: int = 0

    def embed_frame(self, image: Union[Image.Image, Tensor], frame_idx: Optional[int] = None) -> Tensor:
        """
        Extract dense pixel-wise features from a single frame.

        Args:
            image: Input image
            frame_idx: Optional frame index for timeline logging

        Returns:
            Tensor [h, w, D] on self.device
        """
        if isinstance(image, Image.Image):
            img_tensor = self.transform(image).to(self.device, non_blocking=True)  # [C,H,W]
        elif isinstance(image, Tensor):
            img_tensor = image.to(self.device, non_blocking=True)
        else:
            raise ValueError("Input image must be a PIL Image or a Tensor")

        # Log original image to Rerun
        if frame_idx is not None:
            rr.set_time_sequence("frame", frame_idx)
            if isinstance(image, Image.Image):
                rr.log("input/image", rr.Image(np.array(image)))

        # --- OPTIMIZATION ---
        # Simplified AMP context
        amp_dtype = torch.bfloat16 if self.device == "cuda" and torch.cuda.is_bf16_supported() else torch.float16
        with torch.inference_mode(), torch.amp.autocast(device_type=self.device, dtype=amp_dtype):
            x = img_tensor.unsqueeze(0)
            feat_layer = getattr(self, "feat_layer", -1)
            target_layer = (self.n_layers + feat_layer) % self.n_layers

            feats_list = self.model.get_intermediate_layers(x, n=[target_layer], reshape=True, norm=True)
            feats = feats_list[0][0]  # [D, h, w]
            feats = feats.permute(1, 2, 0).contiguous()  # [h, w, D]
        return feats

    def embed_frame_with_masks(
        self,
        image: Union[Image.Image, Tensor],
        masks: np.ndarray,
        return_mask_probs: bool = True,
        frame_idx: Optional[int] = None,
    ) -> Tuple[Tensor, Optional[Tensor], Dict[int, Tensor]]:
        """
        Args:
            image: Input frame
            masks: HxW integer mask image where each value encodes an instance/class.
            return_mask_probs: if True returns one-hot [h, w, M]; can be large.
            frame_idx: Optional frame index for timeline logging
        Returns:
            dense_features: [h, w, D] (on self.device)
            mask_probs: [h, w, M] or None (on self.device, float32)
            mask_embeddings: {mask_id: [D]} (each on self.device)
        """
        dense_features = self.embed_frame(image, frame_idx=frame_idx)  # [h,w,D] on device
        h, w, D = dense_features.shape
        self.feats_height, self.feats_width = int(h), int(w)
        if isinstance(image, Image.Image):
            self.frame_width, self.frame_height = image.size

        masks_tensor = torch.as_tensor(masks, device=self.device, dtype=torch.long)  # [H,W]
        masks_resized = F.interpolate(
            masks_tensor[None, None].float(),
            size=(h, w),
            mode="nearest",
        )[0, 0].long()  # [h,w]

        num_masks = int(masks_resized.max().item()) + 1
        self.num_masks = num_masks

        if frame_idx is not None:
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
        means = sums / counts  # [M,D]

        mask_embeddings: Dict[int, Tensor] = {int(i): means[i] for i in range(1, num_masks) if counts[i].item() > 0}
        return dense_features, mask_probs, mask_embeddings

    @staticmethod
    def mask_to_rgb(mask: Union[np.ndarray, Tensor], num_masks: int) -> np.ndarray:
        """
        Convert segmentation mask to RGB visualization without matplotlib.
        --- OPTIMIZATION ---
        """
        if isinstance(mask, Tensor):
            mask = mask.cpu().numpy().astype(int)
        else:
            mask = mask.astype(int)

        # Generate a simple, deterministic color palette
        colors = np.zeros((num_masks + 1, 3), dtype=np.uint8)
        for i in range(1, num_masks):  # Skip background
            # Simple hash to generate visually distinct colors
            r = i * 123457 % 255
            g = i * 345679 % 255
            b = i * 789013 % 255
            colors[i] = [r, g, b]

        # Apply colors vectorized
        mask_rgb = colors[mask]
        return mask_rgb

    # --- OPTIMIZATION ---
    # This entire section has been simplified.
    # Removed sklearn dependency (PCA, t-SNE) and complex logging.

    def visualize_embeddings(
        self,
        features: Tensor,
        method: Literal["mean", "std", "norm", "naive_rgb"] = "naive_rgb",
        entity_path: str = "features",
        frame_idx: Optional[int] = None,
    ):
        """
        Visualize DINOv3 feature embeddings using simple channel stats in Rerun.
        Args:
            features: Feature tensor of shape [h, w, D]
            method: Reduction method ('mean', 'std', 'norm', 'naive_rgb')
            entity_path: Base path for logging to Rerun
            frame_idx: Optional frame index for timeline
        """
        if frame_idx is not None:
            rr.set_time_sequence("frame", frame_idx)

        vis_features = None
        if method == "mean":
            vis_features = features.mean(axis=-1)
        elif method == "std":
            vis_features = features.std(axis=-1)
        elif method == "norm":
            vis_features = torch.linalg.norm(features, dim=-1)
        elif method == "naive_rgb":
            if features.shape[-1] < 3:
                print("Warning: 'naive_rgb' requires >= 3 feature dimensions. Skipping.")
                return

            # Take first 3 channels and normalize them *independently*
            vis_features_rgb = features[..., :3].cpu().numpy()
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
            # Log single-channel features as a DepthImage for Rerun's colormapping
            rr.log(f"{entity_path}/{method}", rr.DepthImage(vis_features.cpu().numpy()))

    def visualize_feature_similarity(
        self,
        features: Tensor,
        query_points: List[Tuple[int, int]],
        entity_path: str = "similarity",
        frame_idx: Optional[int] = None,
    ):
        """
        Visualize feature similarity maps for specific query points in Rerun.
        --- OPTIMIZATION --- Simplified logging.
        Args:
            features: Feature tensor of shape [h, w, D]
            query_points: List of (y, x) coordinates to use as query points
            entity_path: Base path for logging to Rerun
            frame_idx: Optional frame index for timeline
        """
        if frame_idx is not None:
            rr.set_time_sequence("frame", frame_idx)

        h, w, D = features.shape

        # Log query points
        query_positions = np.array([[x, y] for y, x in query_points], dtype=np.float32)
        rr.log(f"{entity_path}/query_points", rr.Points2D(query_positions, radii=5, colors=[255, 0, 0]))

        features_flat = features.view(-1, D)  # [hw, D]
        for i, (qy, qx) in enumerate(query_points):
            if qy >= h or qx >= w:
                print(f"Query point ({qy}, {qx}) is out of bounds for features shape ({h}, {w})")
                continue

            query_feat = features[qy, qx].unsqueeze(0)  # [1, D]
            similarities = F.cosine_similarity(query_feat, features_flat, dim=1)  # [hw]
            sim_map = similarities.view(h, w).cpu().numpy()

            # Log similarity map as depth image (Rerun will colormap it)
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

    def _extract_dino_features_multiscale(
        self,
        image: Union[Image.Image, Tensor],
        scales: Optional[List[float]] = None,
    ) -> List[Tensor]:
        """
        Extract features at multiple scales.
        --- OPTIMIZATION --- Uses self.pyramid_scales as default.
        """
        scales = scales or self.pyramid_scales
        if isinstance(image, Image.Image):
            base_tensor = self.transform(image).to(self.device)  # [C, H, W]
        else:
            base_tensor = image.to(self.device)

        _, base_h, base_w = base_tensor.shape
        features_pyramid = []
        # print(f"  Multi-scale feature extraction (Base: {base_h}x{base_w}, Scales: {scales}):")

        amp_dtype = torch.bfloat16 if self.device == "cuda" and torch.cuda.is_bf16_supported() else torch.float16

        for scale_idx, scale in enumerate(scales):
            if scale == 1.0:
                scaled_tensor = base_tensor
            else:
                new_h = max(self.patch_size, int(base_h * scale))
                new_w = max(self.patch_size, int(base_w * scale))
                new_h = ((new_h + self.patch_size - 1) // self.patch_size) * self.patch_size
                new_w = ((new_w + self.patch_size - 1) // self.patch_size) * self.patch_size

                scaled_tensor = F.interpolate(
                    base_tensor.unsqueeze(0), size=(new_h, new_w), mode="bicubic", align_corners=False
                ).squeeze(0)

            with torch.inference_mode(), torch.amp.autocast(device_type=self.device, dtype=amp_dtype):
                feat_layer = getattr(self, "feat_layer", -1)
                target_layer = (self.n_layers + feat_layer) % self.n_layers

                feats_list = self.model.get_intermediate_layers(
                    scaled_tensor.unsqueeze(0), n=[target_layer], reshape=True, norm=True
                )
                feats = feats_list[0][0]  # [D, h_i, w_i]
                feat = feats.permute(1, 2, 0).contiguous()  # [h_i, w_i, D]
                features_pyramid.append(feat.detach())
            # print(f"    Scale {scale} ({scale_idx + 1}/{len(scales)}) -> {feat.shape}")
        return features_pyramid

    def upsample_features(
        self,
        features: Union[Tensor, List[Tensor]],
        target_size: Tuple[int, int],
        method: Literal["bilinear", "bicubic", "pyramid_weighted", "pyramid_average", "pyramid_max"] = "bilinear",
    ) -> Tensor:
        """
        Upsample features to target size using specified method.
        Args:
            features: [H, W, D] feature tensor OR
                      List[[H_i, W_i, D], ...] for pyramid methods
            target_size: (H_target, W_target) tuple
            method: Upsampling method
        Returns:
            Upsampled features: [H_target, W_target, D]
        """
        if method.startswith("pyramid"):
            if not isinstance(features, list):
                raise ValueError(
                    f"Method '{method}' requires 'features' to be a List[Tensor]. "
                    "Use _extract_dino_features_multiscale() to generate it."
                )

            # Set the blend mode on the upsampler instance
            blend_mode = method.split("_", 1)[1]  # "weighted", "average", "max"
            self.upsampler.blend_mode = blend_mode

            return self.upsampler.upsample_pyramid(features, target_size)
        else:
            if not isinstance(features, Tensor):
                raise ValueError(f"Method '{method}' requires 'features' to be a single Tensor.")

            if method not in ("bilinear", "bicubic"):
                raise ValueError(f"Unknown single-scale method: {method}")

            return self.upsampler.upsample_single_scale(features, target_size, mode=method)


def demo_usage():
    """Demonstration of DINOv3 with Rerun visualization."""
    tracker = DINOv3EmbeddingEngine(
        model=DinoV3Variant.VITS,  # or "vithp" / "dinov3_vith16plus"
        weights_dir="/home/user/km-vipe/weights/dinov3",  # Update this path
        rerun_app_id="dinov3_demo",
        pyramid_scales=[2.0, 1.5, 1.0, 0.5],
    )

    # --- Paths to your data ---
    image_dir = "/data/Replica/replica/office0/rgb"
    image_files = [f"{image_dir}/frame{str(i).zfill(6)}.jpg" for i in range(20)]

    test_image = None

    # --- Part 1: Per-frame embedding and simple visualization ---
    print("\n--- Part 1: Processing single frames ---")
    for i, img_path in enumerate(image_files):
        try:
            test_image = Image.open(img_path)
        except FileNotFoundError:
            print(f"Warning: Image not found {img_path}. Skipping frame {i}.")
            if i == 0:
                print("Error: Cannot start demo, no images found. Exiting.")
                return
            continue  # Skip to next frame

        print(f"\n{'=' * 60}")
        print(f"Processing Frame {i}")
        print(f"{'=' * 60}")

        t0 = time.time()
        features = tracker.embed_frame(test_image, frame_idx=i)
        t1 = time.time()

        print(f"Frame {i} embedded: features shape {features.shape}")
        print(f"Embedding time: {t1 - t0:.3f} seconds")

        # Log embedding time to Rerun
        rr.set_time_sequence("frame", i)
        rr.log("performance/embedding_time", rr.Scalar(t1 - t0))

        # --- UPDATED VISUALIZATION ---
        # Use one of the simplified methods: 'naive_rgb', 'mean', 'std', 'norm'
        tracker.visualize_embeddings(
            features,
            method="naive_rgb",
            entity_path="visualizations/naive_rgb",
            frame_idx=i,
        )

        # Also show similarity for a query point
        h, w, _ = features.shape
        query_point = (h // 2, w // 2)  # Center point
        tracker.visualize_feature_similarity(
            features, query_points=[query_point], entity_path="visualizations/similarity", frame_idx=i
        )

        # Log memory usage
        mem_stats = tracker.get_memory_usage()
        if "allocated_gb" in mem_stats:
            print(f"GPU Memory: {mem_stats['allocated_gb']:.2f} GB allocated")
            rr.log("performance/gpu_allocated_gb", rr.Scalar(mem_stats["allocated_gb"]))

        # --- Part 2: Multi-scale embedding and pyramid upsampling (on last frame) ---
        if test_image is not None:
            print(f"\n{'=' * 60}")
            print("--- Part 2: Demonstrating Pyramid Upsampling ---")
            print(f"{'=' * 60}")

            # Set Rerun timeline to the last frame index for this static demo
            last_frame_idx = len(image_files) - 1
            rr.set_time_sequence("frame", last_frame_idx)

            # 1. Get the target size (e.g., original image resolution)
            target_h, target_w = test_image.height, test_image.width
            print(f"Original image size (H, W): ({target_h}, {target_w})")

            # 2. Extract features at multiple scales
            # This returns a list of Tensors [H_i, W_i, D]
            print("Extracting multi-scale features...")
            t_start_multi = time.time()
            features_pyramid = tracker._extract_radseg_features_multiscale(test_image)
            t_end_multi = time.time()
            print(f"Multi-scale extraction done in {t_end_multi - t_start_multi:.3f}s")

            # 3. Upsample and blend the pyramid to the target size
            print("Blending pyramid with 'pyramid_weighted'...")
            t_start_blend = time.time()
            blended_features = tracker.upsample_features(
                features=features_pyramid, target_size=(target_h, target_w), method="pyramid_weighted"
            )
            t_end_blend = time.time()
            print(f"Pyramid blending done in {t_end_blend - t_start_blend:.3f}s")
            print(f"Final Blended Feature Shape (H, W, D): {blended_features.shape}")

            # 4. Visualize the high-resolution blended features
            tracker.visualize_embeddings(
                blended_features,
                method="naive_rgb",
                entity_path="visualizations/pyramid_weighted_rgb",
                frame_idx=last_frame_idx,
            )

            # 5. Compare with simple bilinear upsampling of the 1.0 scale features
            print("Comparing with standard 'bilinear' upsample...")
            single_scale_features = features_pyramid[0]
            bilinear_features = tracker.upsample_features(
                features=single_scale_features, target_size=(target_h, target_w), method="bilinear"
            )
            tracker.visualize_embeddings(
                bilinear_features,
                method="naive_rgb",
                entity_path="visualizations/bilinear_rgb",
                frame_idx=last_frame_idx,
            )

    print(f"\n{'=' * 60}")
    print("Demo complete! Check the Rerun viewer for visualizations.")
    print("Look for 'visualizations/naive_rgb' (per-frame) and")
    print("'visualizations/pyramid_weighted_rgb' (hi-res blend).")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    # Run demonstration
    # Make sure to import the necessary classes from your optimized code file
    # from optimized_code import DINOv3EmbeddingEngine, DinoV3Variant

    demo_usage()
