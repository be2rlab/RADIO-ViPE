import logging

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, List

import numpy as np
import rerun as rr
import torch

from vipe.streams.base import VideoFrame

from .dinov2 import DINOv2EmbeddingEngine, DinoV2Variant
from .dinov3 import DINOv3EmbeddingEngine, DinoV3Variant
from .radseg_encoder import RADSegEncoder
from typing import Dict, List, Literal, Optional, Tuple, Union
from torch import Tensor, nn
import torch.nn.functional as F
import torchvision.transforms as TVT
import torchvision.transforms.functional as TVTF

from PIL import Image


if TYPE_CHECKING:
    from vipe.priors.track_anything.yoloe_detector import YOLOEDetector


_REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DINOV3_WEIGHTS = _REPO_ROOT / "weights" / "dinov3"
DEFAULT_DINOV2_WEIGHTS = _REPO_ROOT / "weights" / "dinov2"
DEFAULT_YOLOE_WEIGHTS = _REPO_ROOT / "yoloe-11l-seg-pf.pt"
logger = logging.getLogger(__name__)


class BackboneFamily(str, Enum):
    """Unified selector for supported backbones."""

    DINOV2 = "dinov2"
    DINOV3 = "dinov3"
    RADSEG = "radseg"

    @classmethod
    def from_value(cls, value: "BackboneFamily | str") -> "BackboneFamily":
        if isinstance(value, cls):
            return value
        normalized = str(value).lower()
        for member in cls:
            if member.value == normalized:
                return member
        raise ValueError(f"Unknown backbone family: {value}")

# Keep the old name as an alias for backwards compatibility
DinoBackboneFamily = BackboneFamily


@dataclass
class PCABasis:
    mean: torch.Tensor  # (C,)
    components: torch.Tensor  # (C, K)


class PCAProjector:
    """
    Torch-based PCA projector for per-pixel embeddings.
    - Fit: learns mean and top-K components on sampled pixels.
    - Encode: projects to K-dim.
    - Decode: reconstructs to original C-dim (approximate).
    """

    def __init__(
        self,
        target_dim: int = 128,
        max_samples: int = 200_000,
        seed: int = 0,
        minimum_image_samples: int = 6,
    ):
        self.target_dim = target_dim
        self.max_samples = max_samples
        self.seed = seed
        self.minimum_image_samples = minimum_image_samples
        self._basis: PCABasis | None = None
        self._clear_buffer()

    def _clear_buffer(self) -> None:
        self._samples_buffer: list[torch.Tensor] = []
        self._samples_buffer_idx: list[int] = []
        self._samples_buffer_view_idx: list[int] = []

    @staticmethod
    def _to_batch(feats: torch.Tensor | list[torch.Tensor]) -> torch.Tensor:
        """
        Accepts:
          - (H, W, C)
          - (N, H, W, C)
          - list of (H, W, C) tensors → stacked to (N, H, W, C)
        Returns (N*H*W, C) for PCA computation.
        """
        if isinstance(feats, list):
            feats = torch.stack(feats, dim=0)  # (N, H, W, C)

        if feats.dim() == 3:
            H, W, C = feats.shape
            return feats.reshape(H * W, C)
        elif feats.dim() == 4:
            N, H, W, C = feats.shape
            return feats.reshape(N * H * W, C)
        else:
            raise ValueError("feats must be (H, W, C), (N, H, W, C), or a list of (H, W, C)")

    @torch.no_grad()
    def fit(self, feats: torch.Tensor | list[torch.Tensor]) -> PCABasis:
        """
        feats: (H, W, C), (N, H, W, C), or list of (H, W, C) tensors.
        Randomly samples up to max_samples pixels to learn PCA.
        """
        X = self._to_batch(feats)
        device = X.device
        P, C = X.shape

        g = torch.Generator(device=device)
        g.manual_seed(self.seed)
        if P > self.max_samples:
            idx = torch.randint(low=0, high=P, size=(self.max_samples,), generator=g, device=device)
            Xs = X[idx]
        else:
            Xs = X

        mean = Xs.mean(dim=0)
        Xc = Xs - mean

        _, _, Vh = torch.linalg.svd(Xc, full_matrices=False)
        V = Vh.transpose(0, 1)

        K = min(self.target_dim, V.shape[1])
        components = V[:, :K].contiguous()

        self._basis = PCABasis(mean=mean, components=components)
        return self._basis

    @torch.no_grad()
    def encode(self, feats: torch.Tensor | list[torch.Tensor]) -> list[torch.Tensor]:
        """
        feats: (H, W, C), (N, H, W, C), or list of (H, W, C) tensors.
        Returns: list of (H, W, K) tensors — one per input frame.
        """
        if self._basis is None:
            raise RuntimeError("PCAProjector not fit yet. Call fit() first.")

        if isinstance(feats, list):
            return [self._encode_single(f) for f in feats]

        if feats.dim() == 3:
            return [self._encode_single(feats)]
        elif feats.dim() == 4:
            return [self._encode_single(feats[i]) for i in range(feats.shape[0])]
        else:
            raise ValueError("feats must be (H, W, C), (N, H, W, C), or a list of (H, W, C)")

    def _encode_single(self, feats: torch.Tensor) -> torch.Tensor:
        """feats: (H, W, C) → (H, W, K)"""
        mean = self._basis.mean.to(device=feats.device, dtype=feats.dtype)
        comps = self._basis.components.to(device=feats.device, dtype=feats.dtype)
        H, W, C = feats.shape
        Z = (feats.reshape(-1, C) - mean) @ comps
        return Z.reshape(H, W, comps.shape[1])

    @torch.no_grad()
    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        """codes: (H, W, K) → (H, W, C) approximate reconstruction."""
        if self._basis is None:
            raise RuntimeError("PCAProjector not fit yet. Call fit() first.")
        mean = self._basis.mean.to(device=codes.device, dtype=codes.dtype)
        comps = self._basis.components.to(device=codes.device, dtype=codes.dtype)
        H, W, K = codes.shape
        Xr = codes.reshape(-1, K) @ comps.transpose(0, 1) + mean
        return Xr.reshape(H, W, comps.shape[0])

    def is_fit(self) -> bool:
        return self._basis is not None

    def load_basis(self, path: Path) -> PCABasis:
        if not path.exists():
            raise FileNotFoundError(f"PCA basis file not found: {path}")
        data = torch.load(path, map_location="cpu")
        if "mean" not in data or "components" not in data:
            raise KeyError(
                f"PCA basis file must contain 'mean' and 'components' keys, "
                f"found: {list(data.keys())}"
            )
        self._basis = PCABasis(mean=data["mean"], components=data["components"].T)
        return self._basis

    def state_dict(self) -> dict:
        if self._basis is None:
            raise RuntimeError("Projector has no basis yet.")
        return {"mean": self._basis.mean, "components": self._basis.components}

    def load_state_dict(self, state: dict) -> None:
        self._basis = PCABasis(mean=state["mean"], components=state["components"])


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


class EmbeddingsPipeline:
    """
    Embedding pipeline with selectable backbone (DINOv2, DINOv3, or RADSeg),
    optional PCA compression, and optional YOLOE-based instance pooling.
    """

    def __init__(
        self,
        model_family: BackboneFamily | str = BackboneFamily.DINOV3,
        model_variant: str | DinoV3Variant | DinoV2Variant | None = None,
        weights_dir: str | None = None,
        pca_dim: int | None = None,
        pca_max_samples: int = 200_000,
        pca_seed: int = 0,
        device: str = "cuda",
        segment_with_yoloe: bool = False,
        yolo_model_path: str | None = None,
        yolo_conf_threshold: float = 0.25,
        yolo_iou_threshold: float = 0.45,
        yolo_mask_threshold: float = 0.5,
        yolo_device: str | None = None,
        mask_visualization_entity: str = "yoloe",
        visualize_masks: bool = True,
        # ── RADSeg-specific parameters ──────────────────────────────
        radseg_model_version: str = "c-radio_v3-b",
        radseg_lang_model: str = "siglip2",
        radseg_compile: bool = False,
        radseg_amp: bool = False,
        radseg_scra_scaling: float = 10.0,
        radseg_scga_scaling: float = 10.0,
        radseg_slide_crop: int = 336,
        radseg_slide_stride: int = 224,
        radseg_lang_align: bool = True,
        load_basis_flag: bool = False,
        pyramid_scales: Optional[List[float]] = None,
        enable_rerun: bool = True,
    ) -> None:

        self.family = BackboneFamily.from_value(model_family)
        self.device = device
        self.enable_rerun = enable_rerun
        # ── Build backbone ──────────────────────────────────────────
        if self.family is BackboneFamily.RADSEG:
            print(f"scra_scaling: {radseg_scra_scaling}")
            print(f"scga_scaling: {radseg_scga_scaling}")
            print(f"slide_crop: {radseg_slide_crop}")
            print(f"slide_stride: {radseg_slide_stride}")
            self._radseg_lang_align = radseg_lang_align
            self.engine = self._build_radseg_encoder(
                device=device,
                model_version=radseg_model_version,
                lang_model=radseg_lang_model,
                compile=radseg_compile,
                amp=radseg_amp,
                scra_scaling=radseg_scra_scaling,
                scga_scaling=radseg_scga_scaling,
                slide_crop=radseg_slide_crop,
                slide_stride=radseg_slide_stride,
            )
            # engine/variant/weights are not used for RADSeg but we keep
            # them around so other code that inspects the pipeline doesn't break.
            self.pyramid_scales = pyramid_scales or [1.0, 0.75, 0.5]
            self.upsampler = PyramidUpsampler(scales=self.pyramid_scales, device=self.device)
            self.model_variant = radseg_model_version
            self.weights_dir = None
        else:
            self._radseg_lang_align = False
            self.model_variant = self._resolve_model_variant(model_variant)
            self.weights_dir = self._resolve_weights_dir(weights_dir)
            self.engine = self._build_engine(device)

        # ── ImageNet normalisation (used for DINO backbones) ────────
        self._norm_mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
        self._norm_std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)

        # ── PCA projector ───────────────────────────────────────────
        self.projector: PCAProjector | None = None
        self.load_basis_flag = load_basis_flag
        if pca_dim is not None:
            self.projector = PCAProjector(target_dim=pca_dim, max_samples=pca_max_samples, seed=pca_seed)

        # ── YOLOE segmentation ──────────────────────────────────────
        self.segment_with_yoloe = segment_with_yoloe
        self.yolo_conf_threshold = yolo_conf_threshold
        self.yolo_iou_threshold = yolo_iou_threshold
        self.yolo_mask_threshold = yolo_mask_threshold
        self.yolo_device = yolo_device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.visualize_masks = visualize_masks
        self.mask_visualization_entity = mask_visualization_entity.strip() or "yoloe"
        self._yolo_detector: YOLOEDetector | None = None

        self.latest_mask: torch.Tensor | None = None
        self.latest_mask_info: list[dict[str, Any]] = []
        self.latest_mask_embeddings: dict[int, torch.Tensor] = {}

        self._clear_segmentation_cache()
        if self.segment_with_yoloe:
            self._init_yolo_detector(yolo_model_path)

    # ── RADSeg construction ─────────────────────────────────────────
    @staticmethod
    def _build_radseg_encoder(
        device: str,
        model_version: str,
        lang_model: str,
        compile: bool,
        amp: bool,
        scra_scaling: float,
        scga_scaling: float,
        slide_crop: int,
        slide_stride: int,
    ):
        return RADSegEncoder(
            device=device,
            model_version=model_version,
            lang_model=lang_model,
            return_radio_features=True,
            compile=compile,
            amp=amp,
            predict=False,
            scra_scaling=scra_scaling,
            scga_scaling=scga_scaling,
            slide_crop=slide_crop,
            slide_stride=slide_stride,
        )

    # ── DINO helpers (unchanged) ────────────────────────────────────
    def _resolve_model_variant(
        self, model_variant: str | DinoV3Variant | DinoV2Variant | None
    ) -> str | DinoV3Variant | DinoV2Variant:
        defaults: dict[BackboneFamily, DinoV3Variant | DinoV2Variant] = {
            BackboneFamily.DINOV3: DinoV3Variant.VITSP,
            BackboneFamily.DINOV2: DinoV2Variant.VITL,
        }
        if model_variant is None:
            return defaults[self.family]

        if self.family is BackboneFamily.DINOV3 and isinstance(model_variant, DinoV2Variant):
            logger.warning("Received DINOv2 variant for a DINOv3 backbone. Falling back to %s.", defaults[self.family])
            return defaults[self.family]
        if self.family is BackboneFamily.DINOV2 and isinstance(model_variant, DinoV3Variant):
            logger.warning("Received DINOv3 variant for a DINOv2 backbone. Falling back to %s.", defaults[self.family])
            return defaults[self.family]
        return model_variant

    def _resolve_weights_dir(self, weights_dir: str | None) -> str | None:
        if weights_dir:
            return weights_dir
        default_dir = DEFAULT_DINOV3_WEIGHTS if self.family is BackboneFamily.DINOV3 else DEFAULT_DINOV2_WEIGHTS
        return str(default_dir)

    def _build_engine(self, device) -> DINOv2EmbeddingEngine | DINOv3EmbeddingEngine:
        if self.family is BackboneFamily.DINOV3:
            return DINOv3EmbeddingEngine(
                model=self.model_variant,
                weights_dir=self.weights_dir,
                pyramid_scales=[2.0, 1.0, 0.75],
                device=device,
            )
        return DINOv2EmbeddingEngine(
            model=self.model_variant,
            weights_dir=self.weights_dir,
            pyramid_scales=[2.0, 1.0, 0.75],
            device=device,
        )

    # ── YOLOE helpers (unchanged) ───────────────────────────────────
    def _init_yolo_detector(self, model_path: str | None) -> None:
        try:
            from vipe.priors.track_anything.yoloe_detector import YOLOEDetector as YOLOEDetectorImpl
        except ImportError as exc:
            raise RuntimeError("YOLOE detector requires the 'ultralytics' package. Please install it.") from exc

        resolved_path = Path(model_path) if model_path is not None else DEFAULT_YOLOE_WEIGHTS
        if not resolved_path.exists():
            raise FileNotFoundError(
                f"YOLOE weights not found at {resolved_path}. "
                "Pass `yolo_model_path` or place the checkpoint at the repository root."
            )
        self._yolo_detector = YOLOEDetectorImpl(model_path=str(resolved_path), device=self.yolo_device)

    def _clear_segmentation_cache(self) -> None:
        self.latest_mask = None
        self.latest_mask_info = []
        self.latest_mask_embeddings = {}

    def latest_segmentation(self) -> tuple[torch.Tensor | None, list[dict[str, Any]], dict[int, torch.Tensor]]:
        return self.latest_mask, self.latest_mask_info, self.latest_mask_embeddings

    @staticmethod
    def _frame_to_numpy(frame_data: VideoFrame) -> np.ndarray:
        rgb = frame_data.rgb.detach().cpu().clamp_(0.0, 1.0).numpy()
        rgb_uint8 = np.ascontiguousarray((rgb * 255.0).round().astype(np.uint8))
        return rgb_uint8

    def _segment_with_yoloe(self, frame_data: VideoFrame) -> tuple[np.ndarray | None, list[dict[str, Any]]]:
        if not self.segment_with_yoloe or self._yolo_detector is None:
            return None, []

        origin_frame = self._frame_to_numpy(frame_data)
        _, _, masks, class_names, confidences = self._yolo_detector.run_detection(
            origin_frame,
            conf_threshold=self.yolo_conf_threshold,
            iou_threshold=self.yolo_iou_threshold,
        )

        mask_map = np.zeros(origin_frame.shape[:2], dtype=np.int32)
        mask_info: list[dict[str, Any]] = []
        instance_id = 1

        for idx, mask in enumerate(masks):
            if mask is None:
                continue
            binary_mask = mask > self.yolo_mask_threshold
            if not np.any(binary_mask):
                continue
            mask_map[binary_mask] = instance_id
            label = class_names[idx] if idx < len(class_names) else "object"
            confidence = float(confidences[idx]) if idx < len(confidences) else 0.0
            mask_info.append({"id": instance_id, "class": label, "confidence": confidence})
            instance_id += 1

        if instance_id == 1:
            return None, []
        return mask_map.astype(np.int32), mask_info

    @staticmethod
    def _pool_embeddings_by_mask(
        features: torch.Tensor, mask_np: np.ndarray
    ) -> tuple[torch.Tensor, dict[int, torch.Tensor]]:
        device = features.device
        mask_tensor = torch.from_numpy(mask_np).to(device=device, dtype=torch.long)
        labels = mask_tensor.view(-1)
        valid = labels > 0
        if not torch.any(valid):
            return features, {}

        feats_flat = features.view(-1, features.shape[-1])
        valid_labels = labels[valid]
        valid_feats = feats_flat[valid]
        max_label = int(valid_labels.max().item())

        sums = torch.zeros(max_label + 1, features.shape[-1], device=device, dtype=features.dtype)
        sums.index_add_(0, valid_labels, valid_feats)
        counts = torch.bincount(valid_labels, minlength=max_label + 1).clamp_min(1).to(features.dtype).unsqueeze(1)
        means = sums / counts

        pooled_flat = feats_flat.clone()
        pooled_flat[valid] = means[valid_labels]
        pooled = pooled_flat.view_as(features)

        unique_ids = torch.unique(valid_labels)
        mask_embeddings = {int(idx): means[int(idx)].detach().cpu() for idx in unique_ids}
        return pooled, mask_embeddings

    def _apply_mask_pooling(
        self,
        features: torch.Tensor,
        mask_np: np.ndarray,
        mask_info: list[dict[str, Any]],
        frame_idx: int,
    ) -> torch.Tensor:
        pooled, mask_embeddings = self._pool_embeddings_by_mask(features, mask_np)
        self.latest_mask = torch.from_numpy(mask_np.copy())
        self.latest_mask_info = mask_info
        self.latest_mask_embeddings = mask_embeddings
        self._visualize_masks(mask_np, frame_idx)
        return pooled

    def _visualize_masks(self, mask_np: np.ndarray, frame_idx: int) -> None:
        if not self.visualize_masks or mask_np.max() == 0:
            return
        mask_rgb = self.engine.mask_to_rgb(mask_np, int(mask_np.max()) + 1)
        rr.set_time_sequence("frame", frame_idx)
        rr.log(f"{self.mask_visualization_entity}/segmentation", rr.SegmentationImage(mask_np))
        rr.log(f"{self.mask_visualization_entity}/visualization", rr.Image(mask_rgb))

    def _prepare_image_tensor(self, frame_data: VideoFrame) -> torch.Tensor:
        """Normalise a VideoFrame for DINO backbones (ImageNet stats)."""
        rgb_tensor = frame_data.rgb.permute(2, 0, 1).contiguous().float()
        rgb_tensor = rgb_tensor.clamp_(0.0, 1.0)
        if self.family == BackboneFamily.RADSEG:
            return rgb_tensor.unsqueeze(0)
        elif self.family == BackboneFamily.DINOV3 or self.family == BackboneFamily.DINOV2:
            mean = self._norm_mean.to(rgb_tensor.device)
            std = self._norm_std.to(rgb_tensor.device)
            return (rgb_tensor - mean) / std

    # ── RADSeg feature extraction ───────────────────────────────────
    @torch.no_grad()
    def _extract_radseg_features(self, image: torch.Tensor) -> torch.Tensor:
        """
        Run RADSeg and return features as (H', W', C) on the original device.

        The RADSeg encoder returns a (B, C, H', W') feature map. When
        ``radseg_lang_align`` is True, the spatial features are additionally
        projected into the language-aligned space so that they can be compared
        with text embeddings via cosine similarity.
        """
        image = image.to(self.device)
        feat_map = self.engine.encode_image_to_feat_map(image)  # (1, C, H', W')
        # print(f"image shape is {image.shape}")
        # print(f"feat_map shape is {feat_map.shape}")

        if self._radseg_lang_align:
            feat_map = self.engine.align_spatial_features_with_language(
                feat_map, onehot=False
            )  # (1, C', H', W')

        # (1, C, H', W') → (H', W', C)
        feat_map = feat_map.squeeze(0).permute(1, 2, 0).contiguous().float()
        return feat_map

    # ── Main entry-points ───────────────────────────────────────────

    def process_frame(self, frame_data: VideoFrame, frame_idx: int, view_idx: int):
        features, _ = self.embed_frame(frame_data)

        if self.projector is None:
            return [features], [frame_idx], [view_idx]

        if not self.projector.is_fit():
            self.projector._samples_buffer.append(features)
            self.projector._samples_buffer_idx.append(frame_idx)
            self.projector._samples_buffer_view_idx.append(view_idx)

            if len(self.projector._samples_buffer) >= self.projector.minimum_image_samples:
                self.projector.fit(self.projector._samples_buffer)
                codes = self.projector.encode(self.projector._samples_buffer)
                idx_list = self.projector._samples_buffer_idx.copy()
                view_list = self.projector._samples_buffer_view_idx.copy()
                self.projector._clear_buffer()
                return codes, idx_list, view_list

            return None, None, None

        codes = self.projector.encode(features)
        return codes, [frame_idx], [view_idx]



    def embed_frame(self, frame_data: VideoFrame) -> tuple[torch.Tensor, int]:
        """Process a single video frame to extract embeddings."""
        orig_device = frame_data.rgb.device
        image_tensor = self._prepare_image_tensor(frame_data)

        if self.family is BackboneFamily.RADSEG:
            feats = self._extract_radseg_features(image_tensor)
            feats = feats.to(orig_device, non_blocking=True)
            patch_size = self.engine.model.patch_size
        else:
            feats = self.engine._extract_dino_features_multiscale(image_tensor, scales=[1.0])
            patch_size = self.engine.patch_size

        upfeats = self.upsample_features(
            features=feats,
            target_size=(frame_data.rgb.shape[0] // 8, frame_data.rgb.shape[1] // 8),
            method="bilinear",
        ).detach()
        del feats
        upfeats = upfeats.to(orig_device, non_blocking=True).float()
            

        # ── Visualisation (only when a DINO engine is available) ────
        if self.engine is not None and self.projector is not None:
            self.visualize_embeddings(
                upfeats,
                method="naive_rgb",
                entity_path=f"visualizations/naive_rgb",
                frame_idx=frame_data.raw_frame_idx,
            )

        return upfeats, patch_size

    def encode_labels(self, labels: List[str]) -> torch.Tensor:
        """Encode text labels into embeddings (RADSeg only).

        Returns an (N, C) tensor of L2-normalised text features that can be
        compared with language-aligned spatial features via cosine similarity.
        """
        if self.engine is None:
            raise RuntimeError("encode_labels is only available with the RADSeg backbone.")
        return self.engine.encode_labels(labels, onehot=False)

    def encode_prompts(self, prompts: List[str]) -> torch.Tensor:
        """Encode free-form text prompts (RADSeg only)."""
        if self.engine is None:
            raise RuntimeError("encode_prompts is only available with the RADSeg backbone.")
        return self.engine.encode_prompts(prompts, onehot=False)

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

    def visualize_embeddings(
        self,
        features: Tensor,
        method: Literal["mean", "std", "norm", "naive_rgb"] = "naive_rgb",
        entity_path: str = "features",
        frame_idx: Optional[int] = None,
    ):
        """Visualize features embeddings in Rerun.

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
            # img = Image.fromarray(vis_uint8)
            # img.save(f"feat_image_{frame_idx}.png")

            rr.log(f"{entity_path}/naive_rgb", rr.Image(vis_uint8))
            return
        else:
            raise ValueError(f"Unknown visualization method: {method}")

        if vis_features is not None:
            rr.log(f"{entity_path}/{method}", rr.DepthImage(vis_features))