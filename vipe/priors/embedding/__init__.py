import logging

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import rerun as rr
import torch

from vipe.streams.base import VideoFrame

from .dinov2 import DINOv2EmbeddingEngine, DinoV2Variant
from .dinov3 import DINOv3EmbeddingEngine, DinoV3Variant


if TYPE_CHECKING:
    from vipe.priors.track_anything.yoloe_detector import YOLOEDetector


_REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DINOV3_WEIGHTS = _REPO_ROOT / "weights" / "dinov3"
DEFAULT_DINOV2_WEIGHTS = _REPO_ROOT / "weights" / "dinov2"
DEFAULT_YOLOE_WEIGHTS = _REPO_ROOT / "yoloe-11l-seg-pf.pt"
logger = logging.getLogger(__name__)


class DinoBackboneFamily(str, Enum):
    """Unified selector for the supported DINO backbones."""

    DINOV2 = "dinov2"
    DINOV3 = "dinov3"

    @classmethod
    def from_value(cls, value: "DinoBackboneFamily | str") -> "DinoBackboneFamily":
        if isinstance(value, cls):
            return value
        normalized = str(value).lower()
        for member in cls:
            if member.value == normalized:
                return member
        raise ValueError(f"Unknown DINO backbone family: {value}")


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

    def __init__(self, target_dim: int = 64, max_samples: int = 200_000, seed: int = 0):
        self.target_dim = target_dim
        self.max_samples = max_samples
        self.seed = seed
        self._basis: PCABasis | None = None

    @torch.no_grad()
    def fit(self, feats: torch.Tensor) -> PCABasis:
        """
        feats: (H, W, C) or (N, H, W, C) tensor of per-pixel features (float).
        We randomly sample up to max_samples pixels to learn PCA.
        """
        # Flatten to (P, C)
        if feats.dim() == 3:
            H, W, C = feats.shape
            X = feats.reshape(H * W, C)
        elif feats.dim() == 4:
            N, H, W, C = feats.shape
            X = feats.reshape(N * H * W, C)
        else:
            raise ValueError("feats must be (H, W, C) or (N, H, W, C)")

        device = X.device
        P, C = X.shape

        # Sample pixels
        g = torch.Generator(device=device)
        g.manual_seed(self.seed)
        if P > self.max_samples:
            idx = torch.randint(low=0, high=P, size=(self.max_samples,), generator=g, device=device)
            Xs = X[idx]
        else:
            Xs = X

        # Center
        mean = Xs.mean(dim=0)  # (C,)
        Xc = Xs - mean

        # SVD on (P, C). We only need the right singular vectors (Vh).
        # full_matrices=False is cheaper and is what we want for PCA.
        # Xc = U S V^T  -> top-K components are columns of V (size CxC), take first K.
        # torch.linalg.svd returns U, S, Vh; V = Vh^T
        U, S, Vh = torch.linalg.svd(Xc, full_matrices=False)
        V = Vh.transpose(0, 1)  # (C, C)

        K = min(self.target_dim, V.shape[1])
        components = V[:, :K].contiguous()  # (C, K)

        self._basis = PCABasis(mean=mean, components=components)
        return self._basis

    def load_basis(self, path: Path) -> PCABasis:
        """
        Load PCA basis from a .pt file and initialize PCABasis.
        """

        if not path.exists():
            raise FileNotFoundError(f"PCA basis file not found: {path}")

        data = torch.load(path, map_location="cpu")

        if "mean" not in data or "components" not in data:
            raise KeyError(
                f"PCA basis file must contain 'mean' and 'components' keys, "
                f"found keys: {list(data.keys())}"
            )

        mean = data["mean"]
        components = data["components"]

        self._basis = PCABasis(mean=mean, components=components.T)
        return self._basis

    def is_fit(self) -> bool:
        return self._basis is not None

    @torch.no_grad()
    def encode(self, feats: torch.Tensor) -> torch.Tensor:
        """
        feats: (H, W, C) float tensor
        returns: (H, W, K) float tensor
        """
        if self._basis is None:
            raise RuntimeError("PCAProjector not fit yet. Call fit() first.")
        mean, comps = self._basis.mean, self._basis.components  # (C,), (C, K)
        device = feats.device
        # Move basis to the right device (in case fit occurred elsewhere)
        mean = mean.to(device=device, dtype=feats.dtype)
        comps = comps.to(device=device, dtype=feats.dtype)

        H, W, C = feats.shape
        X = feats.reshape(-1, C)  # (P, C)
        Xc = X - mean
        Z = Xc @ comps  # (P, K)
        return Z.reshape(H, W, comps.shape[1])

    @torch.no_grad()
    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        """
        codes: (H, W, K) float tensor
        returns: (H, W, C) float tensor (approximate reconstruction)
        """
        if self._basis is None:
            raise RuntimeError("PCAProjector not fit yet. Call fit() first.")
        mean, comps = self._basis.mean, self._basis.components  # (C,), (C, K)
        device = codes.device
        mean = mean.to(device=device, dtype=codes.dtype)
        comps = comps.to(device=device, dtype=codes.dtype)

        H, W, K = codes.shape
        Z = codes.reshape(-1, K)  # (P, K)
        Xr = Z @ comps.transpose(0, 1) + mean  # (P, C)
        C = comps.shape[0]
        return Xr.reshape(H, W, C)

    def state_dict(self) -> dict:
        if self._basis is None:
            raise RuntimeError("Projector has no basis yet.")
        return {"mean": self._basis.mean, "components": self._basis.components}

    def load_state_dict(self, state: dict) -> None:
        self._basis = PCABasis(mean=state["mean"], components=state["components"])


class EmbeddingsPipeline:
    """
    Embedding pipeline with selectable DINO backbone, optional PCA compression,
    and optional YOLOE-based instance pooling for object-centric features.
    """

    def __init__(
        self,
        model_family: DinoBackboneFamily | str = DinoBackboneFamily.DINOV3,
        model_variant: str | DinoV3Variant | DinoV2Variant | None = DinoV3Variant.VITSP,
        weights_dir: str | None = None,
        pca_dim: int | None = 32,
        pca_max_samples: int = 200_000,
        pca_seed: int = 0,
        device: str = 'cuda',
        segment_with_yoloe: bool = False,
        yolo_model_path: str | None = None,
        yolo_conf_threshold: float = 0.25,
        yolo_iou_threshold: float = 0.45,
        yolo_mask_threshold: float = 0.5,
        yolo_device: str | None = None,
        mask_visualization_entity: str = "yoloe",
        visualize_masks: bool = True,
    ) -> None:
        self.family = DinoBackboneFamily.from_value(model_family)
        self.model_variant = self._resolve_model_variant(model_variant)
        self.weights_dir = self._resolve_weights_dir(weights_dir)
        self.engine = self._build_engine(device)
        self._norm_mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
        self._norm_std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)
        self.device = device

        self.projector: PCAProjector | None = None
        if pca_dim is not None:
            self.projector = PCAProjector(target_dim=pca_dim, max_samples=pca_max_samples, seed=pca_seed)

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

    def _resolve_model_variant(
        self, model_variant: str | DinoV3Variant | DinoV2Variant | None
    ) -> str | DinoV3Variant | DinoV2Variant:
        defaults: dict[DinoBackboneFamily, DinoV3Variant | DinoV2Variant] = {
            DinoBackboneFamily.DINOV3: DinoV3Variant.VITSP,
            DinoBackboneFamily.DINOV2: DinoV2Variant.VITL,
        }
        if model_variant is None:
            return defaults[self.family]

        if self.family is DinoBackboneFamily.DINOV3 and isinstance(model_variant, DinoV2Variant):
            logger.warning("Received DINOv2 variant for a DINOv3 backbone. Falling back to %s.", defaults[self.family])
            return defaults[self.family]
        if self.family is DinoBackboneFamily.DINOV2 and isinstance(model_variant, DinoV3Variant):
            logger.warning("Received DINOv3 variant for a DINOv2 backbone. Falling back to %s.", defaults[self.family])
            return defaults[self.family]
        return model_variant

    def _resolve_weights_dir(self, weights_dir: str | None) -> str | None:
        if weights_dir:
            return weights_dir
        default_dir = DEFAULT_DINOV3_WEIGHTS if self.family is DinoBackboneFamily.DINOV3 else DEFAULT_DINOV2_WEIGHTS
        return str(default_dir)

    def _build_engine(self,device) -> DINOv2EmbeddingEngine | DINOv3EmbeddingEngine:
        if self.family is DinoBackboneFamily.DINOV3:
            return DINOv3EmbeddingEngine(
                model=self.model_variant,
                weights_dir=self.weights_dir,
                pyramid_scales=[2.0, 1.0, 0.75],
                device = device,
            )
        return DINOv2EmbeddingEngine(
            model=self.model_variant,
            weights_dir=self.weights_dir,
            pyramid_scales=[2.0, 1.0, 0.75],
            device = device,
        )

    def _init_yolo_detector(self, model_path: str | None) -> None:
        try:
            from vipe.priors.track_anything.yoloe_detector import YOLOEDetector as YOLOEDetectorImpl
        except ImportError as exc:  # pragma: no cover - optional dependency
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
        """
        Returns:
            mask tensor on CPU, detection metadata, and pooled embeddings per instance id.
        """
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
        rgb_tensor = frame_data.rgb.permute(2, 0, 1).contiguous().float()
        rgb_tensor = rgb_tensor.clamp_(0.0, 1.0)
        mean = self._norm_mean.to(rgb_tensor.device)
        std = self._norm_std.to(rgb_tensor.device)
        return (rgb_tensor - mean) / std

    def process_frame(self, frame_data: VideoFrame) -> tuple[torch.Tensor, int]:
        """Process a single video frame to extract embeddings."""
        orig_device = frame_data.rgb.device
        normalized = self._prepare_image_tensor(frame_data)
        pyr_feats = self.engine.embed_frame_multiscale(normalized)
        upfeats = self.engine.upsample_features(
            features=pyr_feats,
            target_size=(frame_data.rgb.shape[0], frame_data.rgb.shape[1]),
            method="pyramid_weighted",
        ).detach()
        del pyr_feats

        feats = upfeats.to(orig_device, non_blocking=True).float()
        del upfeats

        # if feats.dim() == 3 and feats.shape[0] < feats.shape[-1]:
        #     feats = feats.permute(1, 2, 0).contiguous()

        if self.projector is not None:
            if not self.projector.is_fit():
                self.projector.load_basis(Path('/home/user/km-vipe/pca_basis.pt'))
            codes = self.projector.encode(feats)
            del feats
        # mask_np, mask_info = self._segment_with_yoloe(frame_data)
        # if mask_np is not None:
        #     feats = self._apply_mask_pooling(features, mask_np, mask_info, frame_data.raw_frame_idx)
        # else:
        #     self._clear_segmentation_cache()

        if self.projector is not None:
            self.engine.visualize_embeddings(
                codes,
                method="naive_rgb",
                entity_path="visualizations/naive_rgb",
                frame_idx=frame_data.raw_frame_idx,
            )

        return codes, self.engine.patch_size
    

    def process_image(self, image_tensor) -> tuple[torch.Tensor, int]:
        """Process a single video frame to extract embeddings."""
        orig_device = image_tensor.device
        normalized = image_tensor
        pyr_feats = self.engine.embed_frame_multiscale(normalized,[1.0])
        return pyr_feats
