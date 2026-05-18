#!/usr/bin/env python3
"""
Visualize a SLAM map with Rerun.

Loads a saved SLAM map, renders the dense RGB point cloud, and (optionally)
grounds a free-form text prompt against per-point language embeddings to
highlight matching object instances as colored points and oriented bounding
boxes (one box per DBSCAN cluster).

Pipeline:
    1. Load map (.pt) and convert to NumPy.
    2. If embeddings are PCA-compressed, decompress them with PcaCompressor.
    3. Optionally re-align the cloud using the first GT pose.
    4. For a given text prompt, encode it with RADSegEncoder, compute cosine
       similarity against all point embeddings, and select matching points
       via an adaptive similarity threshold.
    5. Cluster the matching points with DBSCAN and draw one oriented bbox
       per cluster (aligned to the scene's principal axes).
"""

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import rerun as rr
import torch
from scipy.spatial.transform import Rotation
from sklearn.cluster import DBSCAN

try:
    from typing_extensions import override
except ImportError:
    def override(func):
        return func

from vipe.priors.embedding.radseg_encoder import RADSegEncoder


# ---------------------------------------------------------------------------
# PCA compression
# ---------------------------------------------------------------------------

class PcaCompressor:
    """Compress / decompress feature vectors using a learned PCA basis."""

    def __init__(
        self,
        out_dim: int | None = None,
        in_dim: int | None = None,
        path: str | None = None,
    ):
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.q_val = out_dim
        self.mean = None
        self.basis = None
        if path is not None:
            self.load(path)

    @override
    def fit(self, X: torch.FloatTensor) -> None:
        D = X.shape[-1]
        if self.in_dim is not None and D != self.in_dim:
            raise ValueError("Data feature dimension does not match stored input dim")
        self.in_dim = D

        X_flatten = X.flatten(0, -2)
        self.mean = torch.mean(X_flatten, dim=0)
        X_centered = X_flatten - self.mean
        self.q_val = min(self.out_dim, X_flatten.shape[0], X_flatten.shape[1])
        if self.q_val < self.out_dim:
            print(
                f"⚠️  WARNING: requested out_dim ({self.out_dim}) is larger than "
                f"data rank ({min(X_flatten.shape)}). Capping q to {self.q_val}."
            )
            self.out_dim = self.q_val
        _, _, V = torch.pca_lowrank(X_centered, q=self.out_dim)
        self.basis = V

    @override
    def save(self, fp: str) -> None:
        torch.save(
            dict(
                metadata=dict(in_dim=self.in_dim, out_dim=self.out_dim),
                mean=self.mean,
                basis=self.basis,
            ),
            fp,
        )

    @override
    def load(self, fp: str) -> None:
        d = torch.load(fp)
        self.in_dim = d["metadata"]["in_dim"]
        self.out_dim = d["metadata"]["out_dim"]
        self.mean = d["mean"]
        self.basis = d["basis"]

    @override
    def compress(self, X: torch.Tensor) -> torch.Tensor:
        s = list(X.shape)
        s[-1] = self.out_dim
        return (X.flatten(0, -2) @ self.basis).reshape(*s)

    @override
    def decompress(self, Y: torch.Tensor) -> torch.Tensor:
        s = list(Y.shape)
        s[-1] = self.in_dim
        return (Y.flatten(0, -2) @ self.basis.T).reshape(*s)

    @override
    def is_fitted(self) -> bool:
        return self.mean is not None and self.basis is not None


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def _resolve_device(device: str | torch.device) -> torch.device:
    """Normalize a device input (str or torch.device) to torch.device."""
    return device if isinstance(device, torch.device) else torch.device(device)


def load_slam_map(path: Path, device: str = "cpu") -> dict:
    """Load a SLAM map dict from a .pt file onto the given device."""
    return torch.load(path, map_location=_resolve_device(device))


def load_first_gt_pose(path: Path) -> np.ndarray:
    """Read the first 4x4 pose (flattened on one line) from a GT poses file."""
    return np.loadtxt(path, max_rows=1).reshape(4, 4)


def transform_pointcloud(pose: np.ndarray, pointcloud: np.ndarray) -> np.ndarray:
    """Apply a 4x4 SE(3) pose to an (N, 3) point cloud."""
    rot = pose[:3, :3]
    trans = pose[:3, 3]
    return (pointcloud @ rot.T) + trans


def _extract_pca_basis(state: Any) -> tuple[torch.Tensor, torch.Tensor]:
    """Recursively locate (mean, components) tensors inside an arbitrary
    container saved on disk (dict or object with attributes)."""
    if isinstance(state, dict):
        if "mean" in state and "components" in state:
            return state["mean"], state["components"]
        for value in state.values():
            if isinstance(value, dict):
                try:
                    return _extract_pca_basis(value)
                except (KeyError, TypeError):
                    continue
    elif hasattr(state, "mean") and hasattr(state, "components"):
        return state.mean, state.components
    raise KeyError(
        "Unable to locate 'mean' and 'components' tensors in the provided PCA state."
    )


def decompress_embeddings(
    embeddings: torch.Tensor,
    pca_path: Path,
    device: str,
) -> torch.Tensor:
    """Decompress PCA-compressed embeddings using a saved PCA basis file."""
    data = torch.load(pca_path, map_location="cpu")
    mean, components = _extract_pca_basis(data)

    compressor = PcaCompressor()
    compressor.mean = mean.to(device)
    compressor.basis = components.to(device)
    compressor.in_dim = components.shape[0]    # original dimension
    compressor.out_dim = components.shape[1]   # number of components
    embeddings = embeddings.to(compressor.basis.dtype)
    return compressor.decompress(embeddings)


# ---------------------------------------------------------------------------
# Geometry: scene orientation and oriented bounding boxes
# ---------------------------------------------------------------------------

def get_scene_orientation(all_points: np.ndarray) -> np.ndarray:
    """Compute a rotation matrix aligned to the scene's principal axes via PCA.

    The returned matrix is guaranteed to be a proper rotation (det = +1).
    """
    centroid = np.mean(all_points, axis=0)
    _, _, Vh = np.linalg.svd(all_points - centroid, full_matrices=False)

    rotation_matrix = Vh.T
    if np.linalg.det(rotation_matrix) < 0:
        rotation_matrix[:, 2] *= -1
    return rotation_matrix


def compute_aligned_bbox(
    points: np.ndarray,
    orientation_matrix: np.ndarray,
    expansion: float = 1.0,
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    """Fit an oriented bounding box to ``points`` whose axes are aligned to
    ``orientation_matrix`` (i.e., the scene frame).

    Uses the 7th / 93rd percentiles of the projected extents instead of
    min / max for robustness against outliers.

    Returns ``(world_center, local_size, quat_xyzw)`` or
    ``(None, None, None)`` if ``points`` is empty.
    """
    if len(points) == 0:
        return None, None, None

    points_local = points @ orientation_matrix

    bbox_min = np.percentile(points_local, 7.0, axis=0)
    bbox_max = np.percentile(points_local, 93.0, axis=0)

    local_size = (bbox_max - bbox_min) * expansion
    local_center = (bbox_min + bbox_max) / 2

    world_center = orientation_matrix @ local_center
    quat = Rotation.from_matrix(orientation_matrix).as_quat()

    return world_center, local_size, quat


# ---------------------------------------------------------------------------
# Grounding: adaptive threshold and DBSCAN clustering
# ---------------------------------------------------------------------------

def find_grounding_threshold(
    sim: torch.Tensor,
    initial_threshold: float = 0.25,
    target_std: float = 0.015,      # SigLIP2-calibrated
    min_threshold: float = 0.05,    # noise floor — don't go below this
    threshold_step: float = 0.01,
    min_points: int = 10,
) -> tuple[torch.Tensor, float, dict]:
    """Adaptively pick a cosine-similarity threshold.

    Starts at ``initial_threshold`` and decreases by ``threshold_step`` until
    either:
      - enough points are selected AND their std >= ``target_std`` (real
        cluster found), or
      - threshold reaches ``min_threshold`` (give up, fall back to top-k).

    Returns ``(bool_mask, threshold_used, info_dict)``.
    """
    info: dict = {"initial_threshold": initial_threshold, "steps": 0, "trace": []}
    threshold = initial_threshold

    while threshold >= min_threshold:
        mask = sim > threshold
        n_selected = int(mask.sum().item())
        info["steps"] += 1

        if n_selected < min_points:
            info["trace"].append((round(threshold, 4), n_selected, None))
            threshold -= threshold_step
            continue

        sel_std = float(sim[mask].std().item())
        info["trace"].append((round(threshold, 4), n_selected, round(sel_std, 4)))

        if sel_std >= target_std:
            sel = sim[mask]
            info.update({
                "final_threshold": threshold,
                "n_selected": n_selected,
                "selected_std": sel_std,
                "selected_min": float(sel.min().item()),
                "selected_max": float(sel.max().item()),
                "fallback": False,
            })
            return mask, threshold, info

        threshold -= threshold_step

    # Fallback: top 1% (at least min_points)
    k = max(min_points, int(0.01 * sim.numel()))
    top_vals, _ = sim.topk(k)
    fallback_threshold = float(top_vals[-1].item())
    mask = sim >= top_vals[-1]
    info.update({
        "final_threshold": fallback_threshold,
        "n_selected": int(mask.sum().item()),
        "selected_std": float(sim[mask].std().item()),
        "selected_min": float(top_vals[-1].item()),
        "selected_max": float(top_vals[0].item()),
        "fallback": True,
        "fallback_k": k,
    })
    return mask, fallback_threshold, info


def cluster_object_points(
    points: np.ndarray,
    eps: float = 0.3,
    min_samples: int = 10,
    min_cluster_size: int = 60,
) -> list[np.ndarray]:
    """Spatially cluster grounded points with DBSCAN.

    Returns one point array per cluster, sorted by size (largest first).
    Noise points (DBSCAN label -1) and clusters smaller than
    ``min_cluster_size`` are dropped.
    """
    if len(points) == 0:
        return []
    if len(points) < min_samples:
        return [points] if len(points) >= min_cluster_size else []

    labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(points)

    clusters = []
    for cid in set(labels):
        if cid == -1:  # noise
            continue
        cluster_pts = points[labels == cid]
        if len(cluster_pts) < min_cluster_size:
            continue
        clusters.append(cluster_pts)

    # Largest instance first → instance #0 is the most prominent in the viewer.
    clusters.sort(key=len, reverse=True)
    return clusters


# ---------------------------------------------------------------------------
# Main visualization
# ---------------------------------------------------------------------------

def visualize_slam_map(
    map_path: Path,
    device: str = "cpu",
    pca_basis_path: Path | None = None,
    ground_prompts: list[str] | None = None,
    similarity_threshold: float = 0.25,
    bbox_expansion: float = 1.05,
    # RADSeg parameters
    model_version: str = "c-radio_v3-b",
    lang_model: str = "siglip2",
    scra_scaling: float = 10.0,
    scga_scaling: float = 10.0,
    window_size: int = 336,
    window_stride: int = 224,
    gt_poses_path: Path | None = None,
    # Visualization / clustering flags
    show_object_points: bool = False,
    no_bbox: bool = False,
    cluster_eps: float | None = None,
    cluster_min_samples: int | None = None,
) -> None:
    """Load a SLAM map, render it in Rerun, and optionally highlight a
    grounded text prompt as colored points and per-instance bounding boxes."""

    print(f"Loading SLAM map from: {map_path}")
    data = load_slam_map(map_path, device=device)

    # --- Extract data ---
    xyz = data["dense_disp_xyz"]
    rgb = data["dense_disp_rgb"]
    embeddings_raw = data.get("dense_disp_embeddings")
    embeddings_full = data.get("dense_disp_embeddings_full")
    embedding_valid = data.get("dense_disp_embedding_valid")

    print(f"Point cloud size: {xyz.shape[0]} points")

    # --- Resolve embeddings (decompress if necessary) ---
    if embeddings_full is not None:
        print("Using pre-decompressed embeddings from 'dense_disp_embeddings_full'.")
        embeddings_for_vis = embeddings_full
    elif embeddings_raw is not None and pca_basis_path is not None:
        print("Decompressing embeddings using PcaCompressor...")
        embeddings_for_vis = decompress_embeddings(embeddings_raw, pca_basis_path, device)
        print(f"Decompressed shape: {embeddings_for_vis.shape}")
    else:
        embeddings_for_vis = embeddings_raw
        if embeddings_raw is None:
            print("No embeddings found in map.")

    print(f"Has embeddings: {embeddings_for_vis is not None}")

    # --- Convert to numpy and align ---
    xyz_np = xyz.cpu().numpy()
    rgb_np = (rgb.cpu().numpy() * 255).astype(np.uint8)  # original RGB is in [0, 1]
    if gt_poses_path is not None:
        first_pose = load_first_gt_pose(gt_poses_path)
        xyz_np = transform_pointcloud(first_pose, xyz_np)
        # Aligned to GT frame → use identity; bounding boxes will be axis-aligned
        # to the GT world axes rather than to a PCA-derived scene frame.
        scene_rotation = np.eye(3)
    else:
        scene_rotation = get_scene_orientation(xyz_np)

    # --- Initialize Rerun and log the RGB cloud ---
    rr.init("SLAM Map Grounding", spawn=True)
    rr.log(
        "world/points/rgb",
        rr.Points3D(positions=xyz_np, colors=rgb_np, radii=0.01),
    )

    # --- Grounding ---
    if ground_prompts and embeddings_for_vis is not None:
        print("\n--- Grounding ---")
        main_prompt = ground_prompts[0]
        print(f"Highlighting: '{main_prompt}' (threshold={similarity_threshold})")

        # Filter to valid embeddings if a mask is provided
        if embedding_valid is not None:
            valid_mask = embedding_valid.cpu().numpy()
            grounding_embeddings = embeddings_for_vis[embedding_valid]
            grounding_xyz = xyz_np[valid_mask]
            print(
                f"Valid embeddings: {grounding_embeddings.shape[0]} / "
                f"{embeddings_for_vis.shape[0]}"
            )
        else:
            grounding_embeddings = embeddings_for_vis
            grounding_xyz = xyz_np

        # Initialize RADSegEncoder
        torch_device = _resolve_device(device)
        try:
            encoder = RADSegEncoder(
                model_version=model_version,
                lang_model=lang_model,
                scra_scaling=scra_scaling,
                scga_scaling=scga_scaling,
                slide_crop=window_size,
                slide_stride=window_stride,
                sam_refinement=False,
                predict=False,
                device=torch_device,
            )
        except Exception as e:
            print(f"Failed to initialize RADSegEncoder: {e}")
            encoder = None

        if encoder is not None:
            # Encode text prompt
            text_embeds = encoder.encode_labels([main_prompt]).to(torch_device)
            print(f"Text embedding shape: {text_embeds.shape}")
            print(f"Point embedding shape: {grounding_embeddings.shape}")

            grounding_embeddings = grounding_embeddings.to(
                device=torch_device, dtype=torch.float32
            )
            text_embeds = text_embeds.to(device=torch_device, dtype=torch.float32)

            # Cosine similarity
            point_norm = grounding_embeddings / (
                grounding_embeddings.norm(dim=-1, keepdim=True) + 1e-8
            )
            text_norm = text_embeds / (text_embeds.norm(dim=-1, keepdim=True) + 1e-8)
            sim = (text_norm @ point_norm.T).squeeze(0)  # (N,)

            sim_np = sim.cpu().detach().numpy()
            full_std = float(sim.std().item())
            print(
                f"Similarity stats: min={sim_np.min():.4f}, max={sim_np.max():.4f}, "
                f"mean={sim_np.mean():.4f}, std={full_std:.4f}"
            )

            # Adaptive threshold
            mask_t, used_threshold, info = find_grounding_threshold(
                sim,
                initial_threshold=similarity_threshold,
                target_std=0.015,
                min_threshold=0.05,
                threshold_step=0.01,
                min_points=10,
            )
            mask = mask_t.cpu().numpy()
            num_found = int(mask.sum())

            if abs(used_threshold - similarity_threshold) > 1e-6:
                print(
                    f"Adaptive threshold: {similarity_threshold:.4f} → "
                    f"{used_threshold:.4f} ({info['steps']} step(s))"
                )
            if info.get("fallback"):
                print(
                    f"⚠️  No threshold met target std=0.015; "
                    f"falling back to top-{info['fallback_k']} points."
                )

            print(
                f"Found {num_found} points above {used_threshold:.4f} "
                f"(selected std={info['selected_std']:.4f}, "
                f"range=[{info['selected_min']:.4f}, {info['selected_max']:.4f}])"
            )

            if num_found > 0:
                object_points = grounding_xyz[mask]

                # Highlight matching points in red
                if show_object_points:
                    rr.log(
                        f"world/objects/{main_prompt}_points",
                        rr.Points3D(
                            positions=object_points,
                            colors=[255, 0, 0],
                            radii=0.015,
                        ),
                    )

                # Per-instance bounding boxes via DBSCAN
                if not no_bbox:
                    clusters = cluster_object_points(
                        object_points,
                        eps=cluster_eps,
                        min_samples=cluster_min_samples,
                    )
                    print(f"Found {len(clusters)} cluster(s) for '{main_prompt}'.")

                    for i, cluster_pts in enumerate(clusters):
                        obj_center, obj_size, obj_quat = compute_aligned_bbox(
                            cluster_pts, scene_rotation, expansion=bbox_expansion
                        )
                        if obj_center is None:
                            continue
                        rr.log(
                            f"world/objects/{main_prompt}/instance_{i}",
                            rr.Boxes3D(
                                centers=[obj_center],
                                sizes=[obj_size],
                                quaternions=[obj_quat],
                                colors=[0, 255, 0, 180],
                                labels=[f"{main_prompt} #{i}"],
                            ),
                        )
            else:
                print(f"No points found for '{main_prompt}'.")

    # --- Coordinate frame at origin ---
    rr.log(
        "world/axes",
        rr.LineStrips3D([
            [[0, 0, 0], [1, 0, 0]],
            [[0, 0, 0], [0, 1, 0]],
            [[0, 0, 0], [0, 0, 1]],
        ]),
    )

    print("\nVisualization running in Rerun viewer. Close the viewer to exit.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Visualize a SLAM map with Rerun: RGB point cloud + grounded "
            "object highlighting (points and/or per-instance bounding boxes)."
        )
    )

    # I/O
    parser.add_argument(
        "map_path",
        type=Path,
        help="Path to the saved SLAM map (.pt file).",
    )
    parser.add_argument(
        "--pca-basis",
        type=Path,
        default=None,
        help="Path to a PCA basis file (.pt) for decompression of compressed embeddings.",
    )
    parser.add_argument(
        "--gt-poses-path",
        type=Path,
        default=None,
        help="Path to a GT-poses file. The first 4x4 pose is used to align the cloud.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device to load tensors on (default: cpu).",
    )

    # Grounding
    parser.add_argument(
        "--ground",
        type=str,
        default=None,
        help="Text prompt for grounding (e.g. 'car'). Only the first prompt is used.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.25,
        help=(
            "Initial cosine-similarity threshold for grounding (default: 0.25). "
            "Adaptive logic may lower this automatically; this is the upper bound."
        ),
    )

    # Bounding box geometry
    parser.add_argument(
        "--bbox-expansion",
        type=float,
        default=1.0,
        help="Multiplicative size factor applied to each bounding box (default: 1.0).",
    )

    # RADSeg model parameters
    parser.add_argument(
        "--model-version",
        type=str,
        default="c-radio_v3-b",
        help="RADSeg model version (default: c-radio_v3-b).",
    )
    parser.add_argument(
        "--lang-model",
        type=str,
        default="siglip2",
        help="Language model name (default: siglip2).",
    )
    parser.add_argument(
        "--scra-scaling",
        type=float,
        default=10.0,
        help="SCRA scaling factor (default: 10.0).",
    )
    parser.add_argument(
        "--scga-scaling",
        type=float,
        default=10.0,
        help="SCGA scaling factor (default: 10.0).",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=336,
        help="Sliding window size (default: 336).",
    )
    parser.add_argument(
        "--window-stride",
        type=int,
        default=224,
        help="Sliding window stride (default: 224).",
    )

    # Visualization flags
    parser.add_argument(
        "--show-object-points",
        action="store_true",
        help="Render the grounded points (red) on top of the RGB cloud.",
    )
    parser.add_argument(
        "--no-bbox",
        action="store_true",
        help="Skip bounding boxes entirely (point-cloud / highlight only).",
    )

    # DBSCAN clustering
    parser.add_argument(
        "--cluster-eps",
        type=float,
        default=0.1,
        help="DBSCAN eps in meters (default: 0.1). Larger → fewer, looser clusters.",
    )
    parser.add_argument(
        "--cluster-min-samples",
        type=int,
        default=10,
        help="DBSCAN min_samples for core points (default: 10).",
    )

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if not args.map_path.exists():
        print(f"Error: Map file not found: {args.map_path}")
        return
    if args.pca_basis is not None and not args.pca_basis.exists():
        print(f"Error: PCA basis file not found: {args.pca_basis}")
        return

    # Parse grounding prompts (only first is used)
    ground_prompts: list[str] | None = None
    if args.ground:
        prompts = [p.strip() for p in args.ground.split(";") if p.strip()]
        if prompts:
            ground_prompts = prompts
        else:
            print("Warning: --ground provided but no valid prompts found.")

    visualize_slam_map(
        map_path=args.map_path,
        device=args.device,
        pca_basis_path=args.pca_basis,
        ground_prompts=ground_prompts,
        similarity_threshold=args.threshold,
        bbox_expansion=args.bbox_expansion,
        model_version=args.model_version,
        lang_model=args.lang_model,
        scra_scaling=args.scra_scaling,
        scga_scaling=args.scga_scaling,
        window_size=args.window_size,
        window_stride=args.window_stride,
        gt_poses_path=args.gt_poses_path,
        show_object_points=args.show_object_points,
        no_bbox=args.no_bbox,
        cluster_eps=args.cluster_eps,
        cluster_min_samples=args.cluster_min_samples,
    )


if __name__ == "__main__":
    main()