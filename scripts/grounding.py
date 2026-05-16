#!/usr/bin/env python3
"""
Visualize SLAM map with Rerun.
Shows original RGB point cloud and highlights grounded object with a bounding box.
Uses PcaCompressor for decompression and RADSegEncoder as per documentation.
"""

import argparse
from pathlib import Path
from typing import Any, Optional, List

import numpy as np
import rerun as rr
import torch

try:
    from typing_extensions import override
except ImportError:
    def override(func):
        return func

from vipe.priors.embedding.radseg_encoder import RADSegEncoder


class PcaCompressor:
    """Compress features using Principal Component Analysis"""

    def __init__(self, out_dim: int = None, in_dim: int = None, path: str = None):
        """
        Args:
          out_dim: Output dimension
          in_dim: Input dimension
          path: Path to load pre-fitted compressor
        """
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
            print(f"⚠️  WARNING: Requested out_dim ({self.out_dim}) is larger than data rank ({min(X_flatten.shape)}). "
                  f"Capping q to {self.q_val} to avoid crash.")
            self.out_dim = self.q_val
        _, _, V = torch.pca_lowrank(X_centered, q=self.out_dim)
        self.basis = V

    @override
    def save(self, fp: str) -> None:
        torch.save(dict(
            metadata=dict(in_dim=self.in_dim, out_dim=self.out_dim),
            mean=self.mean, basis=self.basis), fp)

    @override
    def load(self, fp: str) -> None:
        d = torch.load(fp)
        self.in_dim = d["metadata"]["in_dim"]
        self.out_dim = d["metadata"]["out_dim"]
        self.mean = d["mean"]
        self.basis = d["basis"]

    @override
    def compress(self, X):
        s = list(X.shape)
        s[-1] = self.out_dim
        return (X.flatten(0, -2) @ self.basis).reshape(*s)

    @override
    def decompress(self, Y):
        s = list(Y.shape)
        s[-1] = self.in_dim
        return (Y.flatten(0, -2) @ self.basis.T).reshape(*s)

    @override
    def is_fitted(self):
        return self.mean is not None and self.basis is not None


def _resolve_device(device: str | torch.device) -> torch.device:
    """Normalize device inputs to torch.device."""
    return device if isinstance(device, torch.device) else torch.device(device)


def load_slam_map(path: Path, device: str = "cpu"):
    """Load the SLAM map from disk."""
    torch_device = _resolve_device(device)
    data = torch.load(path, map_location=torch_device)
    return data


def _extract_pca_basis(state: Any) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Attempt to locate (mean, components) tensors inside an arbitrary container saved on disk.
    """
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
        return getattr(state, "mean"), getattr(state, "components")
    raise KeyError("Unable to locate 'mean' and 'components' tensors in the provided PCA state.")

def decompress_embeddings(embeddings: torch.Tensor, pca_path: Path, device: str) -> torch.Tensor:
    """
    Decompress PCA-compressed embeddings using PcaCompressor.
    Supports various PCA file formats by extracting mean and components.
    """
    data = torch.load(pca_path, map_location='cpu')
    mean, components = _extract_pca_basis(data)

    # Create compressor and manually set attributes
    compressor = PcaCompressor()
    compressor.mean = mean.to(device)
    compressor.basis = components.to(device)
    compressor.in_dim = components.shape[0]  # original dimension
    compressor.out_dim = components.shape[1] # number of components
    embeddings = embeddings.to(compressor.basis.dtype) 
    return compressor.decompress(embeddings)


from scipy.spatial.transform import Rotation as R

def get_scene_orientation(all_points: np.ndarray) -> np.ndarray:
    centroid = np.mean(all_points, axis=0)
    _, _, Vh = np.linalg.svd(all_points - centroid, full_matrices=False)
    
    rotation_matrix = Vh.T
    if np.linalg.det(rotation_matrix) < 0:
        rotation_matrix[:, 2] *= -1
    return rotation_matrix

def compute_aligned_bbox(points: np.ndarray, orientation_matrix: np.ndarray, expansion: float = 1.0):
    if len(points) == 0: return None, None, None
    
    points_local = points @ orientation_matrix
    
    bbox_min = np.percentile(points_local, 7.0, axis=0)
    bbox_max = np.percentile(points_local, 93.0, axis=0)
    
    local_size = (bbox_max - bbox_min) * expansion
    local_center = (bbox_min + bbox_max) / 2
    
    world_center = orientation_matrix @ local_center
    quat = R.from_matrix(orientation_matrix).as_quat()
    
    return world_center, local_size, quat


def find_grounding_threshold(
    sim: torch.Tensor,
    initial_threshold: float = 0.25,
    target_std: float = 0.015,      # SigLIP2: real clusters spread ≥ ~full-dist std
    min_threshold: float = 0.05,    # noise floor — don't go below this
    threshold_step: float = 0.01,
    min_points: int = 10,           # need enough points for std to be meaningful
) -> tuple[torch.Tensor, float, dict]:
    """
    Adaptively pick a similarity threshold.

    Starts at `initial_threshold` and decreases by `threshold_step` until either:
      - enough points are selected AND their std >= target_std (real cluster found), or
      - threshold reaches `min_threshold` (give up, fall back to top-k).

    Returns (bool_mask, threshold_used, info_dict).
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

def visualize_slam_map(
    map_path: Path,
    device: str = "cpu",
    pca_basis_path: Optional[Path] = None,
    ground_prompts: Optional[List[str]] = None,
    similarity_threshold: float = 0.25,
    bbox_expansion: float = 1.05,
    # RADSeg parameters
    model_version: str = "c-radio_v3-b",
    lang_model: str = "siglip2",
    scra_scaling: float = 10.0,
    scga_scaling: float = 10.0,
    window_size: int = 336,
    window_stride: int = 224,
):
    """
    Main visualization function using Rerun.
    """
    print(f"Loading SLAM map from: {map_path}")
    data = load_slam_map(map_path, device=device)

    # Extract data
    xyz = data["dense_disp_xyz"]
    rgb = data["dense_disp_rgb"]
    embeddings_raw = data.get("dense_disp_embeddings")         
    embeddings_full = data.get("dense_disp_embeddings_full")   
    embedding_valid = data.get("dense_disp_embedding_valid")

    print(f"Point cloud size: {xyz.shape[0]} points")

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

    # Convert to numpy for logging
    xyz_np = xyz.cpu().numpy()
    rgb_np = (rgb.cpu().numpy() * 255).astype(np.uint8)   # original RGB is in [0,1]

    scene_rotation = get_scene_orientation(xyz_np)
    scene_quat = R.from_matrix(scene_rotation).as_quat()

    # Initialize Rerun
    rr.init("SLAM Map Grounding", spawn=True)

    # Log RGB point cloud
    rr.log(
        "world/points/rgb",
        rr.Points3D(
            positions=xyz_np,
            colors=rgb_np,
            radii=0.01,
        ),
    )

    # --- Grounding ---
    ground_bbox_center = None
    ground_bbox_size = None
    if ground_prompts and embeddings_for_vis is not None:

        print("\n--- Grounding ---")
        main_prompt = ground_prompts[0]
        print(f"Highlighting: '{main_prompt}' (threshold={similarity_threshold})")

        # Filter valid embeddings if mask exists
        if embedding_valid is not None:
            valid_mask = embedding_valid.cpu().numpy()
            grounding_embeddings = embeddings_for_vis[embedding_valid]
            grounding_xyz = xyz_np[valid_mask]
            print(f"Valid embeddings: {grounding_embeddings.shape[0]} / {embeddings_for_vis.shape[0]}")
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
            # Get text embeddings using encode_labels (as per documentation)
            text_embeds = encoder.encode_labels([main_prompt])  # (1, D)
            text_embeds = text_embeds.to(torch_device)

            print(f"Text embedding shape: {text_embeds.shape}")
            print(f"Point embedding shape: {grounding_embeddings.shape}")

            grounding_embeddings = grounding_embeddings.to(device=torch_device, dtype=torch.float32)
            text_embeds = text_embeds.to(device=torch_device, dtype=torch.float32)

            # Normalize both
            point_norm = grounding_embeddings / (grounding_embeddings.norm(dim=-1, keepdim=True) + 1e-8)
            text_norm = text_embeds / (text_embeds.norm(dim=-1, keepdim=True) + 1e-8)

            # Cosine similarity: (1, N)
            sim = (text_norm @ point_norm.T).squeeze(0)  # (N,)

            sim_np = sim.cpu().detach().numpy()
            full_std = float(sim.std().item())
            print(
                f"Similarity stats: min={sim_np.min():.4f}, max={sim_np.max():.4f}, "
                f"mean={sim_np.mean():.4f}, std={full_std:.4f}"
            )

            mask_t, used_threshold, info = find_grounding_threshold(
                sim,
                initial_threshold=similarity_threshold,
                target_std=0.015,      # SigLIP2-calibrated
                min_threshold=0.05,
                threshold_step=0.01,
                min_points=10,
            )
            mask = mask_t.cpu().numpy()
            num_found = int(mask.sum())

            if abs(used_threshold - similarity_threshold) > 1e-6:
                print(
                    f"Adaptive threshold: {similarity_threshold:.4f} → {used_threshold:.4f} "
                    f"({info['steps']} step(s))"
                )
            if info.get("fallback"):
                print(
                    f"⚠️  No threshold met target std={0.015}; "
                    f"falling back to top-{info['fallback_k']} points."
                )

            print(
                f"Found {num_found} points above {used_threshold:.4f} "
                f"(selected std={info['selected_std']:.4f}, "
                f"range=[{info['selected_min']:.4f}, {info['selected_max']:.4f}])"
            )

            # Optional: useful for debugging adaptation
            # for t, n, s in info["trace"]:
            #     print(f"  threshold={t}  n={n}  std={s}")

            if num_found > 0:
                object_points = grounding_xyz[mask]
                
                obj_center, obj_size, obj_quat = compute_aligned_bbox(
                    object_points, scene_rotation, expansion=bbox_expansion
                )
                
                if obj_center is not None:
                    rr.log(
                        f"world/objects/{main_prompt}",
                        rr.Boxes3D(
                            centers=[obj_center],
                            sizes=[obj_size],
                            quaternions=[obj_quat], 
                            colors=[0, 255, 0, 180],
                            labels=[main_prompt]
                        ),
                    )
                    
                    # Also log the points as a separate entity for debugging (optional)
                    # Uncomment if you want to see the points as well
                    # rr.log(
                    #     f"world/objects/{main_prompt}_points",
                    #     rr.Points3D(
                    #         positions=object_points,
                    #         colors=[255, 100, 100],
                    #         radii=0.015,
                    #     ),
                    # )
                else:
                    print("Failed to compute bounding box.")
            else:
                print(f"No points found for '{main_prompt}'.")


    # Add coordinate frame at origin
    rr.log("world/axes", rr.LineStrips3D([
        [[0, 0, 0], [1, 0, 0]],
        [[0, 0, 0], [0, 1, 0]],
        [[0, 0, 0], [0, 0, 1]],
    ]))

    print("\nVisualization running in Rerun viewer. Close the viewer to exit.")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize SLAM map with Rerun: RGB + grounded object bounding box."
    )
    parser.add_argument(
        "map_path",
        type=Path,
        help="Path to the saved SLAM map (.pt file)",
    )
    parser.add_argument(
        "--pca-basis",
        type=Path,
        default=None,
        help="Path to PCA basis file (.pt) for decompression (if embeddings are compressed).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device to load tensors on (default: cpu)",
    )
    # Grounding arguments
    parser.add_argument(
        "--ground",
        type=str,
        default=None,
        help="Text prompt for grounding (e.g. 'car'). Only first prompt is used.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.25,
        help="Cosine similarity threshold for grounding (default: 0.25)",
    )
    parser.add_argument(
        "--bbox-expansion",
        type=float,
        default=1.0,
        help="Expand bounding box by this factor (default: 1.05 = 5% larger)",
    )
    # RADSeg model parameters
    parser.add_argument(
        "--model-version",
        type=str,
        default="c-radio_v3-b",
        help="RADSeg model version (default: c-radio_v3-b)",
    )
    parser.add_argument(
        "--lang-model",
        type=str,
        default="siglip2",
        help="Language model name (default: siglip2)",
    )
    parser.add_argument(
        "--scra-scaling",
        type=float,
        default=10.0,
        help="SCRA scaling factor (default: 10.0)",
    )
    parser.add_argument(
        "--scga-scaling",
        type=float,
        default=10.0,
        help="SCGA scaling factor (default: 10.0)",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=336,
        help="Sliding window size (default: 336)",
    )
    parser.add_argument(
        "--window-stride",
        type=int,
        default=224,
        help="Sliding window stride (default: 224)",
    )

    args = parser.parse_args()

    if not args.map_path.exists():
        print(f"Error: Map file not found: {args.map_path}")
        return

    if args.pca_basis is not None and not args.pca_basis.exists():
        print(f"Error: PCA basis file not found: {args.pca_basis}")
        return

    # Parse grounding prompts (only first used)
    ground_prompts = None
    if args.ground:
        prompts = [p.strip() for p in args.ground.split(';') if p.strip()]
        if prompts:
            ground_prompts = prompts
        else:
            print("Warning: --ground provided but no valid prompts found.")

    visualize_slam_map(
        args.map_path,
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
    )


if __name__ == "__main__":
    main()