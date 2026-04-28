import torch
import torch.nn.functional as F


def _semantic_flow_init_unfold(
    Z_i: torch.Tensor,
    Z_j: torch.Tensor,
    mu_ij: torch.Tensor | None = None,
    search_radius: int = 8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Dense semantic correspondence via local cosine-similarity search.
    Original unfold-based implementation (high peak memory).

    Args:
        Z_i: (K, H, W) PCA-compressed Radio embedding for the source frame.
        Z_j: (K, H, W) PCA-compressed Radio embedding for the target frame.
        mu_ij: (H, W, 2) geometric projection from frame i→j in pixel
               coordinates (x, y).  Defaults to the identity grid.
        search_radius: half-size of the square search window.

    Returns:
        omega_semantic: (H, W, 2) absolute pixel coordinates of the best
                        semantic match in frame j.
        max_sim:        (H, W) maximum cosine similarity found per pixel,
                        clamped to [0, 1] (used as semantic confidence).
    """
    K, H, W = Z_i.shape
    r = search_radius
    win = 2 * r + 1
    device = Z_i.device

    if mu_ij is None:
        gy, gx = torch.meshgrid(
            torch.arange(H, device=device, dtype=torch.float32),
            torch.arange(W, device=device, dtype=torch.float32),
            indexing="ij",
        )
        mu_ij = torch.stack([gx, gy], dim=-1)

    Z_i_norm = F.normalize(Z_i.half(), dim=0, eps=1e-8)
    Z_j_norm = F.normalize(Z_j.half(), dim=0, eps=1e-8)

    # Extract (2r+1)x(2r+1) patches centred at every pixel of Z_j.
    patches = F.unfold(
        Z_j_norm.unsqueeze(0), kernel_size=win, padding=r
    ).squeeze(0)  # (K*win^2, H*W)
    patches = patches.view(K, win * win, H * W)

    # For each source pixel u, select the patch centred at round(mu_ij(u)).
    center_x = mu_ij[..., 0].round().long().clamp(0, W - 1)
    center_y = mu_ij[..., 1].round().long().clamp(0, H - 1)
    center_idx = (center_y * W + center_x).reshape(-1)  # (H*W,)

    center_patches = patches[:, :, center_idx]  # (K, win^2, H*W)
    center_patches = F.normalize(center_patches, dim=0, eps=1e-8)

    Z_i_flat = Z_i_norm.reshape(K, H * W)
    sim = torch.einsum("kn,kpn->pn", Z_i_flat, center_patches)  # (win^2, H*W)

    max_sim, max_idx = sim.max(dim=0)  # (H*W,)

    dy = (max_idx // win) - r
    dx = (max_idx % win) - r

    offset = torch.stack([dx.float(), dy.float()], dim=-1).view(H, W, 2)
    omega_semantic = mu_ij + offset
    max_sim = max_sim.view(H, W).clamp(0.0, 1.0)

    return omega_semantic, max_sim


def _semantic_flow_init_gridsample(
    Z_i: torch.Tensor,
    Z_j: torch.Tensor,
    mu_ij: torch.Tensor | None = None,
    search_radius: int = 8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Dense semantic correspondence via local cosine-similarity search.
    Memory-efficient grid_sample implementation that avoids the full unfold.

    Instead of materialising a (K, win², H*W) tensor via F.unfold, this
    iterates over offsets in the search window and uses F.grid_sample to
    gather one slice at a time, accumulating only the running argmax.

    Peak GPU memory: O(K * H * W) instead of O(K * win² * H * W).

    Args:
        Z_i: (K, H, W) PCA-compressed Radio embedding for the source frame.
        Z_j: (K, H, W) PCA-compressed Radio embedding for the target frame.
        mu_ij: (H, W, 2) geometric projection from frame i→j in pixel
               coordinates (x, y).  Defaults to the identity grid.
        search_radius: half-size of the square search window.

    Returns:
        omega_semantic: (H, W, 2) absolute pixel coordinates of the best
                        semantic match in frame j.
        max_sim:        (H, W) maximum cosine similarity found per pixel,
                        clamped to [0, 1] (used as semantic confidence).
    """
    K, H, W = Z_i.shape
    r = search_radius
    win = 2 * r + 1
    device = Z_i.device

    if mu_ij is None:
        gy, gx = torch.meshgrid(
            torch.arange(H, device=device, dtype=torch.float32),
            torch.arange(W, device=device, dtype=torch.float32),
            indexing="ij",
        )
        mu_ij = torch.stack([gx, gy], dim=-1)

    Z_i_norm = F.normalize(Z_i.half(), dim=0, eps=1e-8)  # (K, H, W)
    Z_j_norm = F.normalize(Z_j.half(), dim=0, eps=1e-8)  # (K, H, W)

    # Base grid in pixel coords: the rounded projection centres.
    # mu_ij is (H, W, 2) with [..., 0]=x, [..., 1]=y.
    cx = mu_ij[..., 0].round().clamp(0, W - 1)  # (H, W)
    cy = mu_ij[..., 1].round().clamp(0, H - 1)  # (H, W)

    # Precompute normalisation scale for grid_sample's [-1, 1] convention.
    # align_corners=True  =>  pixel 0 -> -1,  pixel (N-1) -> +1
    scale_x = 2.0 / (W - 1) if W > 1 else 0.0
    scale_y = 2.0 / (H - 1) if H > 1 else 0.0

    # Z_i flattened for dot products: (K, H*W)
    Z_i_flat = Z_i_norm.reshape(K, H * W)

    # Prepare Z_j for grid_sample: needs (1, K, H, W)
    Z_j_4d = Z_j_norm.unsqueeze(0)  # (1, K, H, W)

    # Running best similarity and best offset index.
    best_sim = torch.full((H * W,), -float("inf"), device=device, dtype=torch.float32)
    best_idx = torch.zeros(H * W, device=device, dtype=torch.long)

    # Iterate over each offset in the search window.
    # To keep launch overhead manageable, process a full row of offsets at once
    # (win offsets along x for a single dy), doing win grid_samples per row.
    for p, dy in enumerate(range(-r, r + 1)):
        # Sample coords for this row: all dx offsets at once.
        # Build a grid of shape (1, win, H*W, 2) — one "row" per dx offset.
        sample_y = (cy + dy).reshape(1, 1, H * W).expand(1, win, H * W)
        # dx offsets: -r, -r+1, ..., r
        dx_offsets = torch.arange(-r, r + 1, device=device, dtype=torch.float32)
        sample_x = cx.reshape(1, 1, H * W) + dx_offsets.reshape(1, win, 1)

        # Normalise to [-1, 1]
        grid_x = sample_x * scale_x - 1.0
        grid_y = sample_y * scale_y - 1.0
        grid = torch.stack([grid_x, grid_y], dim=-1).half()  # (1, win, H*W, 2)

        # grid_sample: (1, K, H, W) sampled at (1, win, H*W, 2) -> (1, K, win, H*W)
        sampled = F.grid_sample(
            Z_j_4d, grid, mode="nearest", padding_mode="zeros", align_corners=True,
        )  # (1, K, win, H*W)
        sampled = sampled.squeeze(0)  # (K, win, H*W)

        # Re-normalise after grid_sample (zero-padded border pixels break unit norm).
        sampled = F.normalize(sampled, dim=0, eps=1e-8)

        # Cosine similarity: dot product along K for each (dx_idx, pixel).
        # Z_i_flat: (K, H*W), sampled: (K, win, H*W) -> sim_row: (win, H*W)
        sim_row = torch.einsum("kn,kpn->pn", Z_i_flat, sampled)

        # Update running argmax.
        row_max, row_argmax = sim_row.max(dim=0)  # (H*W,)
        # Global patch index: p * win + dx_index
        global_idx = p * win + row_argmax

        improved = row_max > best_sim
        best_sim = torch.where(improved, row_max, best_sim)
        best_idx = torch.where(improved, global_idx, best_idx)

    # Decode best_idx back to (dy, dx) offsets.
    dy_best = (best_idx // win) - r
    dx_best = (best_idx % win) - r

    offset = torch.stack([dx_best.float(), dy_best.float()], dim=-1).view(H, W, 2)
    omega_semantic = mu_ij + offset
    max_sim = best_sim.view(H, W).clamp(0.0, 1.0)

    return omega_semantic, max_sim


def _semantic_flow_init_batched(
    Z_i: torch.Tensor,       # (B, K, H, W)
    Z_j: torch.Tensor,       # (B, K, H, W)
    mu_ij: torch.Tensor | None = None,  # (B, H, W, 2)
    search_radius: int = 8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Batched dense semantic correspondence — all B edges in one pass."""
    B, K, H, W = Z_i.shape
    r = search_radius
    win = 2 * r + 1
    device = Z_i.device

    if mu_ij is None:
        gy, gx = torch.meshgrid(
            torch.arange(H, device=device, dtype=torch.float32),
            torch.arange(W, device=device, dtype=torch.float32),
            indexing="ij",
        )
        mu_ij = torch.stack([gx, gy], dim=-1).unsqueeze(0).expand(B, -1, -1, -1)

    Z_i_norm = F.normalize(Z_i.half(), dim=1, eps=1e-8)
    Z_j_norm = F.normalize(Z_j.half(), dim=1, eps=1e-8)

    cx = mu_ij[..., 0].round().clamp(0, W - 1)
    cy = mu_ij[..., 1].round().clamp(0, H - 1)

    scale_x = 2.0 / (W - 1) if W > 1 else 0.0
    scale_y = 2.0 / (H - 1) if H > 1 else 0.0

    Z_i_flat = Z_i_norm.reshape(B, K, H * W)

    best_sim = torch.full((B, H * W), -float("inf"), device=device, dtype=torch.float32)
    best_idx = torch.zeros(B, H * W, device=device, dtype=torch.long)

    dx_offsets = torch.arange(-r, r + 1, device=device, dtype=torch.float32)

    for p, dy in enumerate(range(-r, r + 1)):
        sample_y = (cy + dy).reshape(B, 1, H * W).expand(B, win, H * W)
        sample_x = cx.reshape(B, 1, H * W) + dx_offsets.reshape(1, win, 1)

        grid_x = sample_x * scale_x - 1.0
        grid_y = sample_y * scale_y - 1.0
        grid = torch.stack([grid_x, grid_y], dim=-1).half()

        sampled = F.grid_sample(
            Z_j_norm, grid, mode="nearest", padding_mode="zeros", align_corners=True,
        )
        sampled = F.normalize(sampled, dim=1, eps=1e-8)

        sim_row = torch.einsum("bkn,bkpn->bpn", Z_i_flat, sampled)

        row_max, row_argmax = sim_row.max(dim=1)
        global_idx = p * win + row_argmax

        improved = row_max > best_sim
        best_sim = torch.where(improved, row_max, best_sim)
        best_idx = torch.where(improved, global_idx, best_idx)

    dy_best = (best_idx // win) - r
    dx_best = (best_idx % win) - r

    offset = torch.stack([dx_best.float(), dy_best.float()], dim=-1).view(B, H, W, 2)
    omega_semantic = mu_ij + offset
    max_sim = best_sim.view(B, H, W).clamp(0.0, 1.0)

    return omega_semantic, max_sim


def semantic_flow_init(
    Z_i: torch.Tensor,
    Z_j: torch.Tensor,
    mu_ij: torch.Tensor | None = None,
    search_radius: int = 8,
):
    # Support both unbatched (K,H,W) and batched (B,K,H,W)
    if Z_i.ndim == 3:
        return _semantic_flow_init_gridsample(Z_i=Z_i, Z_j=Z_j, mu_ij=mu_ij, search_radius=search_radius)
    return _semantic_flow_init_batched(Z_i=Z_i, Z_j=Z_j, mu_ij=mu_ij, search_radius=search_radius)


def blend_flow_prior(
    omega_prior: torch.Tensor,
    omega_semantic: torch.Tensor,
    photo_conf: torch.Tensor,
    semantic_conf: torch.Tensor,
) -> torch.Tensor:
    """
    Per-pixel confidence-weighted blend of geometric and semantic flow.

    beta(u) = photo_conf(u) / (photo_conf(u) + semantic_conf(u) + eps)
    omega_init(u) = beta(u) * omega_prior(u) + (1 - beta(u)) * omega_semantic(u)

    Args:
        omega_prior:    (H, W, 2) geometric flow field.
        omega_semantic: (H, W, 2) semantic flow field.
        photo_conf:     (H, W) photometric / flow-network confidence.
        semantic_conf:  (H, W) max cosine similarity from semantic search.

    Returns:
        omega_init: (H, W, 2) blended flow field.
    """
    beta = photo_conf / (photo_conf + semantic_conf + 1e-6)
    beta = beta.unsqueeze(-1)  # (H, W, 1) for broadcast over the 2-channel flow
    return beta * omega_prior + (1.0 - beta) * omega_semantic