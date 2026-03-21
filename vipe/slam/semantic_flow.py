import torch
import torch.nn.functional as F


def semantic_flow_init(
    Z_i: torch.Tensor,
    Z_j: torch.Tensor,
    mu_ij: torch.Tensor | None = None,
    search_radius: int = 8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Dense semantic correspondence via local cosine-similarity search.

    For every pixel u in frame i, finds the best matching pixel in frame j
    within a (2*search_radius+1)^2 window centred on the geometric projection
    mu_ij(u).  Uses F.unfold on Z_j for efficient patch extraction and batch
    matmul for the similarity computation.

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

    Z_i_norm = F.normalize(Z_i.float(), dim=0, eps=1e-8)
    Z_j_norm = F.normalize(Z_j.float(), dim=0, eps=1e-8)

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
