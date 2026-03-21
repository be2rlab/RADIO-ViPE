"""Tests for semantic_flow_init and blend_flow_prior."""

import importlib
import sys
from pathlib import Path

import torch
import pytest


# Import the module directly to avoid pulling in the full vipe package
# (which requires compiled C++ extensions).
_MODULE_PATH = Path(__file__).resolve().parents[1] / "vipe" / "slam" / "semantic_flow.py"
_spec = importlib.util.spec_from_file_location("semantic_flow", _MODULE_PATH)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

semantic_flow_init = _mod.semantic_flow_init
blend_flow_prior = _mod.blend_flow_prior


@pytest.fixture
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def test_semantic_flow_init(device: torch.device):
    """
    Construct two random Radio embedding tensors where a 10×10 patch in Z_j is
    a copy of Z_i shifted 3 pixels to the right.  Verify that
    semantic_flow_init recovers the correct (3, 0) offset in >90 % of the
    patch pixels.
    """
    K, H, W = 64, 32, 32
    torch.manual_seed(42)

    Z_i = torch.randn(K, H, W, device=device)
    Z_j = torch.randn(K, H, W, device=device)

    # Inject a known 3-pixel horizontal offset into a 10×10 patch.
    y0, x0 = 10, 8
    patch_h, patch_w = 10, 10
    dx_true = 3

    Z_j[:, y0 : y0 + patch_h, x0 + dx_true : x0 + patch_w + dx_true] = Z_i[
        :, y0 : y0 + patch_h, x0 : x0 + patch_w
    ]

    # Identity geometric projection (each pixel maps to itself).
    gy, gx = torch.meshgrid(
        torch.arange(H, device=device, dtype=torch.float32),
        torch.arange(W, device=device, dtype=torch.float32),
        indexing="ij",
    )
    mu_ij = torch.stack([gx, gy], dim=-1)

    omega_semantic, max_sim = semantic_flow_init(Z_i, Z_j, mu_ij, search_radius=8)

    assert omega_semantic.shape == (H, W, 2)
    assert max_sim.shape == (H, W)

    # Check that the recovered offset in the patch is (dx_true, 0).
    patch_offset = omega_semantic[y0 : y0 + patch_h, x0 : x0 + patch_w] - mu_ij[
        y0 : y0 + patch_h, x0 : x0 + patch_w
    ]
    expected_dx = torch.full((patch_h, patch_w), float(dx_true), device=device)
    expected_dy = torch.zeros(patch_h, patch_w, device=device)

    correct_x = (patch_offset[..., 0] - expected_dx).abs() < 0.5
    correct_y = (patch_offset[..., 1] - expected_dy).abs() < 0.5
    correct = correct_x & correct_y

    accuracy = correct.float().mean().item()
    assert accuracy > 0.90, (
        f"Semantic flow recovered the 3-px offset with only {accuracy:.1%} "
        f"accuracy (need >90%)"
    )

    # The injected pixels should also have high semantic confidence.
    patch_sim = max_sim[y0 : y0 + patch_h, x0 : x0 + patch_w]
    assert patch_sim.mean().item() > 0.9, (
        f"Mean semantic confidence in patch is {patch_sim.mean().item():.3f}, "
        f"expected >0.9 for exact copies"
    )


def test_semantic_flow_init_no_mu_ij(device: torch.device):
    """Calling without mu_ij should default to identity and still work."""
    K, H, W = 32, 16, 16
    torch.manual_seed(0)

    Z_i = torch.randn(K, H, W, device=device)
    Z_j = Z_i.clone()

    omega, sim = semantic_flow_init(Z_i, Z_j, search_radius=4)

    assert omega.shape == (H, W, 2)
    assert sim.shape == (H, W)

    # Perfect copy → offset should be (0, 0) everywhere in the interior.
    margin = 4
    gy, gx = torch.meshgrid(
        torch.arange(H, device=device, dtype=torch.float32),
        torch.arange(W, device=device, dtype=torch.float32),
        indexing="ij",
    )
    identity = torch.stack([gx, gy], dim=-1)
    interior_offset = (
        omega[margin:-margin, margin:-margin]
        - identity[margin:-margin, margin:-margin]
    ).abs()

    assert (interior_offset < 0.5).all(), (
        "With identical embeddings the offset should be zero in the interior."
    )


def test_blend_flow_prior(device: torch.device):
    """Basic sanity check on the blending formula."""
    H, W = 8, 8
    omega_prior = torch.zeros(H, W, 2, device=device)
    omega_semantic = torch.ones(H, W, 2, device=device) * 5.0

    # When photo_conf == 0, should return omega_semantic.
    photo_conf = torch.zeros(H, W, device=device)
    semantic_conf = torch.ones(H, W, device=device)

    result = blend_flow_prior(omega_prior, omega_semantic, photo_conf, semantic_conf)
    assert torch.allclose(result, omega_semantic, atol=1e-4)

    # When semantic_conf == 0, should return omega_prior.
    photo_conf = torch.ones(H, W, device=device)
    semantic_conf = torch.zeros(H, W, device=device)

    result = blend_flow_prior(omega_prior, omega_semantic, photo_conf, semantic_conf)
    assert torch.allclose(result, omega_prior, atol=1e-4)

    # Equal confidences → midpoint.
    photo_conf = torch.ones(H, W, device=device)
    semantic_conf = torch.ones(H, W, device=device)

    result = blend_flow_prior(omega_prior, omega_semantic, photo_conf, semantic_conf)
    expected = 0.5 * omega_prior + 0.5 * omega_semantic
    assert torch.allclose(result, expected, atol=1e-4)
