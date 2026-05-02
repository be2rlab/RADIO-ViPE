"""Utility functions used by semseg evaluation."""

import torch


def compute_cos_sim(
    vec1: torch.FloatTensor, vec2: torch.FloatTensor, softmax: bool = False
) -> torch.FloatTensor:
    """Compute cosine similarity between two sets of vectors."""
    n, c1 = vec1.shape
    m, c2 = vec2.shape
    if c1 != c2:
        raise ValueError(f"Feature dims do not match: {c1} != {c2}")

    vec1 = vec1 / vec1.norm(dim=-1, keepdim=True)
    vec2 = vec2 / vec2.norm(dim=-1, keepdim=True)
    sim = (vec1.reshape(1, n, 1, c1) @ vec2.reshape(m, 1, c1, 1)).reshape(m, n)
    if softmax:
        return torch.softmax(100 * sim, dim=-1)
    return sim
