"""PCA feature compressor."""

import torch

from semseg.feat_compressors.base import FeatCompressor


class PcaCompressor(FeatCompressor):
    def __init__(self, out_dim: int, in_dim: int = None, path: str = None):
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.mean = None
        self.basis = None
        if path is not None:
            self.load(path)

    def fit(self, x: torch.FloatTensor) -> None:
        d = x.shape[-1]
        if self.in_dim is not None and d != self.in_dim:
            raise ValueError("Data feature dimension does not match stored input dim")
        self.in_dim = d
        x_flatten = x.flatten(0, -2)
        self.mean = torch.mean(x_flatten, dim=0)
        x_centered = x_flatten - self.mean
        q = min(self.out_dim, x_flatten.shape[0], x_flatten.shape[1])
        _, _, v = torch.pca_lowrank(x_centered, q=q)
        self.basis = v
        self.out_dim = q

    def save(self, fp: str) -> None:
        torch.save(
            dict(metadata=dict(in_dim=self.in_dim, out_dim=self.out_dim), mean=self.mean, basis=self.basis),
            fp,
        )

    def load(self, fp: str) -> None:
        d = torch.load(fp)
        self.in_dim = d["metadata"]["in_dim"]
        self.out_dim = d["metadata"]["out_dim"]
        self.mean = d["mean"]
        self.basis = d["basis"]

    def compress(self, x: torch.FloatTensor) -> torch.FloatTensor:
        shape = list(x.shape)
        shape[-1] = self.out_dim
        return ((x.flatten(0, -2) - self.mean) @ self.basis).reshape(*shape)

    def decompress(self, y: torch.FloatTensor) -> torch.FloatTensor:
        shape = list(y.shape)
        shape[-1] = self.in_dim
        return (y.flatten(0, -2) @ self.basis.to(y.dtype).T + self.mean).reshape(*shape)

    def is_fitted(self) -> bool:
        return self.mean is not None and self.basis is not None
