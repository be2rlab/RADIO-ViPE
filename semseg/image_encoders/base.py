"""Minimal image encoder interfaces for semseg evaluation."""

import abc
from typing import List

import torch


class ImageSpatialEncoder(abc.ABC):
    def __init__(self, device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    @abc.abstractmethod
    def encode_image_to_feat_map(self, rgb_image: torch.FloatTensor) -> torch.FloatTensor:
        pass


class ImageSemSegEncoder(ImageSpatialEncoder):
    def __init__(self, device: str = None):
        super().__init__(device)
        self.eps = 1e-10

    @property
    @abc.abstractmethod
    def num_classes(self) -> int:
        pass

    @property
    @abc.abstractmethod
    def cat_name_to_index(self):
        pass

    def encode_labels(self, labels: List[str]) -> torch.FloatTensor:
        onehot = torch.full((len(labels), self.num_classes), self.eps, dtype=torch.float, device=self.device)
        for i, c in enumerate(labels):
            onehot[i, self.cat_name_to_index[c]] = 1
        return onehot

    def encode_prompts(self, prompts: List[str]) -> torch.FloatTensor:
        return self.encode_labels(prompts)

    def align_spatial_features_with_language(self, features: torch.FloatTensor) -> torch.FloatTensor:
        return features
