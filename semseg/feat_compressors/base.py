"""Feature compressor abstraction."""

import abc
import torch


class FeatCompressor(abc.ABC):
    @abc.abstractmethod
    def fit(self, x: torch.FloatTensor) -> None:
        pass

    @abc.abstractmethod
    def compress(self, x: torch.FloatTensor) -> torch.FloatTensor:
        pass

    @abc.abstractmethod
    def decompress(self, y: torch.FloatTensor) -> torch.FloatTensor:
        pass

    @abc.abstractmethod
    def is_fitted(self) -> bool:
        pass
