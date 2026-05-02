"""Ground-truth onehot semantic encoder."""

from typing import List
import torch

from semseg.image_encoders.base import ImageSemSegEncoder


class GTEncoder(ImageSemSegEncoder):
    def __init__(self, device: str = None, classes: List[str] = None):
        super().__init__(device)
        self.prompts = classes
        self._cat_index_to_name = {0: ""}
        self._cat_name_to_index = {"": 0}
        if self.prompts:
            for i, name in enumerate(self.prompts, start=1):
                self._cat_index_to_name[i] = name
                self._cat_name_to_index[name] = i

    @property
    def num_classes(self) -> int:
        return len(self._cat_index_to_name)

    @property
    def cat_name_to_index(self):
        return self._cat_name_to_index

    def encode_image_to_feat_map(self, rgb_image: torch.FloatTensor) -> torch.FloatTensor:
        raise NotImplementedError("GTEncoder is used only for label embeddings.")
