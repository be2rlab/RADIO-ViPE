"""Base dataset interfaces for semseg evaluation."""

import abc
from typing import Dict, List, Tuple, Union
import copy

import torch


class PosedRgbdDataset(torch.utils.data.IterableDataset, abc.ABC):
    def __init__(
        self,
        rgb_resolution: Union[Tuple[int], int] = None,
        depth_resolution: Union[Tuple[int], int] = None,
        frame_skip: int = 0,
        interp_mode: str = "bilinear",
    ):
        self.intrinsics_3x3 = None
        if isinstance(rgb_resolution, int):
            self.rgb_h = rgb_resolution
            self.rgb_w = rgb_resolution
        elif hasattr(rgb_resolution, "__len__"):
            self.rgb_h, self.rgb_w = rgb_resolution
        else:
            self.rgb_h, self.rgb_w = -1, -1
        if isinstance(depth_resolution, int):
            self.depth_h = depth_resolution
            self.depth_w = depth_resolution
        elif hasattr(depth_resolution, "__len__"):
            self.depth_h, self.depth_w = depth_resolution
        else:
            self.depth_h, self.depth_w = -1, -1
        self.frame_skip = frame_skip
        self.interp_mode = interp_mode


class SemSegDataset(PosedRgbdDataset, abc.ABC):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._cat_id_to_name: Dict = None
        self._cat_index_to_id = None
        self._cat_id_to_index = None
        self._cat_index_to_name = None
        self._cat_name_to_index = None

    @property
    def cat_index_to_name(self) -> List:
        return self._cat_index_to_name

    def _init_semseg_mappings(
        self,
        cat_id_to_name: Dict[int, str],
        white_list: List[str] = None,
        black_list: List[str] = None,
    ):
        self._cat_id_to_name = cat_id_to_name
        cin = copy.copy(cat_id_to_name)
        cin[0] = ""
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if white_list:
            self._cat_index_to_id = torch.tensor(
                sorted([idx for idx, name in cin.items() if name in white_list or idx == 0]),
                dtype=torch.long,
                device=device,
            )
        else:
            black_list = black_list or []
            self._cat_index_to_id = torch.tensor(
                sorted([idx for idx, name in cin.items() if name not in black_list]),
                dtype=torch.long,
                device=device,
            )
        self._cat_id_to_index = torch.zeros(max(cat_id_to_name.keys()) + 1, dtype=torch.long, device=device)
        self._cat_id_to_index[self._cat_index_to_id] = torch.arange(
            len(self._cat_index_to_id), dtype=torch.long, device=device
        )
        self._cat_index_to_name = [cin[self._cat_index_to_id[i].item()] for i in range(len(self._cat_index_to_id))]
        self._cat_name_to_index = {n: i for i, n in enumerate(self._cat_index_to_name)}
