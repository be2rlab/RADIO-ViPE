import logging

import torch
import torch.nn.functional as F


logger = logging.getLogger(__name__)


class LoopClosureDetector:
    """
    Detects loop closures using global embedding descriptors.

    For each keyframe, computes a global descriptor by mean-pooling the spatial
    embedding features. Queries incoming keyframes against non-recent keyframes
    using cosine similarity to find revisited places.
    """

    def __init__(
        self,
        similarity_thresh: float = 0.85,
        min_interval: int = 20,
        max_candidates: int = 3,
        device: torch.device = torch.device("cuda"),
    ):
        self.similarity_thresh = similarity_thresh
        self.min_interval = min_interval
        self.max_candidates = max_candidates
        self.device = device

        self._desc_storage: torch.Tensor | None = None
        self._n_keyframes: int = 0
        self._loop_edges: list[tuple[int, int]] = []

    @staticmethod
    def _compute_global_descriptor(embeddings: torch.Tensor) -> torch.Tensor:
        """
        Multi-scale pooled descriptor:
        - Global (1x1)
        - 1x2 split
        - 2x2 split
        Returns concatenated L2-normalized descriptor.
        """
        x = embeddings.float()

        pools = []

        # 1x1
        pools.append(F.adaptive_avg_pool2d(x, (1, 1)).reshape(-1))

        # 1x2
        pools.append(F.adaptive_avg_pool2d(x, (1, 2)).reshape(-1))

        # 2x2
        pools.append(F.adaptive_avg_pool2d(x, (2, 2)).reshape(-1))

        desc = torch.cat(pools, dim=0)
        return F.normalize(desc, dim=0)

    def add_keyframe(
        self, kf_idx: int, embeddings: torch.Tensor | None
    ) -> list[tuple[int, int]]:
        """
        Register a keyframe and return any newly detected loop closure edges.

        Args:
            kf_idx: buffer index of the keyframe.
            embeddings: ``(n_views, C, H, W)`` tensor, or *None* if unavailable.

        Returns:
            List of ``(source, target)`` pairs with ``source > target``.
        """
        if embeddings is None:
            self._n_keyframes = max(self._n_keyframes, kf_idx + 1)
            return []

        desc = self._compute_global_descriptor(embeddings)

        if self._desc_storage is None:
            self._desc_storage = torch.zeros(
                1024, desc.shape[0], device=self.device, dtype=torch.float32
            )

        if kf_idx >= self._desc_storage.shape[0]:
            new_storage = torch.zeros(
                self._desc_storage.shape[0] * 2,
                desc.shape[0],
                device=self.device,
                dtype=torch.float32,
            )
            new_storage[: self._desc_storage.shape[0]] = self._desc_storage
            self._desc_storage = new_storage

        self._desc_storage[kf_idx] = desc
        self._n_keyframes = max(self._n_keyframes, kf_idx + 1)

        new_loops = self._detect(kf_idx)
        self._loop_edges.extend(new_loops)
        return new_loops

    def _detect(self, kf_idx: int) -> list[tuple[int, int]]:
        if self._desc_storage is None or kf_idx < self.min_interval:
            return []

        max_candidate = kf_idx - self.min_interval
        if max_candidate <= 0:
            return []

        query = self._desc_storage[kf_idx]
        candidates = self._desc_storage[:max_candidate]
        valid_mask = candidates.norm(dim=1) > 0.5

        if not valid_mask.any():
            return []

        similarities = candidates @ query
        similarities[~valid_mask] = -1.0

        top_k = min(self.max_candidates, int(valid_mask.sum().item()))
        if top_k == 0:
            return []

        values, indices = similarities.topk(top_k)

        loops: list[tuple[int, int]] = []
        for sim, idx in zip(values, indices):
            if sim.item() >= self.similarity_thresh:
                loops.append((kf_idx, idx.item()))
                logger.info(
                    "Loop closure detected: kf %d <-> kf %d (similarity=%.3f)",
                    kf_idx,
                    idx.item(),
                    sim.item(),
                )
        return loops

    def update_after_removal(self, removed_idx: int):
        """Adjust internal state when the frontend drops a keyframe."""
        if self._desc_storage is None:
            return

        if removed_idx < self._n_keyframes - 1:
            self._desc_storage[removed_idx:-1] = self._desc_storage[
                removed_idx + 1 :
            ].clone()
        self._n_keyframes = max(0, self._n_keyframes - 1)

        updated: list[tuple[int, int]] = []
        for src, tgt in self._loop_edges:
            if src == removed_idx or tgt == removed_idx:
                continue
            new_src = src - 1 if src > removed_idx else src
            new_tgt = tgt - 1 if tgt > removed_idx else tgt
            updated.append((new_src, new_tgt))
        self._loop_edges = updated

    def get_loop_edges(self) -> list[tuple[int, int]]:
        return list(self._loop_edges)

    def get_loop_edges_tensors(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Return bidirectional edge tensors suitable for ``FactorGraph.add_factors``."""
        if not self._loop_edges:
            return None

        ii = torch.tensor(
            [e[0] for e in self._loop_edges], dtype=torch.long, device=self.device
        )
        jj = torch.tensor(
            [e[1] for e in self._loop_edges], dtype=torch.long, device=self.device
        )
        return torch.cat([ii, jj]), torch.cat([jj, ii])