"""Utilities for semantic segmentation evaluation."""

import random
import numpy as np
from sklearn.neighbors import BallTree
import torch

from semseg import geometry3d as g3d
from semseg.utils import compute_cos_sim


def reset_seed(seed: int):
    if seed < 0:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_semseg_preds(
    map_feats: torch.FloatTensor,
    text_embeds: torch.FloatTensor,
    prompt_denoising_thresh: float = 0.5,
    prediction_thresh: float = 0.1,
    chunk_size: int = 100000,
):
    num_chunks = int(np.ceil(map_feats.shape[0] / chunk_size))
    preds = []
    for c in range(num_chunks):
        sim_vx = compute_cos_sim(text_embeds, map_feats[c * chunk_size : (c + 1) * chunk_size], softmax=True)
        max_sim = torch.max(sim_vx, dim=0).values
        low_conf_classes = torch.argwhere(max_sim < prompt_denoising_thresh)
        sim_vx[:, low_conf_classes] = -torch.inf
        sim_value, pred = torch.max(sim_vx, dim=-1)
        pred += 1
        pred[sim_value < prediction_thresh] = 0
        preds.append(pred)
    return torch.cat(preds, dim=0)


def eval_gt_pred(pred_ids, gt_ids, num_classes):
    pred_bin_mask = torch.nn.functional.one_hot(pred_ids, num_classes=num_classes).bool()
    gt_bin_mask = torch.nn.functional.one_hot(gt_ids, num_classes=num_classes).bool()
    valid_mask = (gt_ids != 0).unsqueeze(-1)
    tp = torch.sum(gt_bin_mask & pred_bin_mask & valid_mask, dim=0)
    fp = torch.sum(~gt_bin_mask & pred_bin_mask & valid_mask, dim=0)
    fn = torch.sum(gt_bin_mask & ~pred_bin_mask & valid_mask, dim=0)
    tn = torch.sum(~gt_bin_mask & ~pred_bin_mask & valid_mask, dim=0)
    return tp[1:], fp[1:], fn[1:], tn[1:]


def align_labels_with_vox_grid(xyz1, labels1, xyz2, labels2, vox_size):
    labels1_labels2 = torch.zeros(labels1.shape[0] + labels2.shape[0], 2, device=labels1.device, dtype=labels1.dtype)
    labels1_labels2[: labels1.shape[0], 0] = labels1
    labels1_labels2[labels1.shape[0] :, 1] = labels2
    _, labels1_labels2 = g3d.pointcloud_to_sparse_voxels(
        torch.cat([xyz1, xyz2], dim=0), feat_pc=labels1_labels2, vox_size=vox_size, aggregation="sum"
    )
    return labels1_labels2[:, 0], labels1_labels2[:, 1]


def align_labels_with_knn(xyz1, labels1, xyz2, labels2, k=1):
    ball_tree = BallTree(xyz2.cpu())
    matched_indices = torch.from_numpy(ball_tree.query(xyz1.cpu(), k=k)[1]).to(labels2.device)
    aligned_labels2 = torch.mode(labels2[matched_indices], dim=-1).values
    return labels1, aligned_labels2
