"""Minimal 3D geometry helpers required for semseg evaluation."""

import torch
import torch_scatter


def mat_3x3_to_4x4(mat: torch.FloatTensor) -> torch.FloatTensor:
    zeros = torch.zeros(size=(*mat.shape[:-2], 3, 1), device=mat.device)
    mat = torch.cat((mat, zeros), dim=-1)
    row = torch.tensor([[0, 0, 0, 1]], device=mat.device)
    return torch.cat((mat, row.repeat(*mat.shape[:-2], 1, 1)), axis=-2)


def transform_points(points: torch.FloatTensor, transform_mat: torch.FloatTensor) -> torch.FloatTensor:
    is_non_homo = points.shape[-1] == 3
    if is_non_homo:
        points = torch.cat([points, torch.ones_like(points[..., :1])], dim=-1)
    if transform_mat.shape[-2:] == (3, 3):
        transform_mat = mat_3x3_to_4x4(transform_mat)
    transformed = points @ torch.transpose(transform_mat, -2, -1)
    return transformed[..., :3] if is_non_homo else transformed


def depth_to_pointcloud(
    depth_img: torch.FloatTensor,
    pose_4x4: torch.FloatTensor,
    intrinsics_3x3: torch.FloatTensor,
    conf_map: torch.FloatTensor = None,
    max_num_pts: int = -1,
):
    """Unproject valid depth pixels to world xyz points."""
    b, _, h, w = depth_img.shape
    img_xi, img_yi = torch.meshgrid(
        torch.arange(w, device=depth_img.device),
        torch.arange(h, device=depth_img.device),
        indexing="xy",
    )
    img_xi = img_xi.tile((b, 1, 1))
    img_yi = img_yi.tile((b, 1, 1))
    img_plane_pts = torch.stack(
        [img_xi.flatten(-2), img_yi.flatten(-2), torch.ones(b, h * w, device=depth_img.device)],
        axis=-1,
    )
    img_plane_pts = depth_img.reshape(b, h * w, 1) * img_plane_pts
    unproj_mat = pose_4x4 @ mat_3x3_to_4x4(torch.inverse(intrinsics_3x3))
    world_pts_xyz = transform_points(img_plane_pts, unproj_mat)

    valid_depth_mask = torch.logical_and(torch.isfinite(depth_img), depth_img > 0)
    valid_depth_indices = torch.argwhere(valid_depth_mask.flatten()).squeeze(-1)
    max_num_pts *= b
    if max_num_pts > 0 and max_num_pts < len(valid_depth_indices):
        if conf_map is None:
            indices_indices = torch.randperm(len(valid_depth_indices))
        else:
            indices_indices = torch.argsort(conf_map[valid_depth_mask], descending=True)
        selected_indices = valid_depth_indices[indices_indices[:max_num_pts]]
    else:
        selected_indices = valid_depth_indices
    world_pts_xyz = world_pts_xyz.reshape(-1, 3)[selected_indices]
    return world_pts_xyz, selected_indices


def pointcloud_to_sparse_voxels(
    xyz_pc: torch.FloatTensor,
    vox_size: float,
    feat_pc: torch.FloatTensor = None,
    aggregation: str = "mean",
    return_counts: bool = False,
):
    """Voxelize point cloud and aggregate features if provided."""
    d = xyz_pc.device
    xyz_vx = torch.round(xyz_pc / vox_size).type(torch.int64)

    if feat_pc is None:
        xyz_vx, count_vx = torch.unique(xyz_vx, return_counts=True, dim=0)
        xyz_vx = xyz_vx.type(torch.float) * vox_size
        count_vx = count_vx.type(torch.float).unsqueeze(-1)
        return (xyz_vx, count_vx) if return_counts else xyz_vx

    xyz_vx, reduce_ind, counts_vx = torch.unique(
        xyz_vx, return_inverse=True, return_counts=True, dim=0
    )
    feat_vx = torch.zeros((xyz_vx.shape[0], feat_pc.shape[-1]), device=d, dtype=feat_pc.dtype)
    torch_scatter.scatter(src=feat_pc, index=reduce_ind, out=feat_vx, reduce=aggregation, dim=0)
    xyz_vx = xyz_vx.type(torch.float) * vox_size
    counts_vx = counts_vx.type(torch.float).unsqueeze(-1)
    if return_counts:
        return xyz_vx, feat_vx, counts_vx
    return xyz_vx, feat_vx


def add_weighted_sparse_voxels(
    xyz_vx1: torch.FloatTensor,
    feat_cnt_vx1: torch.FloatTensor,
    xyz_vx2: torch.FloatTensor,
    feat_cnt_vx2: torch.FloatTensor,
    vox_size: float,
):
    """Aggregate sparse voxels where last feature dimension is weight/count."""
    feat_cnt_vx1[:, :-1] = feat_cnt_vx1[:, :-1] * feat_cnt_vx1[:, -1:]
    feat_cnt_vx2[:, :-1] = feat_cnt_vx2[:, :-1] * feat_cnt_vx2[:, -1:]
    xyz_vx, feat_cnt_vx = pointcloud_to_sparse_voxels(
        torch.cat((xyz_vx1, xyz_vx2), dim=0),
        vox_size=vox_size,
        feat_pc=torch.cat((feat_cnt_vx1, feat_cnt_vx2), dim=0),
        aggregation="sum",
        return_counts=False,
    )
    feat_cnt_vx[:, :-1] = feat_cnt_vx[:, :-1] / feat_cnt_vx[:, -1:]
    return xyz_vx, feat_cnt_vx
