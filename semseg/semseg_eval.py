"""Semantic segmentation evaluation for RADIO-ViPE pointcloud predictions."""

from collections import OrderedDict
import logging
import os
from dataclasses import dataclass
from typing import Tuple, Union

import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf
import torch
from tqdm import tqdm

from semseg import eval_utils, mapping, image_encoders, feat_compressors, datasets
from semseg import geometry3d as g3d

logger = logging.getLogger(__name__)


@dataclass
class SemSegEvalConfig:
    eval_out: str = "eval_out"
    no_caching: bool = False
    k: int = 0
    prompt_denoising_thresh: float = 0.5
    prediction_thresh: float = 0.1
    classes_to_ignore: Tuple[str] = tuple()
    classes_to_eval: Tuple[str] = tuple()
    chunk_size: int = 10000
    online_eval_period: int = -1
    load_external_gt: bool = False
    gt_dir: str = ""
    first_pose_dir: str = ""
    lang_feats: bool = False
    load_external_pred: bool = False
    pred_dir: str = ""


cs = ConfigStore.instance()
cs.store(name="extras", node=SemSegEvalConfig)


class SemSegEval:
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Explicitly reference local package classes to avoid ambiguous hydra imports.
        _ = datasets, mapping, image_encoders, feat_compressors
        self.dataset = hydra.utils.instantiate(cfg.dataset)
        self.dataset._init_semseg_mappings(self.dataset._cat_id_to_name, cfg.classes_to_eval, cfg.classes_to_ignore)
        self.num_classes = len(self.dataset._cat_index_to_id)
        self.dataloader = torch.utils.data.DataLoader(self.dataset, batch_size=cfg.batch_size)
        self.store_output = bool(cfg.eval_out)
        scene_name = cfg.dataset.scene_name.replace("/", "_")
        cache_dataset_dir = os.path.join(cfg.eval_out, self.dataset.__class__.__name__)
        self.cache_scene_dir = os.path.join(cache_dataset_dir, scene_name)
        os.makedirs(self.cache_scene_dir, exist_ok=True)

    def load_external_semseg_gt(self):
        fn = os.path.join(self.cfg.gt_dir, self.cfg.dataset.scene_name, "external_semseg_gt.pt")
        d = torch.load(fn, weights_only=True)
        gt_xyz = d["semseg_gt_xyz"].to(self.device)
        gt_label = d["semseg_gt_label"].to(self.device)
        gt_label[gt_label < 0] = 0
        gt_label = self.dataset._cat_id_to_index[gt_label]
        gt_onehot = torch.nn.functional.one_hot(gt_label, self.num_classes)
        gt_xyz, gt_onehot = g3d.pointcloud_to_sparse_voxels(
            gt_xyz, self.cfg.mapping.vox_size, gt_onehot, aggregation="sum"
        )
        return gt_xyz, torch.argmax(gt_onehot, dim=-1)

    def compute_semseg_gt(self):
        names = self.dataset.cat_index_to_name[1:]
        gt_encoder = image_encoders.GTEncoder(classes=names)
        semseg_gt_lifter = mapping.SemanticVoxelMap(
            self.dataset.intrinsics_3x3,
            None,
            max_pts_per_frame=self.cfg.mapping.max_pts_per_frame,
            vox_size=self.cfg.mapping.vox_size,
            vox_accum_period=self.cfg.mapping.vox_accum_period,
            encoder=gt_encoder,
        )
        for batch in tqdm(self.dataloader):
            semseg_onehot = torch.nn.functional.one_hot(batch["semseg_img"].cuda(), self.num_classes)
            semseg_onehot = semseg_onehot.squeeze(1).permute(0, 3, 1, 2)
            semseg_gt_lifter.process_posed_rgbd(
                batch["rgb_img"].cuda(), batch["depth_img"].cuda(), batch["pose_4x4"].cuda(), feat_img=semseg_onehot.float()
            )
        semseg_gt_lifter.accum_semantic_voxels()
        text_embeds = gt_encoder.encode_labels(names)
        semseg_gt_xyz = semseg_gt_lifter.global_vox_xyz
        semseg_gt_label = eval_utils.compute_semseg_preds(
            semseg_gt_lifter.global_vox_feat, text_embeds, 0, 0.1, self.cfg.chunk_size
        )
        return semseg_gt_xyz, semseg_gt_label

    def align_points_to_global(self, pred_xyz):
        pose_path = os.path.join(self.cfg.first_pose_dir, self.cfg.dataset.scene_name, "traj.txt")
        with open(pose_path, "r", encoding="utf-8") as f:
            numbers = list(map(float, f.readline().strip().split()))
        pose_matrix = torch.tensor(numbers, dtype=pred_xyz.dtype, device=pred_xyz.device).reshape(4, 4)
        return pred_xyz @ pose_matrix[:3, :3].T + pose_matrix[:3, 3]

    def load_external_preds(self):
        map_path = os.path.join(self.cfg.pred_dir, self.cfg.dataset.scene_name + "_slam_map.pt")
        data = torch.load(map_path, weights_only=False)
        pred_xyz = data["dense_disp_xyz"].to(self.device)
        pred_feats = data["dense_disp_embeddings"].to(self.device)
        return self.align_points_to_global(pred_xyz), pred_feats

    def compute_semseg_metrics(self, gt, preds, gt_xyz=None, preds_xyz=None):
        if gt_xyz is not None:
            if self.cfg.k > 0:
                gt, aligned_preds = eval_utils.align_labels_with_knn(gt_xyz, gt, preds_xyz, preds, k=self.cfg.k)
            else:
                gt, aligned_preds = eval_utils.align_labels_with_vox_grid(gt_xyz, gt, preds_xyz, preds, self.cfg.mapping.vox_size)
        else:
            aligned_preds = preds
        tp, fp, fn, tn = eval_utils.eval_gt_pred(aligned_preds, gt, num_classes=self.num_classes)
        iou = tp / (tp + fp + fn)
        freq = tp + fn
        iou[freq == 0] = torch.nan
        fiou = (freq / torch.sum(freq)) * iou
        return dict(tp=tp, fp=fp, fn=fn, tn=tn, iou=iou, fiou=fiou, miou=torch.nanmean(iou), fmiou=torch.nansum(fiou), acc=torch.mean((aligned_preds[gt != 0] == gt[gt != 0]).float()))

    def save_metrics(self, m: OrderedDict):
        if not self.store_output:
            return
        last_m = next(reversed(m.values()))
        class_wise_metrics = [k for k, v in last_m.items() if v.dim() > 0]
        rows = ["index,label," + ",".join(class_wise_metrics) + "\n"]
        for i in range(self.num_classes - 1):
            fields = [str(i + 1), str(self.dataset._cat_index_to_name[i + 1])]
            fields.extend([str(last_m[k][i].item()) for k in class_wise_metrics])
            rows.append(",".join(fields) + "\n")
        with open(os.path.join(self.cache_scene_dir, "semseg_final_class_results.csv"), "w", encoding="utf-8") as f:
            f.writelines(rows)
        aggregate_metrics = [k for k, v in last_m.items() if v.dim() == 0]
        with open(os.path.join(self.cache_scene_dir, "semseg_final_summary_results.csv"), "w", encoding="utf-8") as f:
            f.write("scene_name," + ",".join(aggregate_metrics) + "\n")
            f.write(",".join([self.dataset.scene_name] + [str(last_m[k].item()) for k in aggregate_metrics]) + "\n")

    def run(self):
        eval_utils.reset_seed(self.cfg.seed)
        if self.cfg.load_external_gt:
            semseg_gt_xyz, semseg_gt_label = self.load_external_semseg_gt()
        else:
            semseg_gt_xyz, semseg_gt_label = self.compute_semseg_gt()
        encoder_kwargs = {}
        if "classes" in self.cfg.encoder:
            encoder_kwargs["classes"] = self.dataset.cat_index_to_name[1:]
        self.encoder = hydra.utils.instantiate(self.cfg.encoder, **encoder_kwargs)

        if self.cfg.load_external_pred and self.cfg.mapping.get("feat_compressor"):
            pca_path = os.path.join(self.cfg.pred_dir, self.cfg.dataset.scene_name + "_pca_basis.pt")
            pca_data = torch.load(pca_path, weights_only=False)
            mean = pca_data["mean"]
            components = pca_data["components"]
            out_dim = pca_data["metadata"]["target_dim"]
            in_dim = components.shape[0]
            self.feat_compressor = feat_compressors.PcaCompressor(out_dim, in_dim)
            self.feat_compressor.mean = mean.to(self.device)
            self.feat_compressor.basis = components.to(self.device)
        else:
            self.feat_compressor = (
                hydra.utils.instantiate(self.cfg.mapping.feat_compressor)
                if self.cfg.mapping.get("feat_compressor") is not None
                else None
            )

        names = self.dataset.cat_index_to_name[1:]
        text_embeds = self.encoder.encode_labels(names) if self.cfg.querying.text_query_mode == "labels" else self.encoder.encode_prompts(names)
        feats_xyz, feats_feats = self.load_external_preds()
        feats_lang = feats_feats.float() if self.cfg.lang_feats else self.encoder.align_spatial_features_with_language(
            feats_feats.unsqueeze(-1).unsqueeze(-1).float()
        ).squeeze(-1).squeeze(-1)
        semseg_pred_label = eval_utils.compute_semseg_preds(
            feats_lang.float(),
            text_embeds.float(),
            self.cfg.prompt_denoising_thresh,
            self.cfg.prediction_thresh,
            self.cfg.chunk_size,
        )
        m = self.compute_semseg_metrics(semseg_gt_label, semseg_pred_label, semseg_gt_xyz, feats_xyz)
        logger.info("miou=%.4f fmiou=%.4f acc=%.4f", m["miou"].item(), m["fmiou"].item(), m["acc"].item())
        self.save_metrics(OrderedDict({-1: m}))
        OmegaConf.save(self.cfg, os.path.join(self.cache_scene_dir, "cfg.yaml"))


@hydra.main(version_base="1.2", config_path="configs", config_name="default")
@torch.inference_mode()
def main(cfg=None):
    SemSegEval(cfg).run()


if __name__ == "__main__":
    main()
