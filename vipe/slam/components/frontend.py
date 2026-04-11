# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# -------------------------------------------------------------------------------------------------
# This file includes code originally from the DROID-SLAM repository:
# https://github.com/cvg/DROID-SLAM
# Licensed under the MIT License. See THIRD_PARTY_LICENSES.md for details.
# -------------------------------------------------------------------------------------------------


# /vipe/slam/components/frontend.py
import logging

import torch

from omegaconf import DictConfig

from vipe.ext.lietorch import SE3

from ..networks.droid_net import DroidNet
from .buffer import GraphBuffer
from .factor_graph import FactorGraph
from .loop_closure import LoopClosureDetector
from vipe.utils.profiler import profile_function, profiler_section

logger = logging.getLogger(__name__)


class SLAMFrontend:
    """
    Frontend is called given every new frame. Currently it is a no-op for non-keyframe frames.
    For keyframe, it handles the system initialization and partial update logic (i.e. use BA to get pose for this kf).
    """

    loop_closure: LoopClosureDetector | None = None

    def __init__(self, net: DroidNet, video: GraphBuffer, args: DictConfig, device: torch.device):
        self.video = video
        self.device = device
        ec_cfg = getattr(args, "embedding_covisibility", None)
        ec_weight = float(getattr(ec_cfg, "weight", 0.0)) if ec_cfg and getattr(ec_cfg, "enabled", False) else 0.0
        self.graph = FactorGraph(
            net,
            video,
            device,
            max_factors=48,
            incremental=True,
            cross_view=args.cross_view,
            use_semantic_flow_init=getattr(args, "use_semantic_flow_init", False),
            embedding_covisibility_weight=ec_weight,
        )

        # Number of frames that the frontend has so far optimized.
        self.t1 = 0

        # frontend variables
        self.is_initialized = False

        self.max_age = 25
        self._idx_buf = torch.zeros(2, dtype=torch.long, device=device)  # [t1-3, t1-2]
        self._ii_min: int = 0  # updated whenever factors are added/removed

        # Adaptive iteration parameters
        adaptive_cfg = getattr(args, "adaptive_frontend_iters", None)
        if adaptive_cfg is not None and getattr(adaptive_cfg, "enabled", False):
            self.adaptive_iters = True
            self.min_iters1 = int(getattr(adaptive_cfg, "min_iters1", 2))
            self.max_iters1 = int(getattr(adaptive_cfg, "max_iters1", 8))
            self.min_iters2 = int(getattr(adaptive_cfg, "min_iters2", 1))
            self.max_iters2 = int(getattr(adaptive_cfg, "max_iters2", 4))
            self.convergence_thresh = float(getattr(adaptive_cfg, "convergence_thresh", 0.025))
        else:
            self.adaptive_iters = False
            self.min_iters1 = 4
            self.max_iters1 = 4
            self.min_iters2 = 2
            self.max_iters2 = 2
            self.convergence_thresh = 0.0

        # Number of frames to wait before initializing (default 8)
        self.args = args
        self.warmup = args.warmup
        self.beta = args.beta
        self.frontend_nms = args.frontend_nms
        self.keyframe_thresh = args.keyframe_thresh
        self.frontend_window = args.frontend_window
        self.frontend_thresh = args.frontend_thresh
        self.frontend_radius = args.frontend_radius
        self.has_init_pose = args.has_init_pose
        self.use_depth_prior_init = getattr(args, "use_depth_prior_init", True)

    def __init_pose(self):
        assert self.t1 > 1
        p1 = SE3(self.video.poses[self.t1 - 2])
        p2 = SE3(self.video.poses[self.t1 - 1])
        w = (p2 * p1.inv()).log() * 0.5
        self.video.poses[self.t1] = (SE3.exp(w) * p2).data
        # self.video.poses[self.t1] = self.video.poses[self.t1 - 1].clone()

    def __update(self):
        self.t1 += 1

        with profiler_section("frontend.rm_old_factors"):
            if self.graph.corr is not None and self.graph.age.max().item() > self.max_age:
                self.graph.rm_factors(self.graph.age > self.max_age, store=True)
                if self.graph.ii is not None and len(self.graph.ii) > 0:
                    self._ii_min = int(self.graph.ii.min().item())

        with profiler_section("frontend.add_proximity"):
            self.graph.add_proximity_factors(
                self.t1 - 5,
                max(self.t1 - self.frontend_window, 0),
                rad=self.frontend_radius,
                nms=self.frontend_nms,
                thresh=self.frontend_thresh,
                beta=self.beta,
                remove=True,
            )

        with profiler_section("frontend.update_iters1"):
            self._run_adaptive_iters(self.min_iters1, self.max_iters1)

        with profiler_section("frontend.frame_distance"):
            self._idx_buf[0] = self.t1 - 3
            self._idx_buf[1] = self.t1 - 2
            d = self.video.frame_distance_dense_disp(
                self._idx_buf[:1], self._idx_buf[1:],
                beta=self.beta, bidirectional=True,
            )

        if d.max().item() < self.keyframe_thresh:
            with profiler_section("frontend.rm_keyframe"):
                removed_idx = self.t1 - 2
                self.graph.rm_second_newest_keyframe(removed_idx)
                self.t1 -= 1
                if self.graph.ii is not None and len(self.graph.ii) > 0:
                    self._ii_min = int(self.graph.ii.min().item())
                if self.loop_closure is not None:
                    self.loop_closure.update_after_removal(removed_idx)
        else:
            with profiler_section("frontend.update_iters2"):
                self._run_adaptive_iters(self.min_iters2, self.max_iters2)

        with profiler_section("frontend.init_pose_and_disp"):
            if not self.has_init_pose:
                self.__init_pose()
            self._predict_next_disp()

        self.video.dirty[self._ii_min : self.t1] = True

    def __initialize(self):
        """initialize the SLAM system with keyframes idx [t0, t1)"""

        self.t1 = self.video.n_frames

        with profiler_section("frontend.init.neighborhood"):
            self.graph.add_neighborhood_factors(0, self.t1, r=1 if self.args.seq_init else 3)

        with profiler_section("frontend.init.update_pass1"):
            for _ in range(8):
                self.graph.update(t0=1, use_inactive=True, fixed_motion=self.has_init_pose)

        if not self.args.seq_init:
            with profiler_section("frontend.init.proximity"):
                self.graph.add_proximity_factors(0, 0, rad=2, nms=2, thresh=self.frontend_thresh, remove=False)
            with profiler_section("frontend.init.update_pass2"):
                for _ in range(8):
                    self.graph.update(t0=1, use_inactive=True, fixed_motion=self.has_init_pose)

        if not self.has_init_pose:
            self.__init_pose()
        self._predict_next_disp()
        self.video.dirty[: self.t1] = True

        # initialization complete
        self.is_initialized = True
        self.graph.rm_factors(self.graph.ii < self.warmup - 4, store=True)

        # Sync _ii_min after initialization's final rm_factors.
        if self.graph.ii is not None and len(self.graph.ii) > 0:
            self._ii_min = int(self.graph.ii.min().item())

    def _run_adaptive_iters(self, min_iters: int, max_iters: int):
        """Run GRU update iterations, stopping early when the flow delta converges.
        """
        for i in range(max_iters):
            delta_norm, energy = self.graph.update(
                use_inactive=True, fixed_motion=self.has_init_pose
            )
            if i < min_iters - 1:
                continue

            if self.adaptive_iters:
                # Check both: flow delta AND energy convergence
                rel_change = abs(prev_energy - energy) / (abs(prev_energy) + 1e-12)
                if delta_norm < self.convergence_thresh or rel_change < 1e-4:
                    logger.debug(
                        "Frontend converged at iter %d/%d "
                        "(delta=%.4f, energy_rel=%.2e)",
                        i + 1, max_iters, delta_norm, rel_change,
                    )
                    break
            prev_energy = energy

    def _predict_next_disp(self):
        """Predict disparity for the next keyframe slot using depth prior when available.
        """
        # disps_sens / disps shape: (T, V, H, W)
        if self.use_depth_prior_init:
            sens = self.video.disps_sens[self.t1 - 1]           # (V, H, W)
            # Boolean mask: True for views that have valid sensor data.
            valid = sens.sum(dim=(-2, -1)) > 0                  # (V,)
            sens_mean = sens.mean(dim=(-2, -1))                  # (V,)
            prev_mean = self.video.disps[self.t1 - 1].mean(dim=(-2, -1))  # (V,)
            # Blend: use sensor mean where valid, previous frame mean elsewhere.
            next_disp_val = torch.where(valid, sens_mean, prev_mean)  # (V,)
            # Broadcast scalar-per-view back to (V, H, W).
            self.video.disps[self.t1] = next_disp_val[:, None, None].expand_as(
                self.video.disps[self.t1]
            )
        else:
            # All views: just take the spatial mean of the previous frame.
            prev_mean = self.video.disps[self.t1 - 1].mean(dim=(-2, -1))  # (V,)
            self.video.disps[self.t1] = prev_mean[:, None, None].expand_as(
                self.video.disps[self.t1]
            )

    def run(self):
        """main update"""

        # do initialization
        if not self.is_initialized and self.video.n_frames == self.warmup:
            self.__initialize()

        # do update if new keyframe is added.
        elif self.is_initialized and self.t1 < self.video.n_frames:
            self.__update()

