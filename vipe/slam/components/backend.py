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

import logging

import torch

from omegaconf import DictConfig

from vipe.priors.depth import DepthEstimationModel

from ..networks.droid_net import DroidNet
from .buffer import GraphBuffer
from .factor_graph import FactorGraph
from .loop_closure import LoopClosureDetector


logger = logging.getLogger(__name__)


class SLAMBackend:
    """
    Mainly used to run a pretty dense bundle adjustment for all the frames in the graph.
    """

    depth_model: DepthEstimationModel | None = None
    loop_closure: LoopClosureDetector | None = None

    def __init__(self, net: DroidNet, video: GraphBuffer, args: DictConfig, device: torch.device):
        self.net = net
        self.video = video
        self.args = args
        self.device = device

    def _iterate_with_depth(self, graph: FactorGraph, steps: int, more_iters: bool):
        steps_preintr = steps // 2
        steps_postintr = steps - steps_preintr
        graph.update_batch(
            itrs=16 if more_iters else 8,
            steps=steps_preintr,
            optimize_intrinsics=self.args.optimize_intrinsics,
            optimize_rig_rotation=self.args.optimize_rig_rotation,
            solver_verbose=True,
        )
        self.video.update_disps_sens(self.depth_model, frame_idx=None)
        # Don't update intrinsics again!
        graph.update_batch(
            itrs=16 if more_iters else 8,
            steps=steps_postintr,
            optimize_intrinsics=False,
            optimize_rig_rotation=self.args.optimize_rig_rotation,
            solver_verbose=True,
        )

    def _iterate_without_depth(self, graph: FactorGraph, steps: int, more_iters: bool):
        graph.update_batch(
            itrs=16 if more_iters else 8,
            steps=steps,
            optimize_intrinsics=self.args.optimize_intrinsics,
            optimize_rig_rotation=self.args.optimize_rig_rotation,
            solver_verbose=True,
        )

    def _inject_loop_closure_edges(self, graph: FactorGraph, t: int) -> bool:
        """Add loop closure edges to the factor graph. Returns True if any were added."""
        if self.loop_closure is None:
            return False

        lc_tensors = self.loop_closure.get_loop_edges_tensors()
        if lc_tensors is None:
            return False

        ii_lc, jj_lc = lc_tensors
        valid = (ii_lc < t) & (jj_lc < t) & (ii_lc >= 0) & (jj_lc >= 0)
        if not valid.any():
            return False

        n_before = graph.ii.shape[0]
        graph.add_factors(ii_lc[valid], jj_lc[valid])
        n_added = graph.ii.shape[0] - n_before
        if n_added > 0:
            logger.info("Added %d loop closure edges to backend graph", n_added)
        return n_added > 0

    @torch.no_grad()
    def run(self, steps: int = 12, update_depth: bool = True, log: bool = False):
        """main update (reset GRU state)"""

        t = self.video.n_frames

        ec_cfg = getattr(self.args, "embedding_covisibility", None)
        ec_weight = float(getattr(ec_cfg, "weight", 0.0)) if ec_cfg and getattr(ec_cfg, "enabled", False) else 0.0
        graph = FactorGraph(
            self.net,
            self.video,
            self.device,
            max_factors=16 * t,
            incremental=False,
            cross_view=self.args.cross_view,
            use_semantic_flow_init=getattr(self.args, "use_semantic_flow_init", False),
            embedding_covisibility_weight=ec_weight,
        )

        graph.add_proximity_factors(
            rad=self.args.backend_radius,
            nms=self.args.backend_nms,
            thresh=self.args.backend_thresh,
            beta=self.args.beta,
        )

        has_loops = self._inject_loop_closure_edges(graph, t)

        if self.args.adaptive_cross_view:
            self.video.build_adaptive_cross_view_idx()

        if len(graph.ii) > 0:
            if has_loops:
                lc_cfg = getattr(self.args, "loop_closure", None)
                pg_steps = int(getattr(lc_cfg, "pose_graph_steps", 2)) if lc_cfg else 2
                pg_iters = int(getattr(lc_cfg, "pose_graph_iters", 4)) if lc_cfg else 4
                logger.info(
                    "Running pose-graph pre-optimization (%d steps, %d iters)",
                    pg_steps, pg_iters,
                )
                graph.update_batch(
                    itrs=pg_iters,
                    steps=pg_steps,
                    optimize_intrinsics=False,
                    optimize_rig_rotation=False,
                    motion_only=True,
                )

            more_iters = self.args.optimize_intrinsics or self.args.optimize_rig_rotation
            if self.depth_model is not None:
                self._iterate_with_depth(graph, steps, more_iters)
            else:
                self._iterate_without_depth(graph, steps, more_iters)
        else:
            # Empty graph with only one keyframe, assign sensor depth
            self.video.disps[0] = torch.where(
                self.video.disps_sens[0] > 0,
                self.video.disps_sens[0],
                self.video.disps[0],
            )

        self.video.dirty[:t] = True

        if log:
            self.video.log(self.args.map_filter_thresh)
            graph.log()

    @torch.no_grad()
    def run_if_necessary(self, steps: int = 12, log: bool = False):
        if self.args.optimize_intrinsics or self.args.optimize_rig_rotation:
            self.run(steps=steps, update_depth=True, log=log)
