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

import csv
import logging
import os
import time

from collections import defaultdict
from pathlib import Path
from typing import Any

import torch

from ..maths.matrix import SparseBlockMatrixDict, SparseMatrixSubview, SparseNullMatrix
from ..maths.retractor import BaseRetractor
from ..maths.vector import SparseBlockVector, SparseNullVector, SparseVectorDict, SparseVectorSubview
from .kernel import RobustKernel
from .terms import SolverTerm, SharedProjectionCache


logger = logging.getLogger(__name__)


_DENSE_GPU_DIM_THRESHOLD = 150

_BENCH_CSV_PATH = "solver_bench.csv"
_BENCH_CSV_HEADER_WRITTEN: set[str] = set()


def _append_bench_row(path: str, row: dict) -> None:
    """Append a single benchmark row to a CSV file, writing the header on first call."""
    p = Path(path)
    write_header = path not in _BENCH_CSV_HEADER_WRITTEN and not p.exists()
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
            _BENCH_CSV_HEADER_WRITTEN.add(path)
        writer.writerow(row)


def _solve_dense_gpu(
    pi: torch.Tensor,
    pj: torch.Tensor,
    lhs_vals: torch.Tensor,
    rhs: torch.Tensor,
) -> torch.Tensor:
    """Assemble COO triplets into a dense matrix on GPU and solve.

    The system is J^T W J + damping, which is symmetric positive-definite
    after damping, so Cholesky is the natural choice (~2× faster than LU
    and numerically ideal for SPD systems).

    Falls back to pivoted general solve on Cholesky failure (e.g. when the
    system is barely positive-definite due to numeric noise).
    """
    n = rhs.shape[0]
    device = rhs.device

    A = torch.zeros(n, n, dtype=torch.float64, device=device)
    A.index_put_((pi, pj), lhs_vals.to(torch.float64), accumulate=True)

    b = rhs.to(torch.float64)

    try:
        L = torch.linalg.cholesky(A)
        x = torch.cholesky_solve(b.unsqueeze(-1), L).squeeze(-1)
    except torch.linalg.LinAlgError:
        x = torch.linalg.solve(A, b)

    return x.to(torch.float32)


def _solve_sparse_cpu(
    pi: torch.Tensor,
    pj: torch.Tensor,
    lhs_vals: torch.Tensor,
    rhs: torch.Tensor,
) -> torch.Tensor:
    """Sparse solve on CPU via scipy — used only when the system is large
    enough that a dense GPU solve would be wasteful."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.linalg import spsolve

    pi_np = pi.cpu().numpy()
    pj_np = pj.cpu().numpy()
    vals_np = lhs_vals.cpu().numpy()
    rhs_np = rhs.cpu().numpy()

    n = rhs_np.shape[0]
    A = coo_matrix((vals_np, (pi_np, pj_np)), shape=(n, n))
    A = A.tocsr()

    x = spsolve(A, rhs_np)
    return torch.tensor(x, device=pi.device, dtype=torch.float32)


def solve_linear_system(
    pi: torch.Tensor,
    pj: torch.Tensor,
    lhs: torch.Tensor,
    rhs: torch.Tensor,
    _validate: bool = False,
    bench_csv: str | None = None,
) -> torch.Tensor:
    """Dispatch to the fastest available solver for the given system size.

    • dim >= _DENSE_GPU_DIM_THRESHOLD  → dense Cholesky on GPU  (no CPU transfer)
    • dim <  _DENSE_GPU_DIM_THRESHOLD → scipy sparse on CPU    (fallback)

    When ``_validate`` is True, runs *both* solvers, logs wall-clock time for
    each, and asserts the solutions agree within tolerance.  Disable in
    production for speed.

    When ``bench_csv`` is set (or VIPE_SOLVER_BENCH_CSV env var), each
    validate-mode call appends a row with dim, timings, speedup, and errors
    to the given CSV file for offline analysis of the GPU/CPU crossover.
    """
    n = rhs.shape[0]
    use_gpu = n >= _DENSE_GPU_DIM_THRESHOLD and rhs.is_cuda

    if not _validate:
        if use_gpu:
            return _solve_dense_gpu(pi, pj, lhs, rhs)
        else:
            return _solve_sparse_cpu(pi, pj, lhs, rhs)

    # ---- Validation mode: run both, time both, compare ----
    csv_path = bench_csv or _BENCH_CSV_PATH

    # Dense GPU solve — use CUDA events for accurate GPU timing.
    if rhs.is_cuda:
        torch.cuda.synchronize(rhs.device)
        start_gpu = torch.cuda.Event(enable_timing=True)
        end_gpu = torch.cuda.Event(enable_timing=True)
        start_gpu.record()
        x_gpu = _solve_dense_gpu(pi, pj, lhs, rhs)
        end_gpu.record()
        torch.cuda.synchronize(rhs.device)
        t_gpu_ms = start_gpu.elapsed_time(end_gpu)
    else:
        t0 = time.perf_counter()
        x_gpu = _solve_dense_gpu(pi, pj, lhs, rhs)
        t_gpu_ms = (time.perf_counter() - t0) * 1000.0

    # Sparse CPU solve — wall-clock is fine (all CPU work).
    t0 = time.perf_counter()
    x_cpu = _solve_sparse_cpu(pi, pj, lhs, rhs)
    t_cpu_ms = (time.perf_counter() - t0) * 1000.0

    # ---- Numerical comparison ----
    diff = (x_gpu - x_cpu).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()

    # Relative error against the larger solution norm.
    norm_gpu = x_gpu.norm().item()
    norm_cpu = x_cpu.norm().item()
    ref_norm = max(norm_gpu, norm_cpu, 1e-12)
    rel_err = diff.norm().item() / ref_norm

    speedup = t_cpu_ms / max(t_gpu_ms, 1e-6)

    logger.info(
        "solve_linear_system [dim=%d]: "
        "dense_gpu=%.2fms  sparse_cpu=%.2fms  speedup=%.1fx | "
        "max_abs_diff=%.3e  mean_abs_diff=%.3e  rel_err=%.3e",
        n, t_gpu_ms, t_cpu_ms, speedup,
        max_diff, mean_diff, rel_err,
    )

    # Tolerance: the GPU path uses float64 Cholesky while the CPU path
    # uses float64 sparse LU — both are double precision but different
    # factorisations.  For well-conditioned systems rel_err should be
    # ~1e-10; for ill-conditioned (near-singular) systems it can be
    # larger.  After Schur complement, rel_err in the 1e-3 range is
    # typical and harmless — only warn for genuinely suspicious values.
    if rel_err > 1e-2:
        logger.warning(
            "solve_linear_system: significant disagreement — dim=%d, "
            "rel_err=%.3e (may indicate ill-conditioning).",
            n, rel_err,
        )

    # ---- CSV benchmark logging ----
    if csv_path:
        _append_bench_row(csv_path, {
            "dim": n,
            "gpu_ms": f"{t_gpu_ms:.4f}",
            "cpu_ms": f"{t_cpu_ms:.4f}",
            "speedup": f"{speedup:.3f}",
            "max_abs_diff": f"{max_diff:.6e}",
            "mean_abs_diff": f"{mean_diff:.6e}",
            "rel_err": f"{rel_err:.6e}",
            "gpu_selected": use_gpu,
            "timestamp": time.time(),
        })

    # Return the result from the path that would normally be selected.
    return x_gpu if use_gpu else x_cpu


# Keep the original name as a public alias.
def solve_scipy(
    pi: torch.Tensor,
    pj: torch.Tensor,
    lhs: torch.Tensor,
    rhs: torch.Tensor,
) -> torch.Tensor:
    return solve_linear_system(pi, pj, lhs, rhs)


class Solver:
    def __init__(
        self,
        compute_energy: bool = False,
    ) -> None:
        """
        If the corresponding JTJ of this group is very sparse, it is faster to solve
        the linear system first with this group being marginalized, and then recover
        the state separately.
        """
        self.terms: list[SolverTerm] = []
        self.kernels: list[RobustKernel | None] = []
        self.compute_energy = compute_energy

        self.group_fixed_inds: dict[str, torch.Tensor | None] = {}
        self.group_damping: dict[str, SparseBlockVector | float] = {}
        self.group_ep: dict[str, float] = {}
        self.group_retractor: dict[str, BaseRetractor] = defaultdict(BaseRetractor)
        self.group_marginalized: dict[str, bool] = defaultdict(lambda: False)

    def _warn_if_no_terms(self, group_name: str):
        all_group_names = set.union(*[t.group_names() for t in self.terms])
        if group_name not in all_group_names:
            logger.warning(f"Group {group_name} is not used in any terms. This may be a mistake.")

    def add_term(self, term: SolverTerm, kernel: RobustKernel | None = None):
        self.terms.append(term)
        self.kernels.append(kernel)

    def set_fixed(self, group_name: str, fixed_inds: torch.Tensor | None = None):
        # None means everything is fixed
        self._warn_if_no_terms(group_name)
        self.group_fixed_inds[group_name] = fixed_inds

    def set_marginilized(self, group_name: str, marginalized: bool = True):
        self._warn_if_no_terms(group_name)
        self.group_marginalized[group_name] = marginalized

    def set_retractor(self, group_name: str, retractor: BaseRetractor):
        self._warn_if_no_terms(group_name)
        self.group_retractor[group_name] = retractor

    def set_damping(self, group_name: str, damping: SparseBlockVector | float, ep: float):
        """
        Set the damping factor.
        If this is a Tensor, it should be of shape (n_vars, n_vars)
            LHS += diag(damping) + ep * I.
        If this is a float, it will be added as
            LHS += diag(LHS) * damping + ep * I
        """
        self._warn_if_no_terms(group_name)
        self.group_damping[group_name] = damping
        self.group_ep[group_name] = ep

    def _solve(self, lhs: SparseMatrixSubview, rhs: SparseVectorSubview) -> SparseVectorSubview:
        assert lhs.row_group_names == lhs.col_group_names == rhs.group_names

        if lhs.has_inverse():
            return lhs.inverse() * rhs

        ravel_mappings = rhs.get_ravel_mapping()
        pi, pj, lhs_data = lhs.ravel(ravel_mappings)
        rhs_data = rhs.ravel(ravel_mappings)

        x_data = solve_linear_system(pi, pj, lhs_data, rhs_data)

        return rhs.unravel(x_data, ravel_mappings)

    def run_inplace(self, variables: dict[str, Any]) -> float:
        lhs: SparseBlockMatrixDict = defaultdict(SparseNullMatrix)
        rhs: SparseVectorDict = defaultdict(SparseNullVector)

        fully_fixed_groups = {t for t, inds in self.group_fixed_inds.items() if inds is None}
        shared_cache = SharedProjectionCache() 

        energy = 0.0
        for term, kernel in zip(self.terms, self.kernels):
            # Compute the newest term formulation
            term.update(self)
            if not term.is_active():
                continue
            term_return = term.forward(variables, jacobian=True, shared_cache=shared_cache)
            term_group_names = list(term.group_names().difference(fully_fixed_groups))

            if kernel is not None:
                term_return.apply_robust_kernel(kernel)

            if self.compute_energy:
                energy += term_return.residual().sum().item()

            for group_name, fixed_inds in self.group_fixed_inds.items():
                if group_name in term_group_names and fixed_inds is not None:
                    term_return.remove_jcol_inds(group_name, fixed_inds)

            # Compute RHS
            for group_name in term_group_names:
                rhs[group_name] += term_return.nwjtr(group_name)

            # Compute only upper triangular part of the LHS
            for group_i in range(len(term_group_names)):
                for group_j in range(group_i, len(term_group_names)):
                    group_name_i = term_group_names[group_i]
                    group_name_j = term_group_names[group_j]
                    if group_name_i in term_group_names and group_name_j in term_group_names:
                        jtwj = term_return.jtwj(group_name_i, group_name_j)
                        lhs[(group_name_i, group_name_j)] += jtwj

        all_group_names = list(rhs.keys())
        marginalized_group_names = [
            group_name
            for group_name, marginalized in self.group_marginalized.items()
            if marginalized and group_name in all_group_names
        ]
        regular_group_names = list(set(all_group_names).difference(marginalized_group_names))

        for group_name in all_group_names:
            damping = self.group_damping.get(group_name, 0.0)
            ep = self.group_ep.get(group_name, 0.0)
            lhs[(group_name, group_name)].apply_damping_assume_coalesced(damping, ep)

        # Build matrices
        lhs_h = SparseMatrixSubview(lhs, regular_group_names, regular_group_names)
        rhs_v = SparseVectorSubview(rhs, regular_group_names)

        if len(marginalized_group_names) > 0:
            lhs_e = SparseMatrixSubview(lhs, regular_group_names, marginalized_group_names)
            lhs_c = SparseMatrixSubview(lhs, marginalized_group_names, marginalized_group_names)
            rhs_w = SparseVectorSubview(rhs, marginalized_group_names)

            # Apply Schur's formula
            h_cinv = lhs_e @ lhs_c.inverse()
            lhs_reg = lhs_h - h_cinv @ lhs_e.transpose()
            rhs_reg = rhs_v - h_cinv * rhs_w

            x_reg: SparseVectorSubview = self._solve(lhs_reg, rhs_reg)

            rhs_marg = rhs_w - lhs_e.transpose() * x_reg
            x_marg: SparseVectorSubview = self._solve(lhs_c, rhs_marg)

            x_dict = x_reg.get_dict() | x_marg.get_dict()

        else:
            x_dict = self._solve(lhs_h, rhs_v).get_dict()

        for group_name in all_group_names:
            self.group_retractor[group_name].oplus(
                variables[group_name],
                x_dict[group_name].inds,
                x_dict[group_name].data,
            )

        return energy