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

from abc import ABC, abstractmethod

import torch
import numpy as np


class RobustKernel:
    @abstractmethod
    def apply(self, x: torch.Tensor) -> torch.Tensor:
        pass


class HuberRobustKernel(RobustKernel):
    def apply(self, x: torch.Tensor) -> torch.Tensor:
        weights = torch.ones_like(x)
        s = x * x
        weights[s > 1] = 1 / torch.sqrt(s)[s > 1]
        return weights


class AdaptiveBarronRobustKernel(RobustKernel):
    def __init__(self, alpha_init: float = 2.0, c: float = 1.0,
                 alpha_range=(-8, 2), num_alpha=50, N=10, steps=2000):
        """
        Args:
            alpha_init: starting alpha value
            c: scale parameter
            alpha_range: (min, max) range of alphas for lookup
            num_alpha: number of alpha samples in lookup table
            N: truncation bound for Z(alpha) integral [-N, N]
            steps: number of integration steps
        """
        self.alpha = alpha_init
        self.c = c
        self.alpha_range = alpha_range
        self.num_alpha = num_alpha
        self.N = N
        self.steps = steps

    def apply(self, x: torch.Tensor, alpha: torch.Tensor | None = None) -> torch.Tensor:
        """
        Compute IRLS weights for residuals x.

        Supports both scalar alpha (single value for all residuals)
        and tensor alpha (per-residual adaptive alpha).
        Handles the special case alpha == 2 (L2 loss) stably.
        Also handles alpha == 0 (Cauchy loss) correctly.
        """
        if alpha is None:
            alpha = torch.tensor(self.alpha, device=x.device, dtype=x.dtype)  # scalar case
        
        # Ensure alpha has same shape as x for broadcasting
        if alpha.dim() == 0:
            alpha = alpha.expand_as(x)
        
        s = (x / self.c) ** 2  # (r/c)^2
        
        # numerical safety epsilon
        eps = 1e-8
        
        # Create masks for special cases
        alpha_is_2 = torch.isclose(alpha, torch.tensor(2.0, device=x.device, dtype=x.dtype), atol=1e-6)
        alpha_is_0 = torch.isclose(alpha, torch.tensor(0.0, device=x.device, dtype=x.dtype), atol=1e-6)
        
        # For general case (alpha != 2 and alpha != 0)
        # Weight formula: w = (1/c²) * ((s/|α-2| + 1)^(α/2 - 1))
        abs_alpha_minus_2 = torch.abs(alpha - 2.0)
        # Add eps to avoid division by zero in general case
        abs_alpha_minus_2_safe = torch.where(alpha_is_2 | alpha_is_0, torch.ones_like(abs_alpha_minus_2), abs_alpha_minus_2 + eps)
        
        # Compute general case weight
        general_weight = (1.0 / self.c**2) * (s / abs_alpha_minus_2_safe + 1.0) ** (alpha / 2.0 - 1.0)
        
        # For alpha == 2 (L2 loss): w = 1
        l2_weight = torch.full_like(general_weight, 1.0)
        
        # For alpha == 0 (Cauchy loss): w = 1/(1 + s/2) = 2/(2 + s)
        cauchy_weight = 2.0 / (2.0 + s)
        
        # Combine using where
        weights = torch.where(alpha_is_2, l2_weight, general_weight)
        weights = torch.where(alpha_is_0, cauchy_weight, weights)
        weights = torch.sqrt(weights)
        return weights