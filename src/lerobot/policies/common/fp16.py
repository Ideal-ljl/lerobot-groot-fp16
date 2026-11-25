#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch


def fp16_safe_rms_norm(hidden_states: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """RMSNorm computed in the input dtype without overflowing FP16.

    ``out = w * y * rsqrt(mean(y^2) + eps / s^2)`` with ``s = max|x|`` per row and ``y = x / s``.
    ``eps / s^2`` is formed as ``(sqrt(eps) / s)^2`` so that ``s^2`` is never materialised.
    """
    scale = hidden_states.abs().amax(dim=-1, keepdim=True)
    scale = torch.where(scale > 0, scale, torch.ones_like(scale))  # all-zero rows: plain eps path
    normed = hidden_states / scale
    variance = normed.pow(2).mean(-1, keepdim=True)
    scaled_eps = (torch.full_like(scale, eps**0.5) / scale).pow(2)
    return weight * (normed * torch.rsqrt(variance + scaled_eps))
