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

"""Pure-FP16 compute for the GR00T N1.7 Qwen3-VL backbone.

Transformers' Qwen3-VL keeps two FP32 islands even when the model runs in FP16: the text RMSNorm
upcasts its input to FP32 before computing the variance, and both the text and the vision RoPE build
cos/sin (and, for vision, apply the rotation) in FP32. ``enable_pure_fp16`` changes those operations
to the model dtype. Other FP32 operations can remain, including vision positional interpolation
and action time embeddings. The Force FP16 experimental context in ``examples/groot_fp16`` also
enforces the dtype of those remaining explicit floating tensor operations.

Simply dropping the RMSNorm upcast is not usable: the GR00T N1.7 backbone carries a massive activation
(|x| ~ 15300 on the first token, channel 1793, from layer 12 on, independent of the input), and x^2
overflows FP16 (max 65504), which zeroes the normalised token. RMSNorm is scale invariant, so the
FP16-safe version normalises ``x / max|x|`` instead: the squares stay <= 1 and nothing overflows, while
the result equals ``w * x * rsqrt(mean(x^2) + eps)`` in exact arithmetic.

Measured on 100 synthetic DROID cases against FP32 (action error relative to each dimension's
q01-q99 range): naive FP16 RMSNorm 98/100 cases above 10%; this module 0/100, max 4.78%, median 0.22%,
run-to-run bit exact. Reduction / GEMM / SDPA accumulators inside the kernels are not affected.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING

import torch
from torch import nn

from lerobot.utils.import_utils import _transformers_available, require_package

if TYPE_CHECKING or _transformers_available:
    import transformers.models.qwen3_vl.modeling_qwen3_vl as _qwen3_vl
else:
    _qwen3_vl = None


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


def _fp16_safe_rms_norm_forward(self: nn.Module, hidden_states: torch.Tensor) -> torch.Tensor:
    return fp16_safe_rms_norm(hidden_states, self.weight, self.variance_epsilon)


@torch.no_grad()
def _text_rope_forward(self: nn.Module, x: torch.Tensor, position_ids: torch.Tensor):
    # Same as Qwen3VLTextRotaryEmbedding.forward, with frequencies and cos/sin in the model dtype.
    position_ids = position_ids.expand(3, -1, -1)
    freqs = position_ids[..., None].to(x.dtype) * self.inv_freq.to(device=x.device, dtype=x.dtype)
    cos = self.recomposition_frequencies(freqs.cos() * self.attention_scaling)
    sin = self.recomposition_frequencies(freqs.sin() * self.attention_scaling)
    return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


@torch.no_grad()
def _vision_rope_forward(self: nn.Module, x: torch.Tensor, position_ids: torch.Tensor):
    # Same as Qwen3VLVisionRotaryEmbedding.forward, with frequencies and cos/sin in the model dtype.
    freqs = position_ids[..., None].to(x.dtype) * self.inv_freq.to(device=x.device, dtype=x.dtype)
    cos = self.recomposition_frequencies(freqs.cos() * self.attention_scaling)
    sin = self.recomposition_frequencies(freqs.sin() * self.attention_scaling)
    return cos, sin


def _apply_rotary_pos_emb_vision_same_dtype(q, k, cos, sin):
    # Same as transformers' apply_rotary_pos_emb_vision without the q/k/cos/sin FP32 upcast.
    rotate_half = _qwen3_vl.rotate_half
    cos, sin = cos.unsqueeze(-2).to(q.dtype), sin.unsqueeze(-2).to(q.dtype)
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)


_PURE_FP16_CLASSES: dict[type, type] = {}


def _pure_fp16_class(cls: type, forward) -> type:
    """Subclass of ``cls`` with ``forward`` replaced; cached so each original class maps to one subclass."""
    if cls not in _PURE_FP16_CLASSES:
        _PURE_FP16_CLASSES[cls] = type(f"PureFP16{cls.__name__}", (cls,), {"forward": forward})
    return _PURE_FP16_CLASSES[cls]


def enable_pure_fp16(module: nn.Module) -> int:
    """Switch the Qwen3-VL RMSNorm and RoPE modules under ``module`` to pure model-dtype compute.

    Only the instances inside ``module`` change (their class is swapped for a subclass), so other
    Qwen3-VL models in the same process keep the stock behaviour. The vision RoPE application is a
    module-level function in transformers; run the backbone forward inside
    :func:`pure_fp16_vision_rope` to cover it as well. Returns the number of modules switched.
    """
    require_package("transformers", extra="groot")
    replacements = {
        _qwen3_vl.Qwen3VLTextRMSNorm: _fp16_safe_rms_norm_forward,
        _qwen3_vl.Qwen3VLTextRotaryEmbedding: _text_rope_forward,
        _qwen3_vl.Qwen3VLVisionRotaryEmbedding: _vision_rope_forward,
    }
    switched = 0
    for submodule in module.modules():
        for cls, forward in replacements.items():
            if type(submodule) is cls:
                submodule.__class__ = _pure_fp16_class(cls, forward)
                switched += 1
    return switched


@contextmanager
def pure_fp16_vision_rope() -> Iterator[None]:
    """Apply the vision RoPE rotation in the model dtype for the duration of the context.

    transformers calls ``apply_rotary_pos_emb_vision`` as a module global, so this temporarily
    replaces that global and restores it on exit.
    """
    if _qwen3_vl is None:
        yield
        return
    original = _qwen3_vl.apply_rotary_pos_emb_vision
    _qwen3_vl.apply_rotary_pos_emb_vision = _apply_rotary_pos_emb_vision_same_dtype
    try:
        yield
    finally:
        _qwen3_vl.apply_rotary_pos_emb_vision = original
