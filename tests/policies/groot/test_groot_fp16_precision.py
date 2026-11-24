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

import pytest
import torch

pytest.importorskip("transformers")

import transformers.models.qwen3_vl.modeling_qwen3_vl as qwen3_vl  # noqa: E402
from transformers import Qwen3VLTextConfig, Qwen3VLVisionConfig  # noqa: E402

from lerobot.policies.groot.configuration_groot import GrootConfig  # noqa: E402
from lerobot.policies.groot.fp16_precision import (  # noqa: E402
    enable_pure_fp16,
    fp16_safe_rms_norm,
    pure_fp16_vision_rope,
)


def _fp32_rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    h = x.float()
    return weight.float() * (h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + eps))


def _hidden_states_with_massive_activation() -> torch.Tensor:
    torch.manual_seed(0)
    x = torch.randn(3, 7, 256) * 3
    x[0, 0, 5] = 15304.0  # magnitude seen in the GR00T N1.7 backbone; x^2 overflows FP16
    x[1, 2] *= 1e-3  # tiny row
    x[2, 3] = 0.0  # all-zero row
    return x.half()


def test_fp16_safe_rms_norm_matches_fp32_with_massive_activation():
    x = _hidden_states_with_massive_activation()
    weight = (torch.rand(256) + 0.5).half()
    reference = _fp32_rms_norm(x, weight, 1e-6)

    out = fp16_safe_rms_norm(x, weight, 1e-6)

    assert out.dtype == torch.float16
    assert torch.isfinite(out).all()
    torch.testing.assert_close(out.float(), reference, rtol=2e-3, atol=2e-3)


def test_naive_fp16_rms_norm_loses_the_massive_activation_token():
    # Documents why the scaled form is needed: the FP16 variance overflows and zeroes the token.
    x = _hidden_states_with_massive_activation()
    naive = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)

    assert torch.count_nonzero(naive[0, 0]) == 0


def test_enable_pure_fp16_switches_only_the_given_module():
    text_config = Qwen3VLTextConfig(
        hidden_size=64,
        num_attention_heads=4,
        num_key_value_heads=4,
        head_dim=16,
        intermediate_size=64,
        num_hidden_layers=1,
        vocab_size=32,
    )
    model = torch.nn.ModuleDict(
        {
            "norm": qwen3_vl.Qwen3VLTextRMSNorm(256),
            "rope": qwen3_vl.Qwen3VLTextRotaryEmbedding(config=text_config),
        }
    ).half()
    untouched = qwen3_vl.Qwen3VLTextRMSNorm(256).half()

    assert enable_pure_fp16(model) == 2
    assert enable_pure_fp16(model) == 0  # idempotent

    assert isinstance(model["norm"], qwen3_vl.Qwen3VLTextRMSNorm)
    assert type(model["norm"]) is not qwen3_vl.Qwen3VLTextRMSNorm
    assert type(untouched) is qwen3_vl.Qwen3VLTextRMSNorm

    x = _hidden_states_with_massive_activation()
    torch.testing.assert_close(model["norm"](x), fp16_safe_rms_norm(x, model["norm"].weight, 1e-6))

    hidden = torch.randn(1, 5, 64).half()
    position_ids = torch.arange(5)[None, None].expand(3, 1, 5)
    cos, sin = model["rope"](hidden, position_ids)
    assert cos.dtype == sin.dtype == torch.float16


def test_pure_fp16_vision_rope_runs_in_model_dtype_and_restores_the_global():
    vision_config = Qwen3VLVisionConfig(
        hidden_size=32, num_heads=2, depth=1, intermediate_size=32, out_hidden_size=64
    )
    rope = qwen3_vl.Qwen3VLVisionRotaryEmbedding(vision_config)
    enable_pure_fp16(rope)
    hidden = torch.randn(6, 32).half()
    position_ids = torch.stack(torch.meshgrid(torch.arange(2), torch.arange(3), indexing="ij"), -1).view(6, 2)
    cos, sin = rope(hidden, position_ids)
    assert cos.dtype == torch.float16

    q = torch.randn(6, 2, 16).half()
    k = torch.randn(6, 2, 16).half()
    original = qwen3_vl.apply_rotary_pos_emb_vision
    expected_q, expected_k = original(q, k, cos, sin)

    with pure_fp16_vision_rope():
        assert qwen3_vl.apply_rotary_pos_emb_vision is not original
        q_embed, k_embed = qwen3_vl.apply_rotary_pos_emb_vision(q, k, cos, sin)

    assert qwen3_vl.apply_rotary_pos_emb_vision is original
    assert q_embed.dtype == k_embed.dtype == torch.float16
    torch.testing.assert_close(q_embed, expected_q, rtol=5e-3, atol=5e-3)
    torch.testing.assert_close(k_embed, expected_k, rtol=5e-3, atol=5e-3)


def test_groot_config_fp16_options_are_validated():
    config = GrootConfig(dtype=torch.float16, use_bf16=False, pure_fp16=True)
    assert config.pure_fp16

    with pytest.raises(ValueError, match="use_bf16=False"):
        GrootConfig(dtype=torch.float16, use_bf16=True)
    with pytest.raises(ValueError, match="pure_fp16=True requires dtype=torch.float16"):
        GrootConfig(dtype=torch.float32, pure_fp16=True)
