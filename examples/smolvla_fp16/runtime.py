"""Shared SmolVLA two-step Force FP16 policy and processor setup."""

import hashlib
from pathlib import Path
from types import MethodType

import torch
from transformers.models.llama.modeling_llama import LlamaRMSNorm

from examples.groot_fp16.full_fp16_experiment import full_fp16
from lerobot.policies.common.fp16 import fp16_safe_rms_norm
from lerobot.policies.smolvla import SmolVLAConfig, SmolVLAPolicy
from lerobot.processor import (
    PolicyProcessorPipeline,
    UnnormalizerProcessorStep,
    hotswap_stats,
    load_pretrained_policy_processors,
)
from lerobot.utils.constants import ACTION

DEFAULT_ACTION_STATS = "so100.buffer.action"


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(2**20), b""):
            h.update(chunk)
    return h.hexdigest()


def scaled_rms_norm(layer: LlamaRMSNorm, x: torch.Tensor) -> torch.Tensor:
    return fp16_safe_rms_norm(x, layer.weight, layer.variance_epsilon)


def load_policy(
    ckpt: Path, backbone: Path, action_stats_key: str = DEFAULT_ACTION_STATS
) -> tuple[SmolVLAPolicy, PolicyProcessorPipeline, PolicyProcessorPipeline]:
    cfg = SmolVLAConfig.from_pretrained(ckpt)
    cfg.num_steps = 2
    cfg.device = "cuda"
    cfg.vlm_model_name = str(backbone.resolve())
    cfg.load_vlm_weights = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    policy = SmolVLAPolicy.from_pretrained(ckpt, config=cfg, strict=True).eval()
    pre, post = load_pretrained_policy_processors(
        ckpt,
        preprocessor_overrides={
            "tokenizer_processor": {"tokenizer_name": cfg.vlm_model_name},
            "device_processor": {"device": cfg.device},
        },
    )
    # The base checkpoint namespaces its statistics by robot, but the processor looks up "action".
    step = next((step for step in post.steps if isinstance(step, UnnormalizerProcessorStep)), None)
    if step is None or not step.stats or action_stats_key not in step.stats:
        raise ValueError(f"Missing checkpoint unnormalization statistics for {action_stats_key!r}")
    stats = step.stats[action_stats_key]
    feature = cfg.action_feature
    if feature is None or any(
        k not in stats or stats[k].shape != tuple(feature.shape) for k in ("mean", "std")
    ):
        raise ValueError(
            f"Expected one mean and standard deviation per action dimension in {action_stats_key}"
        )
    post = hotswap_stats(post, {**step.stats, ACTION: stats})
    return policy, pre, post


def enable_force_fp16(policy: SmolVLAPolicy) -> int:
    torch.backends.cuda.matmul.allow_fp16_accumulation = True
    torch.backends.cuda.allow_fp16_bf16_reduction_math_sdp(True)
    policy.half()
    norms = [layer for layer in policy.modules() if isinstance(layer, LlamaRMSNorm)]
    if not norms:
        raise RuntimeError("No LlamaRMSNorm modules found in the SmolVLA checkpoint")
    for norm in norms:
        norm.forward = MethodType(scaled_rms_norm, norm)
    return len(norms)


def prepare_inputs(
    pre: PolicyProcessorPipeline, observation: dict[str, torch.Tensor], task: str
) -> dict[str, torch.Tensor]:
    with full_fp16():
        prepared = pre({**observation, "task": task})
    return {key: value for key, value in prepared.items() if isinstance(value, torch.Tensor)}


def infer_actions(
    policy: SmolVLAPolicy, post: PolicyProcessorPipeline, batch: dict[str, torch.Tensor], noise: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, int]:
    if noise.dtype != torch.float16 or any(
        value.is_floating_point() and value.dtype != torch.float16 for value in batch.values()
    ):
        raise ValueError("Expected FP16 noise and floating-point policy inputs")
    policy.reset()
    with full_fp16() as audit:
        raw = policy.predict_action_chunk(dict(batch), noise=noise)
        actions = post(raw)
    for value in (raw, actions):
        if value.dtype != torch.float16 or not value.isfinite().all():
            raise ValueError("Expected finite FP16 actions")
    return raw, actions, sum(audit.ops.values())
