"""Compare SmolVLA FP32 and Force FP16 on one fixed synthetic observation, offline."""

import argparse
from pathlib import Path
from types import MethodType

import torch
from transformers.models.llama.modeling_llama import LlamaRMSNorm

from examples.groot_fp16.full_fp16_experiment import full_fp16
from lerobot.policies.common.fp16 import fp16_safe_rms_norm
from lerobot.policies.smolvla import SmolVLAConfig, SmolVLAPolicy
from lerobot.processor import PolicyProcessorPipeline, load_pretrained_policy_processors
from lerobot.utils.constants import OBS_STATE


def scaled_rms_norm(layer: LlamaRMSNorm, x: torch.Tensor) -> torch.Tensor:
    return fp16_safe_rms_norm(x, layer.weight, layer.variance_epsilon)


def load_policy(
    ckpt: Path, backbone: Path
) -> tuple[SmolVLAPolicy, PolicyProcessorPipeline, PolicyProcessorPipeline]:
    cfg = SmolVLAConfig.from_pretrained(ckpt)
    cfg.device = "cuda"
    cfg.vlm_model_name = str(backbone.resolve())
    cfg.load_vlm_weights = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    policy = SmolVLAPolicy.from_pretrained(ckpt, config=cfg, strict=True).float().eval()
    pre, post = load_pretrained_policy_processors(
        ckpt,
        preprocessor_overrides={
            "tokenizer_processor": {"tokenizer_name": cfg.vlm_model_name},
            "device_processor": {"device": cfg.device},
        },
    )
    return policy, pre, post


def enable_force_fp16(policy: SmolVLAPolicy) -> int:
    policy.half()
    norms = [layer for layer in policy.modules() if isinstance(layer, LlamaRMSNorm)]
    if not norms:
        raise RuntimeError("No LlamaRMSNorm modules found in the SmolVLA checkpoint")
    for norm in norms:
        norm.forward = MethodType(scaled_rms_norm, norm)
    return len(norms)


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", type=Path, default=Path("checkpoints/smolvla_base"))
    parser.add_argument("--backbone", type=Path, default=Path("checkpoints/SmolVLM2-500M-Video-Instruct"))
    parser.add_argument(
        "--repeats", type=int, default=1, help="Force FP16 runs with the same input and noise"
    )
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be at least 1")
    for directory in (args.ckpt, args.backbone):
        if not (directory / "config.json").is_file():
            parser.error(f"Missing {directory}/config.json")

    policy, pre, post = load_policy(args.ckpt, args.backbone)
    cfg = policy.config
    rng = torch.Generator().manual_seed(42)
    state_feature = cfg.robot_state_feature
    if state_feature is None:
        raise ValueError("SmolVLA requires a state feature")
    observation = {
        **{key: torch.rand(feature.shape, generator=rng) for key, feature in cfg.image_features.items()},
        OBS_STATE: torch.randn(state_feature.shape, generator=rng),
        "task": "Pick up the block.",
    }
    batch = pre(observation)
    noise = torch.randn((1, cfg.chunk_size, cfg.max_action_dim), generator=rng).to(cfg.device)
    reference = post(policy.predict_action_chunk(dict(batch), noise=noise)).float()
    norm_count = enable_force_fp16(policy)

    first_run = {}
    for run in range(1, args.repeats + 1):
        policy.reset()
        with full_fp16() as audit:
            actions = policy.predict_action_chunk(dict(batch), noise=noise)
        candidate = post(actions.float()).float()
        print(f"Run {run}: {tuple(actions.shape)}, {actions.dtype}, {sum(audit.ops.values())} operator calls")
        for name, tensor in (("raw", actions), ("decoded", candidate)):
            current = tensor.detach().cpu().contiguous()
            if not current.isfinite().all():
                raise RuntimeError(f"Non-finite {name} actions in run {run}")
            if run == 1:
                first_run[name] = current.clone()
                continue
            previous = first_run[name]
            identical = torch.equal(current.view(torch.uint8), previous.view(torch.uint8))
            maximum = (current.float() - previous.float()).abs().max().item()
            print(f"  {name} vs run 1: bit_exact={identical}, max_abs_diff={maximum:.6g}")
    delta = (candidate - reference).abs()
    if not delta.isfinite().all():
        raise RuntimeError("Non-finite actions in the FP32 or Force FP16 result")
    print(f"Actions {tuple(actions.shape)}, {actions.dtype}, {norm_count} scaled RMSNorm modules")
    print(f"Decoded action absolute error: mean={delta.mean().item():.6g}, max={delta.max().item():.6g}")


if __name__ == "__main__":
    main()
