"""Verify two-step SmolVLA Force FP16 inference on one fixed synthetic observation."""

import argparse
from pathlib import Path

import torch

from examples.smolvla_fp16.runtime import (
    DEFAULT_ACTION_STATS,
    enable_force_fp16,
    infer_actions,
    load_policy,
    prepare_inputs,
)
from lerobot.utils.constants import OBS_STATE


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", type=Path, default=Path("checkpoints/smolvla_base"))
    parser.add_argument("--backbone", type=Path, default=Path("checkpoints/SmolVLM2-500M-Video-Instruct"))
    parser.add_argument(
        "--repeats", type=int, default=1, help="Force FP16 runs with the same input and noise"
    )
    parser.add_argument("--compare-fp32", action="store_true", help="Also run a separate FP32 reference")
    parser.add_argument(
        "--action-stats-key", default=DEFAULT_ACTION_STATS, help="Checkpoint action statistics"
    )
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be at least 1")
    for directory in (args.ckpt, args.backbone):
        if not (directory / "config.json").is_file():
            parser.error(f"Missing {directory}/config.json")

    policy, pre, post = load_policy(args.ckpt, args.backbone, args.action_stats_key)
    cfg = policy.config
    rng = torch.Generator().manual_seed(42)
    state_feature = cfg.robot_state_feature
    if state_feature is None:
        raise ValueError("SmolVLA requires a state feature")
    observation = {
        **{
            key: torch.rand(feature.shape, generator=rng, dtype=torch.float16)
            for key, feature in cfg.image_features.items()
        },
        OBS_STATE: torch.randn(state_feature.shape, generator=rng, dtype=torch.float16),
    }
    task = "Pick up the block."
    noise = torch.randn((1, cfg.chunk_size, cfg.max_action_dim), generator=rng, dtype=torch.float16).to(
        cfg.device
    )
    if args.compare_fp32:
        policy.float()
        fp32_batch = pre({**{key: value.float() for key, value in observation.items()}, "task": task})
        reference = post(policy.predict_action_chunk(fp32_batch, noise=noise.float())).float()

    norm_count = enable_force_fp16(policy)
    batch = prepare_inputs(pre, observation, task)
    first_run = {}
    for run in range(1, args.repeats + 1):
        raw, actions, calls = infer_actions(policy, post, batch, noise)
        print(f"Run {run}: {tuple(actions.shape)}, {actions.dtype}, {calls} operator calls")
        for name, tensor in (("raw", raw), ("final actions", actions)):
            current = tensor.detach().cpu().contiguous()
            if run == 1:
                first_run[name] = current.clone()
                continue
            previous = first_run[name]
            identical = torch.equal(current.view(torch.uint8), previous.view(torch.uint8))
            maximum = (current.float() - previous.float()).abs().max().item()
            print(f"  {name} vs run 1: bit_exact={identical}, max_abs_diff={maximum:.6g}")
    print(f"Actions {tuple(actions.shape)}, {actions.dtype}, {norm_count} scaled RMSNorm modules")
    if args.compare_fp32:
        delta = (actions.float() - reference).abs()
        if not delta.isfinite().all():
            raise RuntimeError("Non-finite FP32 reference or comparison")
        print(f"Action absolute error: mean={delta.mean().item():.6g}, max={delta.max().item():.6g}")


if __name__ == "__main__":
    main()
