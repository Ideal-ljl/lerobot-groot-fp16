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

"""FP16 accuracy check of GR00T N1.7 on the current device (e.g. Jetson Orin NX).

Runs a fixed set of synthetic DROID cases (seeded, CPU-generated, so identical on every machine) through
GrootPolicy in the requested precision modes and compares the actions with the FP32 reference recorded on
an A100 (``reference_a100.npz``):

  fp32       dtype=float32                    reference precision (needs ~13 GB for the weights)
  fp16       dtype=float16                    transformers default: RMSNorm / RoPE still upcast to FP32
  fp16_pure  dtype=float16, pure_fp16=True    every backbone op in FP16, overflow-safe scaled RMSNorm

Error per action dimension is |a - ref| / (q99 - q01) in %, E = mean, M = max over a case. A case
violates when M > --limit-pct (10%). ``fp16_pure`` passes when no case violates and all are finite.
"""

import argparse
import gc
import hashlib
import json
import platform
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import transformers
import transformers.models.qwen3_vl.modeling_qwen3_vl as qwen3_vl

from lerobot.policies.groot.modeling_groot import GrootPolicy
from lerobot.policies.groot.processor_groot import make_groot_pre_post_processors

HERE = Path(__file__).resolve().parent
MODES = {
    "fp32": {"dtype": torch.float32, "use_bf16": False},
    "fp16": {"dtype": torch.float16, "use_bf16": False},
    "fp16_pure": {"dtype": torch.float16, "use_bf16": False, "pure_fp16": True},
}
TASKS = [
    "pick up the red cup",
    "put the marker in the bowl",
    "close the drawer",
    "open the cabinet door",
    "move the block to the left",
    "wipe the table with the towel",
]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", required=True, help="local GR00T-N1.7-3B directory (nvidia/GR00T-N1.7-3B)")
    p.add_argument("--out", required=True, help="output directory for summary.json / actions.npz")
    p.add_argument("--modes", nargs="+", default=["fp16", "fp16_pure"], choices=list(MODES))
    p.add_argument("--reference", default=str(HERE / "reference_a100.npz"))
    p.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    p.add_argument("--embodiment-tag", default="oxe_droid_relative_eef_relative_joint")
    p.add_argument("--cameras", nargs="+", default=["exterior_image_1_left", "wrist_image_left"])
    p.add_argument("--cases", type=int, default=100, help="first N of the 100 reference cases")
    p.add_argument("--repeats", type=int, default=2, help="runs per FP16 mode, to check determinism")
    p.add_argument("--limit-pct", type=float, default=10.0)
    p.add_argument("--seed", type=int, default=20261123)
    return p.parse_args()


def smooth_image(g: torch.Generator) -> torch.Tensor:
    low = torch.rand(1, 3, 8, 8, generator=g)
    img = torch.nn.functional.interpolate(low, size=(256, 256), mode="bicubic", align_corners=False)
    return (img + 0.05 * torch.rand(1, 3, 256, 256, generator=g)).clamp(0, 1)


def make_cases(n, s_lo, s_hi, action_horizon, max_action_dim, cameras, seed):
    """Synthetic observations (state uniform in [q01, q99], smooth random images) and FP16 noise."""
    g = torch.Generator().manual_seed(seed)
    cases, digests = [], []
    for i in range(n):
        u = torch.rand(1, len(s_lo), generator=g, dtype=torch.float64).numpy()
        batch = {
            "observation.state": torch.from_numpy(s_lo + u * (s_hi - s_lo)).float(),
            "task": [TASKS[i % len(TASKS)]],
        }
        for cam in cameras:
            batch[f"observation.images.{cam}"] = smooth_image(g)
        noise = torch.randn(1, action_horizon, max_action_dim, generator=g).to(torch.float16)
        h = hashlib.sha256(batch["task"][0].encode())
        for key in ["observation.state", *[f"observation.images.{c}" for c in cameras]]:
            h.update(batch[key].numpy().tobytes())
        h.update(noise.numpy().tobytes())
        cases.append((batch, noise))
        digests.append(h.hexdigest())
    return cases, digests


def sync(device: str):
    if device.startswith("cuda"):
        torch.cuda.synchronize()


def main():
    args = parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    original_vision_rope = qwen3_vl.apply_rotary_pos_emb_vision

    stats = json.loads((Path(args.ckpt) / "statistics.json").read_text())[args.embodiment_tag]

    def cat(d, k):
        return np.concatenate([np.asarray(v[k], dtype=np.float64) for v in d.values()])

    dim_range = np.maximum(cat(stats["action"], "q99") - cat(stats["action"], "q01"), 1e-6)
    s_lo, s_hi = cat(stats["state"], "q01"), cat(stats["state"], "q99")
    ckpt_cfg = json.loads((Path(args.ckpt) / "config.json").read_text())
    cases, digests = make_cases(
        args.cases,
        s_lo,
        s_hi,
        ckpt_cfg["action_horizon"],
        ckpt_cfg["max_action_dim"],
        args.cameras,
        args.seed,
    )

    ref = np.load(args.reference) if Path(args.reference).exists() else None
    inputs_match = None
    if ref is not None:
        ref_digests = [str(d) for d in ref["input_sha256"][: args.cases]]
        inputs_match = int(sum(a == b for a, b in zip(digests, ref_digests, strict=True)))
        print(f"inputs identical to the reference run: {inputs_match}/{args.cases} cases", flush=True)

    def err(a, b):
        if not (np.isfinite(a).all() and np.isfinite(b).all()):
            return {"E": float("nan"), "M": float("nan"), "finite": False}
        rel = np.abs(a.astype(np.float64) - b.astype(np.float64)) / dim_range[None, : a.shape[-1]] * 100
        return {"E": float(rel.mean()), "M": float(rel.max()), "finite": True}

    def summarize(a, b):
        errs = [err(a[i], b[i]) for i in range(len(a))]
        fin = [e for e in errs if e["finite"]]
        ms = np.array([e["M"] for e in fin]) if fin else np.array([np.nan])
        return {
            "nonfinite_cases": len(errs) - len(fin),
            "E_mean": float(np.mean([e["E"] for e in fin])) if fin else float("nan"),
            "M_median": float(np.median(ms)),
            "M_p95": float(np.percentile(ms, 95)),
            "M_max": float(ms.max()),
            "violations_M_gt_limit": int((ms > args.limit_pct).sum()) + len(errs) - len(fin),
        }

    actions, summary = (
        {},
        {
            "config": vars(args),
            "environment": {
                "platform": platform.platform(),
                "python": platform.python_version(),
                "torch": torch.__version__,
                "transformers": transformers.__version__,
                "device_name": torch.cuda.get_device_name(args.device)
                if args.device.startswith("cuda")
                else platform.processor(),
            },
            "inputs_identical_to_reference": inputs_match,
            "modes": {},
        },
    )
    for name in args.modes:
        t0 = time.time()
        policy = GrootPolicy.from_pretrained(
            args.ckpt, strict=False, embodiment_tag=args.embodiment_tag, device=args.device, **MODES[name]
        )
        policy.to(args.device).eval()
        pre, post = make_groot_pre_post_processors(config=policy.config, dataset_stats=None)
        backbone = policy._groot_model.backbone
        entry = {
            "load_s": round(time.time() - t0, 1),
            "param_dtype": str(next(policy.parameters()).dtype),
            "backbone_pure_fp16": bool(backbone.pure_fp16),
            "rmsnorm_classes": sorted(
                {type(m).__name__ for m in backbone.modules() if isinstance(m, qwen3_vl.Qwen3VLTextRMSNorm)}
            ),
        }
        if args.device.startswith("cuda"):
            torch.cuda.reset_peak_memory_stats(args.device)
        noise_dtype = torch.float32 if name == "fp32" else torch.float16
        reps, latencies = [], []
        for _ in range(1 if name == "fp32" else args.repeats):
            row = []
            for batch, noise in cases:
                with torch.no_grad():
                    sync(args.device)
                    t = time.perf_counter()
                    raw = policy.predict_action_chunk(
                        pre(deepcopy(batch)), noise=noise.to(args.device, noise_dtype)
                    )
                    sync(args.device)
                    latencies.append(time.perf_counter() - t)
                    dec = post(raw)
                if isinstance(dec, dict):
                    dec = next(iter(dec.values()))
                row.append(dec.float().cpu().numpy()[0])
            reps.append(np.stack(row))
        a = reps[0]
        actions[name] = a
        # first call of each mode includes warm-up; report the steady-state median
        entry["latency_s_median"] = float(np.median(latencies[1:] if len(latencies) > 1 else latencies))
        if args.device.startswith("cuda"):
            entry["peak_mem_gib"] = round(torch.cuda.max_memory_allocated(args.device) / 2**30, 2)
        entry["vision_rope_global_restored"] = qwen3_vl.apply_rotary_pos_emb_vision is original_vision_rope
        if len(reps) > 1:
            entry["repeat_bitexact_cases"] = int(
                sum(all(np.array_equal(r[i], a[i]) for r in reps) for i in range(len(a)))
            )
        if ref is not None:
            entry["vs_a100_fp32"] = summarize(a, ref["fp32"][: args.cases])
            entry["vs_a100_same_mode_max_abs_diff"] = float(np.abs(a - ref[name][: args.cases]).max())
        summary["modes"][name] = entry
        print(f"{name}: {json.dumps(entry)}", flush=True)
        del policy, pre, post
        gc.collect()
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()

    if "fp32" in actions:
        for name in actions:
            if name != "fp32":
                summary["modes"][name]["vs_local_fp32"] = summarize(actions[name], actions["fp32"])

    if "fp16_pure" in summary["modes"]:
        m = summary["modes"]["fp16_pure"]
        acc = m.get("vs_local_fp32") or m.get("vs_a100_fp32")
        checks = {
            "inputs_identical": inputs_match in (None, args.cases),
            "pure_fp16_enabled": m["backbone_pure_fp16"]
            and m["rmsnorm_classes"] == ["PureFP16Qwen3VLTextRMSNorm"],
            "vision_rope_global_restored": m["vision_rope_global_restored"],
            "no_violation": acc is not None and acc["violations_M_gt_limit"] == 0,
        }
        summary["fp16_pure_checks"] = checks
        summary["fp16_pure_pass"] = all(checks.values())

    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    np.savez_compressed(out / "actions.npz", input_sha256=np.array(digests), **actions)
    print(json.dumps(summary, indent=2), flush=True)
    if "fp16_pure_pass" in summary:
        print("fp16_pure:", "PASS" if summary["fp16_pure_pass"] else "FAIL", flush=True)


if __name__ == "__main__":
    main()
