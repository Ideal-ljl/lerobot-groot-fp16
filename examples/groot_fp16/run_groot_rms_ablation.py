import argparse
import hashlib
import json
import platform
import subprocess
import time
import types
from collections import Counter
from contextlib import nullcontext
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import transformers
import transformers.models.qwen3_vl.modeling_qwen3_vl as qwen
from full_fp16_experiment import full_fp16, naive_rmsnorm
from groot_ablation_common import CHECKPOINT, PROCESSOR, TAG, setup, verify

parser = argparse.ArgumentParser()
parser.add_argument("--cases", type=int, default=100)
parser.add_argument("--repeats", type=int, default=2)
parser.add_argument("--out", default="outputs/groot_rms_ablation")
parser.add_argument("--ckpt", default=CHECKPOINT)
parser.add_argument("--processor", default=PROCESSOR)
args = parser.parse_args()
if not 1 <= args.cases <= 100 or args.repeats < 1:
    parser.error("--cases must be 1..100 and --repeats must be positive")
CHECKPOINT, PROCESSOR = args.ckpt, args.processor
out = Path(args.out)
out.mkdir(parents=True, exist_ok=False)
modes = ["fp16_default", "backbone_scaled", "full_fp16_naive", "full_fp16_scaled"]
print("Loading model and generating fixed local cases...", flush=True)
policy, pre, post, cases, digests, ranges = setup(args.cases, args.ckpt, args.processor)
backbone = policy._groot_model.backbone
switched = [
    (m, type(m), type(m).__bases__[0]) for m in backbone.modules() if type(m).__name__.startswith("PureFP16")
]
norms = [m for m in backbone.modules() if isinstance(m, qwen.Qwen3VLTextRMSNorm)]
original_vision_rope = qwen.apply_rotary_pos_emb_vision
weights_before = {n: (v.data_ptr(), v._version) for n, v in policy.named_parameters()}
print(
    "Loaded; FP16 norms:",
    len(norms),
    "parameter dtypes:",
    dict(Counter(str(p.dtype) for p in policy.parameters())),
    flush=True,
)


def configure(name: str) -> None:
    for m in norms:
        if "forward" in m.__dict__:
            del m.__dict__["forward"]
    for m, pure, standard in switched:
        m.__class__ = standard if name == "fp16_default" else pure
    backbone.pure_fp16 = name != "fp16_default"
    if name == "full_fp16_naive":
        for m in norms:
            m.forward = types.MethodType(naive_rmsnorm, m)


def tensor_hash(batch: dict) -> str:
    h = hashlib.sha256()
    for key in sorted(batch):
        value = batch[key]
        if isinstance(value, torch.Tensor):
            h.update(key.encode())
            h.update(str(value.dtype).encode())
            h.update(np.asarray(value.shape, dtype=np.int64).tobytes())
            h.update(value.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def run(model_input: dict, noise: torch.Tensor, name: str, measure: bool = True) -> tuple:
    configure(name)
    torch.cuda.synchronize()
    if measure:
        torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    with torch.inference_mode():
        with full_fp16() if name.startswith("full_") else nullcontext() as audit:
            raw = policy.predict_action_chunk(model_input, noise=noise)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        decoded = post(raw).float().cpu().numpy()[0]
    assert qwen.apply_rotary_pos_emb_vision is original_vision_rope
    return decoded, raw[:, :, :17].float().cpu().numpy()[0], elapsed, audit.report() if audit else None


# Check the generator is repeatable on this machine; don't use mismatched A100 inputs as a numerical baseline.
stats = json.loads((Path(CHECKPOINT) / "statistics.json").read_text())[TAG]


def cat(d: dict, k: str) -> np.ndarray:
    return np.concatenate([np.asarray(v[k], dtype=np.float64) for v in d.values()])


second_cases, second_digests = verify.make_cases(
    args.cases,
    cat(stats["state"], "q01"),
    cat(stats["state"], "q99"),
    40,
    132,
    ["exterior_image_1_left", "wrist_image_left"],
    20261123,
)
assert digests == second_digests
# Save exact input tensors, not just hashes, for reproducible later experiments on other hardware.
input_arrays = {
    "input_sha256": np.array(digests),
    "tasks": np.array([b["task"][0] for b, n in cases]),
    "state": np.stack([b["observation.state"].numpy() for b, n in cases]),
    "noise": np.stack([n.numpy() for b, n in cases]),
}
for camera in ["exterior_image_1_left", "wrist_image_left"]:
    input_arrays[camera] = np.stack([b["observation.images." + camera].numpy() for b, n in cases])
np.savez(out / "inputs_exact.npz", **input_arrays)
del input_arrays, second_cases

report = {
    "revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
    "environment": {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "device": torch.cuda.get_device_name(),
    },
    "checkpoint": CHECKPOINT,
    "processor": PROCESSOR,
    "cases": args.cases,
    "repeats": args.repeats,
    "seed": 20261123,
    "reference": "local fp16_default: FP16 weights, stock FP32 RMSNorm/RoPE and time embeddings; not full FP32",
    "scope": "Full modes enforce FP16 tensor inputs, explicit dtype requests and all floating tensor outputs during predict_action_chunk; preprocessing/postprocessing and CUDA kernel-internal accumulators are not forced.",
    "latency_note": "Model forward only, excluding preprocessing/postprocessing. Full-mode timings include Python TorchDispatch dtype-enforcement overhead; not deployment benchmarks.",
    "local_inputs_regenerated_bitexact": True,
    "a100_input_hash_matches": sum(
        a == b
        for a, b in zip(
            digests,
            np.load(Path(__file__).with_name("reference_a100.npz"))["input_sha256"][: args.cases],
            strict=True,
        )
    ),
    "a100_outputs_used_as_numerical_reference": False,
    "modes": {},
    "dtype_audits": {},
    "rmsnorm_diagnostics": {},
    "rmsnorm_diagnostic_note": "The hook squares the unscaled input to diagnose hypothetical overflow; this is not the scaled RMSNorm's internal variance calculation.",
    "progress": {},
}


def save() -> None:
    (out / "summary.json").write_text(json.dumps(report, indent=2) + "\n")


# Warm up every mode, and verify the all-FP16 guard before the full run.
batch0, noise0 = cases[0]
with torch.no_grad():
    input0 = pre(deepcopy(batch0))
noise0 = noise0.cuda()
for name in modes:
    a, raw, latency, audit = run(input0, noise0, name)
    if audit:
        report["dtype_audits"][name] = audit
    print(
        "Warmup",
        name,
        "finite",
        bool(np.isfinite(a).all()),
        "seconds",
        round(latency, 3),
        "dtype_audit",
        audit,
        flush=True,
    )
save()

# Capture the actual overflow and zeroed-token mechanism, outside timed inference.
for name in ["full_fp16_naive", "full_fp16_scaled"]:
    trace = []
    hooks = []
    for module_name, module in backbone.named_modules():
        if isinstance(module, qwen.Qwen3VLTextRMSNorm) and module.weight.numel() == 2048:

            def hook(
                m: torch.nn.Module,
                inputs: tuple,
                output: torch.Tensor,
                label: str = module_name,
                trace: list = trace,
            ) -> None:
                x = inputs[0].detach()
                sq = x.square()
                first = x[0, 0]
                trace.append(
                    {
                        "module": label,
                        "input_abs_max": float(x.abs().max()),
                        "first_token_abs_max": float(first.abs().max()),
                        "first_token_max_channel": int(first.abs().argmax()),
                        "squared_inf_elements": int(torch.isinf(sq).sum()),
                        "zero_output_rows": int((output == 0).all(-1).sum()),
                        "first_token_output_all_zero": bool((output[0, 0] == 0).all()),
                    }
                )

            hooks.append(module.register_forward_hook(hook))
    try:
        run(input0, noise0, name, measure=False)
    finally:
        for hook in hooks:
            hook.remove()
    report["rmsnorm_diagnostics"][name] = trace
save()

all_actions = {name: [] for name in modes}
all_raw = {name: [] for name in modes}
latencies = {name: [] for name in modes}
peaks = {name: [] for name in modes}
pre_hashes = []
for repeat in range(args.repeats):
    rows = {name: [] for name in modes}
    raw_rows = {name: [] for name in modes}
    for i, (batch, noise) in enumerate(cases):
        with torch.no_grad():
            model_input = pre(deepcopy(batch))
        digest = tensor_hash(model_input)
        if repeat == 0:
            pre_hashes.append(digest)
        else:
            assert digest == pre_hashes[i], f"Preprocessed input changed for case {i}"
        noise = noise.cuda()
        # Same preprocessed tensor objects and same noise values are used for all modes.
        before = tensor_hash(model_input)
        for name in modes:
            a, raw, latency, audit = run(model_input, noise, name)
            rows[name].append(a)
            raw_rows[name].append(raw)
            latencies[name].append(latency)
            peaks[name].append(torch.cuda.max_memory_allocated() / 2**30)
        assert tensor_hash(model_input) == before, "Model mutated the shared input"
        if (i + 1) % 10 == 0 or i + 1 == args.cases:
            report["progress"] = {"repeat": repeat + 1, "case": i + 1}
            ref_now = np.stack(rows["fp16_default"]).astype(np.float64)
            partial = {}
            for mode in modes:
                current = np.stack(rows[mode]).astype(np.float64)
                maxes = (np.abs(current - ref_now) / ranges[None, None, :] * 100).max((1, 2))
                partial[mode] = {
                    "M_max_pct": float(maxes.max()),
                    "over_10_pct": int((maxes > 10).sum()),
                    "finite": bool(np.isfinite(current).all()),
                }
            report["partial_metrics"] = partial
            save()
            print(f"Progress repeat {repeat + 1}/{args.repeats}: {i + 1}/{args.cases}", flush=True)
    for name in modes:
        all_actions[name].append(np.stack(rows[name]))
        all_raw[name].append(np.stack(raw_rows[name]))
    np.savez_compressed(
        out / "actions.npz",
        input_sha256=np.array(digests),
        preprocessed_sha256=np.array(pre_hashes),
        **{name: np.stack(v) for name, v in all_actions.items()},
        **{"raw_" + name: np.stack(v) for name, v in all_raw.items()},
    )


def metrics(a: np.ndarray, b: np.ndarray) -> dict:
    finite = np.isfinite(a).all((1, 2)) & np.isfinite(b).all((1, 2))
    rel = np.abs(a.astype(np.float64) - b.astype(np.float64)) / ranges[None, None, :] * 100
    per_max = rel.max((1, 2))
    per_mean = rel.mean((1, 2))
    valid = per_max[finite]
    return {
        "nonfinite_cases": int((~finite).sum()),
        "violations_M_gt_10_pct": int(((per_max > 10) | ~finite).sum()),
        "M_max_pct": float(valid.max()) if len(valid) else None,
        "M_median_pct": float(np.median(valid)) if len(valid) else None,
        "M_p95_pct": float(np.percentile(valid, 95)) if len(valid) else None,
        "E_mean_pct": float(per_mean[finite].mean()) if finite.any() else None,
        "per_case_M_pct": per_max.tolist(),
    }


reference = all_actions["fp16_default"][0]
for name in modes:
    data = np.stack(all_actions[name])
    raw = np.stack(all_raw[name])
    same = [all(np.array_equal(r[i], data[0, i]) for r in data[1:]) for i in range(args.cases)]
    raw_same = [all(np.array_equal(r[i], raw[0, i]) for r in raw[1:]) for i in range(args.cases)]
    report["modes"][name] = {
        "vs_local_default_fp16": metrics(data[0], reference),
        "repeat_bitexact_cases": sum(same),
        "raw_repeat_bitexact_cases": sum(raw_same),
        "repeat_max_abs_diff": float(np.max(np.abs(data - data[0]))),
        "all_finite": bool(np.isfinite(data).all()),
        "latency_s_median": float(np.median(latencies[name])),
        "peak_allocated_GiB": max(peaks[name]),
    }
report["naive_vs_scaled"] = metrics(all_actions["full_fp16_naive"][0], all_actions["full_fp16_scaled"][0])
report["weights_unchanged"] = all(
    (v.data_ptr(), v._version) == weights_before[n] for n, v in policy.named_parameters()
)
report["vision_rope_global_restored"] = qwen.apply_rotary_pos_emb_vision is original_vision_rope
report["preprocessed_inputs_bitexact_across_repeats"] = True
report["complete"] = True
save()
print(
    json.dumps(
        {k: {a: b for a, b in v.items() if a != "per_case_M_pct"} for k, v in report["modes"].items()},
        indent=2,
    ),
    flush=True,
)
print("DONE", out / "summary.json", flush=True)
