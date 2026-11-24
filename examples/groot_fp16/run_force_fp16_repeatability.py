"""Replay 20 saved inputs in Force FP16 for three complete rounds."""

import argparse
import csv
import hashlib
import json
import platform
import subprocess
import time
from collections import Counter
from copy import deepcopy
from itertools import combinations
from pathlib import Path

import numpy as np
import torch
import transformers
from full_fp16_experiment import full_fp16
from groot_ablation_common import CHECKPOINT, PROCESSOR, setup


def tensor_hash(batch: dict) -> str:
    digest = hashlib.sha256()
    for key in sorted(batch):
        value = batch[key]
        if isinstance(value, torch.Tensor):
            digest.update(key.encode())
            digest.update(str(value.dtype).encode())
            digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
            digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def byte_equal(a: np.ndarray, b: np.ndarray) -> bool:
    return a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", default=CHECKPOINT)
    parser.add_argument("--processor", default=PROCESSOR)
    parser.add_argument("--source", type=Path, default=Path("outputs/groot_rms_ablation"))
    parser.add_argument("--out", type=Path, default=Path("outputs/force_fp16_20x3"))
    args = parser.parse_args()
    count, repeats = 20, 3
    source = args.source
    out = args.out
    out.mkdir(parents=True, exist_ok=False)
    with np.load(source / "inputs_exact.npz") as saved:
        selected = {k: saved[k][:count].copy() for k in saved.files}
    np.savez(out / "inputs_exact.npz", **selected)
    with np.load(source / "actions.npz") as saved:
        reference = saved["fp16_default"][0, :count].copy()
        old_force = saved["full_fp16_scaled"][0, :count].copy()
        old_hashes = saved["preprocessed_sha256"][:count].copy()
        assert np.array_equal(saved["input_sha256"][:count], selected["input_sha256"])

    print("Loading model; 20 saved cases, 3 rounds, Force FP16", flush=True)
    policy, pre, post, generated, generated_hashes, ranges = setup(count, args.ckpt, args.processor)
    assert generated_hashes == selected["input_sha256"].tolist()
    del generated
    weights_before = {n: (p.data_ptr(), p._version) for n, p in policy.named_parameters()}
    assert all(p.dtype == torch.float16 for p in policy.parameters() if p.is_floating_point())
    cached = []
    raw_batches = []
    hashes = []
    for i in range(count):
        batch = {
            "observation.state": torch.from_numpy(selected["state"][i]),
            "task": [str(selected["tasks"][i])],
        }
        for camera in ("exterior_image_1_left", "wrist_image_left"):
            batch["observation.images." + camera] = torch.from_numpy(selected[camera][i])
        with torch.no_grad():
            model_input = pre(deepcopy(batch))
        noise = torch.from_numpy(selected["noise"][i]).cuda()
        digest = tensor_hash({**model_input, "fixed_noise": noise})
        assert tensor_hash(model_input) == old_hashes[i]
        hashes.append(digest)
        cached.append((model_input, noise))
        raw_batches.append(batch)

    report = {
        "mode": "Force FP16",
        "cases": count,
        "repeats": repeats,
        "selection": "First 20 saved synthetic DROID cases from previous local experiment",
        "input_source": str(source / "inputs_exact.npz"),
        "checkpoint": args.ckpt,
        "revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "device": torch.cuda.get_device_name(),
        },
        "scope": "All explicit model floating tensor operations FP16 with scaled RMSNorm; preprocessing, postprocessing and CUDA internal accumulation unchanged.",
        "execution": "One process and model load, eval/inference mode; one warmup then 3 rounds of cases 1..20; exact saved observations/noise reused, preprocessing refreshed per call for relative-action state cache and hashes checked; no per-call RNG reseeding.",
        "deterministic_algorithms_enabled": torch.are_deterministic_algorithms_enabled(),
        "parameter_dtypes": dict(Counter(str(p.dtype) for p in policy.parameters())),
        "reference": "Previous local default-FP16 output with stock FP32 islands; matching input hashes checked.",
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "force_implementation_sha256": hashlib.sha256(
            Path(__file__).with_name("full_fp16_experiment.py").read_bytes()
        ).hexdigest(),
        "complete": False,
    }

    def save() -> None:
        (out / "summary.json").write_text(json.dumps(report, indent=2) + "\n")

    def infer(model_input: dict, noise: torch.Tensor) -> tuple:
        with torch.inference_mode():
            with full_fp16() as audit:
                raw = policy.predict_action_chunk(model_input, noise=noise)
            decoded = post(raw.clone())
            torch.cuda.synchronize()
            return raw.cpu().numpy().copy(), decoded.cpu().numpy().copy(), audit.report()

    with torch.no_grad():
        warmup_input = pre(deepcopy(raw_batches[0]))
    assert tensor_hash({**warmup_input, "fixed_noise": cached[0][1]}) == hashes[0]
    _, _, audit = infer(warmup_input, cached[0][1])
    report["warmup_dtype_audit"] = audit
    save()
    all_raw, all_actions, audits = [], [], []
    for repeat in range(repeats):
        raw_rows, action_rows = [], []
        for i, (model_input, noise) in enumerate(cached):
            # Packing the observation also refreshes the shared state used by relative-action decoding.
            with torch.no_grad():
                model_input = pre(deepcopy(raw_batches[i]))
            assert tensor_hash({**model_input, "fixed_noise": noise}) == hashes[i]
            start = time.perf_counter()
            raw, action, audit = infer(model_input, noise)
            assert tensor_hash({**model_input, "fixed_noise": noise}) == hashes[i]
            assert not audit["non_fp16_floating_outputs"]
            raw_rows.append(raw[0])
            action_rows.append(action[0])
            audits.append(
                {"round": repeat + 1, "case": i + 1, "seconds": time.perf_counter() - start, **audit}
            )
            if (i + 1) % 5 == 0:
                print(f"Round {repeat + 1}/3: {i + 1}/20 complete", flush=True)
        all_raw.append(np.stack(raw_rows))
        all_actions.append(np.stack(action_rows))
        np.savez_compressed(
            out / "actions.npz",
            raw=np.stack(all_raw),
            decoded=np.stack(all_actions),
            reference=reference,
            input_sha256=selected["input_sha256"],
            model_input_and_noise_sha256=np.asarray(hashes),
        )
        report["completed_rounds"] = repeat + 1
        save()

    raw, actions = np.stack(all_raw), np.stack(all_actions)
    pairs = []
    for a, b in combinations(range(repeats), 2):
        delta = np.abs(actions[a].astype(np.float64) - actions[b].astype(np.float64))
        raw_delta = np.abs(raw[a].astype(np.float64) - raw[b].astype(np.float64))
        pairs.append(
            {
                "rounds": [a + 1, b + 1],
                "decoded_bitexact_cases": sum(byte_equal(actions[a, i], actions[b, i]) for i in range(count)),
                "raw_bitexact_cases": sum(byte_equal(raw[a, i], raw[b, i]) for i in range(count)),
                "decoded_changed_elements": int(np.count_nonzero(delta)),
                "decoded_max_abs_diff": float(delta.max()),
                "decoded_mean_abs_diff": float(delta.mean()),
                "raw_max_abs_diff": float(raw_delta.max()),
                "normalized_max_diff_pct": float((delta / ranges * 100).max()),
            }
        )
    error = np.abs(actions.astype(np.float64) - reference.astype(np.float64)) / ranges * 100
    case_error = error.max(axis=(2, 3))
    rows = []
    for i in range(count):
        rows.append(
            {
                "case": i + 1,
                "all_three_raw_bitexact": all(byte_equal(raw[0, i], raw[r, i]) for r in (1, 2)),
                "all_three_decoded_bitexact": all(byte_equal(actions[0, i], actions[r, i]) for r in (1, 2)),
                "decoded_max_abs_spread": float(np.ptp(actions[:, i].astype(np.float64), axis=0).max()),
                "round1_M_pct": float(case_error[0, i]),
                "round2_M_pct": float(case_error[1, i]),
                "round3_M_pct": float(case_error[2, i]),
                "M_spread_percentage_points": float(np.ptp(case_error[:, i])),
            }
        )
    with (out / "per_case.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    report.update(
        {
            "pairwise": pairs,
            "per_case": rows,
            "raw_shape": list(raw.shape),
            "decoded_shape": list(actions.shape),
            "all_outputs_finite": bool(np.isfinite(raw).all() and np.isfinite(actions).all()),
            "strict_fp16_audit_passed": all(not a["non_fp16_floating_outputs"] for a in audits),
            "inputs_and_noise_unchanged": True,
            "weights_unchanged": all(
                (p.data_ptr(), p._version) == weights_before[n] for n, p in policy.named_parameters()
            ),
            "vs_previous_force_run_max_abs_diff": float(
                np.abs(actions[0].astype(np.float64) - old_force.astype(np.float64)).max()
            ),
            "per_round_reference_error": [
                {
                    "round": r + 1,
                    "M_median_pct": float(np.median(case_error[r])),
                    "M_max_pct": float(case_error[r].max()),
                    "E_mean_pct": float(error[r].mean()),
                }
                for r in range(repeats)
            ],
            "complete": True,
        }
    )
    (out / "dtype_audits.json").write_text(json.dumps(audits, indent=2) + "\n")
    save()
    print(
        json.dumps(
            {
                k: report[k]
                for k in (
                    "pairwise",
                    "per_round_reference_error",
                    "all_outputs_finite",
                    "strict_fp16_audit_passed",
                    "vs_previous_force_run_max_abs_diff",
                )
            },
            indent=2,
        ),
        flush=True,
    )
    print("DONE", out / "summary.json", flush=True)


if __name__ == "__main__":
    main()
