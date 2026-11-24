"""Replay released exact inputs in Force FP16 and measure repeatability."""

import argparse
import json
from copy import deepcopy
from itertools import islice
from pathlib import Path

import numpy as np
import torch
from baseline_tools import iter_cases, verify
from full_fp16_experiment import full_fp16
from groot_ablation_common import setup
from run_force_fp16_repeatability import byte_equal, tensor_hash


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--processor", help="Defaults to processor_assets inside the bundle")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cases", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--input-mode", choices=["model", "raw"], default="model")
    args = parser.parse_args()
    if not 1 <= args.cases <= 100 or args.repeats < 1:
        parser.error("--cases must be 1..100 and --repeats must be positive")
    integrity = verify(args.bundle)
    if not integrity["passed"]:
        raise ValueError(f"Bundle integrity check failed: {integrity}")
    args.out.mkdir(parents=True, exist_ok=False)
    policy, pre, post, _, _, ranges = setup(
        0, args.ckpt, args.processor or str(args.bundle / "processor_assets")
    )
    cases = list(islice(iter_cases(args.bundle), args.cases))
    with np.load(args.bundle / "reference_outputs.npz", allow_pickle=False) as archive:
        reference = {key: archive[key][: args.cases].copy() for key in ("raw", "decoded")}
    before_weights = {name: (p.data_ptr(), p._version) for name, p in policy.named_parameters()}
    all_raw, all_decoded, audits, preprocessing_matches = [], [], [], []
    for repeat in range(args.repeats):
        raw_rows, decoded_rows = [], []
        for i, case in enumerate(cases):
            batch = {
                key: torch.from_numpy(value.copy()) if isinstance(value, np.ndarray) else deepcopy(value)
                for key, value in case["raw_batch"].items()
            }
            # This call must happen immediately before inference/postprocessing: it refreshes
            # the shared current-state cache needed to decode relative actions.
            with torch.no_grad():
                prepared = pre(batch)
            filtered = policy._filter_groot_inputs(prepared, include_action=False)
            matched = set(filtered) == set(case["model_input"]) and all(
                byte_equal(value.detach().cpu().numpy(), case["model_input"][key])
                for key, value in filtered.items()
            )
            preprocessing_matches.append(matched)
            if args.input_mode == "raw":
                if not matched:
                    raise ValueError(f"{case['case_id']}: preprocessed input differs from the saved baseline")
                model_input = filtered
            else:
                model_input = {
                    key: torch.from_numpy(value.copy()).cuda() for key, value in case["model_input"].items()
                }
            noise = torch.from_numpy(case["noise"].copy()).cuda()
            before_input = tensor_hash({**model_input, "fixed_noise": noise})
            with torch.inference_mode():
                with full_fp16() as audit:
                    raw = policy.predict_action_chunk(model_input, noise=noise)
                decoded = post(raw.clone())
                torch.cuda.synchronize()
            assert tensor_hash({**model_input, "fixed_noise": noise}) == before_input
            raw_np, decoded_np = raw[0].cpu().numpy().copy(), decoded[0].cpu().numpy().copy()
            if not (np.isfinite(raw_np).all() and np.isfinite(decoded_np).all()):
                raise ValueError(f"{case['case_id']}: nonfinite model output")
            raw_rows.append(raw_np)
            decoded_rows.append(decoded_np)
            audits.append({"repeat": repeat + 1, "case_id": case["case_id"], **audit.report()})
            if (i + 1) % 10 == 0 or i + 1 == args.cases:
                print(f"Force FP16 round {repeat + 1}/{args.repeats}: {i + 1}/{args.cases}", flush=True)
        all_raw.append(np.stack(raw_rows))
        all_decoded.append(np.stack(decoded_rows))
        np.savez_compressed(
            args.out / f"round_{repeat + 1}.npz",
            case_ids=np.array([case["case_id"] for case in cases]),
            input_sha256=np.array([case["input_sha256"] for case in cases]),
            raw=all_raw[-1],
            decoded=all_decoded[-1],
        )
    report = {
        "mode": "Force FP16",
        "cases": args.cases,
        "repeats": args.repeats,
        "input_mode": args.input_mode,
        "preprocessing_matches_saved_inputs": all(preprocessing_matches),
        "strict_fp16_audit_passed": all(not a["non_fp16_floating_outputs"] for a in audits),
        "weights_unchanged": all(
            (p.data_ptr(), p._version) == before_weights[n] for n, p in policy.named_parameters()
        ),
        "per_round": [],
        "repeatability": {},
    }
    for r, (raw, decoded) in enumerate(zip(all_raw, all_decoded, strict=True)):
        delta = np.abs(decoded.astype(np.float64) - reference["decoded"].astype(np.float64))
        normalized = delta / ranges * 100
        report["per_round"].append(
            {
                "round": r + 1,
                "raw_bitexact_to_reference": sum(
                    byte_equal(raw[i], reference["raw"][i]) for i in range(args.cases)
                ),
                "decoded_bitexact_to_reference": sum(
                    byte_equal(decoded[i], reference["decoded"][i]) for i in range(args.cases)
                ),
                "decoded_max_abs_error": float(delta.max()),
                "M_max_pct": float(normalized.max()),
                "M_median_pct": float(np.median(normalized.max(axis=(1, 2)))),
            }
        )
    for key, values in [("raw", all_raw), ("decoded", all_decoded)]:
        report["repeatability"][key] = {
            "comparisons": max(0, args.repeats - 1),
            "bitexact_cases": sum(
                all(byte_equal(values[0][i], row[i]) for row in values[1:]) for i in range(args.cases)
            )
            if args.repeats > 1
            else None,
            "max_abs_spread": float(np.ptp(np.stack(values).astype(np.float64), axis=0).max()),
        }
    (args.out / "dtype_audits.json").write_text(json.dumps(audits, indent=2) + "\n")
    (args.out / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
