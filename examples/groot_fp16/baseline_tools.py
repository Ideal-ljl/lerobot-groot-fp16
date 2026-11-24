"""Portable NumPy-only loading, checksum verification and output comparison."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Iterator
from pathlib import Path

import numpy as np


def iter_cases(bundle: str | Path) -> Iterator[dict]:
    """Yield exact NumPy inputs. Batch dimensions and source dtypes are retained."""
    root = Path(bundle)
    with np.load(root / "raw_inputs.npz", allow_pickle=False) as archive:
        raw = {k: archive[k] for k in archive.files}
    for i, case_id in enumerate(raw["case_ids"]):
        with np.load(root / "model_inputs" / f"{case_id}.npz", allow_pickle=False) as archive:
            model_input = {k: archive[k].copy() for k in archive.files if k != "noise"}
            noise = archive["noise"].copy()
        batch = {"observation.state": raw["state"][i].copy(), "task": [str(raw["tasks"][i])]}
        for camera in ("exterior_image_1_left", "wrist_image_left"):
            batch["observation.images." + camera] = raw[camera][i].copy()
        yield {
            "case_id": str(case_id),
            "input_sha256": str(raw["input_sha256"][i]),
            "raw_batch": batch,
            "model_input": model_input,
            "noise": noise,
        }


def verify(bundle: str | Path) -> dict:
    root = Path(bundle)
    checked, failures = 0, []
    for line in (root / "SHA256SUMS").read_text().splitlines():
        expected, name = line.split("  ", 1)
        path = root / name
        if not path.is_file():
            failures.append({"file": name, "error": "missing"})
            continue
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(block)
        checked += 1
        if digest.hexdigest() != expected:
            failures.append({"file": name, "error": "SHA256 mismatch"})
    return {"passed": not failures, "checked_files": checked, "failures": failures}


def compare(bundle: str | Path, candidate: str | Path, atol: float = 0.0, rtol: float = 0.0) -> dict:
    """Compare supplied raw and/or decoded outputs; never silently reorder cases."""
    if not np.isfinite([atol, rtol]).all() or atol < 0 or rtol < 0:
        raise ValueError("atol and rtol must be finite and nonnegative")
    with np.load(Path(bundle) / "reference_outputs.npz", allow_pickle=False) as archive:
        ref = {k: archive[k] for k in archive.files}
    with np.load(candidate, allow_pickle=False) as archive:
        got = {k: archive[k] for k in archive.files}
    for key in ("case_ids", "input_sha256"):
        if key not in got or not np.array_equal(got[key], ref[key]):
            raise ValueError(f"{key} missing or differs: expected all 100 cases in original order")
    kinds = [key for key in ("raw", "decoded") if key in got]
    if not kinds:
        raise ValueError("Candidate must contain raw and/or decoded arrays")
    report = {"passed": True, "atol": atol, "rtol": rtol, "outputs": {}}
    for key in kinds:
        expected, actual = ref[key], got[key]
        if actual.shape != expected.shape:
            raise ValueError(f"{key}: expected shape {expected.shape}, got {actual.shape}")
        if not np.issubdtype(actual.dtype, np.floating):
            raise ValueError(f"{key}: candidate must use a floating dtype")
        finite_cases = np.isfinite(actual).all(axis=(1, 2))
        if not finite_cases.all():
            report["passed"] = False
            report["outputs"][key] = {
                "all_finite": False,
                "nonfinite_case_ids": ref["case_ids"][~finite_cases].tolist(),
            }
            continue
        delta = np.abs(actual.astype(np.float64) - expected.astype(np.float64))
        per_case_max = delta.max(axis=(1, 2))
        tolerance_ok = (delta <= atol + rtol * np.abs(expected.astype(np.float64))).all(axis=(1, 2))
        bitexact = [
            actual.dtype == expected.dtype and actual[i].tobytes() == expected[i].tobytes()
            for i in range(len(actual))
        ]
        detail = {
            "all_finite": True,
            "reference_dtype": str(expected.dtype),
            "candidate_dtype": str(actual.dtype),
            "bitexact_cases": sum(bitexact),
            "numeric_exact_cases": int((per_case_max == 0).sum()),
            "within_tolerance_cases": int(tolerance_ok.sum()),
            "changed_elements": int(np.count_nonzero(delta)),
            "max_abs_error": float(delta.max()),
            "mean_abs_error": float(delta.mean()),
            "per_case_max_abs_error": per_case_max.tolist(),
        }
        if key == "decoded":
            normalized = delta / ref["action_ranges"] * 100
            detail["M_max_pct"] = float(normalized.max())
            detail["per_case_M_pct"] = normalized.max(axis=(1, 2)).tolist()
        report["outputs"][key] = detail
        report["passed"] &= bool(tolerance_ok.all())
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=Path(__file__).resolve().parent)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("verify")
    comparison = commands.add_parser("compare")
    comparison.add_argument("candidate", type=Path)
    comparison.add_argument("--atol", type=float, default=0.0)
    comparison.add_argument("--rtol", type=float, default=0.0)
    comparison.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        result = (
            verify(args.bundle)
            if args.command == "verify"
            else compare(args.bundle, args.candidate, args.atol, args.rtol)
        )
    except (ValueError, OSError, KeyError) as exc:
        result = {"passed": False, "error": str(exc)}
    serialized = json.dumps(result, indent=2, allow_nan=False) + "\n"
    print(serialized, end="")
    if getattr(args, "output", None):
        args.output.write_text(serialized)
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
