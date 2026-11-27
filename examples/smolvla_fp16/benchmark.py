"""Replay the ten sealed two-step SmolVLA cases and score three rounds of FP16 final actions."""

import argparse
import csv
import hashlib
import io
import json
from contextlib import nullcontext
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from uuid import uuid4
from zipfile import Path as ZipPath, ZipFile

import numpy as np
import torch
from huggingface_hub import snapshot_download

from examples.smolvla_fp16.runtime import digest, enable_force_fp16, infer_actions, load_policy

BACKBONE = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
REVISION = "4cde9d208e19519c36917aa170fffc7e381aa0ad"
ACTION_SHAPE = (1, 50, 6)


@dataclass
class Case:
    name: str
    batch: dict[str, torch.Tensor]
    noise: torch.Tensor
    reference: np.ndarray
    outputs: dict[int, np.ndarray] = field(default_factory=dict)
    failures: dict[int, str] = field(default_factory=dict)


def plain_name(name: str) -> str:
    if Path(name).name != name or name in {"", ".", ".."}:
        raise ValueError(f"Invalid manifest filename: {name!r}")
    return name


def load_cases(path: Path) -> tuple[dict, list[Case]]:
    cases = []
    source = nullcontext(None) if path.is_dir() else ZipFile(path)
    with source as archive:
        root = ZipPath(archive) if archive is not None else path
        manifest = json.loads((root / "manifest.json").read_bytes())
        if (
            manifest.get("format_version"),
            manifest.get("denoising_steps"),
            manifest.get("scoring_output"),
        ) != (2, 2, "decoded") or not manifest.get("action_stats_key"):
            raise ValueError("需要两步 FP16 反归一化动作基准包（v2），请用当前版本重新导出基准。")
        entries = manifest["cases"]
        if len(entries) != 10 or len({entry["file"] for entry in entries}) != 10:
            raise ValueError("Expected ten distinct cases in sealed replay order")
        for entry in entries:
            name = plain_name(entry["file"])
            content = (root / name).read_bytes()
            if hashlib.sha256(content).hexdigest() != entry["sha256"]:
                raise ValueError(f"Case SHA256 mismatch: {name}")
            with np.load(io.BytesIO(content), allow_pickle=False) as data:
                batch = {k[6:]: torch.from_numpy(data[k]) for k in data.files if k.startswith("input/")}
                noise = torch.from_numpy(data["noise"])
                ref = data["decoded"]
            if not batch or any(v.is_floating_point() and v.dtype != torch.float16 for v in batch.values()):
                raise ValueError(f"Invalid sealed policy inputs: {name}")
            if noise.dtype != torch.float16 or tuple(noise.shape) != (1, 50, 32):
                raise ValueError(f"Invalid sealed noise: {name}")
            if ref.shape != ACTION_SHAPE or ref.dtype != np.float16 or not np.isfinite(ref).all():
                raise ValueError(f"Invalid FP16 reference actions: {name}")
            cases.append(Case(name, batch, noise, ref))
    return manifest, cases


def backbone_assets(checkpoint: Path, hashes: dict[str, str]) -> Path:
    if not hashes or "config.json" not in hashes:
        raise ValueError("Missing backbone hashes in the baseline manifest")
    for name in hashes:
        plain_name(name)
    model_name = BACKBONE.split("/")[-1]
    candidates = [
        checkpoint / model_name,
        checkpoint.parent / model_name,
        *sorted(checkpoint.parent.glob("SmolVLM2-500M*")),
    ]
    for candidate in candidates:
        if all(
            (candidate / name).is_file() and digest(candidate / name) == sha for name, sha in hashes.items()
        ):
            return candidate
    # Only the pinned configuration and tokenizer assets are needed. The policy contains the model weights.
    directory = Path(snapshot_download(BACKBONE, revision=REVISION, allow_patterns=list(hashes)))
    for name, sha in hashes.items():
        if digest(directory / name) != sha:
            raise ValueError(f"Backbone SHA256 mismatch: {name}")
    return directory


def errors(actual: np.ndarray, reference: np.ndarray) -> tuple[np.ndarray, float]:
    ref = reference[0].astype(np.float64)
    norm = np.linalg.norm(ref, axis=0)
    if np.any(norm == 0):
        raise ValueError("参考动作存在全零维度，相对误差无定义")
    per_dim = np.linalg.norm(actual[0].astype(np.float64) - ref, axis=0) / norm
    return per_dim, float(np.sqrt(np.mean(per_dim**2)))


def score(case: Case) -> dict:
    row = {"case": case.name, "base_2A": 0, "score_2A": 0, "overall_error_pct": None, "accuracy_note": ""}
    for dim in range(1, 7):
        row[f"dim_{dim}_error_pct"] = None
    first = case.outputs.get(1)
    if first is not None:
        per_dim, overall = errors(first, case.reference)
        bands = ((0.01, 4), (0.03, 3), (0.05, 2))
        base = next((points for limit, points in bands if overall <= limit), 0)
        row.update(overall_error_pct=100 * overall, base_2A=base, score_2A=base)
        for dim, error in enumerate(per_dim, start=1):
            row[f"dim_{dim}_error_pct"] = 100 * error
        if np.any(per_dim > 0.10):
            row.update(score_2A=0, accuracy_note="单维误差超过 10%")
        elif not base:
            row["accuracy_note"] = "综合误差超过 5%"
    else:
        row["accuracy_note"] = "首轮输出无效"
    row["score_3A"] = 0.5 * sum(r in case.outputs for r in (2, 3))
    repeats = []
    for r in (1, 2, 3):
        row[f"round_{r}_status"] = case.failures.get(r, "OK")
        if r == 1:
            continue
        repeat = None
        if first is not None and r in case.outputs:
            try:
                _, repeat = errors(case.outputs[r], first)
            except ValueError as exc:
                row[f"round_{r}_comparison_note"] = str(exc)
        row.setdefault(f"round_{r}_comparison_note", "")
        row[f"round_{r}_error_pct"] = None if repeat is None else 100 * repeat
        repeats.append(repeat)
    row["score_3B"] = int(all(e is not None and e <= 0.01 for e in repeats))
    return row


@torch.inference_mode()
def run(checkpoint: Path, data: Path) -> None:
    if checkpoint.is_file():
        if checkpoint.name != "model.safetensors":
            raise ValueError("模型文件应为 checkpoint 目录中的 model.safetensors")
        checkpoint = checkpoint.parent
    manifest, cases = load_cases(data)
    if not torch.cuda.is_available():
        raise RuntimeError("请在可用的 CUDA 设备上运行测试。")
    for name, expected in {
        "model.safetensors": manifest["weight_sha256"],
        **manifest["checkpoint_metadata_sha256"],
    }.items():
        if digest(checkpoint / plain_name(name)) != expected:
            raise ValueError(f"Checkpoint SHA256 mismatch: {name}")
    backbone = backbone_assets(checkpoint, manifest["backbone_assets_sha256"])
    device = torch.cuda.get_device_name()
    policy, _, post = load_policy(checkpoint, backbone, manifest["action_stats_key"])
    norms = enable_force_fp16(policy)
    if norms != manifest["scaled_norms"]:
        raise ValueError("Scaled RMSNorm count differs from the reference")
    for case in cases:
        if np.any(np.all(case.reference == 0, axis=1)):
            raise ValueError(f"Zero reference norm in sealed baseline: {case.name}")
    parent = Path.cwd() / "outputs" / "smolvla_tests"
    output = parent / f"{datetime.now():%Y%m%d-%H%M%S}-{uuid4().hex[:8]}"
    output.mkdir(parents=True)
    print(f"测试设备 {device}，基准设备 {manifest['gpu']}，结果目录 {output}", flush=True)
    for r in (1, 2, 3):
        for case in cases:
            try:
                batch = {k: v.cuda() for k, v in case.batch.items()}
                _, actions, _ = infer_actions(policy, post, batch, case.noise.cuda())
                if tuple(actions.shape) != ACTION_SHAPE:
                    raise ValueError(f"Invalid action shape {tuple(actions.shape)}")
                case.outputs[r] = actions.cpu().numpy().copy()
            except Exception as exc:
                # Each failed call counts against the score. Continue with the remaining sealed inputs.
                case.failures[r] = f"{type(exc).__name__}: {exc}"
            print(f"第 {r} 轮 {case.name} {case.failures.get(r, 'OK')}", flush=True)
        np.savez_compressed(
            output / f"round_{r}.npz",
            **{c.name.removesuffix(".npz"): c.outputs[r] for c in cases if r in c.outputs},
        )

    rows = [score(case) for case in cases]
    with (output / "cases.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "reference_gpu": manifest["gpu"],
        "test_gpu": device,
        "weight_sha256": manifest["weight_sha256"],
        "action_stats_key": manifest["action_stats_key"],
        "denoising_steps": 2,
        "dtype": "float16",
        "case_hashes": {e["file"]: e["sha256"] for e in manifest["cases"]},
        "valid_calls": sum(len(c.outputs) for c in cases),
        "expected_calls": 30,
        "task_2A": sum(row["score_2A"] for row in rows),
        "task_3A": sum(row["score_3A"] for row in rows),
        "task_3B": sum(row["score_3B"] for row in rows),
    }
    summary["total_2A_3A_3B"] = summary["task_2A"] + summary["task_3A"] + summary["task_3B"]
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(
        f"精度 2A {summary['task_2A']}/40，稳定性 3A {summary['task_3A']:g}/10、3B {summary['task_3B']}/10。"
        f"合计 {summary['total_2A_3A_3B']:g}/60，有效调用 {summary['valid_calls']}/30。\n结果已保存到 {output}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("data", type=Path)
    args = parser.parse_args()
    run(args.checkpoint.expanduser().absolute(), args.data.expanduser().absolute())


if __name__ == "__main__":
    main()
