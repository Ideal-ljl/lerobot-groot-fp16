"""Export a portable 100-case Force FP16 baseline from exact saved inputs."""

import argparse
import hashlib
import importlib.metadata
import json
import platform
import shutil
import subprocess
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import transformers
from full_fp16_experiment import full_fp16
from groot_ablation_common import CHECKPOINT, PROCESSOR, TAG, setup
from run_force_fp16_repeatability import tensor_hash


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", default=CHECKPOINT)
    parser.add_argument("--processor", default=PROCESSOR)
    parser.add_argument("--source", type=Path, default=Path("outputs/groot_rms_ablation"))
    parser.add_argument("--out", type=Path, default=Path("outputs/force_fp16_baseline_100_v1"))
    args = parser.parse_args()
    out = args.out
    out.mkdir(parents=True, exist_ok=False)
    (out / "model_inputs").mkdir()
    source = args.source
    with np.load(source / "inputs_exact.npz", allow_pickle=False) as archive:
        inputs = {k: archive[k].copy() for k in archive.files}
    count = len(inputs["tasks"])
    assert count == 100
    case_ids = np.array([f"case_{i:03d}" for i in range(count)])
    inputs["case_ids"] = case_ids
    np.savez_compressed(out / "raw_inputs.npz", **inputs)
    with np.load(source / "actions.npz", allow_pickle=False) as archive:
        old_decoded = archive["full_fp16_scaled"][0].copy()
        old_raw = archive["raw_full_fp16_scaled"][0].copy()
        old_hashes = archive["preprocessed_sha256"].copy()
    print("Loading Force FP16; exporting 100 fixed cases", flush=True)
    policy, pre, post, generated, hashes, ranges = setup(count, args.ckpt, args.processor)
    assert hashes == inputs["input_sha256"].tolist()
    del generated
    weights_before = {n: (p.data_ptr(), p._version) for n, p in policy.named_parameters()}
    assert all(p.dtype == torch.float16 for p in policy.parameters() if p.is_floating_point())
    root = Path(args.ckpt)
    metadata = out / "checkpoint_metadata"
    metadata.mkdir()
    for name in (
        "config.json",
        "processor_config.json",
        "statistics.json",
        "embodiment_id.json",
        "model.safetensors.index.json",
        "download_manifest.json",
        "LICENSE",
        "NOTICE",
    ):
        shutil.copyfile(root / name, metadata / name)
    shutil.copytree(root / "experiment_cfg", metadata / "experiment_cfg")
    shutil.copytree(args.processor, out / "processor_assets", ignore=shutil.ignore_patterns(".cache"))
    policy.config.save_pretrained(out / "effective_policy_config")
    source_files = [
        "examples/groot_fp16/export_force_fp16_baseline.py",
        "examples/groot_fp16/groot_ablation_common.py",
        "examples/groot_fp16/full_fp16_experiment.py",
        "examples/groot_fp16/run_force_fp16_repeatability.py",
        "src/lerobot/policies/groot/fp16_precision.py",
        "src/lerobot/policies/groot/modeling_groot.py",
        "src/lerobot/policies/groot/groot_n1_7.py",
        "src/lerobot/policies/groot/processor_groot.py",
        "examples/groot_fp16/verify_fp16.py",
    ]
    for name in source_files:
        target = out / "provenance" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(name, target)
    versions = (
        "\n".join(sorted(f"{d.metadata['Name']}=={d.version}" for d in importlib.metadata.distributions()))
        + "\n"
    )
    (out / "environment_freeze.txt").write_text(versions)
    download = json.loads((root / "download_manifest.json").read_text())
    weight_hashes = {p.name: sha256(p) for p in root.glob("*.safetensors")}
    assert weight_hashes, "No checkpoint weight files found"
    stats = json.loads((root / "statistics.json").read_text())[TAG]
    action_groups = []
    offset = 0
    for key, value in stats["action"].items():
        width = len(value["q01"])
        action_groups.append({"name": key, "start": offset, "stop": offset + width})
        offset += width
    manifest = {
        "format_version": 1,
        "name": "Force FP16 NX baseline",
        "mode": "Force FP16",
        "case_count": count,
        "created_utc": datetime.now(timezone.utc).isoformat(),  # noqa: UP017 - local Orin diagnostics use Python 3.10
        "case_ids": case_ids.tolist(),
        "data_origin": "Exact 100 synthetic DROID observations saved in the previous local experiment; not real robot recordings.",
        "checkpoint_repo": "nvidia/GR00T-N1.7-3B",
        "checkpoint_revision": download["revision"],
        "checkpoint_weight_sha256": weight_hashes,
        "weights_included": False,
        "repository_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(),
            "compute_capability": list(torch.cuda.get_device_capability()),
        },
        "inference": {
            "embodiment_tag": TAG,
            "batch_size": 1,
            "action_horizon": 40,
            "denoising_steps": 4,
            "weight_dtype": "float16",
            "use_bf16": False,
            "use_flash_attention": False,
            "eval_mode": True,
            "tf32": False,
            "noise": "saved exact float16 values; do not regenerate",
            "deterministic_algorithms_enabled": torch.are_deterministic_algorithms_enabled(),
            "precision_scope": "All explicit floating model tensor operations FP16, scaled RMSNorm and FP16 RoPE. Pre/postprocessing and CUDA kernel-internal accumulation unchanged.",
            "implementation": "Existing pure_fp16 patches plus full_fp16() TorchDispatch guard in provenance/examples/groot_fp16/full_fp16_experiment.py",
        },
        "action_groups": action_groups,
        "array_conventions": "Raw inputs have [case,batch,...]; model input files retain batch=1; reference outputs have [case,time,dimension] with batch removed.",
        "model_input_boundary": "Filtered preprocessor tensors accepted by policy.predict_action_chunk or GR00TN17.get_action. Saved original dtypes before Force FP16 enforcement; preserve integer/bool dtypes.",
        "decode_state_requirement": "Run the matching raw observation through pre immediately before post(raw), or supply its raw state explicitly to your decoder. Relative-action decoding reads state cached by pre.",
        "source_sha256": {name: sha256(Path(name)) for name in source_files},
        "cases": [],
        "complete": False,
    }

    def save_manifest() -> None:
        (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    save_manifest()
    raw_outputs, decoded_outputs, audits = [], [], []
    for i, case_id in enumerate(case_ids):
        batch = {"observation.state": torch.from_numpy(inputs["state"][i]), "task": [str(inputs["tasks"][i])]}
        for camera in ("exterior_image_1_left", "wrist_image_left"):
            batch["observation.images." + camera] = torch.from_numpy(inputs[camera][i])
        with torch.no_grad():
            model_input = pre(deepcopy(batch))
        assert tensor_hash(model_input) == old_hashes[i]
        noise = torch.from_numpy(inputs["noise"][i]).cuda()
        before = tensor_hash({**model_input, "fixed_noise": noise})
        filtered = policy._filter_groot_inputs(model_input, include_action=False)
        assert all(isinstance(v, torch.Tensor) for v in filtered.values())
        model_arrays = {k: v.detach().cpu().numpy().copy() for k, v in filtered.items()}
        model_arrays["noise"] = noise.cpu().numpy().copy()
        input_path = out / "model_inputs" / f"{case_id}.npz"
        np.savez_compressed(input_path, **model_arrays)
        with torch.inference_mode():
            with full_fp16() as audit:
                raw = policy.predict_action_chunk(filtered, noise=noise)
            decoded = post(raw.clone())
            torch.cuda.synchronize()
        assert tensor_hash({**model_input, "fixed_noise": noise}) == before
        assert not audit.report()["non_fp16_floating_outputs"]
        raw_np = raw[0].cpu().numpy().copy()
        decoded_np = decoded[0].cpu().numpy().copy()
        assert np.isfinite(raw_np).all() and np.isfinite(decoded_np).all()
        assert np.array_equal(decoded_np, old_decoded[i]), (
            f"Decoded differs from previous Force FP16, case {i}"
        )
        assert np.array_equal(raw_np[:, :17], old_raw[i]), f"Raw active dimensions differ, case {i}"
        raw_outputs.append(raw_np)
        decoded_outputs.append(decoded_np)
        audits.append({"case_id": str(case_id), **audit.report()})
        manifest["cases"].append(
            {
                "case_id": str(case_id),
                "index": i,
                "task": str(inputs["tasks"][i]),
                "raw_input_sha256": str(inputs["input_sha256"][i]),
                "model_input_file": str(input_path.relative_to(out)),
                "model_input_file_sha256": sha256(input_path),
                "model_input_and_noise_sha256": before,
                "model_arrays": {
                    k: {"shape": list(v.shape), "dtype": str(v.dtype)} for k, v in model_arrays.items()
                },
            }
        )
        if (i + 1) % 10 == 0:
            save_manifest()
            print(f"Export {i + 1}/100; exact match to previous Force FP16 outputs", flush=True)
    raw_all, decoded_all = np.stack(raw_outputs), np.stack(decoded_outputs)
    np.savez_compressed(
        out / "reference_outputs.npz",
        case_ids=case_ids,
        raw=raw_all,
        decoded=decoded_all,
        action_ranges=ranges,
        input_sha256=inputs["input_sha256"],
    )
    (out / "dtype_audits.json").write_text(json.dumps(audits, indent=2) + "\n")
    manifest["outputs"] = {
        "raw": {
            "shape": list(raw_all.shape),
            "dtype": str(raw_all.dtype),
            "meaning": "Normalized model action chunk; first 17 dimensions active for this embodiment; keep all 132 for model-level parity.",
        },
        "decoded": {
            "shape": list(decoded_all.shape),
            "dtype": str(decoded_all.dtype),
            "meaning": "Final unnormalized/absolute actions after matching-state postprocessing.",
        },
    }
    manifest["validation"] = {
        "all_finite": True,
        "strict_fp16_audit_passed": True,
        "decoded_exact_match_to_previous_run": 100,
        "raw_active_exact_match_to_previous_run": 100,
        "inputs_unchanged": True,
        "weights_unchanged": all(
            (p.data_ptr(), p._version) == weights_before[n] for n, p in policy.named_parameters()
        ),
    }
    assert manifest["validation"]["weights_unchanged"]
    manifest["complete"] = True
    save_manifest()
    shutil.copyfile(Path(__file__).with_name("baseline_tools.py"), out / "baseline_tools.py")
    shutil.copyfile(Path(__file__).with_name("BASELINE_FORMAT.txt"), out / "README.txt")
    print("DONE", out, json.dumps(manifest["validation"]), flush=True)


if __name__ == "__main__":
    main()
