"""Export and replay ten fixed SmolVLA Force FP16 cases on CUDA."""

import argparse
import hashlib
import json
import platform
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZIP_STORED, ZipFile

import numpy as np
import torch
import transformers

from examples.smolvla_fp16.verify_fp16 import enable_force_fp16, infer_actions, load_policy, prepare_inputs
from lerobot.policies.smolvla import SmolVLAPolicy
from lerobot.processor import PolicyProcessorPipeline
from lerobot.utils.constants import OBS_STATE

TASKS = (
    "Pick up the block.",
    "Place the block in the tray.",
    "Move the cube to the left.",
    "Lift the cup.",
    "Close the drawer.",
    "Push the block forward.",
    "Move the cube to the right.",
    "Open the drawer.",
    "Place the cup on the table.",
    "Pick up the red cube.",
)


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(2**20), b""):
            h.update(chunk)
    return h.hexdigest()


def infer(
    policy: SmolVLAPolicy, post: PolicyProcessorPipeline, batch: dict[str, torch.Tensor], noise: torch.Tensor
) -> tuple[np.ndarray, np.ndarray, int]:
    raw, actions, calls = infer_actions(policy, post, batch, noise)
    return raw.cpu().numpy(), actions.cpu().numpy(), calls


def check_bits(actual: np.ndarray, expected: np.ndarray, label: str) -> None:
    if (
        actual.dtype != expected.dtype
        or actual.shape != expected.shape
        or actual.tobytes() != expected.tobytes()
    ):
        raise ValueError(f"Replay mismatch: {label}")


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", type=Path, default=Path("checkpoints/smolvla_base"))
    parser.add_argument("--backbone", type=Path, default=Path("checkpoints/SmolVLM2-500M-Video-Instruct"))
    parser.add_argument("--out", type=Path, default=Path("outputs/smolvla_force_fp16_2step_baseline_10.zip"))
    args = parser.parse_args()
    if args.out.exists():
        parser.error(f"Output already exists: {args.out}")
    for directory in (args.ckpt, args.backbone):
        if not (directory / "config.json").is_file():
            parser.error(f"Missing {directory}/config.json")
    policy, pre, post = load_policy(args.ckpt, args.backbone)
    norms = enable_force_fp16(policy)
    cfg = policy.config
    state = cfg.robot_state_feature
    if state is None:
        raise ValueError("Missing state feature")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    cases = []
    with TemporaryDirectory(dir=args.out.parent) as tmp:
        root = Path(tmp)
        for i, task in enumerate(TASKS):
            seed = 42 + i
            rng = torch.Generator().manual_seed(seed)
            obs = {
                key: torch.rand(f.shape, generator=rng, dtype=torch.float16)
                for key, f in cfg.image_features.items()
            }
            obs[OBS_STATE] = torch.randn(state.shape, generator=rng, dtype=torch.float16)
            inputs = prepare_inputs(pre, obs, task)
            noise = torch.randn(
                (1, cfg.chunk_size, cfg.max_action_dim), generator=rng, dtype=torch.float16
            ).cuda()
            arrays = {"obs/" + k: v.numpy().copy() for k, v in obs.items()}
            arrays.update({"input/" + k: v.cpu().numpy().copy() for k, v in inputs.items()})
            raw, actions, calls = infer(policy, post, inputs, noise)
            path = root / f"case_{i:03}.npz"
            np.savez_compressed(
                path, **arrays, task=task, noise=noise.cpu().numpy(), raw=raw, decoded=actions
            )

            with np.load(path, allow_pickle=False) as saved:
                observation = {k[4:]: torch.from_numpy(saved[k]) for k in saved.files if k.startswith("obs/")}
                restored = prepare_inputs(pre, observation, saved["task"].item())
                for k in inputs:
                    check_bits(restored[k].cpu().numpy(), saved["input/" + k], k)
                for _ in range(2):
                    replay = {k: torch.from_numpy(saved["input/" + k]).cuda() for k in inputs}
                    r, d, _ = infer(policy, post, replay, torch.from_numpy(saved["noise"]).cuda())
                    check_bits(r, saved["raw"], f"{path.name} raw")
                    check_bits(d, saved["decoded"], f"{path.name} decoded")
            cases.append({"file": path.name, "seed": seed, "sha256": digest(path), "operator_calls": calls})
            print(
                f"{path.name}: inputs verified, raw and final actions bit-exact on both replays", flush=True
            )
        manifest = {
            "format_version": 2,
            "scoring_output": "decoded",
            "model": "lerobot/smolvla_base",
            "weight_sha256": digest(args.ckpt / "model.safetensors"),
            "checkpoint_metadata_sha256": {
                p.name: digest(p)
                for p in sorted(args.ckpt.iterdir())
                if p.suffix in {".json", ".safetensors"} and p.name != "model.safetensors"
            },
            "backbone_assets_sha256": {
                p.name: digest(p) for p in sorted(args.backbone.iterdir()) if p.suffix in {".json", ".txt"}
            },
            "precision": (
                "FP16 observations, noise, normalization, forward and unnormalization; "
                "scaled Llama RMSNorm, TF32 off, FP16 GEMM accumulation enabled"
            ),
            "scaled_norms": norms,
            "denoising_steps": cfg.num_steps,
            "torch": str(torch.__version__),
            "transformers": transformers.__version__,
            "python": platform.python_version(),
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(),
            "verification": "raw observations reprocessed bit-exactly; two disk replays matched the export",
            "cases": cases,
        }
        (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        archive = root / "bundle.zip"
        with ZipFile(archive, "w", compression=ZIP_STORED) as bundle:
            bundle.write(Path(__file__).with_name("BASELINE.md"), "README.md")
            bundle.write(root / "manifest.json", "manifest.json")
            for path in sorted(root.glob("case_*.npz")):
                bundle.write(path, path.name)
        archive.rename(args.out)
    print(f"{args.out}  sha256={digest(args.out)}")


if __name__ == "__main__":
    main()
