"""Shared model loader for the Force FP16 experiments (Python 3.12+)."""

import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

import lerobot.utils.import_utils as available

with patch.object(available, "_datasets_available", False):
    import lerobot.policies.groot.processor_groot as processors
import verify_fp16 as verify

from lerobot.policies.groot.groot_n1_7 import GR00TN17
from lerobot.policies.groot.modeling_groot import GrootPolicy

CHECKPOINT = "checkpoints/GR00T-N1.7-3B"
TAG = "oxe_droid_relative_eef_relative_joint"
PROCESSOR = "checkpoints/Cosmos-Reason2-2B"


def setup(n: int = 100, checkpoint: str = CHECKPOINT, processor: str = PROCESSOR) -> tuple:
    processors.GROOT_N1_7_BACKBONE_MODEL = processor
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    root = Path(checkpoint)
    stats = json.loads((root / "statistics.json").read_text())[TAG]

    def cat(d: dict, k: str) -> np.ndarray:
        return np.concatenate([np.asarray(v[k], dtype=np.float64) for v in d.values()])

    ranges = np.maximum(cat(stats["action"], "q99") - cat(stats["action"], "q01"), 1e-6)
    cases, digests = verify.make_cases(
        n,
        cat(stats["state"], "q01"),
        cat(stats["state"], "q99"),
        40,
        132,
        ["exterior_image_1_left", "wrist_image_left"],
        20261123,
    )
    original = GR00TN17.from_pretrained

    def direct(**kwargs) -> torch.nn.Module:
        return original(
            **kwargs,
            device_map="cuda",
            dtype=torch.float16,
            load_bf16=False,
            backbone_trainable_params_fp32=False,
        )

    with patch.object(GR00TN17, "from_pretrained", side_effect=direct):
        policy = GrootPolicy.from_pretrained(
            checkpoint,
            embodiment_tag=TAG,
            device="cuda",
            dtype=torch.float16,
            use_bf16=False,
            pure_fp16=True,
            use_flash_attention=False,
            tune_llm=False,
            tune_visual=False,
            tune_projector=False,
            tune_diffusion_model=False,
            tune_vlln=False,
        )
    policy.eval().requires_grad_(False)
    pre, post = processors.make_groot_pre_post_processors(config=policy.config, dataset_stats=None)
    return policy, pre, post, cases, digests, ranges
