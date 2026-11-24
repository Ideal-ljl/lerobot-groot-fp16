"""Precision enforcement and portable baseline validation without model weights."""

import numpy as np
import pytest
import torch

from examples.groot_fp16.baseline_tools import compare
from examples.groot_fp16.full_fp16_experiment import full_fp16


def test_force_fp16_overrides_upcasts_and_preserves_integer_inputs():
    original = torch.tensor([1.25, 2.5], dtype=torch.float32)
    indices = torch.tensor([0, 1], dtype=torch.int64)
    saved = original.clone()
    default = torch.get_default_dtype()
    with full_fp16() as audit:
        result = original.double() + torch.ones(2, dtype=torch.float32)
        factory = torch.ones(2)
        selected = result[indices]
        assert result.dtype == factory.dtype == selected.dtype == torch.float16
        assert indices.dtype == torch.int64
    assert torch.get_default_dtype() == default
    assert original.dtype == torch.float32
    torch.testing.assert_close(original, saved)
    torch.testing.assert_close(result.float(), saved + 1, rtol=0, atol=0)
    assert audit.report()["dtype_request_overrides"]
    assert audit.report()["non_fp16_floating_outputs"] == {}


def test_force_fp16_restores_default_dtype_after_exception():
    default = torch.get_default_dtype()
    with pytest.raises(RuntimeError, match="intentional"), full_fp16():
        assert torch.get_default_dtype() == torch.float16
        raise RuntimeError("intentional")
    assert torch.get_default_dtype() == default
    assert torch.ones(1, dtype=torch.float32).dtype == torch.float32


@pytest.fixture
def reference(tmp_path):
    arrays = {
        "case_ids": np.array(["case_000", "case_001"]),
        "input_sha256": np.array(["first", "second"]),
        "decoded": np.zeros((2, 40, 17), dtype=np.float16),
        "action_ranges": np.ones(17, dtype=np.float64),
    }
    np.savez(tmp_path / "reference_outputs.npz", **arrays)
    return tmp_path, arrays


def test_baseline_comparison_detects_numeric_error(reference):
    root, arrays = reference
    assert compare(root, root / "reference_outputs.npz")["outputs"]["decoded"]["bitexact_cases"] == 2
    arrays["decoded"][1, 3, 4] = 0.5
    path = root / "candidate.npz"
    np.savez(path, **arrays)
    result = compare(root, path)
    assert not result["passed"]
    assert result["outputs"]["decoded"]["changed_elements"] == 1
    assert result["outputs"]["decoded"]["M_max_pct"] == 50
    assert compare(root, path, atol=0.5)["passed"]


@pytest.mark.parametrize("field", ["case_ids", "input_sha256", "decoded"])
def test_baseline_comparison_rejects_misaligned_data(reference, field):
    root, arrays = reference
    arrays[field] = arrays[field][::-1] if field != "decoded" else arrays[field][:, :1]
    np.savez(root / "candidate.npz", **arrays)
    with pytest.raises(ValueError):
        compare(root, root / "candidate.npz")


def test_baseline_comparison_rejects_nonfinite_output(reference):
    root, arrays = reference
    arrays["decoded"][0, 0, 0] = np.nan
    np.savez(root / "candidate.npz", **arrays)
    result = compare(root, root / "candidate.npz")
    assert not result["passed"]
    assert result["outputs"]["decoded"]["nonfinite_case_ids"] == ["case_000"]
