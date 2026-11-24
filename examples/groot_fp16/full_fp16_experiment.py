"""Force FP16 tensor enforcement; pair with pure_fp16=True for scaled RMSNorm.

This experimental context changes the process-wide default dtype. Use it for
single-threaded inference, not concurrent model serving. Pre/postprocessing and
CUDA kernel-internal accumulators are outside its scope. Python dispatch adds
overhead, so timings are not deployment benchmarks.
"""

from collections import Counter
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any

import torch
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_flatten, tree_map

HIGH_PRECISION = (torch.float32, torch.float64, torch.bfloat16)


class FP16TensorCompute(TorchDispatchMode):
    def __init__(self) -> None:
        self.converted_inputs = Counter()
        self.dtype_overrides = Counter()
        self.non_fp16_outputs = Counter()
        self.ops = Counter()

    def __torch_dispatch__(
        self, func: Callable, types: tuple[type, ...], args: tuple = (), kwargs: dict | None = None
    ) -> Any:
        name = str(func)
        self.ops[name] += 1

        def convert(value: Any) -> Any:
            if isinstance(value, torch.Tensor) and value.dtype in HIGH_PRECISION:
                self.converted_inputs[name] += 1
                return value.to(torch.float16)
            if isinstance(value, torch.dtype) and value in HIGH_PRECISION:
                self.dtype_overrides[name] += 1
                return torch.float16
            return value

        args = tree_map(convert, args)
        kwargs = tree_map(convert, kwargs or {})
        result = func(*args, **kwargs)
        flat, _ = tree_flatten(result)
        bad = [
            x
            for x in flat
            if isinstance(x, torch.Tensor) and x.is_floating_point() and x.dtype != torch.float16
        ]
        if bad:
            self.non_fp16_outputs[name] += len(bad)
            raise RuntimeError(
                f"Non-FP16 tensor escaped {name}: {[(str(x.dtype), list(x.shape)) for x in bad]}"
            )
        return result

    def report(self) -> dict:
        return {
            "operator_calls": sum(self.ops.values()),
            "input_casts": dict(self.converted_inputs),
            "dtype_request_overrides": dict(self.dtype_overrides),
            "non_fp16_floating_outputs": dict(self.non_fp16_outputs),
        }


@contextmanager
def full_fp16() -> Iterator[FP16TensorCompute]:
    old = torch.get_default_dtype()
    mode = FP16TensorCompute()
    torch.set_default_dtype(torch.float16)
    try:
        with mode:
            yield mode
    finally:
        torch.set_default_dtype(old)


def naive_rmsnorm(self: torch.nn.Module, hidden_states: torch.Tensor) -> torch.Tensor:
    variance = hidden_states.pow(2).mean(-1, keepdim=True)
    return self.weight * (hidden_states * torch.rsqrt(variance + self.variance_epsilon))
