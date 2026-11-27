#!/usr/bin/env bash
set -euo pipefail

if [[ $# != 2 || ${1:-} == --help || ${1:-} == -h ]]; then
    echo "用法: $0 <SmolVLA 权重目录或 model.safetensors> <两步 FP16 基准 ZIP 或解压目录>"
    if [[ ${1:-} == --help || ${1:-} == -h ]]; then exit 0; fi
    exit 2
fi
repo=$(cd -- "$(dirname -- "$0")" && pwd)
export PYTHONPATH="$repo/src:$repo${PYTHONPATH:+:$PYTHONPATH}"
if [[ -n ${VIRTUAL_ENV:-} ]]; then
    runner=("$VIRTUAL_ENV/bin/python")
elif [[ -x "$repo/.venv/bin/python" ]]; then
    runner=("$repo/.venv/bin/python")
elif [[ -n ${CONDA_PREFIX:-} ]]; then
    runner=("$CONDA_PREFIX/bin/python")
elif command -v uv >/dev/null; then
    runner=(uv run --project "$repo" --locked --extra smolvla python)
else
    runner=(python3)
fi
exec "${runner[@]}" -m examples.smolvla_fp16.benchmark "$1" "$2"
