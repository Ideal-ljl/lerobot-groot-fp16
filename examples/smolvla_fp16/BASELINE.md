# SmolVLA Force FP16 两步基准（v2）

每包有 10 条固定的合成算例。逐条推理（batch=1），每次重置策略，以封存的初始噪声执行 2 步 Flow Matching。基准设备见 manifest.json 的 gpu 字段。跨设备评分使用同一份基准包、相同权重及处理器，并比较 FP16 反归一化后的最终动作。

| NPZ 键 | 内容 |
|---|---|
| obs/observation.images.camera1、camera2、camera3（均带相同前缀） | 三路 FP16 合成 RGB 图像，CHW，范围 [0,1] |
| obs/observation.state、task | 6 维 FP16 状态和文本指令 |
| input/ 开头的数组 | 带 batch 维的策略输入。浮点张量为 FP16，词元编号和掩码保留各自类型 |
| noise | 固定的 FP16 初始噪声，形状为 [1,50,32] |
| raw | FP16 归一化动作，形状为 [1,50,6] |
| decoded | **评分用最终动作**，由配套后处理器在 FP16 下反归一化得到，形状为 [1,50,6] |

默认选用权重包中 `so100.buffer.action` 的 6 组均值和标准差，逐维执行 FP16 乘法，再执行 FP16 加法。两台设备需选用同一组参数。使用其他参数时，通过 `--action-stats-key` 指定，导出包的 `action_stats_key` 字段记录实际选择。

manifest.json 记录步数、评分字段、权重及处理器文件的 SHA256、各算例的 SHA256 和运行环境。图像、状态和噪声直接按 FP16 生成，即使种子相同，也应使用 v2 包内实际封存的张量。旧版 10 步基准需要在 NX 上按新配置重新导出后才能用于两步实验评分。

解压后可按下例在目标设备上回放一条算例。请从仓库根目录运行，并先按仓库 README 下载相同版本的策略和处理器。

```python
import json
from pathlib import Path

import numpy as np
import torch

from examples.smolvla_fp16.verify_fp16 import enable_force_fp16, infer_actions, load_policy

manifest = json.loads(Path("manifest.json").read_text())
policy, _, post = load_policy(
    Path("checkpoints/smolvla_base"),
    Path("checkpoints/SmolVLM2-500M-Video-Instruct"),
    manifest["action_stats_key"],
)
enable_force_fp16(policy)
with torch.inference_mode(), np.load("case_000.npz", allow_pickle=False) as case:
    batch = {k[6:]: torch.from_numpy(case[k]).cuda() for k in case.files if k.startswith("input/")}
    noise = torch.from_numpy(case["noise"]).cuda()
    _, actions, _ = infer_actions(policy, post, batch, noise)
    actual = actions.cpu().numpy()
    reference = case["decoded"]
```
