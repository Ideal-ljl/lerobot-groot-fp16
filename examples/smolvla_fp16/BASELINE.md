# SmolVLA Force FP16 数值基准

在 Orin 上逐条运行合成观测，使用 batch=1、固定初始噪声，每条推理前重置策略。每个 case_NNN.npz 文件对应一条完整记录，NumPy 可直接读取。比较时使用文件中保存的张量和噪声，保留 batch 维及各自 dtype。

| NPZ 键 | 含义 |
|---|---|
| obs/observation.images.camera1、camera2、camera3（均带相同前缀） | 三路合成 RGB 图像，各为 float32 CHW，范围 [0,1] |
| obs/observation.state、task | 原始 6 维状态和文本指令 |
| input/ 开头的全部数组 | 策略预处理器的张量输出，含 batch 维。去除 input/ 前缀后作为 predict_action_chunk 的 batch |
| noise | 固定初始噪声，float32 [1,50,32]，经 noise 参数传入 |
| raw | predict_action_chunk 的归一化动作，float16 [1,50,6] |
| decoded | 使用配套后处理器处理 raw.float() 得到的动作，float32 [1,50,6] |

解压后可用以下代码读取一条。复现时使用同一 checkpoint、缩放 RMSNorm 和 full_fp16() 上下文。预处理与后处理在该上下文外运行。manifest.json 记录权重和各样本的 SHA256、运行环境及验证方法。

```python
import numpy as np

with np.load("case_000.npz", allow_pickle=False) as case:
    batch = {k.removeprefix("input/"): case[k] for k in case.files if k.startswith("input/")}
    noise = case["noise"]
    expected_raw = case["raw"]
    expected_decoded = case["decoded"]
```
