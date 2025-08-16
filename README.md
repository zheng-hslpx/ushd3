
# USV-HGNN-PPO

多无人船（USV）任务调度：异构图神经网络（HGNN） + PPO 基线实现。**开箱即用**：解压、`pip install -r requirements.txt`，然后运行：

```bash
python train.py --config config.json
```

## 目录结构

```
usv-hgnn-ppo/
├── config.json
├── config
│   ├── default.json
│   ├── improved_config.json
├── requirements.txt
├── train.py
├── evaluate.py
├── usv_agent/
│   ├── __init__.py
│   ├── data_generator.py
│   ├── hgnn_model.py
│   └── usv_env.py
│   └── ppo_policy.py
├── utils/
│   ├── __init__.py
│   ├── color_config.json
│   └── vis_manager.py
├── data/
│   └── instances/
├── results/
│   ├── figures/
│   └── saved_models/ImprovedTune_5x24_v2/01...
│   └── report/ImprovedTune_5x24_v2/01...
└── scripts/
    └── run_train.sh
```

## 快速开始

1. **安装依赖**（Windows 会安装 `gym`，其他平台安装 `gymnasium`）：
   ```bash
   pip install -r requirements.txt
   ```
2. **训练**：
   ```bash
   python train.py --config config.json
   ```
3. **评估**（可选，加载权重）：
   ```bash
   python evaluate.py --config config.json --checkpoint results/saved_models/agent_ep2.pt
   ```

## 配置说明

- `env_paras`：环境参数（USV数量、任务数量、地图大小等）。
- `model_paras`：HGNN结构与维度。
- `train_paras`：PPO超参数、可视化开关等。

> 本仓库实现了一个**可运行的基线**，方便你逐步替换/升级到你的论文版本。Visdom 可视化默认关闭，若开启需启动 `visdom` 服务器。

## FAQ

- **没有装 Visdom？** 没关系，代码会自动降级，不影响训练。
- **CUDA 不可用？** 默认使用 CPU，可在 `config.json` 里把 `device` 改为 `cuda:0` 并确保 `torch` 可用。
- **动作非法/梯度 NaN？** 已做掩码与回退处理；如果仍出现，请调低 `num_tasks` 或学习率。
