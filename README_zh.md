# RecRL: 推荐大模型强化学习训练框架

RecRL 是一个干净、模块化且高度可扩展的强化学习（RL）框架，专为推荐系统大模型（LLM for Recommendation）设计。受火山引擎 VERL 设计理念的启发，**RecRL** 提供了一套优雅的架构，将数据流、生成采样、奖励计算和训练逻辑进行了彻底的解耦。

*Read this in [English](README.md).*

## ✨ 核心特性

- **模块化“引擎”设计**: 彻底解耦 `DataEngine`（数据）、`RolloutEngine`（生成）、`RewardEngine`（奖励）和 `TrainerEngine`（训练器）。
- **原生多算法支持**: 开箱即用支持 **GRPO** (Group Relative Policy Optimization) 和 **EEPO** (Explore-and-Evaluate Policy Optimization 快速权重探索机制)。
- **可插拔复合奖励**: 像搭积木一样自由组合多个奖励函数（例如：语义相似度、精确匹配度、NDCG 等）并自定义权重分配。
- **强制合规解码 (Constrained Decoding)**: 内置基于前缀树 (Trie) 的 `ConstrainedLogitsProcessor`，严格确保 LLM 只能生成合法的物品 ID（SID），彻底消除推荐时的“幻觉”，防止 RL 初期发生奖励坍塌。
- **极简扩展 API**: 想要添加新算法（如 DPO、PPO）？只需继承核心 Trainer 并重写 `compute_loss()`，代码量不超过 100 行。

## 📦 安装指南

环境要求: Python 3.8+ 和 PyTorch 2.0+。

```bash
# 克隆仓库
https://github.com/zzr9468-sys/CleanOneRec.git
cd RecRL

# 使用可编辑模式安装
pip install -e .
```

## 🚀 快速上手

### 1. 标准 GRPO 训练

```python
from recrl.core import DataEngine, RolloutEngine
from recrl.algorithms.grpo import GRPOTrainer, GRPOConfig
from recrl.rewards import TextSemanticReward
from recrl.data import RepeatRandomSampler

# 加载推荐数据集 (支持 OpenOneRec RecIF / Parquet 格式)
train_dataset = DataEngine.from_parquet("data/train.parquet", format="recif")

# 初始化核心引擎
rollout_engine = RolloutEngine(tokenizer, device="cuda")
reward_engine = TextSemanticReward(recif_path="data/recif_metadata")

# 配置训练器与采样器
config = GRPOConfig(num_generations=16, temperature=0.7)
sampler = RepeatRandomSampler(train_dataset, repeat_count=16)

# 启动训练
trainer = GRPOTrainer(
    model=model,
    ref_model=ref_model,
    config=config,
    train_dataset=train_dataset,
    rollout_engine=rollout_engine,
    reward_engine=reward_engine,
    tokenizer=tokenizer,
    sampler=sampler
)
trainer.train()
```

### 2. EEPO (快速权重探索) 训练

EEPO 通过在生成阶段对 `lm_head` 执行即时的 Fast-Weight 更新，强制模型进行探索，有效打破传统基于模仿学习的“师生曝光偏差”。

```python
from recrl.algorithms.eepo import EEPOTrainer, EEPOConfig

config = EEPOConfig(
    num_generations=16,
    eepo_enabled=True,
    eepo_stage1_ratio=0.5,  # 50% 模型利用 (Exploitation), 50% 强制探索 (Exploration)
    eepo_unlearn_lr=1e-5,
    add_gt=True
)

trainer = EEPOTrainer(
    model=model,
    config=config,
    train_dataset=train_dataset,
    rollout_engine=rollout_engine,
    reward_engine=reward_engine,
    tokenizer=tokenizer,
    sampler=sampler
)
trainer.train()
```

### 3. 组合多个奖励函数

你可以使用 `CompositeReward` 轻松平衡相关性、新颖性和准确率：

```python
from recrl.core import CompositeReward
from recrl.rewards import TextSemanticReward, ExactMatchReward

reward_engine = CompositeReward([
    (TextSemanticReward(recif_path="data/"), 0.8),  # 80% 权重由语义相似度决定
    (ExactMatchReward(), 0.2)                       # 20% 权重由 ID 完全匹配决定
])
```

## 🏗 架构总览

不同于过去将所有逻辑塞在一个巨大的脚本中，RecRL 采用了高度解耦的引擎架构：

```text
recrl/
├── core/                       # 核心基类抽象
│   ├── base_trainer.py         # 训练主循环 (BaseRLTrainer)
│   ├── rollout.py              # 文本生成与 Logprobs 算子 (RolloutEngine)
│   ├── reward.py               # 奖励基类与接口 (CompositeReward)
│   └── data.py                 # 统一数据解析 (DataEngine)
├── algorithms/                 # 具体强化学习算法实现
│   ├── grpo/                   # 标准 GRPO
│   └── eepo/                   # EEPO (Unlearn / 权重突变)
├── rewards/                    # 可插拔的奖励函数库
│   ├── semantic.py             # 引入 Sentence-Transformer 算语义分
│   └── rule.py                 # 精确匹配与 NDCG 计算
├── constraints/                # 强制合规生成控制
│   ├── trie.py                 # 针对有效 SID 的前缀树
│   └── processor.py            # 安全防 NaN 的 ConstrainedLogitsProcessor
└── examples/                   # 开箱即用的参考训练脚本
```

## 🤝 参与贡献

欢迎提交 Issue 和 Pull Request！如果想要加入新算法（例如 DPO）：
1. 在 `recrl/algorithms/dpo/` 创建目录。
2. 继承 `BaseRLTrainer`，只需重写其中的 `compute_loss()`。
3. 你的算法会自动继承来自父类的合规解码约束、组合奖励分配以及批次数据加载能力。

## 📄 开源协议

本项目基于 Apache 2.0 协议开源。
