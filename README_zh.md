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

环境要求: Python 3.10+ 和 PyTorch 2.0+。

### 使用 uv 安装（推荐）

```bash
# 安装 uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# 克隆仓库
git clone https://github.com/zzr9468-sys/CleanOneRec.git
cd CleanOneRec

# 安装依赖
uv sync

# 激活环境
source .venv/bin/activate
```

### 使用 pip 安装

```bash
git clone https://github.com/zzr9468-sys/CleanOneRec.git
cd CleanOneRec
pip install -e .
```

详细安装说明请参考 [安装指南](docs/INSTALLATION.md)。

## 🚀 快速上手

### 单卡训练（推荐）

```bash
# GRPO + 约束解码
bash train_constrained.sh
```

使用前缀树约束解码，确保只生成合法的物品 ID。

### 多卡训练

```bash
# 4 卡 Accelerate DDP
bash train_multi_gpu.sh
```

### EEPO 训练（增强探索）

```bash
# EEPO 快速权重探索
bash train_eepo.sh
```

### 自定义训练

```python
from recrl.core import DataEngine, RolloutEngine
from recrl.algorithms.grpo import GRPOTrainer, GRPOConfig
from recrl.rewards import CompositeReward, LongviewReward, SemanticReward
from recrl.constraints import SIDTrie

# 加载数据
train_dataset = DataEngine.from_recif_parquet(“data/train.parquet”)

# 配置引擎
rollout_engine = RolloutEngine(tokenizer, device=”cuda”)
trie = SIDTrie.from_recif(“path/to/RecIF”, tokenizer)
rollout_engine.set_trie(trie)

# 复合奖励
reward_engine = CompositeReward([
    (LongviewReward(“path/to/RecIF”), 0.5),
    (SemanticReward(“path/to/RecIF”), 0.3),
])

# 训练
config = GRPOConfig(num_generations=4, temperature=0.7)
trainer = GRPOTrainer(
    model=model,
    config=config,
    train_dataset=train_dataset,
    rollout_engine=rollout_engine,
    reward_engine=reward_engine,
    tokenizer=tokenizer,
)
trainer.train()
```

详细训练教程请参考 [训练指南](docs/TRAINING.md)。

## 🏗 架构总览

不同于过去将所有逻辑塞在一个巨大的脚本中，RecRL 采用了高度解耦的引擎架构：

```text
recrl/
├── core/                       # 核心基类抽象
│   ├── base_trainer.py         # 训练主循环（支持多卡）
│   ├── rollout.py              # 文本生成与 Logprobs 算子
│   ├── reward.py               # 奖励基类与接口
│   └── data.py                 # 统一数据解析
├── algorithms/                 # 具体强化学习算法实现
│   ├── grpo/                   # 标准 GRPO
│   │   ├── trainer.py          # GRPO 训练器
│   │   └── config.py           # GRPO 配置
│   └── eepo/                   # EEPO（快速权重探索）
│       ├── trainer.py          # EEPO 训练器
│       ├── unlearn.py          # 快速权重反学习
│       └── config.py           # EEPO 配置
├── rewards/                    # 可插拔的奖励函数库
│   ├── longview_reward.py      # 用户历史对齐
│   ├── semantic.py             # 语义相似度
│   ├── novelty_reward.py       # 新颖性奖励
│   └── diversity_reward.py     # 多样性奖励
├── constraints/                # 强制合规生成控制
│   ├── trie.py                 # 针对有效 SID 的前缀树
│   └── processor.py            # ConstrainedLogitsProcessor
└── data/                       # 数据工具
    └── sampler.py              # RepeatRandomSampler
```

## 📊 性能表现

### 约束解码 vs 非约束解码

| 指标 | 非约束 | 约束 | 提升 |
|------|--------|------|------|
| 平均奖励 | -0.19 | **0.78** | **+488%** |
| 有效 SID 率 | ~50% | **100%** | **+100%** |
| 正奖励率 | 48.9% | **99.7%** | **+104%** |

**结论**: 约束解码对 RecRL 至关重要。

### 训练效率

| 配置 | 步数/秒 | GPU 显存 | 每轮耗时 |
|------|---------|----------|----------|
| 单卡 | 0.12 | 20GB | ~46 小时 |
| 4 卡 (DDP) | 0.45 | 20GB/卡 | ~12 小时 |

详细性能分析请参考 [性能基准](docs/BENCHMARKS.md)。

## 📚 文档

- [安装指南](docs/INSTALLATION.md) - 环境配置和安装步骤
- [训练指南](docs/TRAINING.md) - 单卡/多卡训练、超参数调优
- [算法文档](docs/ALGORITHMS.md) - GRPO、EEPO、约束解码、奖励设计
- [性能基准](docs/BENCHMARKS.md) - 性能对比和消融实验

## 🎯 核心功能详解

### 约束解码

基于前缀树的 logits 处理器，确保只生成合法的语义 ID（SID）：

```python
from recrl.constraints import SIDTrie

# 从 RecIF 元数据构建前缀树
trie = SIDTrie.from_recif("path/to/RecIF", tokenizer)
rollout_engine.set_trie(trie)
```

**效果**: 相比非约束生成，奖励提升 488%。

### EEPO（探索-评估）

两阶段生成，带快速权重探索：

1. **利用阶段**（50%）：用当前策略生成
2. **探索阶段**（50%）：应用快速权重突变后生成

**优势**: 跳出局部最优，提升多样性。

### 复合奖励

灵活的奖励组合：

```python
reward_engine = CompositeReward([
    (LongviewReward(recif_path), 0.50),   # 用户历史对齐
    (SemanticReward(recif_path), 0.30),   # 内容相似度
    (NoveltyReward(), 0.15),              # 探索奖励
    (DiversityReward(), 0.05),            # 组内多样性
])
```

### 内存优化

- **梯度检查点**: 节省约 40% GPU 显存
- **参考模型 CPU 卸载**: 节省约 3.4GB GPU 显存
- **混合精度（bf16）**: 节省约 50% 显存

**结果**: 1.7B 模型可在 24GB GPU 上训练。

## 🤝 参与贡献

欢迎提交 Issue 和 Pull Request！如果想要加入新算法（例如 DPO）：

1. 在 `recrl/algorithms/dpo/` 创建目录
2. 继承 `BaseRLTrainer`，只需重写其中的 `compute_loss()`
3. 你的算法会自动继承来自父类的合规解码约束、组合奖励分配以及批次数据加载能力

详细贡献指南请参考 [CONTRIBUTING.md](CONTRIBUTING.md)。

## 📝 引用

如果你在研究中使用了 RecRL，请引用：

```bibtex
@software{recrl2025,
  title = {RecRL: Reinforcement Learning Framework for Recommendation LLMs},
  author = {Zhou, Ziren},
  year = {2025},
  url = {https://github.com/zzr9468-sys/CleanOneRec}
}
```

## 🙏 致谢

- [OpenOneRec](https://github.com/OpenOneRec) 提供 RecIF 基准和 OneRec 模型
- [TRL](https://github.com/huggingface/trl) 提供 RL 训练工具
- [Accelerate](https://github.com/huggingface/accelerate) 提供多卡训练支持
- [SwanLab](https://swanlab.cn) 提供实验跟踪平台

## 📄 开源协议

本项目基于 Apache 2.0 协议开源 - 详见 [LICENSE](LICENSE)。
