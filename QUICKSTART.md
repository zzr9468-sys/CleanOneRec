# RecRL 快速使用指南

## 🚀 快速开始

### 1. 数据加载

```python
from dataset_recif import build_recif_dataset

# 构建训练数据集
train_dataset = build_recif_dataset(
    recif_path="/Users/zhouziren/onerec/OpenOneRec-RecIF",
    split="train",
    sample_num=1000  # 采样 1000 条，-1 表示全部
)

# 数据格式
# {
#     'prompt': str,                    # LLM 输入
#     'target_pids': List[int],         # 目标 PID 列表
#     'target_sids': List[str],         # 目标 SID 列表
#     'longview_history': List[int],    # Longview 历史
#     'behavior_signals': Dict[str, float]  # 行为信号
# }
```

### 2. Reward 函数

```python
from recrl.rewards.recrl_composite_reward import RecRLCompositeReward

# 初始化 Reward 函数
reward_fn = RecRLCompositeReward(
    recif_path="/Users/zhouziren/onerec/OpenOneRec-RecIF",
    device="cuda",
    weights={
        'longview': 0.5,   # Longview 隐式反馈
        'novelty': 0.15,   # 新颖性
        # 'semantic': 0.3, # TODO: 实现
        # 'diversity': 0.05 # TODO: 实现
    }
)

# 计算 Reward
rewards = reward_fn(
    prompts=batch['prompt'],
    completions=generated_sids,  # LLM 生成的 SID
    longview_history=batch['longview_history']
)
```

### 3. 训练循环（伪代码）

```python
from recrl.algorithms.grpo.trainer import GRPOTrainer

# 初始化 Trainer
trainer = GRPOTrainer(
    model=your_llm_model,
    reward_fn=reward_fn,
    config=RLConfig(
        learning_rate=1e-5,
        num_samples=4,  # 每个 prompt 生成 4 个样本
        kl_coef=0.1
    )
)

# 训练
for batch in train_dataset:
    # 1. 生成多个样本
    completions = trainer.generate(
        prompts=batch['prompt'],
        num_samples=4
    )

    # 2. 计算 Reward
    rewards = reward_fn(
        prompts=batch['prompt'],
        completions=completions,
        longview_history=batch['longview_history']
    )

    # 3. 更新模型
    loss = trainer.step(
        prompts=batch['prompt'],
        completions=completions,
        rewards=rewards
    )
```

## 📊 Reward 设计理念

### 为什么不用 CTR？

❌ **CTR 的问题**：
- 受曝光偏差影响严重
- 只能评估已曝光的物品
- 会让新模型退化成老模型（师生偏见）

✅ **Longview 的优势**：
- 用户主动长时观看 = 真实兴趣
- 不受曝光偏差影响
- 可以评估未曝光但相似的物品

### Reward 组合策略

```
RecRL Reward = 0.5 * Longview 隐式反馈
             + 0.3 * 语义相似度 (TODO)
             + 0.15 * 新颖性
             + 0.05 * 多样性 (TODO)
```

**Longview 隐式反馈 (50%)**：
- 计算推荐物品与用户 longview 历史的语义相似度
- 最大相似度作为 reward

**新颖性 (15%)**：
- 基于物品曝光频率
- 公式：`novelty = 1 - log(freq + 1) / log(max_freq + 1)`
- 鼓励推荐长尾物品

## 🔧 常见问题

### Q1: 如何处理 SID 解析失败？

```python
# Reward 函数会自动处理
# 无效的 SID 返回 -10.0 作为惩罚
if gen_caption is None:
    rewards.append(-10.0)
```

### Q2: 如何调整 Reward 权重？

```python
# 根据实验结果调整
reward_fn = RecRLCompositeReward(
    recif_path="...",
    weights={
        'longview': 0.6,   # 增加 Longview 权重
        'novelty': 0.1,    # 减少新颖性权重
    }
)
```

### Q3: 如何处理 Longview 历史稀疏？

```python
# 在 Reward 函数中已处理
# 如果 longview_history 为空，返回 0.0
if len(hist_pids) == 0:
    rewards.append(0.0)
```

## 📈 评估指标

除了 NDCG，还应该关注：

1. **长尾覆盖率**：推荐的低频物品比例
2. **Categorical Diversity**：推荐列表的多样性
3. **Longview 命中率**：推荐物品与 longview 历史的相似度
4. **新颖性分数**：推荐物品的平均新颖性

## 🎯 实验 Roadmap

按照你的 `Experimental_Roadmap.md`：

### Phase 1.1: Baseline
- 关闭 EEPO
- 使用简单的语义相似度 Reward
- 验证基础流程

### Phase 1.2: EEPO 工程效率
- 开启 EEPO
- 验证 Fast-Weight 是否有效
- 对比训练速度

### Phase 2.1: Semantic Reward 链路
- 使用 Longview + 语义相似度
- 验证 Reward 计算是否合理

### Phase 2.2: 最终形态
- EEPO + Longview + 新颖性
- 完整的组合 Reward
- 评估长尾覆盖率和多样性

## 📝 注意事项

1. **数据格式**：确保使用 `dataset_recif.py` 而不是旧的 `dataset.py`
2. **列名**：`pid2caption.parquet` 的列名是 `dense_caption` 而不是 `caption`
3. **SID 匹配**：`sid2pid.json` 的键是字符串类型
4. **模型加载**：首次运行会下载 sentence-transformers 模型（~80MB）

## 🔗 相关文档

- `REWARD_DESIGN_ANALYSIS.md` - 详细的 Reward 设计分析
- `FINAL_SUMMARY.md` - 完整的问题总结和解决方案
- `ARCHITECTURE.md` - RecRL 框架架构
