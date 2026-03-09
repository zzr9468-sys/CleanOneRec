# RecRL 框架问题分析和 Reward Model 设计建议

## 🔴 发现的问题

### 1. **数据加载器不匹配（严重）**

**问题**：
- RecRL 的 `DataEngine.from_parquet()` 假设数据有 `messages` 和 `metadata` 列
- 但实际的 `onerec_bench_release.parquet` 没有这些列
- 实际列包括：`hist_video_pid`, `target_video_pid`, `inter_user_profile_with_sid` 等

**影响**：
- 当前的 DataEngine 无法加载你的数据
- 需要重写数据加载逻辑

**解决方案**：创建专门的 RecIF 数据加载器

### 2. **SID 匹配问题（已确认无问题）**

✅ `sid2pid.json` 的键是字符串类型
✅ `sid_to_hash_key()` 返回字符串
✅ 类型匹配正确

### 3. **Reward Model 设计问题（核心）**

根据你的文档，你的核心思路是：
- **打破师生偏见**：不能用老模型（CTR/SASRec）作为 RM
- **探索长尾**：EEPO 强制探索未曝光的物品
- **无偏见评价**：用内容语义而非 ID 匹配

**当前问题**：
1. 简单的语义相似度可能不够
2. 缺少对用户行为信号的利用
3. 没有考虑多样性和新颖性

---

## 🎯 Reward Model 设计建议

基于你的研究目标（打破师生偏见 + 探索长尾），我建议设计一个**多维度组合 Reward**：

### 方案 1：基于行为信号的加权 Reward（推荐）

```python
class BehaviorWeightedSemanticReward(BaseReward):
    """
    利用用户的多种行为信号（longview, like, forward）
    对语义相似度进行加权
    """

    def __init__(self, recif_path, device="cuda"):
        super().__init__("behavior_weighted_semantic")
        self.semantic_reward = TextSemanticReward(recif_path, device)

        # 行为权重（根据重要性）
        self.behavior_weights = {
            'longview': 1.0,    # 长时观看最重要
            'like': 0.8,        # 点赞次之
            'forward': 0.6,     # 转发
            'follow': 0.4       # 关注
        }

    def __call__(self, prompts, completions, **kwargs):
        # 1. 基础语义相似度
        semantic_scores = self.semantic_reward(prompts, completions, **kwargs)

        # 2. 获取用户历史行为
        hist_behaviors = kwargs.get("hist_behaviors", {})
        target_pid = kwargs.get("target_pid")

        # 3. 计算行为加权
        behavior_boost = 0.0
        for behavior, weight in self.behavior_weights.items():
            if behavior in hist_behaviors:
                # 如果推荐的物品类型与用户高频行为一致，给予奖励
                behavior_boost += weight * hist_behaviors[behavior]

        # 4. 组合：语义相似度 + 行为偏好
        final_rewards = []
        for i, sem_score in enumerate(semantic_scores):
            if sem_score > 0:  # 有效推荐
                # 语义相似度 * (1 + 行为加权)
                final_score = sem_score * (1.0 + 0.3 * behavior_boost)
            else:
                final_score = sem_score  # 无效推荐保持惩罚
            final_rewards.append(final_score)

        return final_rewards
```

### 方案 2：基于 CoT 的 LLM-as-Judge Reward（未来方向）

```python
class CoTJudgeReward(BaseReward):
    """
    利用 reco_cot（推荐理由）训练一个 Judge 模型
    评估推荐的合理性
    """

    def __init__(self, judge_model_path):
        super().__init__("cot_judge")
        self.judge_model = AutoModelForSequenceClassification.from_pretrained(
            judge_model_path
        )

    def __call__(self, prompts, completions, **kwargs):
        # 构造输入：用户历史 + 推荐物品 + 推荐理由
        judge_inputs = []
        for prompt, completion in zip(prompts, completions):
            judge_input = f"用户历史: {prompt}\n推荐: {completion}\n是否合理?"
            judge_inputs.append(judge_input)

        # Judge 模型打分
        scores = self.judge_model.predict(judge_inputs)
        return scores
```

### 方案 3：多目标组合 Reward（最推荐）

```python
class MultiObjectiveReward(CompositeReward):
    """
    组合多个目标：
    1. 语义相似度（内容匹配）
    2. 新颖性（探索长尾）
    3. 多样性（避免重复）
    4. 行为一致性（用户偏好）
    """

    def __init__(self, recif_path, device="cuda"):
        # 1. 语义相似度 (40%)
        semantic_reward = TextSemanticReward(recif_path, device)

        # 2. 新颖性奖励 (30%)
        novelty_reward = NoveltyReward(recif_path)

        # 3. 多样性奖励 (20%)
        diversity_reward = DiversityReward()

        # 4. 行为一致性 (10%)
        behavior_reward = BehaviorConsistencyReward(recif_path)

        super().__init__([
            (semantic_reward, 0.4),
            (novelty_reward, 0.3),
            (diversity_reward, 0.2),
            (behavior_reward, 0.1)
        ])
```

### 方案 4：基于 Longview 的隐式反馈 Reward

```python
class LongviewBasedReward(BaseReward):
    """
    利用 longview 作为隐式正反馈
    如果推荐的物品与用户 longview 过的物品语义相似，给高分
    """

    def __init__(self, recif_path, device="cuda"):
        super().__init__("longview_based")
        self.recif_path = recif_path
        self.device = device

        # 加载映射
        self.sid2pid_dict = self._load_sid2pid()
        self.pid2caption_dict = self._load_pid2caption()
        self.text_model = self._load_text_model()

    def __call__(self, prompts, completions, **kwargs):
        # 获取用户 longview 历史
        hist_longview_pids = kwargs.get("hist_longview_video_list", [])

        if len(hist_longview_pids) == 0:
            return [0.0] * len(completions)

        # 获取 longview 物品的 captions
        longview_captions = []
        for pid in hist_longview_pids:
            if str(pid) in self.pid2caption_dict:
                longview_captions.append(self.pid2caption_dict[str(pid)])

        if len(longview_captions) == 0:
            return [0.0] * len(completions)

        # 计算推荐物品与 longview 历史的相似度
        rewards = []
        for completion in completions:
            # 解析 SID -> PID -> Caption
            gen_caption = self._get_caption_from_sid(completion)

            if gen_caption is None:
                rewards.append(-10.0)  # 无效推荐
                continue

            # 计算与所有 longview 物品的最大相似度
            max_sim = 0.0
            for lv_caption in longview_captions:
                sim = self._compute_similarity(gen_caption, lv_caption)
                max_sim = max(max_sim, sim)

            rewards.append(max_sim)

        return rewards
```

---

## 🎯 推荐的最终方案

基于你的研究目标，我建议使用 **方案 3（多目标组合）+ 方案 4（Longview 隐式反馈）**：

```python
class RecRLReward(CompositeReward):
    """
    RecRL 推荐的 Reward 组合

    核心思想：
    1. 用 Longview 作为主要信号（用户真实兴趣）
    2. 语义相似度作为辅助（内容匹配）
    3. 新颖性鼓励探索长尾
    4. 多样性避免重复推荐
    """

    def __init__(self, recif_path, device="cuda"):
        # 主要：Longview 隐式反馈 (50%)
        longview_reward = LongviewBasedReward(recif_path, device)

        # 辅助：语义相似度 (30%)
        semantic_reward = TextSemanticReward(recif_path, device)

        # 探索：新颖性 (15%)
        novelty_reward = NoveltyReward(recif_path)

        # 质量：多样性 (5%)
        diversity_reward = DiversityReward()

        super().__init__([
            (longview_reward, 0.5),
            (semantic_reward, 0.3),
            (novelty_reward, 0.15),
            (diversity_reward, 0.05)
        ])
```

**为什么这样设计？**

1. **Longview 是最强信号**：用户长时间观看 = 真实兴趣，不受曝光偏差影响
2. **语义相似度是保底**：确保推荐内容相关
3. **新颖性鼓励探索**：配合 EEPO 打破茧房
4. **多样性提升体验**：避免推荐重复内容

---

## 🔧 需要修复的代码

### 1. 修复 DataEngine（紧急）

需要创建专门的 RecIF 数据加载器，处理实际的列格式。

### 2. 实现新的 Reward 函数

实现上述的 `LongviewBasedReward` 和 `NoveltyReward`。

### 3. 修复 SID 解析

确保 SID 解析逻辑与 RecIF 数据格式完全匹配。

---

## 📊 评估指标建议

除了 NDCG，还应该关注：

1. **长尾覆盖率**：推荐的物品中，有多少是低频/冷门物品
2. **多样性**：推荐列表的 Categorical Diversity
3. **新颖性**：推荐的物品与训练数据的重叠度
4. **Longview 命中率**：推荐的物品是否与用户 longview 历史相似

---

需要我帮你实现这些修复吗？
