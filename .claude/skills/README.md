# Skills 配置

本目录包含 Gluttonous 量化交易系统的自动化工作流技能定义。

## 可用技能

### 0. 数据采集 (collect)

**命令**: `/collect`

从掘金 API 采集股票数据，支持单日采集和每日定时任务。

**输入**: 掘金 API 配置、目标日期
**输出**: NAS 上的 K 线数据、`logs/{date}/data_collection.log`

详细说明：[collect.md](./collect.md)

---

### 1. 数据清洗 (clean)

**命令**: `/clean`

从 NAS 原始数据中提取和清洗股票数据，生成标准化的日线数据。

**输入**: NAS 原始分钟 K 线数据
**输出**: `.pipeline_data/cleaned/*.parquet`

详细说明：[clean.md](./clean.md)

---

### 2. 数据校验 (validate)

**命令**: `/validate`

验证清洗后数据和特征数据的质量，确保数据可用于模型训练。

**输入**: 清洗数据、特征数据
**输出**: `.pipeline_data/validation_report.json`

详细说明：[validate.md](./validate.md)

---

### 3. 模型训练 (train)

**命令**: `/train`

支持 LSTM 框架和多模型框架（LightGBM/MLP/Ensemble）进行模型训练。

**输入**: 特征数据、价格数据
**输出**: 模型检查点、实验结果

**支持的模型**:
- LSTM（时序深度学习）
- LightGBM（梯度提升树）
- MLP（多层感知器）
- Ensemble（集成模型）

详细说明：[train.md](./train.md)

---

### 4. 策略回测 (backtest)

**命令**: `/backtest`

支持 LSTM 框架和多模型框架进行策略回测和性能评估。

**输入**: 特征数据、价格数据
**输出**: 回测结果、性能指标

详细说明：[backtest.md](./backtest.md)

---

### 5. 多模型对比 (compare)

**命令**: `/compare`

运行多个模型并生成对比分析报告。

**输入**: 特征数据、价格数据
**输出**: 对比报告、对比图表

**可用模型**: lightgbm, mlp, ensemble

详细说明：[compare.md](./compare.md)

---

### 6. 策略归档 (archive)

**命令**: `/archive`

将成功的策略完整归档，记录所有关键配置和结果。

**输入**: 实验结果、模型配置、回测数据
**输出**: 完整的策略归档文件（代码、配置、结果、文档）

详细说明：[archive.md](./archive.md)

---

### 7. 代码审查 (review)

**命令**: `/review`

在代码修改后进行全面审查，评估框架合理性、代码质量、可复用性和规范性。

**审查维度**:
- 框架合理性（模块划分、依赖关系、数据流）
- 代码实现（正确性、健壮性、效率、可读性）
- 可复用性（DRY原则、通用组件、配置参数化）
- 规范性（命名规范、类型注解、文档字符串）
- 项目特定规范（时间序列安全、配置集中）

**输出**: 审查报告（问题列表、修改建议、优先级）

详细说明：[review.md](./review.md)

---

## 典型工作流

### 完整流程（LSTM 框架）

```bash
# 0. 数据采集（每日 17:00 后自动执行，或手动运行）
/collect --date 2026-01-17

# 1. 数据清洗
/clean

# 2. 特征工程
python -m pipeline.data_cleaning.features

# 3. 数据校验
/validate

# 4. 模型训练与回测
/train

# 5. 策略归档（可选）
/archive --version v0.3 --strategy expanding_window
```

### 多模型对比流程

```bash
# 1. 确保数据已准备（同上步骤 0-3）

# 2. 多模型对比
/compare

# 或直接运行
python src/models/scripts/compare_models.py \
    --models lightgbm,mlp,ensemble \
    --start_date 2025-04-01 \
    --end_date 2026-01-15

# 3. 查看对比报告
cat src/models/data/results/reports/comparison_*.md
```

### 快速测试

```bash
# 测试 LSTM 框架
python src/lstm/scripts/test_framework.py

# 运行 LSTM 实验
python src/lstm/scripts/run_experiments.py \
    --strategies expanding_window \
    --calculate_metrics

# 运行多模型实验
python src/models/scripts/run_lightgbm.py --start_date 2025-04-01 --end_date 2026-01-15
```

---

## Skills 配置格式

每个 skill 文件遵循以下结构：

1. **标题和简介**：技能的名称和用途
2. **职责**：技能的主要功能
3. **输入**：所需的输入数据和文件
4. **输出**：生成的输出文件和格式
5. **配置**：相关配置参数
6. **运行命令**：具体的执行命令
7. **验证要点**：需要检查的关键指标
8. **常见问题**：常见错误和解决方案
9. **下一步**：后续操作建议

---

## 技能依赖关系

```
collect (数据采集) [每日自动]
  ↓
clean (数据清洗)
  ↓                    ←── review (代码审查) [修改后触发]
features (特征工程)
  ↓
validate (数据校验)
  ↓
train/backtest (训练与回测)
  ↓
compare (多模型对比) [可选，多模型框架]
  ↓
archive (策略归档) [可选]

review 可在任意阶段触发，用于审查代码修改
compare 用于多模型框架的横向对比分析
```

---

## 实验管理系统

### 统一日志输出

所有实验日志统一输出到 `logs/{date}/` 目录：

```
logs/{date}/
├── lstm_training.log    # 训练过程日志
├── lstm_backtest.log    # 回测过程日志
├── features.log         # 特征工程日志
├── data_collection.log  # 数据采集日志
└── experiments/
    └── {experiment_id}/
        ├── config.json       # 完整配置快照
        ├── orders.csv        # 所有订单
        ├── daily_records.csv # 每日资金记录
        └── metrics.json      # 回测指标
```

### 实验注册表

实验注册表存储在 `logs/experiments_registry.json`，记录所有实验的元数据：

```json
{
  "experiments": [
    {
      "id": "exp_20260117_143022_abc123",
      "strategy": "expanding_window",
      "start_date": "2025-04-01",
      "end_date": "2026-01-15",
      "config_snapshot": {...},
      "results_path": "logs/2026-01-17/experiments/exp_20260117_143022_abc123/",
      "metrics": {
        "total_return": 1.0707,
        "sharpe_ratio": 1.818
      },
      "status": "completed",
      "created_at": "2026-01-17T14:30:22"
    }
  ]
}
```

### 实验 ID 格式

```
exp_{YYYYMMDD}_{HHMMSS}_{hash6}
例如: exp_20260117_143022_a1b2c3
```

### 查看实验

```bash
# 查看注册表
cat logs/experiments_registry.json

# 查看今日日志
ls logs/2026-01-17/

# 查看实验结果
ls logs/2026-01-17/experiments/
```

---

## 更新日志

- **2026-01-18**: 支持多模型框架
  - 新增 `compare.md` - 多模型对比技能
  - 更新 `train.md` - 支持双框架训练（LSTM + LightGBM/MLP/Ensemble）
  - 更新 `backtest.md` - 支持双框架回测
  - 更新技能依赖关系图和典型工作流

- **2026-01-17**: 新增代码审查技能
  - 新增 `review.md` - 代码审查技能
  - 支持框架合理性、代码实现、可复用性、规范性审查
  - 提供标准化审查报告格式

- **2026-01-16**: Skills 迁移和扩展
  - 将 skills 从 `.github/` 迁移到 `.claude/`
  - 新增 `archive.md` - 策略归档技能
  - 更新 train.md 使用新框架
  - 更新 backtest.md 使用新框架
  - 移除对旧 pipeline 模块的引用

- **初始版本**: 定义 4 个核心技能
  - clean: 数据清洗
  - validate: 数据校验
  - train: 模型训练
  - backtest: 策略回测

---

## 贡献指南

添加新技能时，请：
1. 在此目录创建 `{skill_name}.md` 文件
2. 按照统一格式编写文档
3. 更新本 README 添加技能索引
4. 更新 CLAUDE.md 中的 skills 使用说明
