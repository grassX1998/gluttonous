# Skills 配置

本目录包含 Gluttonous 量化交易系统的自动化工作流技能定义。

## 可用技能

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

使用 LSTM 实验框架进行模型训练和回测。

**输入**: 特征数据、价格数据
**输出**: 模型检查点、实验结果

详细说明：[train.md](./train.md)

---

### 4. 策略回测 (backtest)

**命令**: `/backtest`

使用 LSTM 实验框架进行策略回测和性能评估。

**输入**: 特征数据、价格数据
**输出**: 回测结果、性能指标

详细说明：[backtest.md](./backtest.md)

---

### 5. 策略归档 (archive)

**命令**: `/archive`

将成功的策略完整归档，记录所有关键配置和结果。

**输入**: 实验结果、模型配置、回测数据
**输出**: 完整的策略归档文件（代码、配置、结果、文档）

详细说明：[archive.md](./archive.md)

---

## 典型工作流

### 完整流程

```bash
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

### 快速测试

```bash
# 测试框架
python src/lstm/scripts/test_framework.py

# 运行实验
python src/lstm/scripts/run_experiments.py \
    --strategies expanding_window \
    --calculate_metrics
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
clean (数据清洗)
  ↓
features (特征工程)
  ↓
validate (数据校验)
  ↓
train/backtest (训练与回测)
  ↓
archive (策略归档) [可选]
```

---

## 更新日志

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
