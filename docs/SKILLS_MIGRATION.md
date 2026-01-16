# Skills 迁移和扩展总结

**日期**: 2026-01-16
**版本**: v1.1.0

---

## 📋 迁移概述

将 Skills 配置从 `.github/skills/` 迁移到 `.claude/skills/`，并新增策略归档技能。

---

## 🔄 迁移详情

### 迁移路径

```
.github/skills/  →  .claude/skills/
```

### 迁移文件

- ✅ `clean.md` - 数据清洗技能
- ✅ `validate.md` - 数据校验技能
- ✅ `train.md` - 模型训练技能
- ✅ `backtest.md` - 策略回测技能
- ✅ `README.md` - Skills 索引

### 迁移原因

1. **统一配置位置**: `.claude/` 目录是 Claude Code 的标准配置目录
2. **更好的组织**: 将 Claude 专用配置与 GitHub 配置分离
3. **易于维护**: 所有 Claude 相关配置集中管理

---

## ✨ 新增技能

### 策略归档 (archive)

**文件**: `.claude/skills/archive.md`
**命令**: `/archive`

#### 功能描述

将成功验证的策略进行完整归档，确保策略可复现和可追溯。

#### 归档内容

1. **策略基本信息**
   - 策略名称、版本号、归档日期
   - 策略类型、描述、适用市场

2. **数据清洗方式**
   - 数据源配置
   - 清洗规则和过滤器
   - 股票池选择

3. **特征工程**
   - 完整特征列表（25+ 特征）
   - 特征分类（价格、均线、波动率、技术指标等）
   - 标准化方法

4. **模型配置**
   - 模型架构（LSTM 层数、隐藏层大小）
   - 训练超参数（学习率、批次大小、轮数）
   - 硬件配置（GPU、混合精度）

5. **训练策略**
   - 策略类型（扩展窗口、K折验证等）
   - Walk-Forward 配置
   - 样本权重计算方式

6. **回测配置**
   - 交易参数（持仓数、阈值、持有天数）
   - 成本设置（手续费、滑点）
   - 回测周期和方法

7. **仓位管理**
   - 仓位配置（初始资金、最大持仓）
   - 仓位分配方式（等权、动态等）
   - 风险控制（止损、止盈、回撤预警）
   - 交易规则（买卖时机、特殊情况处理）

8. **回测结果**
   - 收益指标（总收益、年化、月度）
   - 风险指标（夏普、回撤、波动率）
   - 交易统计（胜率、盈亏比、交易次数）

9. **代码和检查点**
   - 代码位置
   - 数据位置
   - 检查点位置
   - 归档位置

10. **复现步骤**
    - 完整的命令序列
    - 参数配置
    - 验证方法

#### 归档结构

```
archive/v{version}/
├── README.md                   # 策略摘要
├── STRATEGY.md                 # 完整策略文档
├── code/                       # 代码快照
│   ├── lstm/
│   └── pipeline/
├── config/                     # 配置文件
│   └── full_config.json
├── data/                       # 数据样本
├── results/                    # 回测结果
│   └── {strategy}_result.json
├── docs/                       # 相关文档
└── requirements.txt            # 依赖清单
```

#### 使用方式

```bash
# 方式1：使用归档脚本
python scripts/archive_strategy.py \
    --version v0.3 \
    --strategy expanding_window \
    --result_file src/lstm/data/results/experiments/expanding_window_20260116.json

# 方式2：使用 skill 命令
/archive --version v0.3 --strategy expanding_window
```

---

## 📊 Skills 清单

### 更新后的技能列表

| # | 技能 | 命令 | 功能 | 状态 |
|---|------|------|------|------|
| 1 | 数据清洗 | `/clean` | 从 NAS 提取和清洗数据 | ✅ 已迁移 |
| 2 | 数据校验 | `/validate` | 验证数据质量 | ✅ 已迁移 |
| 3 | 模型训练 | `/train` | LSTM 训练和回测 | ✅ 已迁移 |
| 4 | 策略回测 | `/backtest` | 策略性能评估 | ✅ 已迁移 |
| 5 | 策略归档 | `/archive` | 完整策略归档 | ✨ 新增 |

### 技能依赖关系

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

## 📝 更新的文档

### 主要文档

1. **CLAUDE.md**
   - 更新 skills 路径：`.github/` → `.claude/`
   - 添加第 5 个技能：`/archive`
   - 更新 skills 数量：4 → 5

2. **.claude/skills/README.md**
   - 添加 `archive` 技能索引
   - 更新技能依赖关系图
   - 更新完整工作流示例
   - 更新更新日志

3. **docs/CLEANUP_SUMMARY.md**
   - 更新 skills 路径引用

### 新增文档

1. **.claude/skills/archive.md**
   - 17KB 的完整归档技能文档
   - 包含 10 个归档内容类别
   - 详细的归档结构说明
   - 归档脚本示例代码

2. **scripts/archive_strategy.py**
   - 自动化归档脚本
   - 支持命令行参数
   - 完整的归档流程实现

3. **docs/SKILLS_MIGRATION.md**
   - 本文档，记录迁移和扩展过程

---

## 🎯 迁移验证

### 验证清单

- [x] 所有文件已从 `.github/skills/` 移动到 `.claude/skills/`
- [x] `.github/skills/` 目录已删除
- [x] 新增 `archive.md` 技能文档
- [x] 新增 `archive_strategy.py` 归档脚本
- [x] 更新 `CLAUDE.md` 中的引用
- [x] 更新 `.claude/skills/README.md`
- [x] 所有路径引用已更新

### 文件统计

- **迁移文件**: 5 个
- **新增文件**: 3 个（archive.md, archive_strategy.py, SKILLS_MIGRATION.md）
- **更新文件**: 3 个（CLAUDE.md, README.md, CLEANUP_SUMMARY.md）

---

## 🚀 使用示例

### 完整工作流（含归档）

```bash
# 1. 数据清洗
/clean
# 或
python -m pipeline.data_cleaning.clean

# 2. 特征工程
python -m pipeline.data_cleaning.features

# 3. 数据校验
/validate
# 或
python -m pipeline.data_validation.validate

# 4. 模型训练与回测
/train
# 或
python src/lstm/scripts/run_experiments.py \
    --strategies expanding_window \
    --calculate_metrics

# 5. 策略归档（可选）
/archive --version v0.3 --strategy expanding_window
# 或
python scripts/archive_strategy.py \
    --version v0.3 \
    --strategy expanding_window \
    --result_file src/lstm/data/results/experiments/expanding_window_20260116.json
```

### 查看归档

```bash
# 查看归档列表
ls archive/

# 查看特定版本
cd archive/v0.3/
cat README.md
cat STRATEGY.md

# 查看配置
cat config/full_config.json

# 查看结果
cat results/expanding_window_result.json
```

---

## 📚 相关文档

- **Skills 索引**: `.claude/skills/README.md`
- **策略归档**: `.claude/skills/archive.md`
- **清理总结**: `docs/CLEANUP_SUMMARY.md`
- **项目说明**: `CLAUDE.md`
- **框架文档**: `src/lstm/README.md`

---

## 🎉 总结

### 完成的工作

1. ✅ 将 5 个 skills 文件从 `.github/` 迁移到 `.claude/`
2. ✅ 新增策略归档技能和文档（17KB）
3. ✅ 创建自动化归档脚本
4. ✅ 更新所有相关文档的路径引用
5. ✅ 建立完整的归档工作流

### 技能体系特点

1. **完整性**: 覆盖从数据到归档的完整流程
2. **标准化**: 所有 skills 遵循统一格式
3. **可追溯**: 策略归档确保完整信息保留
4. **自动化**: 提供归档脚本简化操作
5. **文档化**: 详细的使用说明和示例

### 项目收益

- 🎯 **更好的组织**: 配置位置更合理
- 📦 **完整归档**: 策略信息永久保存
- 🔄 **可复现**: 完整的复现步骤记录
- 📈 **可追溯**: 每个版本都有完整文档
- 🛠️ **易维护**: 标准化的 skills 格式

---

**迁移完成时间**: 2026-01-16
**负责人**: Claude Code
**版本**: v1.1.0
