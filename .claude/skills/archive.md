# 策略归档 (Archive)

将成功的策略完整归档，包括数据处理、模型训练、回测结果和所有关键配置。

## 职责

对已验证的有效策略进行完整归档，记录所有关键信息，确保策略可复现和可追溯。策略归档包括完整的数据清洗流程、特征工程、模型配置、训练方式、回测结果、仓位管理等。

## 归档内容

### 1. 策略基本信息

- **策略名称**: 策略的唯一标识名称
- **版本号**: 如 v0.3, v1.0
- **归档日期**: 策略完成验证的日期
- **策略类型**: 如扩展窗口、K折验证、多尺度集成等
- **策略描述**: 策略的核心思路和特点
- **适用市场**: 如 A股、港股、美股等
- **适用品种**: 如小盘股、大盘股、全市场等

### 2. 数据清洗方式

记录完整的数据处理流程：

```python
# 数据源
data_source = {
    "nas_path": "\\\\DXP8800PRO-A577\\data\\stock\\gm\\",
    "time_range": "2024-06 ~ 2026-01",
    "data_type": "分钟K线"
}

# 清洗配置
cleaning_config = {
    "universe": ["SZSE.000905", "SZSE.000852"],  # 中证500+1000
    "exclude": ["SZSE.000300"],                   # 排除中证300
    "min_trading_days": 60,
    "remove_limit": True,                         # 去除涨跌停
    "remove_st": True,                            # 去除ST股票
    "filters": [
        "涨跌幅 < 20%",
        "成交量 > 0",
        "价格逻辑正确"
    ]
}

# 清洗脚本
cleaning_command = "python -m pipeline.data_cleaning.clean"
```

### 3. 特征工程

记录所有使用的特征：

```python
# 特征列表
features = {
    # 价格特征
    "price": [
        "return_1d",   # 1日收益率
        "return_5d",   # 5日收益率
        "return_10d",  # 10日收益率
        "return_20d",  # 20日收益率
    ],

    # 移动平均
    "ma": [
        "ma5_ratio",   # 5日均线比率
        "ma10_ratio",  # 10日均线比率
        "ma20_ratio",  # 20日均线比率
        "ma60_ratio",  # 60日均线比率
    ],

    # 波动率
    "volatility": [
        "volatility_5d",
        "volatility_10d",
        "volatility_20d",
    ],

    # 技术指标
    "indicators": [
        "rsi",         # 相对强弱指标
        "macd",        # MACD
        "macd_signal", # MACD信号线
        "bb_upper",    # 布林带上轨
        "bb_lower",    # 布林带下轨
    ],

    # 成交量特征
    "volume": [
        "volume_ratio_5d",
        "turnover_rate",
    ],

    # 策略特征
    "strategy": [
        "breakout_20d",    # 20日突破
        "stop_loss_trigger", # 止损触发
    ],

    # 市场特征
    "market": [
        "market_return",     # 市场收益
        "relative_strength", # 相对强度
    ],

    # 标记
    "flags": [
        "is_limit_up",   # 涨停标记
        "is_limit_down", # 跌停标记
    ]
}

# 特征工程脚本
feature_command = "python -m pipeline.data_cleaning.features"

# 特征统计
feature_stats = {
    "total_features": 25,
    "feature_selection": "全部使用",
    "normalization": "StandardScaler (从训练集计算)",
}
```

### 4. 模型配置

记录模型的完整配置：

```python
# 模型类型
model_type = "LSTM"

# 模型架构
model_config = {
    "hidden_size": 128,        # LSTM隐藏层大小
    "num_layers": 2,           # LSTM层数
    "dropout": 0.3,            # Dropout率
    "bidirectional": False,    # 是否双向
    "attention": False,        # 是否使用注意力机制
}

# 训练超参数
training_config = {
    "batch_size": 1024,        # 批次大小
    "epochs": 10,              # 训练轮数
    "learning_rate": 0.001,    # 学习率
    "optimizer": "Adam",       # 优化器
    "loss_function": "BCELoss", # 损失函数
    "early_stop_patience": 3,   # 早停耐心值
    "weight_decay": 0.0001,    # L2正则化
    "grad_clip": 1.0,          # 梯度裁剪
}

# 硬件配置
hardware_config = {
    "device": "CUDA",
    "mixed_precision": True,   # 混合精度训练
    "num_workers": 4,
}
```

### 5. 训练策略

记录训练的具体方式：

```python
# 策略类型
strategy_type = "ExpandingWindow"  # 扩展窗口策略

# 策略配置
strategy_config = {
    "min_train_days": 60,         # 最小训练天数
    "max_train_days": 500,        # 最大训练天数
    "val_days": 1,                # 验证天数
    "use_sample_weight": True,    # 使用样本权重
    "weight_decay_days": 30,      # 权重衰减周期
    "weight_decay_rate": 0.98,    # 权重衰减率
    "retrain_interval": 1,        # 重训练间隔（天）
}

# Walk-Forward 配置
walkforward_config = {
    "method": "expanding",        # 扩展窗口
    "start_date": "2025-04-01",
    "end_date": "2026-01-15",
    "validation_method": "时序验证",
    "no_lookahead_bias": True,
}

# 样本权重计算
def sample_weight_formula(days_ago):
    """
    近期数据权重更高，指数衰减
    """
    return 0.98 ** (days_ago / 30)
```

### 6. 回测配置

记录完整的回测设置：

```python
# 交易配置
trading_config = {
    "top_n": 10,                  # 每日持仓数
    "prob_threshold": 0.60,       # 买入概率阈值
    "holding_days": 5,            # 持有天数
    "commission": 0.001,          # 手续费 0.1%
    "slippage": 0.001,            # 滑点 0.1%
    "min_price": 0,               # 最低价格（0=无限制）
    "max_price": 0,               # 最高价格（0=无限制）
}

# 回测周期
backtest_period = {
    "start_date": "2025-04-01",
    "end_date": "2026-01-15",
    "total_days": 180,
}

# 回测方法
backtest_method = {
    "type": "Walk-Forward",
    "rebalance": "每日",
    "execution": "收盘价",
    "entry": "T日收盘",
    "exit": "T+5日收盘",
}
```

### 7. 仓位管理

记录仓位管理的详细规则：

```python
# 仓位配置
position_config = {
    "initial_cash": 1000000,      # 初始资金（回测用）
    "max_positions": 10,          # 最大持仓数
    "position_sizing": "等权",    # 仓位分配方式
    "rebalance_frequency": "每日", # 调仓频率
}

# 仓位分配方式
def position_sizing_method():
    """
    等权分配：每只股票分配相等的资金
    """
    return initial_cash / max_positions

# 选股逻辑
selection_logic = {
    "step1": "筛选预测概率 > threshold 的股票",
    "step2": "按概率从高到低排序",
    "step3": "选取前 top_n 只",
    "step4": "等权分配资金",
}

# 风险控制
risk_management = {
    "stop_loss": None,            # 无止损（持有固定天数）
    "take_profit": None,          # 无止盈
    "max_drawdown_alert": 0.50,   # 最大回撤预警
    "单股最大仓位": 0.10,         # 10%
}

# 交易规则
trading_rules = {
    "买入时机": "T日收盘前",
    "卖出时机": "T+holding_days日收盘前",
    "是否追涨": False,
    "是否抄底": False,
    "涨跌停处理": "跳过",
}
```

### 8. 回测结果

记录详细的回测表现：

```python
# 收益指标
return_metrics = {
    "total_return": 0.7484,       # 总收益率 74.84%
    "annual_return": 0.8542,      # 年化收益率 85.42%
    "monthly_return": 0.0577,     # 月均收益率 5.77%
    "cumulative_return": 0.7484,  # 累计收益率
}

# 风险指标
risk_metrics = {
    "sharpe_ratio": 1.566,        # 夏普比率
    "max_drawdown": 0.4704,       # 最大回撤 47.04%
    "volatility": 0.0456,         # 波动率
    "downside_deviation": 0.0312, # 下行偏差
    "calmar_ratio": 1.815,        # 卡玛比率
    "sortino_ratio": 2.145,       # 索提诺比率
}

# 交易统计
trade_stats = {
    "total_trades": 1250,         # 总交易次数
    "win_rate": 0.548,            # 胜率 54.8%
    "avg_profit": 0.0342,         # 平均盈利
    "avg_loss": -0.0289,          # 平均亏损
    "profit_factor": 1.623,       # 盈亏比
    "avg_holding_days": 5,        # 平均持有天数
}

# 日度统计
daily_stats = {
    "trading_days": 180,          # 交易日数
    "daily_win_rate": 0.612,      # 日胜率
    "best_day": 0.0856,           # 最佳单日收益
    "worst_day": -0.0734,         # 最差单日收益
    "avg_daily_return": 0.00312,  # 平均日收益
}

# 月度收益
monthly_returns = {
    "2025-04": 0.0823,
    "2025-05": 0.0645,
    "2025-06": 0.0512,
    "2025-07": 0.0789,
    "2025-08": 0.0456,
    "2025-09": 0.0678,
    "2025-10": 0.0734,
    "2025-11": 0.0598,
    "2025-12": 0.0623,
    "2026-01": 0.0626,
}
```

### 9. 代码和检查点

记录所有相关文件的位置：

```python
# 代码位置
code_location = {
    "framework": "src/lstm/",
    "config": "src/lstm/config.py",
    "model": "src/lstm/models/lstm_model.py",
    "executor": "src/lstm/experiments/executors/expanding_window.py",
    "scripts": "src/lstm/scripts/run_experiments.py",
}

# 数据位置
data_location = {
    "raw_data": "\\\\DXP8800PRO-A577\\data\\stock\\gm\\",
    "cleaned": ".pipeline_data/cleaned/",
    "features": ".pipeline_data/features_monthly/",
    "daily": ".pipeline_data/daily/",
}

# 检查点位置
checkpoint_location = {
    "models": "src/lstm/data/checkpoints/",
    "results": "src/lstm/data/results/experiments/",
    "logs": "src/lstm/data/logs/",
}

# 归档位置
archive_location = {
    "root": "archive/v{version}/",
    "code": "archive/v{version}/code/",
    "data": "archive/v{version}/data/",
    "results": "archive/v{version}/results/",
    "docs": "archive/v{version}/docs/",
}
```

### 10. 复现步骤

记录完整的复现流程：

```bash
# 步骤1: 数据清洗
python -m pipeline.data_cleaning.clean \
    --start_date 2024-06-18 \
    --end_date 2026-01-13

# 步骤2: 特征工程
python -m pipeline.data_cleaning.features

# 步骤3: 数据校验
python -m pipeline.data_validation.validate

# 步骤4: 运行实验
python src/lstm/scripts/run_experiments.py \
    --strategies expanding_window \
    --start_date 2025-04-01 \
    --end_date 2026-01-15 \
    --calculate_metrics \
    --update_claude_md

# 步骤5: 查看结果
ls src/lstm/data/results/experiments/
cat src/lstm/data/results/experiments/expanding_window_*.json
```

## 归档结构

### 目录结构

```
archive/
└── v0.3/                           # 版本号
    ├── README.md                   # 策略总结
    ├── STRATEGY.md                 # 完整策略文档（本文件）
    ├── code/                       # 代码快照
    │   ├── src/
    │   ├── pipeline/
    │   └── requirements.txt
    ├── config/                     # 配置文件
    │   ├── model_config.json
    │   ├── trading_config.json
    │   └── strategy_config.json
    ├── data/                       # 数据样本
    │   ├── features_sample.parquet
    │   └── predictions_sample.json
    ├── results/                    # 回测结果
    │   ├── backtest_report.json
    │   ├── equity_curve.png
    │   ├── drawdown.png
    │   └── metrics.json
    └── docs/                       # 相关文档
        ├── backtest_report.md
        ├── optimization_log.md
        └── lessons_learned.md
```

## 运行命令

```bash
# 创建策略归档
python scripts/archive_strategy.py \
    --version v0.3 \
    --strategy expanding_window \
    --result_file src/lstm/data/results/experiments/expanding_window_20260116.json

# 或使用 skill 命令
/archive --version v0.3 --strategy expanding_window
```

## 归档脚本示例

```python
# scripts/archive_strategy.py

import sys
import json
import shutil
from pathlib import Path
from datetime import datetime

def archive_strategy(version: str, strategy: str, result_file: Path):
    """
    归档策略的完整信息

    Args:
        version: 版本号，如 v0.3
        strategy: 策略名称，如 expanding_window
        result_file: 实验结果文件路径
    """
    # 创建归档目录
    archive_dir = Path(f"archive/{version}")
    archive_dir.mkdir(parents=True, exist_ok=True)

    # 创建子目录
    (archive_dir / "code").mkdir(exist_ok=True)
    (archive_dir / "config").mkdir(exist_ok=True)
    (archive_dir / "data").mkdir(exist_ok=True)
    (archive_dir / "results").mkdir(exist_ok=True)
    (archive_dir / "docs").mkdir(exist_ok=True)

    # 1. 复制代码
    shutil.copytree("src/lstm", archive_dir / "code" / "lstm",
                    dirs_exist_ok=True, ignore=shutil.ignore_patterns("data"))
    shutil.copytree("pipeline", archive_dir / "code" / "pipeline",
                    dirs_exist_ok=True)

    # 2. 保存配置
    from src.lstm.config import (
        MODEL_CONFIG, TRADING_CONFIG,
        ALL_STRATEGY_CONFIGS
    )

    config_data = {
        "model": MODEL_CONFIG,
        "trading": TRADING_CONFIG,
        "strategy": ALL_STRATEGY_CONFIGS[strategy]().to_dict(),
    }

    with open(archive_dir / "config" / "full_config.json", "w") as f:
        json.dump(config_data, f, indent=2, ensure_ascii=False)

    # 3. 复制结果
    shutil.copy(result_file, archive_dir / "results" / f"{strategy}_result.json")

    # 4. 生成策略文档
    generate_strategy_doc(archive_dir, version, strategy, config_data)

    # 5. 生成 README
    generate_readme(archive_dir, version, strategy)

    print(f"策略归档完成: {archive_dir}")

def generate_strategy_doc(archive_dir, version, strategy, config_data):
    """生成完整的策略文档"""
    # 这里可以根据模板生成完整的 STRATEGY.md
    pass

def generate_readme(archive_dir, version, strategy):
    """生成归档 README"""
    readme_content = f"""# 策略归档 - {version}

**策略名称**: {strategy}
**归档日期**: {datetime.now().strftime('%Y-%m-%d')}

## 目录说明

- `code/`: 代码快照
- `config/`: 配置文件
- `data/`: 数据样本
- `results/`: 回测结果
- `docs/`: 相关文档

## 快速复现

详见 `STRATEGY.md` 中的复现步骤。
"""

    with open(archive_dir / "README.md", "w", encoding="utf-8") as f:
        f.write(readme_content)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", required=True)
    parser.add_argument("--strategy", required=True)
    parser.add_argument("--result_file", required=True)
    args = parser.parse_args()

    archive_strategy(args.version, args.strategy, Path(args.result_file))
```

## 验证要点

归档后应检查：

- [ ] 所有配置文件已保存
- [ ] 代码快照完整（包含所有依赖）
- [ ] 回测结果文件已复制
- [ ] 策略文档完整记录所有关键信息
- [ ] 复现步骤经过验证
- [ ] 数据样本已保存（可选）
- [ ] 相关图表已保存

## 归档清单

### 必须包含

- ✅ 策略配置（模型、训练、交易）
- ✅ 代码快照（src/lstm/, pipeline/）
- ✅ 回测结果（JSON + 指标）
- ✅ 复现步骤（完整命令）
- ✅ 性能指标（收益、风险、交易统计）

### 建议包含

- 📊 可视化图表（收益曲线、回撤、分布）
- 📝 优化日志（调参过程）
- 💡 经验总结（lessons learned）
- 📦 数据样本（特征和预测结果）
- 🔧 依赖清单（requirements.txt）

### 可选包含

- 🎯 对比分析（与其他策略对比）
- 📈 实盘跟踪（如有）
- 🐛 已知问题（bugs & limitations）
- 🔮 改进方向（future work）

## 最佳实践

### 1. 版本命名

- 使用语义化版本：`v{major}.{minor}.{patch}`
- 重大变更：v1.0, v2.0
- 功能增强：v0.1, v0.2
- Bug修复：v0.1.1, v0.1.2

### 2. 归档时机

- ✅ 策略开发完成并通过验证
- ✅ 回测结果稳定且可复现
- ✅ 准备实盘测试前
- ✅ 策略即将被替换前

### 3. 文档质量

- 使用清晰的中文描述
- 包含足够的代码示例
- 记录所有关键参数
- 解释参数选择的原因

### 4. 数据保留

- 保留关键检查点（不是全部）
- 保存配置和结果（必须）
- 数据样本用于快速验证
- 大数据集保留索引即可

## 下一步

策略归档完成后：

1. 在 `CLAUDE.md` 中更新最佳策略记录
2. 如果是新版本，更新 `README.md` 中的版本号
3. 提交 git commit 并打 tag：`git tag v0.3`
4. 可以开始下一个策略的开发

## 相关文档

- 快速入门：`docs/QUICKSTART_LSTM.md`
- 实验框架：`docs/EXPERIMENT_FRAMEWORK.md`
- 已归档策略：`archive/` 目录
- 项目说明：`CLAUDE.md`
