# 深度学习量化策略挖掘系统

本模块使用深度学习技术从历史行情数据中挖掘交易策略。

## 目录结构

```
ml/
├── README.md              # 本文档
├── data/
│   └── dataset.py         # 数据集构建
├── features/
│   └── technical.py       # 技术指标特征工程
├── models/
│   └── lstm.py            # 模型定义 (LSTM/Transformer)
├── train.py               # 模型训练脚本
├── backtest.py            # 策略回测脚本
├── checkpoints/           # 模型检查点保存目录
└── results/               # 回测结果输出目录
```

## 整体流程

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│  原始数据    │ -> │  特征工程    │ -> │  模型训练    │ -> │  策略回测    │
│  (分钟K线)   │    │  (技术指标)   │    │  (LSTM)     │    │  (买卖信号)   │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
```

---

## 1. 数据处理模块 (data/dataset.py)

### 1.1 数据源

使用预处理好的中证1000成分股数据：
- 路径: `/data/backtest_1/{date}/mkline.parquet`
- 格式: 分钟K线数据
- 字段: symbol, open, high, low, close, volume, turnover, date, time

### 1.2 数据加载流程

```python
# 1. 获取交易日列表
trading_days = get_trading_days(start_date, end_date)

# 2. 加载每日数据并聚合为日线
daily = load_daily_data(date)  # 分钟K线 -> 日线OHLCV

# 3. 构建时序样本
X, y = build_stock_dataset(symbol, start_date, end_date, lookback=20)
```

### 1.3 样本构建逻辑

```
时间窗口 (lookback=20天):

Day 1  Day 2  ... Day 19  Day 20  | Day 21 (预测目标)
[          特征序列 X           ] | [  标签 y  ]

X.shape = (N, 20, 24)  # N个样本, 20天历史, 24个特征
y.shape = (N,)         # N个标签 (0=跌, 1=涨)
```

### 1.4 标签定义

```python
# 分类任务：预测次日涨跌
y = (future_return > 0).astype(int)  # 1=涨, 0=跌

# 也可设置阈值，如涨幅>2%才算正类
y = (future_return > 0.02).astype(int)
```

---

## 2. 特征工程模块 (features/technical.py)

### 2.1 特征列表 (共24个)

| 类别 | 特征名 | 说明 |
|------|--------|------|
| **收益率** | ret_1, ret_5, ret_10, ret_20 | 1/5/10/20日收益率 |
| **均线偏离** | ma_5_ratio, ma_10_ratio, ma_20_ratio, ma_60_ratio | 价格相对均线偏离度 |
| **波动率** | volatility_5, volatility_10, volatility_20 | 5/10/20日波动率 |
| **RSI** | rsi_14 | 14日相对强弱指标 |
| **MACD** | macd_dif, macd_dea, macd_hist | MACD三线 |
| **布林带** | bb_position | 价格在布林带中的位置(0-1) |
| **量比** | vol_ratio_5, vol_ratio_10, vol_ratio_20 | 成交量相对均量比值 |
| **价格形态** | amplitude, upper_shadow, lower_shadow, body | 振幅、上下影线、实体 |

### 2.2 特征计算示例

```python
# 收益率
ret_5 = close / close.shift(5) - 1

# MA偏离度
ma_20_ratio = close / close.rolling(20).mean() - 1

# RSI
rsi_14 = 100 - 100 / (1 + avg_gain / avg_loss)

# 布林带位置
bb_position = (close - bb_lower) / (bb_upper - bb_lower)

# 量比
vol_ratio_5 = volume / volume.rolling(5).mean()
```

### 2.3 特征标准化

```python
# 训练时计算均值和标准差
X_mean = X.mean(axis=(0, 1), keepdims=True)
X_std = X.std(axis=(0, 1), keepdims=True)
X_normalized = (X - X_mean) / X_std

# 保存参数用于推理
np.save("X_mean.npy", X_mean)
np.save("X_std.npy", X_std)
```

---

## 3. 模型架构 (models/lstm.py)

### 3.1 LSTM分类器 (主模型)

```
输入: (batch, 20, 24) - 20天历史，24个特征

       ┌─────────────────────────────────────┐
       │         Bidirectional LSTM          │
       │   hidden_size=64, num_layers=2      │
       └─────────────────────────────────────┘
                        │
                        ▼
       ┌─────────────────────────────────────┐
       │        Attention Mechanism          │
       │   对时间维度加权聚合                  │
       └─────────────────────────────────────┘
                        │
                        ▼
       ┌─────────────────────────────────────┐
       │     Classifier (FC Layers)          │
       │   Linear(128->64) -> ReLU           │
       │   Dropout(0.3)                      │
       │   Linear(64->2)                     │
       └─────────────────────────────────────┘
                        │
                        ▼
输出: (batch, 2) - 涨/跌的logits
```

### 3.2 Attention机制

```python
# 计算每个时间步的注意力权重
attn_weights = softmax(tanh(W1 @ lstm_out) @ W2)  # (batch, seq_len, 1)

# 加权聚合
context = sum(lstm_out * attn_weights, dim=1)  # (batch, hidden*2)
```

### 3.3 模型参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| input_size | 24 | 特征数量 |
| hidden_size | 64 | LSTM隐藏层大小 |
| num_layers | 2 | LSTM层数 |
| dropout | 0.3 | Dropout比例 |
| bidirectional | True | 双向LSTM |

### 3.4 其他模型

- **LSTMRegressor**: LSTM回归模型，直接预测收益率
- **TransformerClassifier**: 基于Transformer的分类模型

---

## 4. 训练流程 (train.py)

### 4.1 数据划分

```
总数据
├── 训练集 (70%) - 用于训练模型
├── 验证集 (15%) - 用于调参和早停
└── 测试集 (15%) - 用于最终评估
```

### 4.2 训练配置

```python
config = {
    "model_type": "lstm_classifier",
    "lookback": 20,              # 历史窗口长度
    "predict_days": 1,           # 预测未来天数
    "batch_size": 64,            # 批大小
    "learning_rate": 0.001,      # 学习率
    "epochs": 50,                # 最大训练轮数
    "early_stop_patience": 10,   # 早停耐心值
}
```

### 4.3 训练技巧

1. **梯度裁剪**: 防止梯度爆炸
   ```python
   torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
   ```

2. **学习率调度**: 验证损失不下降时降低学习率
   ```python
   scheduler = ReduceLROnPlateau(optimizer, factor=0.5, patience=5)
   ```

3. **早停**: 验证损失连续10轮不下降时停止训练

4. **L2正则化**: 防止过拟合
   ```python
   optimizer = Adam(params, weight_decay=1e-5)
   ```

### 4.4 评估指标

| 指标 | 说明 |
|------|------|
| Accuracy | 整体准确率 |
| Precision | 预测为涨的股票中，实际涨的比例 |
| Recall | 实际涨的股票中，被预测为涨的比例 |

### 4.5 运行训练

```bash
cd /home/grasszhang/workspace/projects/gluttonous
source .venv/bin/activate

# 默认配置训练
python ml/train.py

# 自定义参数
python ml/train.py \
    --model lstm_classifier \
    --start_date 2025-01-01 \
    --end_date 2025-10-31 \
    --epochs 50 \
    --batch_size 64 \
    --lr 0.001 \
    --max_stocks 200
```

---

## 5. 策略回测 (backtest.py)

### 5.1 交易策略

```
每个交易日:
1. 收盘前，用模型预测所有股票次日涨跌概率
2. 选择概率最高的 top_k 只股票（概率需超过阈值）
3. 以当天收盘价买入
4. 次日以开盘价卖出

参数:
- top_k: 每天最多买入的股票数量 (默认10)
- prob_threshold: 买入的概率阈值 (默认0.6)
```

### 5.2 回测流程

```python
for date in trading_days:
    # 1. 获取当天所有股票
    symbols = get_all_symbols(date)
    
    # 2. 模型预测上涨概率
    predictions = model.predict(symbols)  # {symbol: prob}
    
    # 3. 筛选买入标的
    candidates = sorted(predictions, key=prob, reverse=True)
    candidates = [s for s in candidates if prob[s] >= 0.6][:10]
    
    # 4. 执行买入
    for symbol in candidates:
        buy(symbol, close_price)
    
    # 5. 次日开盘卖出
    for symbol in holdings:
        sell(symbol, next_day_open_price)
```

### 5.3 运行回测

```bash
cd /home/grasszhang/workspace/projects/gluttonous
source .venv/bin/activate

python ml/backtest.py
```

### 5.4 回测报告

输出到 `ml/results/`:
- `ml_trades.csv`: 所有交易记录明细
- `ml_report.md`: 回测统计报告

---

## 6. 快速开始

### 6.1 完整工作流

```bash
# 1. 激活虚拟环境
cd /home/grasszhang/workspace/projects/gluttonous
source .venv/bin/activate

# 2. 训练模型
python ml/train.py --max_stocks 200 --epochs 50

# 3. 运行回测
python ml/backtest.py
```

### 6.2 调参建议

| 场景 | 调整方向 |
|------|----------|
| 过拟合 | 增加dropout, 减少hidden_size, 增加数据量 |
| 欠拟合 | 增加hidden_size, 增加num_layers, 增加epochs |
| 训练慢 | 减少max_stocks, 减少batch_size |
| 类别不平衡 | 调整分类阈值, 使用加权损失函数 |

---

## 7. 扩展方向

1. **更多特征**: 添加财务指标、情绪指标、资金流向等
2. **更复杂模型**: TCN, GNN, 多任务学习
3. **强化学习**: 端到端学习交易策略
4. **集成学习**: 多模型投票/加权

---

## 8. 代码示例

### 8.1 自定义特征

```python
# 在 features/technical.py 中添加

def calc_custom_features(df: pl.DataFrame) -> pl.DataFrame:
    """自定义特征"""
    return df.with_columns([
        # 动量指标
        (pl.col("close").diff(10) / pl.col("close").shift(10)).alias("momentum_10"),
        # 成交额占比
        (pl.col("turnover") / pl.col("turnover").rolling_sum(5)).alias("turnover_ratio"),
    ])

# 更新 FEATURE_COLS
FEATURE_COLS.extend(["momentum_10", "turnover_ratio"])
```

### 8.2 自定义模型

```python
# 在 models/lstm.py 中添加

class TCNClassifier(nn.Module):
    """时序卷积网络"""
    def __init__(self, input_size, hidden_size=64, num_classes=2):
        super().__init__()
        # ... 实现TCN结构
        pass

# 在 create_model 中注册
def create_model(model_type, input_size, **kwargs):
    if model_type == "tcn":
        return TCNClassifier(input_size, **kwargs)
    # ...
```

---

## 9. 注意事项

1. **数据质量**: 确保数据无缺失、无异常值
2. **过拟合风险**: 金融数据噪声大，容易过拟合
3. **未来信息泄露**: 特征计算时注意不要使用未来数据
4. **交易成本**: 实际交易需考虑手续费、滑点、冲击成本
5. **市场变化**: 模型需要定期重训练以适应市场变化

---

## 10. 依赖环境

```
Python >= 3.11
torch >= 2.0
polars >= 0.20
numpy >= 1.24
```

安装依赖:
```bash
pip install torch polars numpy
```
