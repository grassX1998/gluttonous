# 基金突破策略 (StrFundMax1)

## 策略概述

该策略用于交易 ETF 基金，通过监控指数价格和成交量来判断买卖时机。核心逻辑是在指数突破 25 日最高价且成交量放大时买入对应 ETF，在价格回落或成交量萎缩时卖出。

## 交易标的

| ETF 代码 | ETF 名称 | 对应指数代码 | 指数名称 |
|---------|---------|-------------|---------|
| SH.512880 | 证券ETF | SZ.399975 | 证券公司指数 |
| SH.510300 | 沪深300ETF | SH.000300 | 沪深300指数 |
| SH.512100 | 中证1000ETF | SH.000852 | 中证1000指数 |

## 策略参数

| 参数 | 值 | 说明 |
|-----|---|------|
| max_window | 25 | 计算指数最高价的滚动窗口（天） |
| vol_avg_window | 10 | 计算成交量均值的滚动窗口（天） |
| vol_ratio | 1.2 | 买入时成交量需达到均量的倍数 |
| stop_loss_ratio | 0.95 | 止损线：跌破最高价的 95% |
| check_time | 14:50:00 | 每日信号检查时间 |

## 数据计算

### 输入数据
- **基金分钟K线**：获取 ETF 的分钟级别 K 线数据
- **指数分钟K线**：获取对应指数的分钟级别 K 线数据

### 指标计算

```python
# 1. 提取 14:50 的基金收盘价
fund_price = fund_df.filter(time == "14:50:00")["price"]

# 2. 计算指数每日成交量
volume = index_df.group_by("date").agg(volume.sum())

# 3. 计算 14:50 前的累计成交量（用于实时判断）
check_volume = index_df.filter(time <= "14:50:00").group_by("date").agg(volume.sum())

# 4. 计算 14:50 的指数价格
check_index_price = index_df.filter(time == "14:50:00")["price"]

# 5. 计算 25 日滚动最高价
max_25 = index_price.rolling_max(window=25)

# 6. 计算 10 日成交量均值
avg_vol_10 = check_volume.rolling_mean(window=10)
```

## 交易信号

### 买入条件（全部满足）
1. **价格突破**：当日 14:50 指数价格 >= 25 日最高价
2. **成交量放大**：当日 14:50 前累计成交量 > 10 日均量 × 1.2

```python
if check_index_price >= max_25 and check_volume > avg_vol_10 * 1.2:
    signal = "Buy"
```

### 卖出条件（满足任一）
1. **价格跌破**：当日 14:50 指数价格 <= 25 日最高价 × 0.95
2. **成交量萎缩**：当日 14:50 前累计成交量 < 10 日均量

```python
if check_index_price <= max_25 * 0.95 or check_volume < avg_vol_10:
    signal = "Sell"
```

## 策略状态

```python
class SecSignal:
    on_position: bool    # 是否持仓
    price: float         # 买入价格
    rate: float          # 收益率
    max_price: float     # 持仓期间最高价
    max_25: float        # 25日最高价（前一日）
    avg_vol_10: int      # 10日均量（前一日）
```

## 实现代码

源文件：`str_fund_max_1.py`

### 核心类
- `SecRecord`: 数据记录类
- `SecSignal`: 信号状态类，维护单个标的的交易状态
- `StrFundMax1`: 策略主类，管理多个标的

### 主要方法
- `load()`: 加载历史数据，初始化策略状态
- `check(code, price, vol, data_time)`: 实时检查信号
- `handle_records()`: 处理并输出交易信号

## 注意事项

1. 策略使用 **T+1** 逻辑，当日信号在次日执行
2. 检查时间为 14:50，留出 10 分钟执行交易
3. 25 日最高价和 10 日均量使用 **前一日收盘后的数据**
4. 同时持有多个 ETF 时，各标的独立管理
