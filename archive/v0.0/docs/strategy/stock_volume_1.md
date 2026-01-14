# 个股放量策略 (StockOnRecv)

## 策略概述

该策略用于交易个股，通过监控开盘后 30 分钟内的分钟级别成交量异常放大来捕捉短线交易机会。适用于中证 1000 成分股。

## 交易标的

- **股票池**：中证 1000 指数成分股 (`SHSE.000852`)
- **筛选后标的**：满足前置过滤条件的股票

## 策略参数

| 参数 | 值 | 说明 |
|-----|---|------|
| ma_short | 10 | 短期均线周期（天） |
| ma_long | 20 | 长期均线周期（天） |
| vol_ratio_threshold | 10 | 分钟成交量需达到历史均量的倍数 |
| price_change_threshold | 0.5% | 分钟涨幅阈值 |
| check_start_time | 09:35:00 | 信号检查开始时间 |
| check_end_time | 10:00:00 | 信号检查结束时间 |

## 前置过滤条件

在开盘前对股票池进行筛选，只监控满足以下条件的股票：

```python
def shouldFocus(self) -> bool:
    # 1. 短期均线 > 长期均线（多头排列）
    if self.day_10ma <= self.day_20ma:
        return False
    
    # 2. 近2日涨幅在合理范围内（-10% ~ 5%）
    if self.day_2delta > 5.0 or self.day_2delta < -10.0:
        return False
    
    # 3. 前日收盘价 > 10日均线（价格站上均线）
    if self.pre_day_close < self.day_10ma:
        return False
    
    return True
```

## 数据计算

### 日线指标

```python
# 从掘金数据获取日K线
dkline = get_dkline(code, start_date, end_date)

# 计算指标
dkline = dkline.with_columns(
    # 10日均线
    pl.col("close").rolling_mean(window_size=10).alias("day_10ma"),
    # 20日均线
    pl.col("close").rolling_mean(window_size=20).alias("day_20ma"),
    # 近2日最高价
    pl.col("close").rolling_max(window_size=2).alias("day_2max"),
    # 前日开盘价（用于计算涨幅）
    pl.col("open").shift(1).alias("pre_open"),
)

# 近2日涨幅
day_2delta = (close - pre_open) * 100 / pre_open
```

### 分钟线指标

```python
# 获取分钟K线并计算历史均量
df_mkline = get_mkline(code, start_date, end_date)

# 按时间分组，计算每个时间点的10日均量
for time_group in df_mkline.group_by(["time"]):
    volume_avg_10 = time_group["volume"].rolling_mean(window_size=10)
```

## 交易信号

### 买入条件（全部满足）

1. **时间窗口**：09:35:00 ~ 10:00:00
2. **开盘不跳空**：当日开盘价 <= 近2日最高价
3. **分钟涨幅**：当根K线涨幅 >= 0.5%
4. **成交量异常**：当根K线成交量 >= 历史同时刻均量 × 10

```python
def check(self, record: KlineRecord) -> bool:
    # 已买入则不重复触发
    if self.buy_flag:
        return False
    
    # 开盘跳空过高
    if self.day_open > self.day_2max:
        return False
    
    # 时间窗口外
    if record.time < "09:35:00" or record.time > "10:00:00":
        return False
    
    # 分钟涨幅不足
    price_change = (record.close - record.open) * 100 / record.open
    if price_change < 0.5:
        return False
    
    # 成交量放大倍数
    vol_rate = record.volume / self.minmap[record.time]["volume_avg_10"]
    if vol_rate <= 10:
        return False
    
    return True  # 触发买入信号
```

### 卖出条件

该策略为短线策略，买入后的卖出逻辑需根据实际情况制定（当前代码中未实现自动卖出）。

## 策略状态

```python
class SecSignal:
    code: str            # 股票代码
    day_10ma: float      # 10日均线
    day_20ma: float      # 20日均线
    day_2max: float      # 近2日最高价
    day_2delta: float    # 近2日涨幅
    day_open: float      # 当日开盘价
    pre_day_close: float # 前日收盘价
    minmap: dict         # 分钟级别历史均量映射
    buy_flag: bool       # 是否已触发买入
    date: str            # 当前日期
```

## 实现代码

源文件：`stock_on_recv.py`（已归档至 `archive/`）

### 核心类
- `KlineRecord`: 分钟K线数据记录
- `SecSignal`: 个股信号状态类
- `CurKlineTest`: Futu 实时K线回调处理类

### 数据依赖
- 掘金量化数据：日K线、分钟K线、指数成分股列表
- Futu API：实时分钟K线推送

## 注意事项

1. 策略仅在 **09:35 ~ 10:00** 时间窗口内触发
2. 每只股票每天 **最多触发一次** 买入信号
3. 成交量放大 10 倍是较严格的条件，实际触发频率较低
4. 需要 Futu OpenD 本地服务运行在 `127.0.0.1:11111`
5. 由于 Futu API 订阅限制，最多同时监控 290 只股票
