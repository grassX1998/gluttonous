# 数据目录结构

项目使用的行情数据存储在 `/data` 目录下。

## 目录结构

```
/data/
├── stock/
│   ├── gm/                          # 掘金量化数据
│   │   ├── cfg/
│   │   │   └── trading_days.toml    # 交易日历配置
│   │   ├── meta/
│   │   │   ├── instruments.parquet  # 证券基础信息
│   │   │   └── index/               # 指数成分股
│   │   │       └── {date}/          # 按日期存储
│   │   │           └── {index_code}.toml  # 指数成分股列表
│   │   ├── mkline/                  # 分钟K线数据
│   │   │   └── {symbol}/            # 按证券代码存储
│   │   │       └── {date}.parquet   # 每日分钟K线
│   │   └── tick_l1/                 # Level1 Tick 数据
│   │       └── {symbol}/
│   │           └── {date}.parquet
│   └── cal/                         # 计算结果缓存
│       └── sor1/                    # 策略1的缓存数据
└── temp/                            # 临时文件
```

## 数据格式

### 交易日历 (trading_days.toml)

```toml
trading_days = ["2024-01-02", "2024-01-03", ...]
```

### 指数成分股 ({index_code}.toml)

```toml
sec_ids = ["SHSE.600000", "SHSE.600004", ...]
```

### 分钟K线 (mkline/{symbol}/{date}.parquet)

| 字段 | 类型 | 说明 |
|------|------|------|
| symbol | str | 证券代码 (如 SHSE.600000) |
| open | f64 | 开盘价 |
| high | f64 | 最高价 |
| low | f64 | 最低价 |
| close | f64 | 收盘价 |
| volume | i64 | 成交量 |
| turnover | f64 | 成交额 |
| date | date | 日期 |
| time | time | 时间 |

## 证券代码格式

- **上交所**: `SHSE.{code}` (如 SHSE.600000, SHSE.000300)
- **深交所**: `SZSE.{code}` (如 SZSE.000001, SZSE.399975)

### 常用指数代码

| 代码 | 名称 |
|------|------|
| SZSE.000852 | 中证1000指数 |
| SZSE.000905 | 中证500指数 |

> 注意：指数数据存储在 `SZSE.` 前缀下

## 数据时间范围

- 分钟K线: 2024-06 至 2026-01
- 交易日历: 2024-01 至 2025-12
