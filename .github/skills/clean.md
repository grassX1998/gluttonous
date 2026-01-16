# 数据清洗 (Data Cleaning)

从NAS原始数据中提取、清洗和预处理股票行情数据，生成标准化的清洗数据。

## 职责

从NAS原始分钟K线数据中提取和清洗股票数据，生成标准化的日线数据，为后续特征工程做准备。

## 输入

- **数据源**: `\\DXP8800PRO-A577\data\stock\gm` (NAS存储，只读)
- **交易日历**: `cfg/trading_days.toml`
- **证券列表**: `meta/instruments.parquet`
- **分钟K线**: `mkline/{symbol}/{date}.parquet`

## 输出

- **位置**: `.pipeline_data/cleaned/{symbol}.parquet`
- **格式**: Parquet + ZSTD压缩
- **字段**: symbol, date, time, open, high, low, close, volume, turnover

## 处理逻辑

1. 过滤非股票证券（保留SHSE/SZSE股票）
2. 移除空值和异常值（涨跌幅>20%、成交量<=0）
3. 验证价格逻辑（high >= low, close在high/low之间）
4. 按日期+时间排序
5. 检查数据完整性（最少60个交易日）

## 运行命令

```powershell
# 默认使用 CSI500 + CSI1000 成分股（约1500只）
python -m pipeline.data_cleaning.clean

# 指定日期范围
python -m pipeline.data_cleaning.clean --start_date 2024-06-01 --end_date 2025-12-31

# 使用所有股票（不推荐，数据量太大）
python -m pipeline.data_cleaning.clean --all_stocks

# 快速测试（限制股票数量）
python -m pipeline.data_cleaning.clean --limit 50
```

## 关键注意事项

⚠️ **重要**: NAS数据源是只读的，绝对不能修改原始数据！

⚠️ **缓存管理**: 修改清洗逻辑后，必须清除缓存：
```powershell
Remove-Item -Recurse -Force ".pipeline_data\cleaned\*"
```

## 验证要点

运行完成后，数据校验角色应检查：

- [ ] 输出文件数量与有效股票数量一致
- [ ] 每个文件包含所有必需字段
- [ ] 无空值和异常值
- [ ] 价格逻辑正确 (high >= low, high >= close >= low)
- [ ] 数据量满足最小天数要求（>=60天）

## 下一步

数据清洗完成后，应该：
1. 运行数据校验确保质量：`python -m pipeline.data_validation.validate`
2. 运行特征工程：`python -m pipeline.data_cleaning.features`
