# 数据采集 (collect)

## 简介

从掘金量化 API 采集股票数据，支持单日采集、范围采集和每日定时任务。

## 职责

1. 更新交易日历
2. 下载股票元数据
3. 下载指数成分股快照
4. 采集分钟 K 线数据
5. 错误自动重试
6. 记录采集日志

## 输入

- 掘金 API 配置（`pipeline/shared/config.py` 中的 `JUEJIN_CONFIG`）
- 目标日期或日期范围

## 输出

- 交易日历：`\\NAS\data\stock\gm\cfg\trading_days.toml`
- 股票元数据：`\\NAS\data\stock\gm\meta\instruments.parquet`
- 指数成分股：`\\NAS\data\stock\gm\meta\index/{date}/*.toml`
- 分钟 K 线：`\\NAS\data\stock\gm\mkline/{symbol}/{date}.parquet`
- 采集日志：`logs/{date}/data_collection.log`
- 状态文件：`DATA_COLLECTION_STATUS.json`

## 配置

```python
# pipeline/shared/config.py

JUEJIN_CONFIG = {
    "server": "192.168.31.252:7001",
    "token": "...",
}

DAILY_TASK_CONFIG = {
    "run_after": "17:00:00",    # 17:00 后开始采集
    "check_interval": 600,      # 检查间隔（秒）
    "max_retries": 3,           # 最大重试次数
    "retry_interval": 1800,     # 重试间隔（秒）
}
```

## 运行命令

### 手动采集

```bash
# 采集指定日期
python -m pipeline.data_collection.collector --date 2026-01-17

# 采集日期范围
python -m pipeline.data_collection.collector --start 2026-01-01 --end 2026-01-17

# 采集当天
python -m pipeline.data_collection.collector
```

### 每日定时任务

```bash
# 启动每日任务（17:00 后自动执行）
python -m pipeline.data_collection.daily_task

# 后台运行
nohup python -m pipeline.data_collection.daily_task &

# 查看状态
python -m pipeline.data_collection.daily_task --status

# 测试模式（立即执行一次）
python -m pipeline.data_collection.daily_task --test
```

## 验证要点

1. 检查 `DATA_COLLECTION_STATUS.json` 状态是否为 `completed`
2. 检查 `logs/{date}/data_collection.log` 是否有错误
3. 确认失败数为 0

## 功能特性

### 单实例保护

通过 PID 锁文件（`.pipeline_data/daily_task.lock`）确保同时只有一个采集任务运行。

### 错误重试

- 单股票：失败自动重试 3 次
- 整体任务：失败后 30 分钟重试，最多 3 次

### 日志管理

所有日志按日期组织在 `logs/{date}/` 目录下。

## 常见问题

### Q: 非交易日会采集吗？
A: 不会。系统会自动检查交易日历，跳过非交易日。

### Q: 已有数据会重复下载吗？
A: 不会。如果文件已存在，会自动跳过。

### Q: 如何查看采集进度？
A: 查看 `logs/{date}/data_collection.log` 或控制台输出。

## 下一步

采集完成后，运行数据清洗：
```bash
/clean
```
