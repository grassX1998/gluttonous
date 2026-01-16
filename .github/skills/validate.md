# 数据校验 (Data Validation)

验证清洗后数据和特征数据的质量，确保数据可用于模型训练。

## 职责

对数据流水线中的各个阶段产出进行质量检查，发现潜在问题，确保下游任务能正常运行。

## 输入

- **清洗数据**: `.pipeline_data/cleaned/*.parquet`
- **特征数据**: `.pipeline_data/features/*.parquet` (如存在)
- **训练数据**: `.pipeline_data/train/*.npz` (如存在)

## 输出

- **验证报告**: `.pipeline_data/validation_report.json`
- **日志**: 控制台输出详细验证结果

## 验证项目

### 对清洗数据的校验

| 检查项 | 标准 | 严重程度 |
|--------|------|----------|
| 缺失值检查 | 无缺失值 | Error |
| 数据量检查 | 每只股票 >= 1000条记录 | Warning |
| 价格范围 | close > 0 | Error |
| 数据类型 | 符合预期dtype | Error |
| 价格逻辑 | high >= low, high >= close >= low | Error |

### 对特征数据的校验

| 检查项 | 标准 | 严重程度 |
|--------|------|----------|
| NaN检查 | 特征无NaN | Error |
| Inf检查 | 特征无Inf | Error |
| 分布检查 | 无极端异常值 | Warning |
| 标签平衡 | 正负样本比例合理(0.3-0.7) | Warning |
| 特征数量 | 包含所有必需特征 | Error |

## 运行命令

```powershell
# 运行数据校验
python -m pipeline.data_validation.validate

# 查看验证报告
cat .pipeline_data/validation_report.json
```

## 交叉验证

本角色负责验证其他角色的输出：

- **验证清洗结果**: 检查 `.pipeline_data/cleaned/` 目录
- **验证特征结果**: 检查 `.pipeline_data/features/` 目录
- **验证训练数据**: 检查 `.pipeline_data/train/` 目录的npz文件

## 验证报告示例

```json
{
  "timestamp": "2025-01-16T12:00:00",
  "cleaned_data": {
    "total_files": 1500,
    "passed": 1480,
    "warnings": 20,
    "errors": 0
  },
  "feature_data": {
    "total_samples": 500000,
    "nan_count": 0,
    "inf_count": 0,
    "label_distribution": {
      "0": 0.48,
      "1": 0.52
    }
  }
}
```

## 问题处理

### Error级别问题
必须立即修复，否则无法继续训练：
- 清除缓存并重新运行数据清洗
- 检查原始数据质量
- 修改清洗逻辑处理异常情况

### Warning级别问题
可以继续，但可能影响模型效果：
- 评估影响范围
- 必要时调整参数
- 记录问题用于后续优化

## 下一步

校验通过后，可以：
1. 如果是清洗数据校验通过：运行特征工程 `python -m pipeline.data_cleaning.features`
2. 如果是特征数据校验通过：开始模型训练 `python -m pipeline.training.train`
