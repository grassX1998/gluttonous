# 13小时持续优化系统

自动化多策略训练、参数调优和仓位管理优化框架。

## 📋 系统组件

### 1. 多策略配置管理器 (`multi_strategy_config.py`)

管理6种训练策略及其参数空间：

| 策略 | 描述 | 参数组合数 |
|------|------|----------|
| **expanding_window** | 扩展窗口 - 累积历史数据 | 2,916 |
| **rolling_window** | 滚动窗口 - 固定长度训练集 | 486 |
| **adaptive_weight** | 自适应权重 - 根据准确率调整 | 128 |
| **ensemble_multi_scale** | 多尺度集成 - 短中长期组合 | 256 |
| **volatility_adaptive** | 波动率自适应 - 市场状态调整 | 256 |
| **momentum_enhanced** | 动量增强 - 结合短期信号 | 576 |

### 2. 并行执行器 (`parallel_executor.py`)

- 多线程并行训练和回测
- 建议2-3个worker（避免GPU冲突）
- 自动任务队列管理
- 综合评分系统（收益40% + 夏普30% + 回撤20% + 胜率10%）

### 3. 仓位管理优化器 (`position_manager_optimizer.py`)

8种仓位管理策略：

| 策略 | 描述 |
|------|------|
| **equal_weight** | 等权重 - 平均分配 |
| **prob_weighted** | 概率加权 - 根据预测概率 |
| **kelly_criterion** | Kelly公式 - 根据胜率和盈亏比 |
| **volatility_adjusted** | 波动率调整 - 低波动高仓位 |
| **risk_parity** | 风险平价 - 均衡风险贡献 |
| **dynamic_sizing** | 动态仓位 - 根据盈亏调整 |
| **concentration_limit** | 集中度限制 - 限制单股/行业占比 |
| **adaptive_leverage** | 自适应杠杆 - 根据市场状态 |

### 4. 持续优化主控 (`continuous_optimization.py`)

- 13小时持续运行
- 智能参数采样（70% exploitation + 30% exploration）
- 每30分钟自动保存进度
- 自动生成总结报告

## 🚀 快速开始

### 1. 测试框架

```bash
python scripts/test_optimization_framework.py
```

验证所有组件是否正常工作。

### 2. 启动13小时优化

```bash
python scripts/start_13hour_optimization.py
```

或使用自定义参数：

```bash
cd src/lstm/optimization
python continuous_optimization.py --hours 13 --workers 2 --save-interval 30
```

## 📊 输出结构

```
src/lstm/data/results/optimization/
├── optimization_log_{timestamp}.json        # 优化日志
├── best_params_{strategy}.json              # 各策略最佳参数
├── summary_report_{timestamp}.md            # 总结报告
├── {task_id}_config.json                    # 任务配置
├── {task_id}_result.json                    # 任务结果
└── position_mgmt/
    └── position_mgmt_state.json             # 仓位管理状态
```

## 🔧 参数调优策略

### Exploitation（70%）

基于当前最佳参数进行小幅扰动：
- 选择1-2个参数修改
- 在参数空间中选择相邻值
- 逐步优化，稳步提升

### Exploration（30%）

随机探索参数空间：
- 完全随机采样
- 发现新的优化方向
- 避免局部最优

## 📈 评分系统

综合评分公式：
```python
score = (
    total_return * 0.4 +      # 总收益率（40%）
    sharpe * 0.1 * 0.3 +      # 夏普比率（30%）
    (1 - max_dd) * 0.2 +      # 最大回撤（20%，反向）
    win_rate * 0.1            # 胜率（10%）
)
```

## 🎯 优化流程

```
开始
  ↓
启动并行执行器（2-3 workers）
  ↓
迭代循环（13小时）
  │
  ├─ 收集完成的结果
  │  ├─ 更新最佳参数
  │  └─ 记录优化历史
  │
  ├─ 生成新任务
  │  ├─ 智能采样参数
  │  └─ 提交到任务队列
  │
  └─ 定期保存（每30分钟）
     ├─ 保存优化日志
     └─ 保存最佳参数
  ↓
生成总结报告
  ↓
结束
```

## ⚙️ 配置说明

### 训练参数示例（expanding_window）

```python
train_params = {
    'min_train_days': 60,           # 最小训练天数
    'max_train_days': 500,          # 最大训练天数
    'weight_decay_days': 20,        # 权重衰减周期
    'weight_decay_rate': 0.95,      # 衰减率
    'retrain_interval': 1,          # 重训练间隔
}
```

### 交易参数示例

```python
trading_params = {
    'prob_threshold': 0.70,         # 概率阈值
    'trailing_stop_pct': 0.05,      # 动态回撤5%
    'max_holding_days': 10,         # 最长持有
    'min_holding_days': 1,          # 最短持有
}
```

## 📝 使用建议

### 1. 调整并行数

根据硬件配置调整：
- **单GPU**: max_workers = 1-2
- **多GPU**: max_workers = GPU数量
- **高内存**: 可以增加到3-4

### 2. 保存间隔

根据需求调整：
- **快速实验**: 10-15分钟
- **正常运行**: 30分钟
- **长期运行**: 60分钟

### 3. 运行时长

根据目标调整：
- **快速测试**: 1-2小时
- **标准优化**: 13小时
- **深度优化**: 24-48小时

## 🔍 监控和调试

### 查看实时日志

```bash
# 查看优化日志
tail -f src/lstm/data/results/optimization/optimization_log_*.json

# 查看最佳参数
cat src/lstm/data/results/optimization/best_params_expanding_window.json
```

### 中断和恢复

按 `Ctrl+C` 安全中断：
- 自动保存当前进度
- 生成部分总结报告
- 不会丢失已完成的结果

恢复运行：
- 系统会自动加载之前的最佳参数
- 继续优化（暂不支持断点续传完整状态）

## 📚 扩展开发

### 添加新策略

1. 在 `multi_strategy_config.py` 中添加策略配置
2. 在 `parallel_executor.py` 中实现训练逻辑
3. 更新参数空间定义

### 添加新仓位管理策略

1. 在 `position_manager_optimizer.py` 中添加策略
2. 实现 `_xxx_sizing()` 方法
3. 更新策略池

### 自定义评分函数

修改 `parallel_executor.py` 中的 `_calculate_score()` 方法，调整各指标权重。

## ⚠️ 注意事项

1. **GPU内存**: 多worker可能导致OOM，建议监控GPU使用率
2. **磁盘空间**: 大量训练会产生大量检查点，定期清理
3. **时间估算**: 单次训练约1-2小时，规划好总任务数
4. **参数冲突**: 多worker可能修改相同配置文件，当前版本暂未完全解决

## 📊 预期效果

13小时优化预期完成：
- 约10-15轮完整迭代
- 每个策略测试2-3个参数组合
- 生成完整的优化报告
- 找到每个策略的局部最优参数

## 🎉 成果展示

优化完成后，查看：
- `summary_report_{timestamp}.md` - 总结报告
- `best_params_{strategy}.json` - 各策略最佳参数
- `optimization_log_{timestamp}.json` - 完整优化历史

根据报告选择最佳策略和参数，用于后续实盘或进一步优化。

---

**版本**: v0.4
**创建时间**: 2026-01-16
**维护者**: Gluttonous Team
