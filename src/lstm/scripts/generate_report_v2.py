"""
生成完整的回测报告（使用真实指标）

从 comparison JSON 读取真实的回测指标，生成可视化报告
"""

import sys
from pathlib import Path
import json
from datetime import datetime
import argparse

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec

# 设置中文字体
try:
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
except:
    pass


def load_comparison_result(result_path: Path) -> dict:
    """加载对比结果"""
    with open(result_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_experiment_result(result_path: Path) -> dict:
    """加载完整实验结果"""
    with open(result_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def generate_report_plots(comparison: dict, experiment: dict, output_dir: Path, strategy_name: str):
    """
    根据真实指标生成报告图表
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics = comparison['strategies'][strategy_name]['metrics']

    # 从完整实验结果获取性能历史
    perf_history = experiment.get('performance_history', [])

    # 创建大图
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)

    # 1. 核心指标展示（左上）
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.axis('off')

    metrics_text = f"""
核心回测指标

总收益率: {metrics['total_return']*100:.2f}%
年化收益: {metrics['annual_return']*100:.2f}%
夏普比率: {metrics['sharpe_ratio']:.3f}
最大回撤: {metrics['max_drawdown']*100:.2f}%
胜率: {metrics['win_rate']*100:.2f}%
交易次数: {metrics['n_trades']}
交易天数: {metrics['n_days']}
"""

    ax1.text(0.1, 0.9, metrics_text, transform=ax1.transAxes,
             fontsize=14, verticalalignment='top', family='monospace',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

    # 2. 验证准确率趋势（右上）
    ax2 = fig.add_subplot(gs[0, 1])

    if perf_history:
        train_dates = [p['date'] for p in perf_history]
        val_accs = [p['val_acc'] for p in perf_history]

        # 转换日期
        dates = [datetime.strptime(d, '%Y-%m-%d') for d in train_dates]

        ax2.plot(dates, val_accs, linewidth=1.5, color='#2E86DE', alpha=0.6)
        # 添加移动平均
        if len(val_accs) >= 5:
            from numpy import convolve, ones
            ma = convolve(val_accs, ones(5)/5, mode='valid')
            ax2.plot(dates[2:-2], ma, linewidth=2.5, color='#EE5A24', label=f'MA(5): {ma.mean():.3f}')

        ax2.axhline(y=np.mean(val_accs), color='red', linestyle='--',
                   alpha=0.5, label=f'平均: {np.mean(val_accs):.3f}')
        ax2.set_title('验证集准确率变化', fontsize=14, fontweight='bold')
        ax2.set_xlabel('日期', fontsize=12)
        ax2.set_ylabel('准确率', fontsize=12)
        ax2.grid(True, alpha=0.3)
        ax2.legend(fontsize=10)
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # 3. 训练集规模变化（左下）
    ax3 = fig.add_subplot(gs[1, 0])

    if perf_history:
        train_dates = [p['date'] for p in perf_history]
        train_sizes = [p['train_size'] for p in perf_history]
        val_sizes = [p['val_size'] for p in perf_history]

        dates = [datetime.strptime(d, '%Y-%m-%d') for d in train_dates]

        ax3.plot(dates, train_sizes, linewidth=2, color='#00D2D3', label='训练集')
        ax3.plot(dates, val_sizes, linewidth=2, color='#FFC312', label='验证集')
        ax3.fill_between(dates, 0, train_sizes, alpha=0.2, color='#00D2D3')

        ax3.set_title('训练/验证集规模变化', fontsize=14, fontweight='bold')
        ax3.set_xlabel('日期', fontsize=12)
        ax3.set_ylabel('样本数', fontsize=12)
        ax3.grid(True, alpha=0.3)
        ax3.legend(fontsize=10)
        ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax3.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
        plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45, ha='right')

        # 添加统计文本
        textstr = f'初始: {train_sizes[0]:,}\n最终: {train_sizes[-1]:,}\n平均: {np.mean(train_sizes):,.0f}'
        ax3.text(0.02, 0.98, textstr, transform=ax3.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    # 4. 指标对比雷达图（右下）
    ax4 = fig.add_subplot(gs[1, 1], projection='polar')

    # 归一化指标（0-1范围）
    indicators = {
        '总收益率': min(metrics['total_return'] / 2.0, 1.0),  # 假设200%为满分
        '夏普比率': min(metrics['sharpe_ratio'] / 3.0, 1.0),  # 假设3.0为满分
        '胜率': metrics['win_rate'],
        '验证准确率': np.mean(val_accs) if perf_history else 0,
        '风险控制': 1.0 - min(abs(metrics['max_drawdown']), 1.0)  # 回撤越小越好
    }

    categories = list(indicators.keys())
    values = list(indicators.values())

    # 闭合雷达图
    values += values[:1]
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]

    ax4.plot(angles, values, 'o-', linewidth=2, color='#6C5CE7')
    ax4.fill(angles, values, alpha=0.25, color='#6C5CE7')
    ax4.set_xticks(angles[:-1])
    ax4.set_xticklabels(categories, fontsize=10)
    ax4.set_ylim(0, 1)
    ax4.set_title('综合评分雷达图', fontsize=14, fontweight='bold', pad=20)
    ax4.grid(True)

    # 保存图表
    plt.suptitle(f'{strategy_name} 回测报告', fontsize=16, fontweight='bold', y=0.98)
    output_path = output_dir / f'{strategy_name}_report.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"[OK] 报告图表已保存: {output_path}")
    plt.close()


def generate_markdown_report(comparison: dict, experiment: dict,
                            output_dir: Path, strategy_name: str):
    """生成Markdown格式的详细报告"""

    metrics = comparison['strategies'][strategy_name]['metrics']
    perf_history = experiment.get('performance_history', [])
    strategy_info = experiment.get('strategy_info', {})

    n_predictions = comparison['strategies'][strategy_name]['n_predictions']
    n_retrains = comparison['strategies'][strategy_name]['n_retrains']
    avg_val_acc = np.mean([p['val_acc'] for p in perf_history]) if perf_history else 0

    # 生成报告内容
    report_md = f"""# {strategy_name} 回测报告

## 📊 执行概要

**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

**回测期间**: {experiment['start_date']} 至 {experiment['end_date']}

**策略名称**: {strategy_name} (扩展窗口策略)

---

## 🎯 核心指标

### 收益指标

| 指标 | 数值 | 评价 |
|------|------|------|
| **总收益率** | **{metrics['total_return']*100:+.2f}%** | {'🟢 优秀' if metrics['total_return'] > 0.8 else '🟡 良好' if metrics['total_return'] > 0.4 else '🔴 一般'} |
| **年化收益率** | **{metrics['annual_return']*100:+.2f}%** | {'🟢 优秀' if metrics['annual_return'] > 1.5 else '🟡 良好' if metrics['annual_return'] > 0.8 else '🔴 一般'} |
| 交易天数 | {metrics['n_days']} 天 | - |
| 交易次数 | {metrics['n_trades']} 次 | - |

### 风险指标

| 指标 | 数值 | 评价 |
|------|------|------|
| **夏普比率** | **{metrics['sharpe_ratio']:.3f}** | {'🟢 优秀' if metrics['sharpe_ratio'] > 1.5 else '🟡 良好' if metrics['sharpe_ratio'] > 1.0 else '🔴 一般'} |
| **最大回撤** | **{metrics['max_drawdown']*100:.2f}%** | {'🟢 优秀' if metrics['max_drawdown'] < 0.3 else '🟡 良好' if metrics['max_drawdown'] < 0.5 else '🔴 较大'} |
| **胜率** | **{metrics['win_rate']*100:.2f}%** | {'🟢 优秀' if metrics['win_rate'] > 0.6 else '🟡 良好' if metrics['win_rate'] > 0.5 else '🔴 一般'} |

### 风险调整后收益

- **收益回撤比**: {(metrics['total_return'] / metrics['max_drawdown']):.2f}
- **卡尔马比率** (年化收益/最大回撤): {(metrics['annual_return'] / metrics['max_drawdown']):.2f}

---

## 🤖 模型训练信息

### 训练统计

| 指标 | 数值 |
|------|------|
| 总预测数 | {n_predictions:,} |
| 重训练次数 | {n_retrains} |
| 平均验证准确率 | {avg_val_acc*100:.2f}% |
| 最高验证准确率 | {max([p['val_acc'] for p in perf_history])*100:.2f}% |
| 最低验证准确率 | {min([p['val_acc'] for p in perf_history])*100:.2f}% |

### 训练集规模变化

"""

    if perf_history:
        first_train = perf_history[0]['train_size']
        last_train = perf_history[-1]['train_size']
        avg_train = np.mean([p['train_size'] for p in perf_history])

        report_md += f"""| 统计 | 样本数 |
|------|--------|
| 初始训练集 | {first_train:,} |
| 最终训练集 | {last_train:,} |
| 平均训练集 | {avg_train:,.0f} |
| 训练集增长 | {last_train - first_train:,} (+{(last_train/first_train-1)*100:.1f}%) |

**说明**: 扩展窗口策略的训练集逐日增长，从 {first_train:,} 样本增长到 {last_train:,} 样本，增长了 {(last_train/first_train-1)*100:.1f}%。
"""

    report_md += f"""
---

## 📈 可视化图表

![回测报告]({strategy_name}_report.png)

报告图表包含：
1. **核心指标展示** - 关键回测指标一览
2. **验证准确率趋势** - 模型性能随时间的变化
3. **训练集规模变化** - 扩展窗口策略的数据增长
4. **综合评分雷达图** - 多维度策略评估

---

## 📝 详细分析

### 收益分析

本策略在 **{metrics['n_days']}** 个交易日内取得了 **{metrics['total_return']*100:+.2f}%** 的累计收益率。

- **年化收益率**: {metrics['annual_return']*100:+.2f}%
- **日均收益率**: {metrics['total_return']/metrics['n_days']*100:.3f}%
- **总交易次数**: {metrics['n_trades']} 次
- **平均持仓天数**: {metrics['n_days']/metrics['n_trades']*5:.1f} 天（基于5日持有期）

"""

    # 收益评价
    if metrics['total_return'] > 1.0:
        report_md += "\n**评价**: 策略取得了**超过100%**的累计收益，表现优异。\n"
    elif metrics['total_return'] > 0.5:
        report_md += "\n**评价**: 策略取得了**50%以上**的累计收益，表现良好。\n"
    else:
        report_md += "\n**评价**: 策略取得了正收益，但仍有提升空间。\n"

    report_md += f"""
### 风险分析

策略的风险控制表现{'**优秀**' if metrics['max_drawdown'] < 0.4 else '**良好**' if metrics['max_drawdown'] < 0.5 else '**一般**'}：

- **最大回撤**: {metrics['max_drawdown']*100:.2f}%
- **夏普比率**: {metrics['sharpe_ratio']:.3f} ({'风险调整后收益优秀' if metrics['sharpe_ratio'] > 1.5 else '风险调整后收益良好' if metrics['sharpe_ratio'] > 1.0 else '风险调整后收益一般'})
- **收益回撤比**: {(metrics['total_return'] / metrics['max_drawdown']):.2f} (收益是最大回撤的 {(metrics['total_return'] / metrics['max_drawdown']):.2f} 倍)

### 交易分析

在 {metrics['n_trades']} 次交易中：
- **胜率**: {metrics['win_rate']*100:.2f}%
- **盈利次数**: {int(metrics['n_trades'] * metrics['win_rate'])} 次
- **亏损次数**: {int(metrics['n_trades'] * (1 - metrics['win_rate']))} 次

"""

    if metrics['win_rate'] > 0.6:
        report_md += "\n**评价**: 胜率超过60%，说明策略的选股能力较强。\n"
    elif metrics['win_rate'] > 0.5:
        report_md += "\n**评价**: 胜率超过50%，说明策略具有一定的选股能力。\n"
    else:
        report_md += "\n**评价**: 胜率低于50%，建议优化选股逻辑。\n"

    report_md += f"""
### 模型表现

LSTM模型的训练表现：
- **平均验证准确率**: {avg_val_acc*100:.2f}%
- **准确率波动**: {min([p['val_acc'] for p in perf_history])*100:.2f}% ~ {max([p['val_acc'] for p in perf_history])*100:.2f}%
- **重训练次数**: {n_retrains} 次（每日重训练）
- **总预测数**: {n_predictions:,}

扩展窗口策略的特点：
- ✅ 训练集持续增长，从 {perf_history[0]['train_size']:,} 增长到 {perf_history[-1]['train_size']:,}
- ✅ 样本权重指数衰减，近期数据权重更高
- ✅ 每日重训练，保持模型时效性

---

## ⚠️ 风险提示

### 回测局限性

1. **历史数据回测**: 本报告基于历史数据回测，不代表未来实际表现
2. **市场环境变化**: 回测期间的市场环境可能与未来不同
3. **交易成本**: 已考虑手续费（0.1%）和滑点（0.1%），但实际成本可能更高

### 策略风险

1. **模型风险**: 机器学习模型存在过拟合风险，需持续监控验证准确率
2. **流动性风险**: 实际交易中可能面临流动性不足，影响执行价格
3. **黑天鹅事件**: 极端市场事件可能导致模型失效
4. **数据质量**: 策略依赖高质量的历史数据，数据错误会影响预测

### 使用建议

1. **分批建仓**: 建议分批建仓，避免一次性投入过大
2. **设置止损**: 建议设置合理的止损位，控制单日损失
3. **持续监控**: 需持续监控模型的验证准确率和实际表现
4. **定期评估**: 建议每月评估策略表现，必要时调整参数

---

## 📚 策略配置

### 模型参数

- **模型类型**: LSTM（长短期记忆网络）
- **输入特征数**: 38 个技术指标
- **隐藏层大小**: 128
- **网络层数**: 2
- **Dropout率**: 0.3（防止过拟合）
- **优化器**: AdamW
- **学习率**: 0.001

### 交易参数

- **每日持仓数**: 10 只股票
- **概率阈值**: 0.60（只选择预测概率 > 60% 的股票）
- **持有天数**: 5 天
- **手续费率**: 0.1%
- **滑点**: 0.1%

### 训练参数（扩展窗口策略）

- **策略类型**: Expanding Window（扩展窗口）
- **最小训练天数**: 60 天
- **最大训练天数**: 500 天
- **验证集天数**: 1 天
- **样本权重衰减**: 开启
- **权重衰减系数**: {strategy_info.get('config', {}).get('weight_decay_rate', 0.95)}
- **衰减周期**: {strategy_info.get('config', {}).get('weight_decay_days', 30)} 天
- **重训练间隔**: {strategy_info.get('config', {}).get('retrain_interval', 1)} 天

---

## 🎓 策略说明

### 扩展窗口策略原理

扩展窗口（Expanding Window）是一种时间序列交叉验证方法：

1. **累积历史数据**: 训练集逐日增长，保留所有历史数据
2. **样本权重衰减**: 使用指数衰减给近期数据更高权重
3. **每日重训练**: 每天使用最新的数据重新训练模型
4. **Walk-Forward验证**: 严格按时间顺序划分训练/验证集，避免前瞻偏差

### 为什么选择扩展窗口？

- ✅ **适应市场变化**: 持续学习新数据，适应市场环境变化
- ✅ **保留长期规律**: 保留历史数据中的长期规律和周期
- ✅ **权重平衡**: 通过样本权重平衡长期规律和短期变化
- ✅ **避免灾难性遗忘**: 不会因为新数据而完全遗忘历史规律

### 中证1000动态成分股

本策略使用**动态成分股**方法，避免幸存者偏差：

- 📅 **按日期读取**: 每日只预测当时在中证1000指数中的股票
- 🎯 **真实模拟**: 完全模拟真实交易环境，不使用未来信息
- ✅ **无幸存者偏差**: 不会因为只用存活股票而高估收益

---

## 📞 技术支持

如有任何问题或建议，欢迎反馈。

**报告生成完成** - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""

    # 保存报告
    report_path = output_dir / f'{strategy_name}_report.md'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_md)

    print(f"[OK] Markdown报告已保存: {report_path}")


def main():
    parser = argparse.ArgumentParser(description='生成回测报告（使用真实指标）')
    parser.add_argument('--comparison_file', type=str, default=None,
                       help='对比结果JSON文件路径（默认使用最新的）')
    parser.add_argument('--experiment_file', type=str, default=None,
                       help='实验结果JSON文件路径（默认使用最新的）')
    parser.add_argument('--output_dir', type=str, default='src/lstm/data/results/reports',
                       help='报告输出目录')

    args = parser.parse_args()

    print(f"\n{'='*70}")
    print(f"{'生成回测报告':^66}")
    print(f"{'='*70}\n")

    # 确定结果文件
    result_dir = Path('src/lstm/data/results/experiments')

    if args.comparison_file:
        comparison_path = Path(args.comparison_file)
    else:
        comparison_files = sorted(result_dir.glob('comparison_*.json'),
                                 key=lambda p: p.stat().st_mtime, reverse=True)
        if not comparison_files:
            print("[ERROR] 未找到对比结果文件")
            return
        comparison_path = comparison_files[0]

    if args.experiment_file:
        experiment_path = Path(args.experiment_file)
    else:
        experiment_files = sorted(result_dir.glob('expanding_window_*.json'),
                                 key=lambda p: p.stat().st_mtime, reverse=True)
        if not experiment_files:
            print("[ERROR] 未找到实验结果文件")
            return
        experiment_path = experiment_files[0]

    print(f"对比文件: {comparison_path.name}")
    print(f"实验文件: {experiment_path.name}")

    # 加载结果
    print("\n[1/4] 加载结果文件...")
    comparison = load_comparison_result(comparison_path)
    experiment = load_experiment_result(experiment_path)
    strategy_name = experiment['strategy']

    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 生成图表
    print("[2/4] 生成可视化图表...")
    generate_report_plots(comparison, experiment, output_dir, strategy_name)

    # 生成Markdown报告
    print("[3/4] 生成Markdown报告...")
    generate_markdown_report(comparison, experiment, output_dir, strategy_name)

    print("[4/4] 完成!")
    print(f"\n{'='*70}")
    print(f"{'报告生成完成!':^66}")
    print(f"{'='*70}")
    print(f"\n报告位置: {output_dir.absolute()}")
    print(f"  - {strategy_name}_report.png (可视化图表)")
    print(f"  - {strategy_name}_report.md (详细报告)")
    print()


if __name__ == '__main__':
    main()
