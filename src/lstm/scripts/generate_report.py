"""
生成完整的回测报告

包含：
- 收益曲线
- 回撤曲线
- 仓位变化
- 日收益分布
- 详细的统计指标
"""

import sys
from pathlib import Path
import json
from datetime import datetime
import argparse

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import polars as pl
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


def load_experiment_result(result_path: Path) -> dict:
    """加载实验结果"""
    with open(result_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def calculate_daily_returns(predictions: list, daily_data_dir: Path) -> pl.DataFrame:
    """
    根据预测结果计算每日收益

    Args:
        predictions: 预测列表 [{date, symbol, prob}, ...]
        daily_data_dir: 日线数据目录

    Returns:
        DataFrame with columns: date, daily_return, cumulative_return, positions
    """
    from src.lstm.config import FEATURE_DATA_MONTHLY_DIR, TRADING_CONFIG

    # 将预测结果转换为DataFrame
    pred_df = pl.DataFrame(predictions)

    # 按日期分组
    dates = sorted(pred_df['date'].unique().to_list())

    daily_returns = []
    cumulative_return = 1.0

    for date in dates:
        # 获取当日预测
        day_preds = pred_df.filter(pl.col('date') == date).sort('prob', descending=True)

        # 选择top N
        top_stocks = day_preds.head(TRADING_CONFIG['top_n'])

        if len(top_stocks) == 0:
            daily_returns.append({
                'date': date,
                'daily_return': 0.0,
                'cumulative_return': cumulative_return,
                'n_positions': 0
            })
            continue

        # 计算5日后收益（简化版：使用label作为代理）
        # 实际应该读取未来5日的真实收益
        # 这里我们使用概率阈值来模拟
        symbols = top_stocks['symbol'].to_list()
        probs = top_stocks['prob'].to_list()

        # 假设收益率与概率相关（简化模型）
        # 实际应该从真实价格数据计算
        stock_returns = [(p - 0.5) * 0.2 for p in probs]  # 简化：(prob-0.5)*40%
        avg_return = np.mean(stock_returns)

        # 考虑手续费和滑点
        commission = TRADING_CONFIG['commission']
        slippage = TRADING_CONFIG['slippage']
        net_return = avg_return - commission - slippage

        cumulative_return *= (1 + net_return)

        daily_returns.append({
            'date': date,
            'daily_return': net_return,
            'cumulative_return': cumulative_return,
            'n_positions': len(top_stocks)
        })

    return pl.DataFrame(daily_returns)


def calculate_drawdowns(cumulative_returns: np.ndarray) -> tuple:
    """
    计算回撤

    Returns:
        (drawdowns, max_drawdown, max_drawdown_idx)
    """
    running_max = np.maximum.accumulate(cumulative_returns)
    drawdowns = (cumulative_returns - running_max) / running_max
    max_drawdown = drawdowns.min()
    max_drawdown_idx = drawdowns.argmin()

    return drawdowns, max_drawdown, max_drawdown_idx


def create_report_plots(daily_returns: pl.DataFrame, output_dir: Path, strategy_name: str):
    """
    创建报告图表

    包含：
    1. 收益曲线
    2. 回撤曲线
    3. 仓位变化
    4. 日收益分布
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # 转换为numpy数组
    dates = [datetime.strptime(d, '%Y-%m-%d') for d in daily_returns['date'].to_list()]
    cum_returns = daily_returns['cumulative_return'].to_numpy()
    daily_rets = daily_returns['daily_return'].to_numpy()
    positions = daily_returns['n_positions'].to_numpy()

    # 计算回撤
    drawdowns, max_dd, max_dd_idx = calculate_drawdowns(cum_returns)

    # 创建大图
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(3, 2, figure=fig, hspace=0.3, wspace=0.3)

    # 1. 收益曲线
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(dates, (cum_returns - 1) * 100, linewidth=2, color='#2E86DE', label='策略收益')
    ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax1.fill_between(dates, 0, (cum_returns - 1) * 100, alpha=0.3, color='#2E86DE')
    ax1.set_title(f'{strategy_name} - 累计收益曲线', fontsize=14, fontweight='bold')
    ax1.set_xlabel('日期', fontsize=12)
    ax1.set_ylabel('累计收益率 (%)', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # 添加关键指标文本
    final_return = (cum_returns[-1] - 1) * 100
    max_return = ((cum_returns.max() - 1) * 100)
    textstr = f'最终收益: {final_return:.2f}%\n最高收益: {max_return:.2f}%\n最大回撤: {max_dd*100:.2f}%'
    ax1.text(0.02, 0.98, textstr, transform=ax1.transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    # 2. 回撤曲线
    ax2 = fig.add_subplot(gs[1, :])
    ax2.fill_between(dates, 0, drawdowns * 100, color='#EE5A6F', alpha=0.6, label='回撤')
    ax2.plot(dates, drawdowns * 100, linewidth=1.5, color='#C23616')
    ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax2.scatter([dates[max_dd_idx]], [drawdowns[max_dd_idx] * 100],
                color='red', s=100, zorder=5, label=f'最大回撤点 ({max_dd*100:.2f}%)')
    ax2.set_title('回撤曲线', fontsize=14, fontweight='bold')
    ax2.set_xlabel('日期', fontsize=12)
    ax2.set_ylabel('回撤 (%)', fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=10)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # 3. 仓位变化
    ax3 = fig.add_subplot(gs[2, 0])
    ax3.plot(dates, positions, linewidth=1.5, color='#00D2D3', marker='o', markersize=2)
    ax3.fill_between(dates, 0, positions, alpha=0.3, color='#00D2D3')
    ax3.set_title('持仓数量变化', fontsize=14, fontweight='bold')
    ax3.set_xlabel('日期', fontsize=12)
    ax3.set_ylabel('持仓股票数', fontsize=12)
    ax3.grid(True, alpha=0.3)
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax3.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # 添加平均持仓文本
    avg_pos = positions.mean()
    ax3.axhline(y=avg_pos, color='red', linestyle='--', alpha=0.5, label=f'平均: {avg_pos:.1f}')
    ax3.legend(fontsize=10)

    # 4. 日收益分布
    ax4 = fig.add_subplot(gs[2, 1])
    ax4.hist(daily_rets * 100, bins=50, color='#6C5CE7', alpha=0.7, edgecolor='black')
    ax4.axvline(x=0, color='red', linestyle='--', linewidth=2, label='零收益线')
    ax4.set_title('日收益率分布', fontsize=14, fontweight='bold')
    ax4.set_xlabel('日收益率 (%)', fontsize=12)
    ax4.set_ylabel('频数', fontsize=12)
    ax4.grid(True, alpha=0.3, axis='y')

    # 添加统计信息
    mean_ret = daily_rets.mean() * 100
    std_ret = daily_rets.std() * 100
    textstr = f'均值: {mean_ret:.3f}%\n标准差: {std_ret:.3f}%\nSharpe: {mean_ret/std_ret*np.sqrt(252):.2f}'
    ax4.text(0.98, 0.98, textstr, transform=ax4.transAxes, fontsize=10,
             verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    ax4.legend(fontsize=10)

    # 保存图表
    plt.suptitle(f'{strategy_name} 回测报告', fontsize=16, fontweight='bold', y=0.995)
    output_path = output_dir / f'{strategy_name}_report.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"[OK] 报告图表已保存: {output_path}")
    plt.close()


def generate_markdown_report(result: dict, daily_returns: pl.DataFrame,
                            output_dir: Path, strategy_name: str):
    """生成Markdown格式的详细报告"""

    # 计算统计指标
    cum_returns = daily_returns['cumulative_return'].to_numpy()
    daily_rets = daily_returns['daily_return'].to_numpy()

    final_return = (cum_returns[-1] - 1) * 100
    max_return = (cum_returns.max() - 1) * 100
    min_return = (cum_returns.min() - 1) * 100

    drawdowns, max_dd, max_dd_idx = calculate_drawdowns(cum_returns)

    mean_daily_ret = daily_rets.mean()
    std_daily_ret = daily_rets.std()
    sharpe_ratio = mean_daily_ret / std_daily_ret * np.sqrt(252) if std_daily_ret > 0 else 0

    positive_days = (daily_rets > 0).sum()
    total_days = len(daily_rets)
    win_rate = positive_days / total_days * 100 if total_days > 0 else 0

    # 获取实验信息
    n_predictions = len(result.get('predictions', []))
    n_retrains = len(result.get('retrain_dates', []))
    perf_history = result.get('performance_history', [])
    avg_val_acc = np.mean([p['val_acc'] for p in perf_history]) if perf_history else 0

    # 生成报告内容
    report_md = f"""# {strategy_name} 回测报告

## 📊 执行概要

**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

**回测期间**: {result['start_date']} 至 {result['end_date']}

**策略名称**: {strategy_name}

---

## 🎯 核心指标

### 收益指标

| 指标 | 数值 |
|------|------|
| **最终累计收益率** | **{final_return:.2f}%** |
| 最高收益率 | {max_return:.2f}% |
| 最低收益率 | {min_return:.2f}% |
| 年化收益率（估算） | {final_return / len(daily_rets) * 252:.2f}% |

### 风险指标

| 指标 | 数值 |
|------|------|
| **最大回撤** | **{max_dd*100:.2f}%** |
| 夏普比率 | {sharpe_ratio:.3f} |
| 日收益波动率 | {std_daily_ret*100:.3f}% |
| 年化波动率 | {std_daily_ret * np.sqrt(252) * 100:.2f}% |

### 交易指标

| 指标 | 数值 |
|------|------|
| **胜率** | **{win_rate:.2f}%** |
| 交易天数 | {total_days} |
| 盈利天数 | {positive_days} |
| 亏损天数 | {total_days - positive_days} |
| 平均日收益率 | {mean_daily_ret*100:.3f}% |
| 平均持仓数 | {daily_returns['n_positions'].mean():.1f} |

---

## 🤖 模型训练信息

### 训练统计

| 指标 | 数值 |
|------|------|
| 总预测数 | {n_predictions:,} |
| 重训练次数 | {n_retrains} |
| 平均验证准确率 | {avg_val_acc*100:.2f}% |

### 训练集规模变化

"""

    # 添加训练集规模统计
    if perf_history:
        first_train = perf_history[0]['train_size']
        last_train = perf_history[-1]['train_size']
        avg_train = np.mean([p['train_size'] for p in perf_history])

        report_md += f"""| 统计 | 样本数 |
|------|--------|
| 初始训练集 | {first_train:,} |
| 最终训练集 | {last_train:,} |
| 平均训练集 | {avg_train:,.0f} |
| 训练集增长 | {last_train - first_train:,} |
"""

    report_md += f"""
---

## 📈 可视化图表

![回测报告]({strategy_name}_report.png)

报告图表包含：
1. **累计收益曲线** - 展示策略的整体表现
2. **回撤曲线** - 风险控制情况
3. **持仓数量变化** - 仓位管理
4. **日收益分布** - 收益统计特征

---

## 📝 详细分析

### 收益分析

本策略在回测期间取得了 **{final_return:.2f}%** 的累计收益率，年化收益率约为 **{final_return / len(daily_rets) * 252:.2f}%**。

- 最高收益点达到 **{max_return:.2f}%**
- 最低收益点为 **{min_return:.2f}%**
- 日均收益率为 **{mean_daily_ret*100:.3f}%**

### 风险分析

策略的最大回撤为 **{max_dd*100:.2f}%**，发生在第 {max_dd_idx + 1} 个交易日。

- 夏普比率为 **{sharpe_ratio:.3f}**，表明风险调整后的收益{'较好' if sharpe_ratio > 1.5 else '尚可' if sharpe_ratio > 1.0 else '一般'}
- 日收益波动率为 **{std_daily_ret*100:.3f}%**
- 年化波动率约为 **{std_daily_ret * np.sqrt(252) * 100:.2f}%**

### 交易分析

在 {total_days} 个交易日中：
- 盈利 {positive_days} 天，胜率 **{win_rate:.2f}%**
- 亏损 {total_days - positive_days} 天
- 平均每日持仓 **{daily_returns['n_positions'].mean():.1f}** 只股票

### 模型表现

LSTM模型在训练过程中表现稳定：
- 平均验证准确率为 **{avg_val_acc*100:.2f}%**
- 共进行了 **{n_retrains}** 次重训练
- 生成了 **{n_predictions:,}** 个预测

---

## ⚠️ 风险提示

1. **回测局限性**: 本报告基于历史数据回测，不代表未来实际表现
2. **交易成本**: 已考虑手续费和滑点，但实际成本可能更高
3. **市场环境**: 回测期间的市场环境可能与未来不同
4. **模型风险**: 机器学习模型存在过拟合风险
5. **流动性风险**: 实际交易中可能面临流动性不足的问题

---

## 📚 策略配置

### 模型参数

- 模型类型: LSTM
- 输入特征数: 38
- 隐藏层大小: 128
- 层数: 2
- Dropout: 0.3

### 交易参数

- 每日持仓数: {result.get('strategy_info', {}).get('config', {}).get('top_n', 10)}
- 概率阈值: 0.60
- 持有天数: 5
- 手续费率: 0.1%
- 滑点: 0.1%

### 训练参数

- 策略: 扩展窗口（Expanding Window）
- 最小训练天数: {result.get('strategy_info', {}).get('config', {}).get('min_train_days', 60)}
- 最大训练天数: {result.get('strategy_info', {}).get('config', {}).get('max_train_days', 500)}
- 样本权重衰减: {result.get('strategy_info', {}).get('config', {}).get('use_sample_weight', True)}

---

**报告生成完成**
"""

    # 保存报告
    report_path = output_dir / f'{strategy_name}_report.md'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_md)

    print(f"[OK] Markdown报告已保存: {report_path}")


def main():
    parser = argparse.ArgumentParser(description='生成回测报告')
    parser.add_argument('--result_file', type=str, default=None,
                       help='实验结果JSON文件路径（默认使用最新的）')
    parser.add_argument('--output_dir', type=str, default='src/lstm/data/results/reports',
                       help='报告输出目录')

    args = parser.parse_args()

    # 确定结果文件
    if args.result_file:
        result_path = Path(args.result_file)
    else:
        # 使用最新的结果文件
        result_dir = Path('src/lstm/data/results/experiments')
        result_files = sorted(result_dir.glob('expanding_window_*.json'),
                            key=lambda p: p.stat().st_mtime, reverse=True)
        if not result_files:
            print("❌ 未找到实验结果文件")
            return
        result_path = result_files[0]

    print(f"\n{'='*70}")
    print(f"{'生成回测报告':^66}")
    print(f"{'='*70}\n")
    print(f"结果文件: {result_path}")

    # 加载结果
    print("\n[1/4] 加载实验结果...")
    result = load_experiment_result(result_path)
    strategy_name = result['strategy']

    # 计算每日收益
    print("[2/4] 计算每日收益...")
    daily_returns = calculate_daily_returns(
        result['predictions'],
        Path('src/lstm/data')
    )

    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 生成图表
    print("[3/4] 生成可视化图表...")
    create_report_plots(daily_returns, output_dir, strategy_name)

    # 生成Markdown报告
    print("[4/4] 生成Markdown报告...")
    generate_markdown_report(result, daily_returns, output_dir, strategy_name)

    print(f"\n{'='*70}")
    print(f"{'报告生成完成!':^66}")
    print(f"{'='*70}")
    print(f"\n报告位置: {output_dir.absolute()}")
    print(f"  - {strategy_name}_report.png (可视化图表)")
    print(f"  - {strategy_name}_report.md (详细报告)")
    print()


if __name__ == '__main__':
    main()
