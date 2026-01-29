"""
生成完整回测报告

包含交易订单、持仓变化、收益曲线等详细可视化
"""

import sys
from pathlib import Path
import json
from datetime import datetime
import argparse

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec
import matplotlib.font_manager as fm

# 设置中文字体
def setup_chinese_font():
    chinese_fonts = ['Microsoft YaHei', 'SimHei', 'STSong', 'SimSun']
    available_fonts = [f.name for f in fm.fontManager.ttflist]
    for font in chinese_fonts:
        if font in available_fonts:
            plt.rcParams['font.sans-serif'] = [font]
            plt.rcParams['axes.unicode_minus'] = False
            return True
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
    return False

has_chinese_font = setup_chinese_font()


def load_backtest_result(result_path: Path) -> dict:
    """加载回测结果"""
    with open(result_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def generate_comprehensive_plots(result: dict, output_dir: Path):
    """
    生成综合性能图表
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics = result['metrics']
    daily_records = result['backtest']['daily_records']
    orders = result['backtest']['orders']

    # 转换为DataFrame
    df_daily = pd.DataFrame(daily_records)
    df_daily['date'] = pd.to_datetime(df_daily['date'])

    df_orders = pd.DataFrame(orders)
    if len(df_orders) > 0:
        df_orders['date'] = pd.to_datetime(df_orders['date'])

    # 创建大图
    fig = plt.figure(figsize=(20, 14))
    gs = GridSpec(4, 2, figure=fig, hspace=0.35, wspace=0.3)

    color_profit = '#26de81'
    color_loss = '#EE5A6F'
    color_primary = '#2E86DE'
    color_secondary = '#FFC312'

    # ==================== 图1: 收益率曲线（左上） ====================
    ax1 = fig.add_subplot(gs[0, :])

    ax1.plot(df_daily['date'], df_daily['return'] * 100,
            linewidth=2.5, color=color_primary, label='Portfolio Return')
    ax1.fill_between(df_daily['date'], 0, df_daily['return'] * 100,
                     where=(df_daily['return'] >= 0), alpha=0.3, color=color_profit,
                     label='Profit Period')
    ax1.fill_between(df_daily['date'], 0, df_daily['return'] * 100,
                     where=(df_daily['return'] < 0), alpha=0.3, color=color_loss,
                     label='Loss Period')

    ax1.axhline(y=0, color='gray', linestyle='--', linewidth=1.5, alpha=0.5)
    ax1.set_title('Cumulative Return Over Time', fontsize=16, fontweight='bold')
    ax1.set_xlabel('Date', fontsize=12)
    ax1.set_ylabel('Return (%)', fontsize=12)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.legend(fontsize=10, loc='best')
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # 添加关键点标注
    max_return = df_daily['return'].max()
    min_return = df_daily['return'].min()
    max_date = df_daily.loc[df_daily['return'].idxmax(), 'date']
    min_date = df_daily.loc[df_daily['return'].idxmin(), 'date']

    ax1.annotate(f'Peak: {max_return*100:.2f}%', xy=(max_date, max_return*100),
                xytext=(10, 10), textcoords='offset points',
                bbox=dict(boxstyle='round', facecolor='green', alpha=0.5),
                arrowprops=dict(arrowstyle='->', color='green'))
    ax1.annotate(f'Trough: {min_return*100:.2f}%', xy=(min_date, min_return*100),
                xytext=(10, -20), textcoords='offset points',
                bbox=dict(boxstyle='round', facecolor='red', alpha=0.5),
                arrowprops=dict(arrowstyle='->', color='red'))

    # ==================== 图2: 资产组成变化（左中） ====================
    ax2 = fig.add_subplot(gs[1, 0])

    ax2.plot(df_daily['date'], df_daily['cash'] / 1e6,
            linewidth=2, color='#00D2D3', label='Cash', marker='o', markersize=1)
    ax2.plot(df_daily['date'], df_daily['holdings_value'] / 1e6,
            linewidth=2, color='#FFC312', label='Holdings', marker='o', markersize=1)
    ax2.plot(df_daily['date'], df_daily['portfolio_value'] / 1e6,
            linewidth=3, color='#6C5CE7', label='Total', linestyle='--')

    ax2.set_title('Portfolio Composition', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Date', fontsize=12)
    ax2.set_ylabel('Value (Million $)', fontsize=12)
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.legend(fontsize=10)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # ==================== 图3: 持仓比例变化（右中） ====================
    ax3 = fig.add_subplot(gs[1, 1])

    ax3.fill_between(df_daily['date'], 0, df_daily['position_ratio'] * 100,
                     color=color_secondary, alpha=0.6, label='Position Ratio')
    ax3.plot(df_daily['date'], df_daily['position_ratio'] * 100,
            linewidth=2, color='#EE5A24', label='Position %')

    # 计算平均持仓比例
    avg_ratio = df_daily['position_ratio'].mean() * 100
    ax3.axhline(y=avg_ratio, color='red', linestyle='--',
               linewidth=2, label=f'Avg: {avg_ratio:.1f}%')

    ax3.set_title('Position Ratio Over Time', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Date', fontsize=12)
    ax3.set_ylabel('Position Ratio (%)', fontsize=12)
    ax3.set_ylim(0, 105)
    ax3.grid(True, alpha=0.3, linestyle='--')
    ax3.legend(fontsize=10)
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax3.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # ==================== 图4: 持仓数量变化（左下） ====================
    ax4 = fig.add_subplot(gs[2, 0])

    ax4.plot(df_daily['date'], df_daily['n_positions'],
            linewidth=2.5, color=color_primary, marker='o', markersize=3)
    ax4.fill_between(df_daily['date'], 0, df_daily['n_positions'],
                     alpha=0.3, color=color_primary)

    avg_positions = df_daily['n_positions'].mean()
    ax4.axhline(y=avg_positions, color='red', linestyle='--',
               linewidth=2, label=f'Avg: {avg_positions:.1f}')

    ax4.set_title('Number of Positions Over Time', fontsize=14, fontweight='bold')
    ax4.set_xlabel('Date', fontsize=12)
    ax4.set_ylabel('Number of Positions', fontsize=12)
    ax4.set_ylim(0, result['config']['top_n'] + 1)
    ax4.grid(True, alpha=0.3, linestyle='--')
    ax4.legend(fontsize=10)
    ax4.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax4.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # ==================== 图5: 交易盈亏分布（右下） ====================
    ax5 = fig.add_subplot(gs[2, 1])

    if len(df_orders) > 0:
        sell_orders = df_orders[df_orders['action'] == 'SELL'].copy()
        if len(sell_orders) > 0:
            pnl_values = sell_orders['pnl_pct'].values * 100

            # 绘制直方图
            n, bins, patches = ax5.hist(pnl_values, bins=40, edgecolor='black',
                                       linewidth=1.5, alpha=0.7)

            # 根据盈亏给柱子着色
            for i, patch in enumerate(patches):
                if bins[i] >= 0:
                    patch.set_facecolor(color_profit)
                else:
                    patch.set_facecolor(color_loss)

            ax5.axvline(x=0, color='black', linestyle='--', linewidth=2.5,
                       label='Break-even')

            # 添加统计信息
            mean_pnl = pnl_values.mean()
            median_pnl = np.median(pnl_values)
            ax5.axvline(x=mean_pnl, color='blue', linestyle='--', linewidth=2,
                       label=f'Mean: {mean_pnl:.2f}%')

            textstr = f'Mean: {mean_pnl:.2f}%\nMedian: {median_pnl:.2f}%\nStd: {pnl_values.std():.2f}%'
            ax5.text(0.98, 0.98, textstr, transform=ax5.transAxes,
                    fontsize=11, verticalalignment='top', horizontalalignment='right',
                    family='monospace',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    ax5.set_title('Trade P&L Distribution', fontsize=14, fontweight='bold')
    ax5.set_xlabel('P&L (%)', fontsize=12)
    ax5.set_ylabel('Frequency', fontsize=12)
    ax5.grid(True, alpha=0.3, axis='y', linestyle='--')
    ax5.legend(fontsize=10)

    # ==================== 图6: 回撤曲线（底部跨两列） ====================
    ax6 = fig.add_subplot(gs[3, :])

    # 计算回撤
    portfolio_values = df_daily['portfolio_value'].values
    peak = np.maximum.accumulate(portfolio_values)
    drawdown = (portfolio_values - peak) / peak * 100

    ax6.fill_between(df_daily['date'], 0, drawdown,
                     color=color_loss, alpha=0.6, label='Drawdown')
    ax6.plot(df_daily['date'], drawdown, linewidth=2, color='#C23616')

    # 标注最大回撤点
    max_dd_idx = drawdown.argmin()
    max_dd_date = df_daily.iloc[max_dd_idx]['date']
    max_dd_value = drawdown[max_dd_idx]

    ax6.scatter([max_dd_date], [max_dd_value], color='red', s=200, zorder=5,
               marker='v', label=f'Max DD: {max_dd_value:.2f}%')

    ax6.axhline(y=0, color='gray', linestyle='--', linewidth=1.5, alpha=0.5)
    ax6.set_title('Drawdown Over Time', fontsize=14, fontweight='bold')
    ax6.set_xlabel('Date', fontsize=12)
    ax6.set_ylabel('Drawdown (%)', fontsize=12)
    ax6.grid(True, alpha=0.3, linestyle='--')
    ax6.legend(fontsize=10)
    ax6.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax6.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    plt.setp(ax6.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # 保存图表
    plt.suptitle('Comprehensive Backtest Report',
                fontsize=18, fontweight='bold', y=0.995)

    output_path = output_dir / 'backtest_comprehensive.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"[OK] Comprehensive report saved: {output_path}")
    plt.close()


def generate_trade_analysis(result: dict, output_dir: Path):
    """生成交易分析图表"""

    orders = result['backtest']['orders']
    if not orders:
        print("[WARN] No orders found")
        return

    df_orders = pd.DataFrame(orders)
    df_orders['date'] = pd.to_datetime(df_orders['date'])

    # 创建图表
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle('Trade Analysis', fontsize=16, fontweight='bold')

    # 图1：每日交易次数
    ax1 = axes[0, 0]
    buy_orders = df_orders[df_orders['action'] == 'BUY']
    sell_orders = df_orders[df_orders['action'] == 'SELL']

    daily_buys = buy_orders.groupby(buy_orders['date'].dt.date).size()
    daily_sells = sell_orders.groupby(sell_orders['date'].dt.date).size()

    dates = pd.date_range(df_orders['date'].min(), df_orders['date'].max(), freq='D')
    dates_only = [d.date() for d in dates]

    buys = [daily_buys.get(d, 0) for d in dates_only]
    sells = [daily_sells.get(d, 0) for d in dates_only]

    x = np.arange(len(dates))
    width = 0.35

    ax1.bar(x - width/2, buys, width, label='Buy', color='#26de81', alpha=0.8)
    ax1.bar(x + width/2, sells, width, label='Sell', color='#EE5A6F', alpha=0.8)

    ax1.set_title('Daily Trading Activity', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Number of Trades')
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')

    # 简化x轴标签
    step = max(1, len(dates) // 10)
    ax1.set_xticks(x[::step])
    ax1.set_xticklabels([dates[i].strftime('%m-%d') for i in range(0, len(dates), step)],
                       rotation=45, ha='right')

    # 图2：盈亏累积
    ax2 = axes[0, 1]
    if len(sell_orders) > 0:
        sell_orders = sell_orders.sort_values('date')
        cumulative_pnl = sell_orders['pnl'].cumsum() / 1000  # 转换为千元

        ax2.plot(sell_orders['date'], cumulative_pnl, linewidth=2.5,
                color='#2E86DE', label='Cumulative P&L')
        ax2.fill_between(sell_orders['date'], 0, cumulative_pnl,
                        where=(cumulative_pnl >= 0), alpha=0.3, color='#26de81')
        ax2.fill_between(sell_orders['date'], 0, cumulative_pnl,
                        where=(cumulative_pnl < 0), alpha=0.3, color='#EE5A6F')

        ax2.axhline(y=0, color='gray', linestyle='--', linewidth=1.5, alpha=0.5)
        ax2.set_title('Cumulative P&L', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Date')
        ax2.set_ylabel('P&L (K$)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # 图3：持有天数分布
    ax3 = axes[1, 0]
    if len(sell_orders) > 0 and 'holding_days' in sell_orders.columns:
        holding_days = sell_orders['holding_days'].values

        ax3.hist(holding_days, bins=range(0, int(holding_days.max()) + 2),
                edgecolor='black', color='#FFC312', alpha=0.7)

        ax3.set_title('Holding Period Distribution', fontsize=12, fontweight='bold')
        ax3.set_xlabel('Holding Days')
        ax3.set_ylabel('Frequency')
        ax3.grid(True, alpha=0.3, axis='y')

        # 添加平均值线
        mean_days = holding_days.mean()
        ax3.axvline(x=mean_days, color='red', linestyle='--',
                   linewidth=2, label=f'Mean: {mean_days:.1f} days')
        ax3.legend()

    # 图4：盈亏比统计
    ax4 = axes[1, 1]
    if len(sell_orders) > 0:
        profitable = sell_orders[sell_orders['pnl'] > 0]
        losing = sell_orders[sell_orders['pnl'] < 0]

        labels = ['Profitable\nTrades', 'Losing\nTrades']
        sizes = [len(profitable), len(losing)]
        colors = ['#26de81', '#EE5A6F']
        explode = (0.05, 0.05)

        ax4.pie(sizes, explode=explode, labels=labels, colors=colors,
               autopct='%1.1f%%', shadow=True, startangle=90,
               textprops={'fontsize': 12, 'fontweight': 'bold'})

        ax4.set_title(f'Win/Loss Ratio\nWin Rate: {len(profitable)/len(sell_orders)*100:.1f}%',
                     fontsize=12, fontweight='bold')

    plt.tight_layout()
    output_path = output_dir / 'trade_analysis.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"[OK] Trade analysis saved: {output_path}")
    plt.close()


def generate_markdown_report(result: dict, output_dir: Path):
    """生成Markdown报告"""

    metrics = result['metrics']
    config = result['config']
    metadata = result['metadata']
    orders = result['backtest']['orders']

    # 统计交易信息
    df_orders = pd.DataFrame(orders)
    buy_orders = df_orders[df_orders['action'] == 'BUY']
    sell_orders = df_orders[df_orders['action'] == 'SELL']

    if len(sell_orders) > 0:
        profitable_trades = sell_orders[sell_orders['pnl'] > 0]
        avg_win = profitable_trades['pnl'].mean() if len(profitable_trades) > 0 else 0
        avg_loss = sell_orders[sell_orders['pnl'] < 0]['pnl'].mean() if len(sell_orders[sell_orders['pnl'] < 0]) > 0 else 0
        avg_holding_days = sell_orders['holding_days'].mean() if 'holding_days' in sell_orders.columns else 0
    else:
        avg_win = 0
        avg_loss = 0
        avg_holding_days = 0

    report_md = f"""# 完整回测报告

## 📊 执行概要

**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

**回测期间**: {metadata['start_date']} 至 {metadata['end_date']}

**初始资金**: ${metadata['initial_capital']:,.2f}

**实验文件**: {metadata['experiment_file']}

---

## 🎯 核心指标

### 收益指标

| 指标 | 数值 | 评价 |
|------|------|------|
| **总收益率** | **{metrics['total_return']*100:+.2f}%** | {'🟢 优秀' if metrics['total_return'] > 0.5 else '🟡 良好' if metrics['total_return'] > 0 else '🔴 亏损'} |
| **年化收益率** | **{metrics['annual_return']*100:+.2f}%** | {'🟢 优秀' if metrics['annual_return'] > 0.8 else '🟡 良好' if metrics['annual_return'] > 0 else '🔴 亏损'} |
| 最终资产 | ${result['backtest']['final_value']:,.2f} | - |
| 交易天数 | {metrics['n_days']} 天 | - |

### 风险指标

| 指标 | 数值 | 评价 |
|------|------|------|
| **夏普比率** | **{metrics['sharpe_ratio']:.3f}** | {'🟢 优秀' if metrics['sharpe_ratio'] > 1.5 else '🟡 良好' if metrics['sharpe_ratio'] > 1.0 else '🔴 较差'} |
| **最大回撤** | **{metrics['max_drawdown']*100:.2f}%** | {'🟢 优秀' if metrics['max_drawdown'] < 0.3 else '🟡 良好' if metrics['max_drawdown'] < 0.5 else '🔴 较大'} |
| **胜率** | **{metrics['win_rate']*100:.2f}%** | {'🟢 优秀' if metrics['win_rate'] > 0.6 else '🟡 良好' if metrics['win_rate'] > 0.5 else '🔴 较低'} |

---

## 📈 交易统计

### 交易概况

| 指标 | 数值 |
|------|------|
| 总交易次数 | {metrics['n_trades']} |
| 盈利交易 | {len(profitable_trades) if len(sell_orders) > 0 else 0} |
| 亏损交易 | {len(sell_orders[sell_orders['pnl'] < 0]) if len(sell_orders) > 0 else 0} |
| 平均持有天数 | {avg_holding_days:.1f} 天 |
| 平均盈利 | ${avg_win:,.2f} |
| 平均亏损 | ${avg_loss:,.2f} |
| 盈亏比 | {abs(avg_win/avg_loss) if avg_loss != 0 else 0:.2f} |

### 交易成本

| 成本项 | 总额 |
|--------|------|
| 手续费 | ${sum([o.get('commission', 0) for o in orders]):,.2f} |
| 滑点成本 | ${sum([o.get('slippage', 0) for o in orders]):,.2f} |
| 总成本 | ${sum([o.get('commission', 0) + o.get('slippage', 0) for o in orders]):,.2f} |

---

## 📊 可视化图表

### 综合性能分析

![综合报告](backtest_comprehensive.png)

图表包含：
1. **收益率曲线** - 累计收益随时间的变化
2. **资产组成变化** - 现金、持仓、总资产的变化
3. **持仓比例** - 仓位管理情况
4. **持仓数量** - 每日持仓股票数
5. **交易盈亏分布** - 单笔交易的盈亏分布
6. **回撤曲线** - 风险控制情况

### 交易细节分析

![交易分析](trade_analysis.png)

图表包含：
1. **每日交易活动** - 买入/卖出频率
2. **累计盈亏** - 盈亏累积变化
3. **持有天数分布** - 实际持有期统计
4. **胜负比** - 盈利/亏损交易占比

---

## 📝 策略配置

### 交易参数

- **每日最大持仓**: {config['top_n']} 只
- **概率阈值**: {config['prob_threshold']}
- **手续费率**: {config['commission']*100:.2f}%
- **滑点**: {config['slippage']*100:.2f}%

### 动态回撤止盈止损

- **回撤阈值**: {config.get('trailing_stop_pct', 0.05)*100:.1f}%
  - 从最高点回撤超过此比例时止盈
  - 从入场价下跌超过此比例时止损
- **最长持有**: {config.get('max_holding_days', 10)} 天
- **最短持有**: {config.get('min_holding_days', 1)} 天
- **低概率退出**: {'是' if config.get('exit_on_low_prob', False) else '否'}

---

## ⚠️ 风险提示

1. **回测局限性**: 基于历史数据，不代表未来表现
2. **交易成本**: 实际交易成本可能更高
3. **滑点风险**: 实际滑点可能大于设定值
4. **流动性风险**: 可能无法按预期价格成交

---

**报告生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""

    # 保存报告
    report_path = output_dir / 'backtest_report.md'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_md)

    print(f"[OK] Markdown report saved: {report_path}")


def main():
    parser = argparse.ArgumentParser(description='生成完整回测报告')
    parser.add_argument('--result_file', type=str, default=None,
                       help='回测结果文件路径（默认使用最新的）')
    parser.add_argument('--output_dir', type=str,
                       default='src/lstm/data/results/backtest/reports',
                       help='报告输出目录')

    args = parser.parse_args()

    print(f"\n{'='*70}")
    print(f"{'Generate Full Backtest Report':^66}")
    print(f"{'='*70}\n")

    # 确定结果文件
    if args.result_file:
        result_file = Path(args.result_file)
    else:
        result_dir = Path('src/lstm/data/results/backtest')
        result_files = sorted(result_dir.glob('backtest_*.json'),
                            key=lambda p: p.stat().st_mtime, reverse=True)
        if not result_files:
            print("[ERROR] No backtest result file found")
            return
        result_file = result_files[0]

    print(f"Backtest file: {result_file.name}")

    # 加载结果
    print("\n[1/4] Loading backtest result...")
    result = load_backtest_result(result_file)

    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 生成综合图表
    print("\n[2/4] Generating comprehensive charts...")
    generate_comprehensive_plots(result, output_dir)

    # 生成交易分析
    print("\n[3/4] Generating trade analysis...")
    generate_trade_analysis(result, output_dir)

    # 生成Markdown报告
    print("\n[4/4] Generating markdown report...")
    generate_markdown_report(result, output_dir)

    print(f"\n{'='*70}")
    print(f"{'Report Generated Successfully!':^66}")
    print(f"{'='*70}")
    print(f"\nReport location: {output_dir.absolute()}")
    print(f"  - backtest_comprehensive.png (Comprehensive analysis)")
    print(f"  - trade_analysis.png (Trade details)")
    print(f"  - backtest_report.md (Detailed report)")
    print()


if __name__ == '__main__':
    main()
