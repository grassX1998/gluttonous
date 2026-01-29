"""
统一回测报告生成器

功能：
1. 生成标准化的可视化报告（PNG）
2. 生成 Markdown 格式报告
3. 支持多种策略的通用接口
"""

import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


class BacktestReportGenerator:
    """回测报告生成器"""

    def __init__(self, output_dir: Path = None):
        """
        初始化报告生成器

        Args:
            output_dir: 输出目录，默认为 src/lstm/data/results/reports/
        """
        if output_dir is None:
            output_dir = Path(__file__).parent.parent.parent / "data" / "results" / "reports"
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_report(
        self,
        daily_data: List[Dict],
        results: Dict[str, Any],
        config: Dict[str, Any],
        strategy_name: str = "unknown",
        show: bool = False
    ) -> Dict[str, Path]:
        """
        生成完整回测报告

        Args:
            daily_data: 每日数据列表，每项包含:
                - date: 日期字符串 (YYYY-MM-DD)
                - cum_return: 累计收益率
                - cum_value: 累计净值
                - drawdown: 当前回撤
                - daily_return: 日收益率
                - n_positions: 持仓数量
            results: 汇总结果字典，包含:
                - total_return: 总收益率
                - annual_return: 年化收益率
                - sharpe_ratio: 夏普比率
                - max_drawdown: 最大回撤
                - daily_win_rate: 日胜率
                - trade_win_rate: 交易胜率
                - n_trades: 交易次数
            config: 策略配置字典，包含:
                - top_n: 持仓数量
                - prob_threshold: 概率阈值
                - holding_days: 持有天数
            strategy_name: 策略名称
            show: 是否显示图表

        Returns:
            生成的文件路径字典 {png: Path, md: Path}
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 生成PNG报告
        png_path = self._generate_png_report(
            daily_data, results, config, strategy_name, timestamp, show
        )

        # 生成Markdown报告
        md_path = self._generate_md_report(
            daily_data, results, config, strategy_name, timestamp
        )

        return {
            'png': png_path,
            'md': md_path
        }

    def _generate_png_report(
        self,
        daily_data: List[Dict],
        results: Dict[str, Any],
        config: Dict[str, Any],
        strategy_name: str,
        timestamp: str,
        show: bool
    ) -> Path:
        """生成PNG可视化报告"""
        if not daily_data:
            print("[ReportGenerator] ERROR: No daily_data provided!")
            return None

        # 提取数据
        dates = [d['date'] for d in daily_data]
        cum_returns = [d['cum_return'] * 100 for d in daily_data]
        cum_values = [d['cum_value'] for d in daily_data]
        drawdowns = [d['drawdown'] * 100 for d in daily_data]
        daily_returns = [d['daily_return'] * 100 for d in daily_data]
        n_positions = [d['n_positions'] for d in daily_data]

        # 转换日期
        date_objs = [datetime.strptime(d, '%Y-%m-%d') for d in dates]

        # 创建图表
        fig = plt.figure(figsize=(16, 14))
        fig.suptitle(f'{strategy_name} 回测报告', fontsize=16, fontweight='bold', y=0.98)

        # ===== 1. 累计收益曲线 =====
        ax1 = fig.add_subplot(3, 2, 1)
        ax1.plot(date_objs, cum_returns, 'b-', linewidth=1.5, label='累计收益')
        ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax1.fill_between(date_objs, 0, cum_returns, alpha=0.3)
        ax1.set_title('累计收益曲线', fontsize=14, fontweight='bold')
        ax1.set_ylabel('收益率 (%)')
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax1.xaxis.set_major_locator(mdates.MonthLocator())
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='upper left')

        total_return = results.get('total_return', 0) * 100
        ax1.annotate(f'总收益: {total_return:.2f}%',
                    xy=(0.02, 0.95), xycoords='axes fraction',
                    fontsize=11, fontweight='bold',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        # ===== 2. 净值曲线 =====
        ax2 = fig.add_subplot(3, 2, 2)
        ax2.plot(date_objs, cum_values, 'g-', linewidth=1.5, label='净值')
        ax2.axhline(y=1, color='gray', linestyle='--', alpha=0.5)
        ax2.set_title('净值曲线', fontsize=14, fontweight='bold')
        ax2.set_ylabel('净值')
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax2.xaxis.set_major_locator(mdates.MonthLocator())
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc='upper left')

        # ===== 3. 回撤曲线 =====
        ax3 = fig.add_subplot(3, 2, 3)
        ax3.fill_between(date_objs, 0, [-d for d in drawdowns], color='red', alpha=0.5, label='回撤')
        ax3.set_title('回撤曲线', fontsize=14, fontweight='bold')
        ax3.set_ylabel('回撤 (%)')
        ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax3.xaxis.set_major_locator(mdates.MonthLocator())
        plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45)
        ax3.grid(True, alpha=0.3)
        ax3.legend(loc='lower left')

        max_dd = results.get('max_drawdown', 0) * 100
        ax3.annotate(f'最大回撤: {max_dd:.2f}%',
                    xy=(0.02, 0.05), xycoords='axes fraction',
                    fontsize=11, fontweight='bold',
                    bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.5))

        # ===== 4. 每日仓位数量 =====
        ax4 = fig.add_subplot(3, 2, 4)
        ax4.bar(date_objs, n_positions, color='steelblue', alpha=0.7, label='持仓数')
        target_n = config.get('top_n', 10)
        ax4.axhline(y=target_n, color='red', linestyle='--', alpha=0.7,
                   label=f'目标持仓 ({target_n})')
        ax4.set_title('每日持仓数量', fontsize=14, fontweight='bold')
        ax4.set_ylabel('持仓数')
        ax4.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax4.xaxis.set_major_locator(mdates.MonthLocator())
        plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45)
        ax4.grid(True, alpha=0.3)
        ax4.legend(loc='upper right')

        # ===== 5. 日收益分布 =====
        ax5 = fig.add_subplot(3, 2, 5)
        ax5.hist(daily_returns, bins=30, color='steelblue', alpha=0.7, edgecolor='black')
        ax5.axvline(x=0, color='red', linestyle='--', alpha=0.7)
        mean_ret = np.mean(daily_returns)
        ax5.axvline(x=mean_ret, color='green', linestyle='-', alpha=0.7,
                   label=f'均值: {mean_ret:.3f}%')
        ax5.set_title('日收益分布', fontsize=14, fontweight='bold')
        ax5.set_xlabel('日收益率 (%)')
        ax5.set_ylabel('频次')
        ax5.grid(True, alpha=0.3)
        ax5.legend(loc='upper right')

        # ===== 6. 统计指标摘要 =====
        ax6 = fig.add_subplot(3, 2, 6)
        ax6.axis('off')

        stats_text = f"""
========== 回测统计 ==========

策略: {strategy_name}
测试期间: {dates[0]} ~ {dates[-1]}
交易天数: {len(dates)}

---------- 收益指标 ----------
总收益率:    {results.get('total_return', 0)*100:+.2f}%
年化收益率:  {results.get('annual_return', 0)*100:+.2f}%
夏普比率:    {results.get('sharpe_ratio', 0):.3f}

---------- 风险指标 ----------
最大回撤:    {results.get('max_drawdown', 0)*100:.2f}%
日胜率:      {results.get('daily_win_rate', 0)*100:.2f}%
交易胜率:    {results.get('trade_win_rate', 0)*100:.2f}%

---------- 交易统计 ----------
总交易次数:  {results.get('n_trades', 0)}
每日平均:    {results.get('n_trades', 0) / max(len(dates), 1):.1f} 笔

---------- 策略配置 ----------
每日持仓:    {config.get('top_n', 'N/A')} 只
概率阈值:    {config.get('prob_threshold', 'N/A')}
持有天数:    {config.get('holding_days', 'N/A')} 天
        """

        ax6.text(0.1, 0.95, stats_text, transform=ax6.transAxes,
                fontsize=10, verticalalignment='top',
                fontfamily='Microsoft YaHei',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.3))

        plt.tight_layout()

        # 保存图表
        output_file = self.output_dir / f"{strategy_name}_report_{timestamp}.png"
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"[ReportGenerator] PNG saved: {output_file}")

        if show:
            plt.show()
        else:
            plt.close()

        return output_file

    def _generate_md_report(
        self,
        daily_data: List[Dict],
        results: Dict[str, Any],
        config: Dict[str, Any],
        strategy_name: str,
        timestamp: str
    ) -> Path:
        """生成Markdown格式报告"""
        dates = [d['date'] for d in daily_data]

        md_content = f"""# {strategy_name} 回测报告

**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 概要

| 项目 | 值 |
|------|-----|
| 策略名称 | {strategy_name} |
| 测试期间 | {dates[0]} ~ {dates[-1]} |
| 交易天数 | {len(dates)} |

## 收益指标

| 指标 | 值 |
|------|-----|
| 总收益率 | {results.get('total_return', 0)*100:+.2f}% |
| 年化收益率 | {results.get('annual_return', 0)*100:+.2f}% |
| 夏普比率 | {results.get('sharpe_ratio', 0):.3f} |

## 风险指标

| 指标 | 值 |
|------|-----|
| 最大回撤 | {results.get('max_drawdown', 0)*100:.2f}% |
| 日胜率 | {results.get('daily_win_rate', 0)*100:.2f}% |
| 交易胜率 | {results.get('trade_win_rate', 0)*100:.2f}% |

## 交易统计

| 指标 | 值 |
|------|-----|
| 总交易次数 | {results.get('n_trades', 0)} |
| 每日平均交易 | {results.get('n_trades', 0) / max(len(dates), 1):.1f} 笔 |

## 策略配置

| 参数 | 值 |
|------|-----|
| 每日持仓数 | {config.get('top_n', 'N/A')} |
| 概率阈值 | {config.get('prob_threshold', 'N/A')} |
| 持有天数 | {config.get('holding_days', 'N/A')} |
| 手续费 | {config.get('commission', 0)*100:.2f}% |
| 滑点 | {config.get('slippage', 0)*100:.2f}% |

## 可视化报告

![回测报告]({strategy_name}_report_{timestamp}.png)

---
*报告由 BacktestReportGenerator 自动生成*
"""

        output_file = self.output_dir / f"{strategy_name}_report_{timestamp}.md"
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(md_content)

        print(f"[ReportGenerator] MD saved: {output_file}")

        return output_file


def generate_report_from_json(json_path: Path, output_dir: Path = None, show: bool = False):
    """
    从JSON结果文件生成报告

    Args:
        json_path: 回测结果JSON文件路径
        output_dir: 输出目录
        show: 是否显示图表
    """
    import json

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if 'daily_data' not in data or not data['daily_data']:
        print(f"ERROR: No daily_data in {json_path}")
        return None

    generator = BacktestReportGenerator(output_dir)

    return generator.generate_report(
        daily_data=data['daily_data'],
        results=data.get('results', {}),
        config=data.get('config', {}),
        strategy_name=data.get('strategy', 'unknown'),
        show=show
    )


def main():
    """命令行入口"""
    import argparse

    parser = argparse.ArgumentParser(description='生成回测报告')
    parser.add_argument('--result', type=str, required=True, help='回测结果JSON文件路径')
    parser.add_argument('--output', type=str, help='输出目录')
    parser.add_argument('--show', action='store_true', help='显示图表')

    args = parser.parse_args()

    result_path = Path(args.result)
    output_dir = Path(args.output) if args.output else None

    generate_report_from_json(result_path, output_dir, args.show)


if __name__ == "__main__":
    main()
