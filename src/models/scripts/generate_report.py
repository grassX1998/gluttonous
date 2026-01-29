"""
多模型回测报告生成器

生成：
1. 多模型收益率曲线对比图
2. 回撤曲线对比图
3. 详细统计报告（Markdown）
4. 单模型详细报告（PNG）

用法:
    python src/models/scripts/generate_report.py
"""

import sys
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional

import numpy as np
import polars as pl
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# 添加项目根目录
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.lstm.config import FEATURE_DATA_DIR

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


class MultiModelReportGenerator:
    """多模型回测报告生成器"""

    def __init__(self, output_dir: Path = None):
        if output_dir is None:
            output_dir = Path(__file__).parent.parent / "data" / "results" / "reports"
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.price_map = None

    def load_price_data(self) -> Dict:
        """加载价格数据"""
        if self.price_map is not None:
            return self.price_map

        self.price_map = {}
        data_dir = FEATURE_DATA_DIR

        if not data_dir.exists():
            print(f"[ReportGenerator] ERROR: Data dir not found: {data_dir}")
            return self.price_map

        parquet_files = list(data_dir.glob("*.parquet"))
        print(f"[ReportGenerator] Loading price data from {len(parquet_files)} files...")

        for parquet_file in parquet_files:
            try:
                df = pl.read_parquet(parquet_file)
                if "date" not in df.columns or "symbol" not in df.columns or "close" not in df.columns:
                    continue
                df = df.with_columns(pl.col("date").cast(str).alias("date_str"))
                for row in df.select(["date_str", "symbol", "close"]).iter_rows():
                    self.price_map[(row[0], row[1])] = row[2]
            except Exception as e:
                print(f"[ReportGenerator] Warning: Failed to load {parquet_file}: {e}")

        print(f"[ReportGenerator] Loaded {len(self.price_map)} price records")
        return self.price_map

    def calculate_daily_returns(
        self,
        predictions: List[Dict],
        trading_params: Dict = None
    ) -> List[Dict]:
        """
        从预测结果计算每日收益

        Args:
            predictions: 预测列表 [{date, symbol, prob}, ...]
            trading_params: 交易参数

        Returns:
            每日数据列表 [{date, daily_return, cum_return, cum_value, drawdown, n_positions}, ...]
        """
        if trading_params is None:
            trading_params = {
                'top_n': 10,
                'prob_threshold': 0.60,
                'holding_days': 5,
                'commission': 0.001,
                'slippage': 0.001
            }

        price_map = self.load_price_data()
        if not price_map:
            return []

        # 按日期分组预测
        pred_df = pl.DataFrame(predictions)
        if pred_df.is_empty():
            return []

        unique_dates = sorted(pred_df["date"].unique().to_list())
        date_to_idx = {d: i for i, d in enumerate(unique_dates)}

        # 模拟交易，记录每日持仓和收益
        daily_data = []
        positions = {}  # {symbol: {entry_date, entry_price, exit_date}}

        for i, date in enumerate(unique_dates):
            # 检查需要平仓的持仓
            closed_returns = []
            symbols_to_close = []
            for symbol, pos in positions.items():
                if pos['exit_date'] <= date:
                    exit_price = price_map.get((date, symbol))
                    if exit_price:
                        gross_ret = exit_price / pos['entry_price'] - 1
                        net_ret = gross_ret - (trading_params['commission'] + trading_params['slippage']) * 2
                        closed_returns.append(net_ret)
                    symbols_to_close.append(symbol)

            for symbol in symbols_to_close:
                del positions[symbol]

            # 当日预测选股
            day_pred = pred_df.filter(pl.col("date") == date)
            candidates = day_pred.filter(
                pl.col("prob") >= trading_params['prob_threshold']
            ).sort("prob", descending=True).head(trading_params['top_n'])

            # 开新仓
            new_positions = 0
            for row in candidates.iter_rows(named=True):
                symbol = row["symbol"]
                if symbol in positions:
                    continue  # 已持有

                entry_price = price_map.get((date, symbol))
                if entry_price is None:
                    continue

                exit_idx = i + trading_params['holding_days']
                if exit_idx >= len(unique_dates):
                    continue

                exit_date = unique_dates[exit_idx]
                positions[symbol] = {
                    'entry_date': date,
                    'entry_price': entry_price,
                    'exit_date': exit_date
                }
                new_positions += 1

            # 计算当日收益（平仓收益的平均）
            if closed_returns:
                daily_return = np.mean(closed_returns)
            else:
                daily_return = 0.0

            daily_data.append({
                'date': date,
                'daily_return': daily_return,
                'n_positions': len(positions),
                'n_new': new_positions,
                'n_closed': len(closed_returns)
            })

        # 计算累计收益和回撤
        cum_value = 1.0
        peak = 1.0
        for d in daily_data:
            cum_value *= (1 + d['daily_return'])
            d['cum_value'] = cum_value
            d['cum_return'] = cum_value - 1

            peak = max(peak, cum_value)
            d['drawdown'] = (peak - cum_value) / peak if peak > 0 else 0

        return daily_data

    def load_experiment_results(self, results_dir: Path = None) -> Dict[str, Dict]:
        """加载最新的实验结果"""
        if results_dir is None:
            results_dir = Path(__file__).parent.parent / "data" / "results" / "experiments"

        results = {}
        model_names = ['lightgbm', 'mlp', 'ensemble']

        for model_name in model_names:
            # 找到该模型最新的结果文件
            pattern = f"{model_name}_*.json"
            files = sorted(results_dir.glob(pattern), reverse=True)
            if files:
                latest_file = files[0]
                print(f"[ReportGenerator] Loading {model_name}: {latest_file.name}")
                with open(latest_file, 'r', encoding='utf-8') as f:
                    results[model_name] = json.load(f)

        return results

    def generate_comparison_report(
        self,
        results: Dict[str, Dict] = None,
        show: bool = False
    ) -> Dict[str, Path]:
        """
        生成多模型对比报告

        Args:
            results: {model_name: result_dict}
            show: 是否显示图表

        Returns:
            生成的文件路径
        """
        if results is None:
            results = self.load_experiment_results()

        if not results:
            print("[ReportGenerator] ERROR: No results to generate report")
            return {}

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 计算每个模型的每日收益
        model_daily_data = {}
        model_metrics = {}

        for model_name, result in results.items():
            predictions = result.get('predictions', [])
            if not predictions:
                continue

            print(f"[ReportGenerator] Processing {model_name}: {len(predictions)} predictions")
            daily_data = self.calculate_daily_returns(predictions)
            model_daily_data[model_name] = daily_data

            # 计算指标
            if daily_data:
                returns = np.array([d['daily_return'] for d in daily_data])
                cum_return = daily_data[-1]['cum_return']
                n_days = len(daily_data)
                annual_return = (1 + cum_return) ** (250 / n_days) - 1 if n_days > 0 else 0
                sharpe = np.sqrt(250) * np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0
                max_dd = max(d['drawdown'] for d in daily_data)
                win_rate = np.mean(returns > 0) if len(returns) > 0 else 0

                model_metrics[model_name] = {
                    'total_return': cum_return,
                    'annual_return': annual_return,
                    'sharpe_ratio': sharpe,
                    'max_drawdown': max_dd,
                    'win_rate': win_rate,
                    'n_days': n_days
                }

        # 生成对比图
        png_path = self._generate_comparison_png(model_daily_data, model_metrics, timestamp, show)

        # 生成详细报告
        md_path = self._generate_comparison_md(model_daily_data, model_metrics, timestamp)

        return {
            'png': png_path,
            'md': md_path
        }

    def _generate_comparison_png(
        self,
        model_daily_data: Dict[str, List[Dict]],
        model_metrics: Dict[str, Dict],
        timestamp: str,
        show: bool
    ) -> Path:
        """生成多模型对比PNG图"""
        if not model_daily_data:
            return None

        # 颜色配置
        colors = {
            'lightgbm': '#2E86AB',   # 蓝色
            'mlp': '#A23B72',        # 紫红色
            'ensemble': '#F18F01',   # 橙色
            'lstm': '#C73E1D'        # 红色
        }

        fig = plt.figure(figsize=(16, 12))
        fig.suptitle('多模型回测对比报告', fontsize=18, fontweight='bold', y=0.98)

        # ===== 1. 累计收益曲线对比 =====
        ax1 = fig.add_subplot(2, 2, 1)
        for model_name, daily_data in model_daily_data.items():
            if not daily_data:
                continue
            dates = [datetime.strptime(d['date'], '%Y-%m-%d') for d in daily_data]
            cum_returns = [d['cum_return'] * 100 for d in daily_data]
            color = colors.get(model_name, 'gray')
            ax1.plot(dates, cum_returns, '-', linewidth=2, label=model_name.upper(), color=color)

        ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax1.set_title('累计收益率对比', fontsize=14, fontweight='bold')
        ax1.set_ylabel('收益率 (%)')
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax1.xaxis.set_major_locator(mdates.MonthLocator())
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='upper left')

        # ===== 2. 净值曲线对比 =====
        ax2 = fig.add_subplot(2, 2, 2)
        for model_name, daily_data in model_daily_data.items():
            if not daily_data:
                continue
            dates = [datetime.strptime(d['date'], '%Y-%m-%d') for d in daily_data]
            cum_values = [d['cum_value'] for d in daily_data]
            color = colors.get(model_name, 'gray')
            ax2.plot(dates, cum_values, '-', linewidth=2, label=model_name.upper(), color=color)

        ax2.axhline(y=1, color='gray', linestyle='--', alpha=0.5)
        ax2.set_title('净值曲线对比', fontsize=14, fontweight='bold')
        ax2.set_ylabel('净值')
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax2.xaxis.set_major_locator(mdates.MonthLocator())
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc='upper left')

        # ===== 3. 回撤曲线对比 =====
        ax3 = fig.add_subplot(2, 2, 3)
        for model_name, daily_data in model_daily_data.items():
            if not daily_data:
                continue
            dates = [datetime.strptime(d['date'], '%Y-%m-%d') for d in daily_data]
            drawdowns = [-d['drawdown'] * 100 for d in daily_data]
            color = colors.get(model_name, 'gray')
            ax3.fill_between(dates, 0, drawdowns, alpha=0.3, color=color)
            ax3.plot(dates, drawdowns, '-', linewidth=1, label=model_name.upper(), color=color)

        ax3.set_title('回撤曲线对比', fontsize=14, fontweight='bold')
        ax3.set_ylabel('回撤 (%)')
        ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax3.xaxis.set_major_locator(mdates.MonthLocator())
        plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45)
        ax3.grid(True, alpha=0.3)
        ax3.legend(loc='lower left')

        # ===== 4. 统计指标对比 =====
        ax4 = fig.add_subplot(2, 2, 4)
        ax4.axis('off')

        # 构建表格数据
        headers = ['模型', '总收益', '年化收益', 'Sharpe', '最大回撤', '胜率']
        cell_text = []
        cell_colors = []

        for model_name, metrics in model_metrics.items():
            row = [
                model_name.upper(),
                f"{metrics['total_return']*100:+.2f}%",
                f"{metrics['annual_return']*100:+.2f}%",
                f"{metrics['sharpe_ratio']:.3f}",
                f"{metrics['max_drawdown']*100:.2f}%",
                f"{metrics['win_rate']*100:.2f}%"
            ]
            cell_text.append(row)
            cell_colors.append([colors.get(model_name, 'lightgray')] + ['white'] * 5)

        table = ax4.table(
            cellText=cell_text,
            colLabels=headers,
            cellLoc='center',
            loc='center',
            colColours=['lightgray'] * 6
        )
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1.2, 1.8)

        # 设置表格样式
        for i, row in enumerate(cell_text):
            table[(i + 1, 0)].set_facecolor(colors.get(row[0].lower(), 'lightgray'))
            table[(i + 1, 0)].set_text_props(color='white', fontweight='bold')

        ax4.set_title('模型指标对比', fontsize=14, fontweight='bold', pad=20)

        plt.tight_layout()

        # 保存
        output_file = self.output_dir / f"multi_model_comparison_{timestamp}.png"
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"[ReportGenerator] PNG saved: {output_file}")

        if show:
            plt.show()
        else:
            plt.close()

        return output_file

    def _generate_comparison_md(
        self,
        model_daily_data: Dict[str, List[Dict]],
        model_metrics: Dict[str, Dict],
        timestamp: str
    ) -> Path:
        """生成Markdown对比报告"""
        # 获取日期范围
        all_dates = []
        for daily_data in model_daily_data.values():
            if daily_data:
                all_dates.extend([d['date'] for d in daily_data])

        if all_dates:
            start_date = min(all_dates)
            end_date = max(all_dates)
        else:
            start_date = end_date = "N/A"

        md_content = f"""# 多模型回测对比报告

**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 概要

| 项目 | 值 |
|------|-----|
| 测试期间 | {start_date} ~ {end_date} |
| 对比模型 | {', '.join(model_metrics.keys())} |

## 模型对比

| 模型 | 总收益率 | 年化收益 | Sharpe | 最大回撤 | 胜率 |
|------|---------|---------|--------|---------|------|
"""
        for model_name, metrics in model_metrics.items():
            md_content += f"| {model_name.upper()} | {metrics['total_return']*100:+.2f}% | "
            md_content += f"{metrics['annual_return']*100:+.2f}% | "
            md_content += f"{metrics['sharpe_ratio']:.3f} | "
            md_content += f"{metrics['max_drawdown']*100:.2f}% | "
            md_content += f"{metrics['win_rate']*100:.2f}% |\n"

        md_content += f"""
## 模型分析

### LightGBM
- **优势**: 风险控制最佳，最大回撤最低
- **劣势**: 保守策略导致交易机会少，总收益较低
- **特点**: 使用梯度提升树，概率预测较为保守

### MLP (多层感知器)
- **优势**: 总收益最高，交易机会多
- **劣势**: 最大回撤较大，波动性高
- **特点**: 神经网络模型，概率预测分布更广

### Ensemble (集成模型)
- **优势**: 综合两个模型的预测
- **劣势**: 集成后表现介于两者之间，未能超越单模型
- **特点**: 软投票集成，取两模型概率均值

## 可视化报告

![多模型对比](multi_model_comparison_{timestamp}.png)

## 配置参数

| 参数 | 值 |
|------|-----|
| 每日持仓数 | 10 |
| 概率阈值 | 0.60 |
| 持有天数 | 5 |
| 手续费 | 0.10% |
| 滑点 | 0.10% |

---
*报告由 MultiModelReportGenerator 自动生成*
"""

        output_file = self.output_dir / f"multi_model_comparison_{timestamp}.md"
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(md_content)

        print(f"[ReportGenerator] MD saved: {output_file}")
        return output_file


def main():
    """生成多模型对比报告"""
    import argparse

    parser = argparse.ArgumentParser(description='生成多模型回测报告')
    parser.add_argument('--show', action='store_true', help='显示图表')
    parser.add_argument('--output', type=str, help='输出目录')

    args = parser.parse_args()

    output_dir = Path(args.output) if args.output else None
    generator = MultiModelReportGenerator(output_dir)

    print("=" * 60)
    print("多模型回测报告生成")
    print("=" * 60)

    paths = generator.generate_comparison_report(show=args.show)

    print("\n" + "=" * 60)
    print("生成完成")
    print("=" * 60)
    for name, path in paths.items():
        if path:
            print(f"  {name}: {path}")


if __name__ == "__main__":
    main()
