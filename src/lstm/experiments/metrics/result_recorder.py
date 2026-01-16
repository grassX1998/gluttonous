"""
结果记录器

负责：
1. 保存实验结果到JSON
2. 计算回测指标（收益率、夏普、回撤等）
3. 更新 CLAUDE.md 实验记录
4. 生成对比报告
"""

import sys
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime
import json

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.lstm.config import (
    EXPERIMENT_RESULT_DIR,
    DAILY_DATA_DIR
)


class ResultRecorder:
    """结果记录器"""

    def __init__(self, output_dir: Path = None):
        """
        初始化记录器

        Args:
            output_dir: 输出目录
        """
        self.output_dir = output_dir or (EXPERIMENT_RESULT_DIR / "experiments")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def save_experiment_result(self, strategy_name: str, result: Dict[str, Any],
                              timestamp: str = None) -> Path:
        """
        保存单个实验结果

        Args:
            strategy_name: 策略名称
            result: 实验结果
            timestamp: 时间戳

        Returns:
            保存的文件路径
        """
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        filename = f"{strategy_name}_{timestamp}.json"
        filepath = self.output_dir / filename

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        print(f"[ResultRecorder] Saved: {filepath}")

        return filepath

    def load_experiment_result(self, filepath: Path) -> Dict[str, Any]:
        """加载实验结果"""
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)

    def calculate_backtest_metrics(self, predictions: List[Dict],
                                   trading_params: Dict = None) -> Dict[str, Any]:
        """
        从预测结果计算回测指标

        Args:
            predictions: 预测列表 [{date, symbol, prob}, ...]
            trading_params: 交易参数 {top_n, prob_threshold, holding_days, ...}

        Returns:
            回测指标字典
        """
        if trading_params is None:
            trading_params = {
                'top_n': 10,
                'prob_threshold': 0.60,
                'holding_days': 5,
                'commission': 0.001,
                'slippage': 0.001
            }

        # 加载价格数据
        price_map = self._load_price_data()

        # 模拟交易
        trades = self._simulate_trades(predictions, price_map, trading_params)

        if not trades:
            return {
                'total_return': 0,
                'annual_return': 0,
                'sharpe_ratio': 0,
                'max_drawdown': 0,
                'win_rate': 0,
                'n_trades': 0
            }

        # 计算每日收益
        trades_df = pl.DataFrame(trades)
        daily_returns = trades_df.group_by("entry_date").agg([
            pl.col("net_return").mean().alias("return")
        ]).sort("entry_date")

        returns = daily_returns["return"].to_numpy()

        # 计算指标
        metrics = self._calculate_metrics(returns, len(trades))

        return metrics

    def _load_price_data(self) -> Dict:
        """加载价格数据"""
        price_map = {}

        if not DAILY_DATA_DIR.exists():
            print("[WARNING] Daily data directory not found")
            return price_map

        # 加载所有日线数据
        for parquet_file in DAILY_DATA_DIR.glob("*.parquet"):
            try:
                df = pl.read_parquet(parquet_file)
                for row in df.select(["date", "symbol", "close"]).iter_rows():
                    price_map[(row[0], row[1])] = row[2]
            except:
                pass

        return price_map

    def _simulate_trades(self, predictions: List[Dict], price_map: Dict,
                        params: Dict) -> List[Dict]:
        """
        模拟交易

        Args:
            predictions: 预测列表
            price_map: 价格映射 {(date, symbol): price}
            params: 交易参数

        Returns:
            交易列表
        """
        # 按日期分组
        pred_df = pl.DataFrame(predictions)

        if pred_df.is_empty():
            return []

        unique_dates = sorted(pred_df["date"].unique().to_list())

        # 构建日期索引
        date_to_idx = {d: i for i, d in enumerate(unique_dates)}

        trades = []

        for date in unique_dates:
            # 当日预测
            day_pred = pred_df.filter(pl.col("date") == date)

            # 选股
            candidates = day_pred.filter(
                pl.col("prob") >= params['prob_threshold']
            ).sort("prob", descending=True).head(params['top_n'])

            if candidates.is_empty():
                continue

            for row in candidates.iter_rows(named=True):
                symbol = row["symbol"]
                entry_date = row["date"]
                prob = row["prob"]

                # 买入价格
                entry_price = price_map.get((entry_date, symbol))
                if entry_price is None:
                    continue

                # 计算卖出日期
                entry_idx = date_to_idx.get(entry_date)
                if entry_idx is None:
                    continue

                exit_idx = entry_idx + params['holding_days']
                if exit_idx >= len(unique_dates):
                    continue

                exit_date = unique_dates[exit_idx]
                exit_price = price_map.get((exit_date, symbol))

                if exit_price is None:
                    continue

                # 计算收益
                gross_return = exit_price / entry_price - 1
                net_return = gross_return - (params['commission'] + params['slippage']) * 2

                trades.append({
                    "entry_date": entry_date,
                    "exit_date": exit_date,
                    "symbol": symbol,
                    "prob": prob,
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "gross_return": gross_return,
                    "net_return": net_return
                })

        return trades

    def _calculate_metrics(self, returns: np.ndarray, n_trades: int) -> Dict[str, float]:
        """计算回测指标"""
        if len(returns) == 0:
            return {
                'total_return': 0,
                'annual_return': 0,
                'sharpe_ratio': 0,
                'max_drawdown': 0,
                'win_rate': 0,
                'n_trades': 0
            }

        # 累计收益
        cum_returns = np.cumprod(1 + returns) - 1
        total_return = cum_returns[-1]

        # 年化收益
        n_days = len(returns)
        annual_return = (1 + total_return) ** (250 / n_days) - 1 if n_days > 0 else 0

        # 夏普比率
        daily_std = np.std(returns)
        sharpe = np.sqrt(250) * np.mean(returns) / daily_std if daily_std > 0 else 0

        # 最大回撤
        cum_values = np.cumprod(1 + returns)
        peak = np.maximum.accumulate(cum_values)
        drawdown = (peak - cum_values) / peak
        max_drawdown = np.max(drawdown)

        # 胜率
        win_rate = np.mean(returns > 0)

        return {
            'total_return': float(total_return),
            'annual_return': float(annual_return),
            'sharpe_ratio': float(sharpe),
            'max_drawdown': float(max_drawdown),
            'win_rate': float(win_rate),
            'n_trades': n_trades,
            'n_days': n_days
        }

    def compare_strategies(self, results: Dict[str, Dict]) -> Dict[str, Any]:
        """
        对比多个策略的结果

        Args:
            results: {strategy_name: result_dict}

        Returns:
            对比统计
        """
        comparison = {
            'timestamp': datetime.now().isoformat(),
            'strategies': {}
        }

        for strategy_name, result in results.items():
            # 如果结果中已有metrics，直接使用
            if 'metrics' in result:
                metrics = result['metrics']
            else:
                # 否则从预测结果计算
                metrics = self.calculate_backtest_metrics(result.get('predictions', []))

            comparison['strategies'][strategy_name] = {
                'metrics': metrics,
                'n_predictions': len(result.get('predictions', [])),
                'n_retrains': len(result.get('retrain_dates', [])),
            }

        return comparison

    def generate_markdown_table(self, comparison: Dict[str, Any]) -> str:
        """
        生成 Markdown 格式的对比表格

        Args:
            comparison: 对比结果

        Returns:
            Markdown 字符串
        """
        lines = []

        lines.append("## 实验结果对比")
        lines.append("")
        lines.append(f"**更新时间**: {comparison['timestamp']}")
        lines.append("")
        lines.append("| 策略 | 总收益率 | 年化收益 | Sharpe | 最大回撤 | 胜率 | 交易次数 |")
        lines.append("|------|---------|---------|--------|---------|------|---------|")

        for strategy_name, data in comparison['strategies'].items():
            metrics = data['metrics']
            line = f"| {strategy_name} | "
            line += f"{metrics['total_return']*100:+.2f}% | "
            line += f"{metrics['annual_return']*100:+.2f}% | "
            line += f"{metrics['sharpe_ratio']:.3f} | "
            line += f"{metrics['max_drawdown']*100:.2f}% | "
            line += f"{metrics['win_rate']*100:.2f}% | "
            line += f"{metrics['n_trades']} |"
            lines.append(line)

        lines.append("")

        return "\n".join(lines)

    def update_claude_md(self, comparison: Dict[str, Any]):
        """
        更新 CLAUDE.md 中的实验结果

        Args:
            comparison: 对比结果
        """
        claude_md_path = Path("CLAUDE.md")

        if not claude_md_path.exists():
            print("[WARNING] CLAUDE.md not found")
            return

        # 读取现有内容
        with open(claude_md_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # 生成新的结果表格
        table_md = self.generate_markdown_table(comparison)

        # 查找或创建实验结果章节
        marker = "## 实验结果对比"

        if marker in content:
            # 替换现有章节
            parts = content.split(marker)
            # 找到下一个 ## 章节
            next_section_idx = parts[1].find('\n## ')
            if next_section_idx != -1:
                new_content = parts[0] + table_md + parts[1][next_section_idx:]
            else:
                new_content = parts[0] + table_md
        else:
            # 追加新章节
            new_content = content + '\n\n' + table_md

        # 写回文件
        with open(claude_md_path, 'w', encoding='utf-8') as f:
            f.write(new_content)

        print(f"[ResultRecorder] Updated: {claude_md_path}")


def main():
    """示例用法"""
    recorder = ResultRecorder()

    # 示例：生成对比表格
    dummy_comparison = {
        'timestamp': datetime.now().isoformat(),
        'strategies': {
            'expanding_window': {
                'metrics': {
                    'total_return': 0.75,
                    'annual_return': 0.85,
                    'sharpe_ratio': 1.82,
                    'max_drawdown': 0.38,
                    'win_rate': 0.58,
                    'n_trades': 1520
                }
            }
        }
    }

    table = recorder.generate_markdown_table(dummy_comparison)
    print(table)


if __name__ == "__main__":
    main()
