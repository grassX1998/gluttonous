"""
运行完整回测

记录所有交易订单、持仓信息，生成详细报告
"""

import sys
from pathlib import Path
import json
import argparse
from datetime import datetime

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.lstm.experiments.metrics.backtest_engine import BacktestEngine
from src.lstm.config import TRADING_CONFIG


def load_predictions(result_file: Path) -> list:
    """加载预测结果"""
    with open(result_file, 'r', encoding='utf-8') as f:
        result = json.load(f)
    return result['predictions']


def save_backtest_result(result: dict, output_path: Path):
    """保存回测结果"""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False, default=str)

    print(f"\n[OK] Backtest result saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='运行完整回测')
    parser.add_argument('--result_file', type=str, default=None,
                       help='实验结果文件路径（默认使用最新的）')
    parser.add_argument('--initial_capital', type=float, default=1000000.0,
                       help='初始资金（默认100万）')
    parser.add_argument('--output_dir', type=str,
                       default='src/lstm/data/results/backtest',
                       help='回测结果输出目录')

    args = parser.parse_args()

    print(f"\n{'='*70}")
    print(f"{'Full Backtest Engine':^66}")
    print(f"{'='*70}\n")

    # 确定结果文件
    if args.result_file:
        result_file = Path(args.result_file)
    else:
        result_dir = Path('src/lstm/data/results/experiments')
        result_files = sorted(result_dir.glob('expanding_window_*.json'),
                            key=lambda p: p.stat().st_mtime, reverse=True)
        if not result_files:
            print("[ERROR] No experiment result file found")
            return
        result_file = result_files[0]

    print(f"Experiment file: {result_file.name}")
    print(f"Initial capital: ${args.initial_capital:,.2f}")

    # 加载预测结果
    print("\n[1/4] Loading predictions...")
    with open(result_file, 'r', encoding='utf-8') as f:
        experiment_result = json.load(f)

    predictions = experiment_result['predictions']
    start_date = experiment_result['start_date']
    end_date = experiment_result['end_date']

    print(f"  Predictions: {len(predictions):,}")
    print(f"  Period: {start_date} to {end_date}")

    # 初始化回测引擎
    print("\n[2/4] Initializing backtest engine...")
    engine = BacktestEngine(
        initial_capital=args.initial_capital,
        top_n=TRADING_CONFIG['top_n'],
        prob_threshold=TRADING_CONFIG['prob_threshold'],
        commission=TRADING_CONFIG['commission'],
        slippage=TRADING_CONFIG['slippage'],
        trailing_stop_pct=TRADING_CONFIG['trailing_stop_pct'],
        max_holding_days=TRADING_CONFIG['max_holding_days'],
        min_holding_days=TRADING_CONFIG['min_holding_days'],
        exit_on_low_prob=TRADING_CONFIG['exit_on_low_prob']
    )

    print(f"  Top N: {TRADING_CONFIG['top_n']}")
    print(f"  Prob threshold: {TRADING_CONFIG['prob_threshold']}")
    print(f"  Trailing stop: {TRADING_CONFIG['trailing_stop_pct']*100:.1f}%")
    print(f"  Max holding: {TRADING_CONFIG['max_holding_days']} days")

    # 加载价格数据
    print("\n[3/4] Loading price data...")
    price_data = engine.load_price_data(Path('src/lstm/data'), start_date, end_date)

    if price_data is None:
        print("[ERROR] Failed to load price data")
        return

    print(f"  Price data loaded: {price_data.height:,} rows")

    # 获取交易日列表
    trading_dates = sorted(price_data['date'].cast(str).unique().to_list())
    print(f"  Trading dates: {len(trading_dates)}")

    # 运行回测
    print("\n[4/4] Running backtest...")
    backtest_result = engine.run_backtest(predictions, price_data, trading_dates)

    # 计算指标
    metrics = engine.calculate_metrics()

    # 汇总结果
    full_result = {
        'metadata': {
            'experiment_file': str(result_file.name),
            'backtest_time': datetime.now().isoformat(),
            'start_date': start_date,
            'end_date': end_date,
            'initial_capital': args.initial_capital
        },
        'config': {
            'top_n': TRADING_CONFIG['top_n'],
            'prob_threshold': TRADING_CONFIG['prob_threshold'],
            'commission': TRADING_CONFIG['commission'],
            'slippage': TRADING_CONFIG['slippage'],
            'trailing_stop_pct': TRADING_CONFIG['trailing_stop_pct'],
            'max_holding_days': TRADING_CONFIG['max_holding_days'],
            'min_holding_days': TRADING_CONFIG['min_holding_days'],
            'exit_on_low_prob': TRADING_CONFIG['exit_on_low_prob']
        },
        'metrics': metrics,
        'backtest': backtest_result
    }

    # 保存结果
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_path = Path(args.output_dir) / f'backtest_{timestamp}.json'
    save_backtest_result(full_result, output_path)

    # 打印摘要
    print(f"\n{'='*70}")
    print(f"{'Backtest Summary':^66}")
    print(f"{'='*70}\n")
    print(f"Total Return:    {metrics['total_return']*100:+.2f}%")
    print(f"Annual Return:   {metrics['annual_return']*100:+.2f}%")
    print(f"Sharpe Ratio:    {metrics['sharpe_ratio']:.3f}")
    print(f"Max Drawdown:    {metrics['max_drawdown']*100:.2f}%")
    print(f"Win Rate:        {metrics['win_rate']*100:.2f}%")
    print(f"Total Trades:    {metrics['n_trades']}")
    print(f"Trading Days:    {metrics['n_days']}")
    print(f"\nFinal Value:     ${backtest_result['final_value']:,.2f}")
    print(f"Total Orders:    {len(backtest_result['orders'])}")
    print(f"\n{'='*70}\n")


if __name__ == '__main__':
    main()
