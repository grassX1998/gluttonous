"""
运行实验脚本

用于运行多个训练策略实验并生成对比报告。

用法示例：
    # 运行所有策略
    python scripts/run_experiments.py --start_date 2025-04-01 --end_date 2026-01-15

    # 运行特定策略
    python scripts/run_experiments.py --strategies expanding_window --start_date 2025-04-01 --end_date 2026-01-15

    # 生成回测指标并更新CLAUDE.md
    python scripts/run_experiments.py --strategies expanding_window --start_date 2025-04-01 --end_date 2026-01-15 --calculate_metrics --update_claude_md
"""

import sys
from pathlib import Path
import argparse
from datetime import datetime

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.lstm.experiments.experiment_manager import ExperimentManager
from src.lstm.experiments.metrics.result_recorder import ResultRecorder
from src.lstm.config import ALL_STRATEGY_CONFIGS


def main():
    parser = argparse.ArgumentParser(
        description="运行多策略实验",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 运行扩展窗口策略
  python scripts/run_experiments.py --strategies expanding_window

  # 运行所有可用策略
  python scripts/run_experiments.py

  # 指定日期范围
  python scripts/run_experiments.py --start_date 2025-04-01 --end_date 2026-01-15

  # 计算回测指标并更新CLAUDE.md
  python scripts/run_experiments.py --strategies expanding_window --calculate_metrics --update_claude_md
        """
    )

    parser.add_argument("--strategies", nargs='+', default=None,
                       help=f"要运行的策略列表。可选: {', '.join(ALL_STRATEGY_CONFIGS.keys())}")
    parser.add_argument("--start_date", type=str, default="2025-04-01",
                       help="开始日期 (YYYY-MM-DD)")
    parser.add_argument("--end_date", type=str, default="2026-01-15",
                       help="结束日期 (YYYY-MM-DD)")
    parser.add_argument("--calculate_metrics", action="store_true",
                       help="计算回测指标（收益率、夏普比率等）")
    parser.add_argument("--update_claude_md", action="store_true",
                       help="更新 CLAUDE.md 中的实验结果")
    parser.add_argument("--trading_params", type=str, default=None,
                       help="交易参数 JSON 字符串，如 '{\"top_n\": 10, \"prob_threshold\": 0.60}'")

    args = parser.parse_args()

    print("\n" + "=" * 70)
    print(" " * 20 + "多策略实验运行器")
    print("=" * 70)
    print(f"日期范围: {args.start_date} 至 {args.end_date}")
    print(f"策略: {args.strategies or '全部'}")
    print(f"计算指标: {'是' if args.calculate_metrics else '否'}")
    print(f"更新 CLAUDE.md: {'是' if args.update_claude_md else '否'}")
    print("=" * 70 + "\n")

    # 创建实验管理器
    manager = ExperimentManager(strategies=args.strategies)

    # 运行所有实验
    manager.run_all_experiments(args.start_date, args.end_date)

    # 打印摘要
    manager.print_summary()

    # 计算回测指标
    if args.calculate_metrics:
        print("\n" + "=" * 70)
        print("计算回测指标...")
        print("=" * 70 + "\n")

        # 解析交易参数
        trading_params = None
        if args.trading_params:
            import json
            trading_params = json.loads(args.trading_params)

        recorder = ResultRecorder()

        # 为每个策略计算指标
        for strategy_name, result in manager.results.items():
            print(f"\n策略: {strategy_name}")

            metrics = recorder.calculate_backtest_metrics(
                result['predictions'],
                trading_params
            )

            result['metrics'] = metrics

            print(f"  总收益率: {metrics['total_return']*100:+.2f}%")
            print(f"  年化收益: {metrics['annual_return']*100:+.2f}%")
            print(f"  夏普比率: {metrics['sharpe_ratio']:.3f}")
            print(f"  最大回撤: {metrics['max_drawdown']*100:.2f}%")
            print(f"  胜率: {metrics['win_rate']*100:.2f}%")
            print(f"  交易次数: {metrics['n_trades']}")

        # 生成对比报告
        print("\n" + "=" * 70)
        print("生成对比报告...")
        print("=" * 70)

        comparison = recorder.compare_strategies(manager.results)

        # 打印对比表格
        table = recorder.generate_markdown_table(comparison)
        print("\n" + table)

        # 保存对比结果
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        comparison_file = recorder.output_dir / f"comparison_{timestamp}.json"

        import json
        with open(comparison_file, 'w', encoding='utf-8') as f:
            json.dump(comparison, f, indent=2, ensure_ascii=False)

        print(f"\n对比结果已保存: {comparison_file}")

        # 更新 CLAUDE.md
        if args.update_claude_md:
            print("\n更新 CLAUDE.md...")
            recorder.update_claude_md(comparison)
            print("CLAUDE.md 已更新")

    print("\n" + "=" * 70)
    print(" " * 25 + "实验完成!")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
