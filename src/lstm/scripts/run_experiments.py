"""
运行实验脚本

用于运行多个训练策略实验并生成对比报告。

用法示例：
    # 运行所有策略
    python src/lstm/scripts/run_experiments.py --start_date 2025-04-01 --end_date 2026-01-15

    # 运行特定策略
    python src/lstm/scripts/run_experiments.py --strategies expanding_window --start_date 2025-04-01 --end_date 2026-01-15

    # 生成回测指标并更新CLAUDE.md
    python src/lstm/scripts/run_experiments.py --strategies expanding_window --calculate_metrics --update_claude_md
"""

import sys
from pathlib import Path
import argparse
from datetime import datetime
import json

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.lstm.experiments.experiment_manager import ExperimentManager
from src.lstm.experiments.metrics.result_recorder import ResultRecorder
from src.lstm.experiments.registry import get_registry
from src.lstm.config import ALL_STRATEGY_CONFIGS
from pipeline.shared.logging_config import get_lstm_training_logger


def main():
    parser = argparse.ArgumentParser(
        description="运行多策略实验",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 运行扩展窗口策略
  python src/lstm/scripts/run_experiments.py --strategies expanding_window

  # 运行所有可用策略
  python src/lstm/scripts/run_experiments.py

  # 指定日期范围
  python src/lstm/scripts/run_experiments.py --start_date 2025-04-01 --end_date 2026-01-15

  # 计算回测指标并更新CLAUDE.md
  python src/lstm/scripts/run_experiments.py --strategies expanding_window --calculate_metrics --update_claude_md
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

    # 获取日志记录器
    logger = get_lstm_training_logger()

    logger.info("=" * 70)
    logger.info(" " * 20 + "多策略实验运行器")
    logger.info("=" * 70)
    logger.info(f"日期范围: {args.start_date} 至 {args.end_date}")
    logger.info(f"策略: {args.strategies or '全部'}")
    logger.info(f"计算指标: {'是' if args.calculate_metrics else '否'}")
    logger.info(f"更新 CLAUDE.md: {'是' if args.update_claude_md else '否'}")
    logger.info("=" * 70)

    # 获取注册表
    registry = get_registry()

    # 确定要运行的策略
    strategies = args.strategies or list(ALL_STRATEGY_CONFIGS.keys())

    # 为每个策略创建实验记录
    experiment_ids = {}
    for strategy_name in strategies:
        # 获取策略配置
        config_class = ALL_STRATEGY_CONFIGS.get(strategy_name)
        if config_class is None:
            logger.warning(f"Unknown strategy: {strategy_name}, skipping")
            continue

        config = config_class()
        config_snapshot = {
            'strategy_name': config.strategy_name,
            'min_train_days': getattr(config, 'min_train_days', None),
            'max_train_days': getattr(config, 'max_train_days', None),
            'val_days': getattr(config, 'val_days', None),
            'retrain_interval': getattr(config, 'retrain_interval', None),
        }

        # 创建实验记录
        experiment_id = registry.create_experiment(
            strategy=strategy_name,
            start_date=args.start_date,
            end_date=args.end_date,
            config_snapshot=config_snapshot
        )
        experiment_ids[strategy_name] = experiment_id
        logger.info(f"Created experiment: {experiment_id} for {strategy_name}")

    # 创建实验管理器
    manager = ExperimentManager(strategies=strategies, logger=logger)

    # 运行所有实验
    manager.run_all_experiments(args.start_date, args.end_date)

    # 打印摘要
    manager.print_summary()

    # 计算回测指标
    if args.calculate_metrics:
        logger.info("=" * 70)
        logger.info("计算回测指标...")
        logger.info("=" * 70)

        # 解析交易参数
        trading_params = None
        if args.trading_params:
            trading_params = json.loads(args.trading_params)

        recorder = ResultRecorder(logger=logger)

        # 为每个策略计算指标
        for strategy_name, result in manager.results.items():
            logger.info(f"策略: {strategy_name}")

            metrics = recorder.calculate_backtest_metrics(
                result['predictions'],
                trading_params
            )

            result['metrics'] = metrics

            logger.info(f"  总收益率: {metrics['total_return']*100:+.2f}%")
            logger.info(f"  年化收益: {metrics['annual_return']*100:+.2f}%")
            logger.info(f"  夏普比率: {metrics['sharpe_ratio']:.3f}")
            logger.info(f"  最大回撤: {metrics['max_drawdown']*100:.2f}%")
            logger.info(f"  胜率: {metrics['win_rate']*100:.2f}%")
            logger.info(f"  交易次数: {metrics['n_trades']}")

            # 更新注册表
            experiment_id = experiment_ids.get(strategy_name)
            if experiment_id:
                registry.update_experiment(
                    experiment_id=experiment_id,
                    metrics=metrics,
                    status="completed"
                )
                logger.info(f"  Updated experiment registry: {experiment_id}")

        # 生成对比报告
        logger.info("=" * 70)
        logger.info("生成对比报告...")
        logger.info("=" * 70)

        comparison = recorder.compare_strategies(manager.results, experiment_ids)

        # 打印对比表格
        table = recorder.generate_markdown_table(comparison)
        logger.info(table)

        # 保存对比结果
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        comparison_file = recorder.output_dir / f"comparison_{timestamp}.json"

        with open(comparison_file, 'w', encoding='utf-8') as f:
            json.dump(comparison, f, indent=2, ensure_ascii=False)

        logger.info(f"对比结果已保存: {comparison_file}")

        # 更新 CLAUDE.md
        if args.update_claude_md:
            logger.info("更新 CLAUDE.md...")
            recorder.update_claude_md(comparison)
            logger.info("CLAUDE.md 已更新")

    else:
        # 即使不计算指标，也更新实验状态
        for strategy_name in strategies:
            experiment_id = experiment_ids.get(strategy_name)
            if experiment_id:
                registry.update_experiment(
                    experiment_id=experiment_id,
                    status="completed"
                )

    logger.info("=" * 70)
    logger.info(" " * 25 + "实验完成!")
    logger.info("=" * 70)

    # 打印注册表信息
    logger.info(f"实验注册表位置: logs/experiments_registry.json")
    logger.info(f"日志输出目录: logs/{datetime.now().strftime('%Y-%m-%d')}/")


if __name__ == "__main__":
    main()
