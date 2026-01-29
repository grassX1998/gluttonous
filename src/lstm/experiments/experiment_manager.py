"""
实验管理器

管理多个训练策略实验的运行、结果记录和对比分析。
"""

import sys
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
import json

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.lstm.config import (
    ALL_STRATEGY_CONFIGS,
    TrainStrategyConfig,
    EXPERIMENT_RESULT_DIR
)
from pipeline.shared.logging_config import get_lstm_training_logger


class ExperimentManager:
    """实验管理器"""

    def __init__(self, strategies: List[str] = None, logger: Optional[logging.Logger] = None):
        """
        初始化实验管理器

        Args:
            strategies: 要运行的策略名称列表，如果为None则运行所有策略
            logger: 日志记录器，如果为 None 则使用默认的 LSTM 训练日志器
        """
        if strategies is None:
            strategies = list(ALL_STRATEGY_CONFIGS.keys())

        self.strategy_names = strategies
        self.results = {}  # {strategy_name: result_dict}
        self.output_dir = EXPERIMENT_RESULT_DIR / "experiments"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logger or get_lstm_training_logger()

    def _get_executor(self, config: TrainStrategyConfig):
        """
        根据配置创建对应的执行器

        Args:
            config: 策略配置

        Returns:
            策略执行器实例
        """
        from src.lstm.experiments.executors.expanding_window import ExpandingWindowExecutor
        from src.lstm.experiments.executors.v03_repro import V03ReproExecutor
        from src.lstm.experiments.executors.v03_daily import V03DailyExecutor

        executor_map = {
            "expanding_window": ExpandingWindowExecutor,
            "v03_repro": V03ReproExecutor,
            "v03_daily": V03DailyExecutor,
        }

        executor_class = executor_map.get(config.strategy_name)
        if executor_class is None:
            raise ValueError(f"Unknown strategy: {config.strategy_name}")

        # V03ReproExecutor 和 V03DailyExecutor 不继承 BaseStrategyExecutor，不接受 logger 参数
        if config.strategy_name in ["v03_repro", "v03_daily"]:
            return executor_class(config)
        return executor_class(config, logger=self.logger)

    def run_single_experiment(self, strategy_name: str, start_date: str, end_date: str) -> Dict[str, Any]:
        """
        运行单个实验

        Args:
            strategy_name: 策略名称
            start_date: 开始日期
            end_date: 结束日期

        Returns:
            实验结果
        """
        self.logger.info("=" * 60)
        self.logger.info(f"Running experiment: {strategy_name}")
        self.logger.info("=" * 60)

        # 创建策略配置
        config_class = ALL_STRATEGY_CONFIGS[strategy_name]
        config = config_class()

        # 创建执行器
        executor = self._get_executor(config)

        # 运行回测
        result = executor.run(start_date, end_date)

        # 保存结果
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = self.output_dir / f"{strategy_name}_{timestamp}.json"

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        self.logger.info(f"Results saved to: {output_file}")

        return result

    def run_all_experiments(self, start_date: str, end_date: str):
        """
        运行所有实验

        Args:
            start_date: 开始日期
            end_date: 结束日期
        """
        self.logger.info("=" * 60)
        self.logger.info("EXPERIMENT MANAGER - Running All Strategies")
        self.logger.info("=" * 60)
        self.logger.info(f"Period: {start_date} to {end_date}")
        self.logger.info(f"Strategies: {', '.join(self.strategy_names)}")
        self.logger.info("=" * 60)

        for strategy_name in self.strategy_names:
            try:
                result = self.run_single_experiment(strategy_name, start_date, end_date)
                self.results[strategy_name] = result
            except Exception as e:
                self.logger.error(f"Error running {strategy_name}: {e}")
                import traceback
                self.logger.error(traceback.format_exc())

        self.logger.info("=" * 60)
        self.logger.info("All experiments completed")
        self.logger.info("=" * 60)

    def compare_results(self) -> Dict[str, Any]:
        """
        对比所有策略的结果

        Returns:
            对比统计
        """
        if not self.results:
            self.logger.warning("No results to compare")
            return {}

        comparison = {
            'strategies': {},
            'summary': {}
        }

        for strategy_name, result in self.results.items():
            # 防御性处理：不同策略的结果结构可能不同
            predictions = result.get('predictions', [])
            retrain_dates = result.get('retrain_dates', [])
            performance_history = result.get('performance_history', [])

            n_predictions = len(predictions) if predictions else 0
            n_retrains = len(retrain_dates) if retrain_dates else 0

            avg_val_acc = 0
            if performance_history:
                val_accs = [h.get('val_acc', 0) for h in performance_history if h.get('val_acc') is not None]
                if val_accs:
                    avg_val_acc = sum(val_accs) / len(val_accs)

            comparison['strategies'][strategy_name] = {
                'n_predictions': n_predictions,
                'n_retrains': n_retrains,
                'avg_val_acc': avg_val_acc,
            }

        return comparison

    def print_summary(self):
        """打印实验摘要"""
        comparison = self.compare_results()

        self.logger.info("=" * 60)
        self.logger.info("EXPERIMENT SUMMARY")
        self.logger.info("=" * 60)

        for strategy_name, stats in comparison['strategies'].items():
            self.logger.info(f"{strategy_name}:")
            self.logger.info(f"  Predictions: {stats['n_predictions']}")
            self.logger.info(f"  Retrains: {stats['n_retrains']}")
            self.logger.info(f"  Avg Val Acc: {stats['avg_val_acc']:.4f}")

        self.logger.info("=" * 60)


def main():
    """示例：运行实验"""
    import argparse

    parser = argparse.ArgumentParser(description="实验管理器")
    parser.add_argument("--strategies", nargs='+', default=None,
                       help="要运行的策略列表")
    parser.add_argument("--start_date", type=str, default="2025-04-01",
                       help="开始日期")
    parser.add_argument("--end_date", type=str, default="2026-01-15",
                       help="结束日期")

    args = parser.parse_args()

    manager = ExperimentManager(strategies=args.strategies)
    manager.run_all_experiments(args.start_date, args.end_date)
    manager.print_summary()


if __name__ == "__main__":
    main()
