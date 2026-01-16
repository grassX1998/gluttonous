"""
实验管理器

管理多个训练策略实验的运行、结果记录和对比分析。
"""

import sys
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime
import json

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.lstm.config import (
    ALL_STRATEGY_CONFIGS,
    TrainStrategyConfig,
    EXPERIMENT_RESULT_DIR
)


class ExperimentManager:
    """实验管理器"""

    def __init__(self, strategies: List[str] = None):
        """
        初始化实验管理器

        Args:
            strategies: 要运行的策略名称列表，如果为None则运行所有策略
        """
        if strategies is None:
            strategies = list(ALL_STRATEGY_CONFIGS.keys())

        self.strategy_names = strategies
        self.results = {}  # {strategy_name: result_dict}
        self.output_dir = EXPERIMENT_RESULT_DIR / "experiments"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _get_executor(self, config: TrainStrategyConfig):
        """
        根据配置创建对应的执行器

        Args:
            config: 策略配置

        Returns:
            策略执行器实例
        """
        from src.lstm.experiments.executors.expanding_window import ExpandingWindowExecutor
        # 其他执行器将在实现后导入

        executor_map = {
            "expanding_window": ExpandingWindowExecutor,
            # 其他策略将在实现后添加
        }

        executor_class = executor_map.get(config.strategy_name)
        if executor_class is None:
            raise ValueError(f"Unknown strategy: {config.strategy_name}")

        return executor_class(config)

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
        print("=" * 60)
        print(f"Running experiment: {strategy_name}")
        print("=" * 60)

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

        print(f"\nResults saved to: {output_file}\n")

        return result

    def run_all_experiments(self, start_date: str, end_date: str):
        """
        运行所有实验

        Args:
            start_date: 开始日期
            end_date: 结束日期
        """
        print("\n" + "=" * 60)
        print("EXPERIMENT MANAGER - Running All Strategies")
        print("=" * 60)
        print(f"Period: {start_date} to {end_date}")
        print(f"Strategies: {', '.join(self.strategy_names)}")
        print("=" * 60 + "\n")

        for strategy_name in self.strategy_names:
            try:
                result = self.run_single_experiment(strategy_name, start_date, end_date)
                self.results[strategy_name] = result
            except Exception as e:
                print(f"Error running {strategy_name}: {e}")
                import traceback
                traceback.print_exc()

        print("\n" + "=" * 60)
        print("All experiments completed")
        print("=" * 60)

    def compare_results(self) -> Dict[str, Any]:
        """
        对比所有策略的结果

        Returns:
            对比统计
        """
        if not self.results:
            print("No results to compare")
            return {}

        comparison = {
            'strategies': {},
            'summary': {}
        }

        for strategy_name, result in self.results.items():
            n_predictions = len(result['predictions'])
            n_retrains = len(result['retrain_dates'])

            avg_val_acc = 0
            if result['performance_history']:
                avg_val_acc = sum(h['val_acc'] for h in result['performance_history']) / len(result['performance_history'])

            comparison['strategies'][strategy_name] = {
                'n_predictions': n_predictions,
                'n_retrains': n_retrains,
                'avg_val_acc': avg_val_acc,
            }

        return comparison

    def print_summary(self):
        """打印实验摘要"""
        comparison = self.compare_results()

        print("\n" + "=" * 60)
        print("EXPERIMENT SUMMARY")
        print("=" * 60)

        for strategy_name, stats in comparison['strategies'].items():
            print(f"\n{strategy_name}:")
            print(f"  Predictions: {stats['n_predictions']}")
            print(f"  Retrains: {stats['n_retrains']}")
            print(f"  Avg Val Acc: {stats['avg_val_acc']:.4f}")

        print("=" * 60 + "\n")


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
