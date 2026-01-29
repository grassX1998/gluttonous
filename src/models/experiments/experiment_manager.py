"""
实验管理器

管理模型实验的运行、保存和指标计算。
复用 LSTM 框架的 ResultRecorder 进行指标计算。
"""

import sys
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime
import json

# 添加项目根目录
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.config import EXPERIMENT_RESULT_DIR
from src.lstm.experiments.metrics.result_recorder import ResultRecorder
from pipeline.shared.logging_config import get_lstm_backtest_logger


class ExperimentManager:
    """实验管理器"""

    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        初始化管理器

        Args:
            logger: 日志记录器
        """
        self.logger = logger or get_lstm_backtest_logger()
        self.result_recorder = ResultRecorder(
            output_dir=EXPERIMENT_RESULT_DIR / "experiments",
            logger=self.logger
        )
        self.results: Dict[str, Any] = {}

    def run_experiment(self, executor, start_date: str, end_date: str,
                       calculate_metrics: bool = True) -> Dict[str, Any]:
        """
        运行单个实验

        Args:
            executor: 模型执行器
            start_date: 开始日期
            end_date: 结束日期
            calculate_metrics: 是否计算回测指标

        Returns:
            实验结果
        """
        strategy_name = getattr(executor.config, 'strategy_name', 'unknown')
        self.logger.info(f"Running experiment: {strategy_name}")

        # 运行回测
        result = executor.run(start_date, end_date)

        # 计算指标
        if calculate_metrics and result['predictions']:
            trading_params = {
                'top_n': getattr(executor.config, 'top_n', 10),
                'prob_threshold': getattr(executor.config, 'prob_threshold', 0.60),
                'holding_days': getattr(executor.config, 'holding_days', 5),
                'commission': getattr(executor.config, 'commission', 0.001),
                'slippage': getattr(executor.config, 'slippage', 0.001),
            }
            metrics = self.result_recorder.calculate_backtest_metrics(
                result['predictions'],
                trading_params
            )
            result['metrics'] = metrics

            self.logger.info(f"Metrics for {strategy_name}:")
            self.logger.info(f"  Total Return: {metrics['total_return']*100:.2f}%")
            self.logger.info(f"  Sharpe Ratio: {metrics['sharpe_ratio']:.3f}")
            self.logger.info(f"  Max Drawdown: {metrics['max_drawdown']*100:.2f}%")
            self.logger.info(f"  Win Rate: {metrics['win_rate']*100:.2f}%")
            self.logger.info(f"  N Trades: {metrics['n_trades']}")

        # 保存结果
        self.results[strategy_name] = result

        return result

    def save_results(self, strategy_name: str = None) -> Path:
        """
        保存实验结果

        Args:
            strategy_name: 策略名称，如果为 None 则保存所有结果

        Returns:
            保存的文件路径
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        if strategy_name:
            result = self.results.get(strategy_name)
            if result:
                return self.result_recorder.save_experiment_result(
                    strategy_name, result, timestamp
                )
        else:
            # 保存所有结果
            for name, result in self.results.items():
                self.result_recorder.save_experiment_result(name, result, timestamp)

        return EXPERIMENT_RESULT_DIR / "experiments"

    def compare_results(self) -> Dict[str, Any]:
        """
        对比所有实验结果

        Returns:
            对比统计
        """
        return self.result_recorder.compare_strategies(self.results)

    def print_comparison_table(self):
        """打印对比表格"""
        comparison = self.compare_results()
        table = self.result_recorder.generate_markdown_table(comparison)
        print(table)

    def update_claude_md(self):
        """更新 CLAUDE.md"""
        comparison = self.compare_results()
        self.result_recorder.update_claude_md(comparison)
