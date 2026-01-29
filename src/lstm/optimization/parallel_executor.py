"""
并行训练和回测执行器

支持多策略并行训练和回测
"""

import json
import threading
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Tuple
import queue


class ParallelExecutor:
    """并行执行器"""

    def __init__(self, max_workers: int = 3, output_dir: Path = None):
        """
        Args:
            max_workers: 最大并行数（建议2-3个，避免GPU冲突）
            output_dir: 输出目录
        """
        self.max_workers = max_workers
        self.output_dir = output_dir or Path("src/lstm/data/results/optimization")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.task_queue = queue.Queue()
        self.result_queue = queue.Queue()
        self.workers = []
        self.running = False

    def submit_task(self, strategy_name: str, train_params: Dict,
                   trading_params: Dict, position_strategy: str = 'equal_weight',
                   task_id: str = None):
        """提交训练+回测任务（支持仓位管理策略）"""
        if task_id is None:
            task_id = f"{strategy_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        task = {
            'task_id': task_id,
            'strategy_name': strategy_name,
            'train_params': train_params,
            'trading_params': trading_params,
            'position_strategy': position_strategy,
            'submit_time': datetime.now().isoformat(),
        }

        self.task_queue.put(task)
        return task_id

    def start(self):
        """启动并行执行"""
        self.running = True

        for i in range(self.max_workers):
            worker = threading.Thread(target=self._worker, args=(i,), daemon=True)
            worker.start()
            self.workers.append(worker)

        print(f"[ParallelExecutor] Started {self.max_workers} workers")

    def stop(self):
        """停止执行"""
        self.running = False
        for _ in range(self.max_workers):
            self.task_queue.put(None)  # 发送停止信号

        for worker in self.workers:
            worker.join(timeout=10)

        print("[ParallelExecutor] Stopped")

    def get_results(self, timeout: float = 1.0) -> List[Dict]:
        """获取完成的结果"""
        results = []
        while True:
            try:
                result = self.result_queue.get(timeout=timeout)
                results.append(result)
            except queue.Empty:
                break
        return results

    def get_pending_count(self) -> int:
        """获取待处理任务数"""
        return self.task_queue.qsize()

    def has_capacity(self, max_pending: int = None) -> bool:
        """检查是否有空余容量提交新任务"""
        if max_pending is None:
            max_pending = self.max_workers * 2  # 默认最大排队数为 worker 数的2倍
        return self.get_pending_count() < max_pending

    def _worker(self, worker_id: int):
        """工作线程"""
        print(f"[Worker {worker_id}] Started")

        while self.running:
            try:
                task = self.task_queue.get(timeout=1.0)
            except queue.Empty:
                continue

            if task is None:  # 停止信号
                break

            print(f"[Worker {worker_id}] Processing task: {task['task_id']}")

            try:
                result = self._execute_task(task, worker_id)
                self.result_queue.put(result)
            except Exception as e:
                print(f"[Worker {worker_id}] Task {task['task_id']} failed: {e}")
                result = {
                    'task_id': task['task_id'],
                    'status': 'failed',
                    'error': str(e),
                }
                self.result_queue.put(result)

        print(f"[Worker {worker_id}] Stopped")

    def _execute_task(self, task: Dict, worker_id: int) -> Dict:
        """执行单个任务（训练+回测）"""
        task_id = task['task_id']
        strategy_name = task['strategy_name']
        train_params = task['train_params']
        trading_params = task['trading_params']

        start_time = time.time()

        # 1. 保存配置到临时文件
        config_file = self.output_dir / f"{task_id}_config.json"
        with open(config_file, 'w') as f:
            json.dump({
                'strategy': strategy_name,
                'train_params': train_params,
                'trading_params': trading_params,
            }, f, indent=2)

        # 2. 运行训练
        print(f"[Worker {worker_id}] Training {strategy_name}...")

        # 根据策略类型选择训练方法
        if strategy_name == "expanding_window":
            train_result = self._train_strategy(task, worker_id, "ExpandingWindowExecutor")
        elif strategy_name == "rolling_window":
            train_result = self._train_strategy(task, worker_id, "RollingWindowExecutor")
        elif strategy_name == "adaptive_weight":
            train_result = self._train_strategy(task, worker_id, "AdaptiveWeightExecutor")
        elif strategy_name == "ensemble_multi_scale":
            train_result = self._train_strategy(task, worker_id, "EnsembleMultiScaleExecutor")
        elif strategy_name == "volatility_adaptive":
            train_result = self._train_strategy(task, worker_id, "VolatilityAdaptiveExecutor")
        elif strategy_name == "momentum_enhanced":
            train_result = self._train_strategy(task, worker_id, "MomentumEnhancedExecutor")
        else:
            # 未知策略
            train_result = {
                'status': 'skipped',
                'message': f'Unknown strategy: {strategy_name}',
            }

        if train_result.get('status') == 'failed':
            return {
                'task_id': task_id,
                'status': 'failed',
                'error': train_result.get('error', 'Training failed'),
                'duration': time.time() - start_time,
            }

        # 3. 运行回测（支持仓位管理策略）
        print(f"[Worker {worker_id}] Backtesting {strategy_name}...")
        position_strategy = task.get('position_strategy', 'equal_weight')
        backtest_result = self._run_backtest(task, worker_id, train_result, position_strategy)

        # 4. 计算评分
        score = self._calculate_score(backtest_result)

        # 5. 保存结果
        result = {
            'task_id': task_id,
            'status': 'completed',
            'strategy': strategy_name,
            'train_params': train_params,
            'trading_params': trading_params,
            'train_result': train_result,
            'backtest_result': backtest_result,
            'score': score,
            'duration': time.time() - start_time,
            'worker_id': worker_id,
            'complete_time': datetime.now().isoformat(),
        }

        # 保存到文件
        result_file = self.output_dir / f"{task_id}_result.json"
        with open(result_file, 'w') as f:
            json.dump(result, f, indent=2, default=str)

        print(f"[Worker {worker_id}] Task {task_id} completed. Score: {score:.4f}")

        return result

    def _train_strategy(self, task: Dict, worker_id: int, executor_class_name: str) -> Dict:
        """通用策略训练方法（直接调用执行器）"""
        try:
            import sys
            from datetime import datetime as dt

            # 导入配置和执行器
            from src.lstm.config import (
                ExpandingWindowConfig, RollingKFoldConfig, MODEL_CONFIG, FEATURE_DATA_MONTHLY_DIR
            )

            # 导入所有执行器
            from src.lstm.experiments.executors.expanding_window import ExpandingWindowExecutor
            from src.lstm.experiments.executors.rolling_window import RollingWindowExecutor
            from src.lstm.experiments.executors.adaptive_weight import AdaptiveWeightExecutor
            from src.lstm.experiments.executors.ensemble_multi_scale import EnsembleMultiScaleExecutor
            from src.lstm.experiments.executors.volatility_adaptive import VolatilityAdaptiveExecutor
            from src.lstm.experiments.executors.momentum_enhanced import MomentumEnhancedExecutor

            # 创建配置对象
            strategy_name = task['strategy_name']
            train_params = task['train_params']

            # 根据策略选择配置类
            if strategy_name == "expanding_window":
                config = ExpandingWindowConfig()
            elif strategy_name == "rolling_window":
                config = RollingKFoldConfig()
            else:
                # 其他策略使用 ExpandingWindowConfig 作为基础配置
                # 因为所有策略都继承自相似的参数结构
                config = ExpandingWindowConfig()
                config.strategy_name = strategy_name
                config.description = f"{strategy_name} strategy"

            # 更新配置参数
            for key, value in train_params.items():
                setattr(config, key, value)

            print(f"[Worker {worker_id}] Training {strategy_name} with params: {train_params}")

            # 创建执行器
            executor_map = {
                "ExpandingWindowExecutor": ExpandingWindowExecutor,
                "RollingWindowExecutor": RollingWindowExecutor,
                "AdaptiveWeightExecutor": AdaptiveWeightExecutor,
                "EnsembleMultiScaleExecutor": EnsembleMultiScaleExecutor,
                "VolatilityAdaptiveExecutor": VolatilityAdaptiveExecutor,
                "MomentumEnhancedExecutor": MomentumEnhancedExecutor,
            }

            executor_class = executor_map.get(executor_class_name)
            if not executor_class:
                return {
                    'status': 'failed',
                    'error': f'Unknown executor: {executor_class_name}',
                }

            executor = executor_class(config)

            # 运行实验
            result = executor.run(start_date="2025-04-01", end_date="2026-01-15")

            # 保存结果到文件
            import json
            from datetime import datetime
            output_dir = Path("src/lstm/data/results/experiments")
            output_dir.mkdir(parents=True, exist_ok=True)

            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_file = output_dir / f"{strategy_name}_{timestamp}.json"

            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False, default=str)

            # 计算平均验证准确率
            avg_val_acc = 0
            if result.get('performance_history'):
                accs = [p['val_acc'] for p in result['performance_history']]
                avg_val_acc = sum(accs) / len(accs) if accs else 0

            return {
                'status': 'success',
                'experiment_file': str(output_file),
                'predictions': len(result.get('predictions', [])),
                'avg_val_acc': avg_val_acc,
                'retrain_dates': result.get('retrain_dates', []),
            }

        except Exception as e:
            import traceback
            return {
                'status': 'failed',
                'error': str(e),
                'traceback': traceback.format_exc(),
            }

    def _run_backtest(self, task: Dict, worker_id: int, train_result: Dict,
                     position_strategy: str = 'equal_weight') -> Dict:
        """运行回测（直接调用Python函数，支持仓位管理）"""
        try:
            from src.lstm.experiments.metrics.backtest_engine import BacktestEngine
            from src.lstm.optimization.position_manager_optimizer import PositionManagerOptimizer
            from src.lstm.config import FEATURE_DATA_MONTHLY_DIR
            import polars as pl

            # 获取实验结果文件
            exp_file = train_result.get('experiment_file', '')
            if not exp_file or not Path(exp_file).exists():
                return {
                    'status': 'failed',
                    'error': f'Experiment file not found: {exp_file}',
                }

            # 读取预测结果
            with open(exp_file, 'r', encoding='utf-8') as f:
                experiment_result = json.load(f)

            predictions = experiment_result['predictions']
            start_date = experiment_result['start_date']
            end_date = experiment_result['end_date']

            # 创建仓位管理优化器
            position_mgr = PositionManagerOptimizer()

            # 应用仓位管理策略到预测结果
            # 注：这里暂时使用等权重，完整实现需要在BacktestEngine中集成
            # TODO: 在BacktestEngine中集成仓位管理逻辑

            # 创建回测引擎（使用任务的交易参数）
            trading_params = task['trading_params']

            engine = BacktestEngine(
                initial_capital=1000000.0,
                top_n=10,  # 固定为10
                prob_threshold=trading_params.get('prob_threshold', 0.70),
                commission=0.001,
                slippage=0.001,
                trailing_stop_pct=trading_params.get('trailing_stop_pct', 0.05),
                max_holding_days=trading_params.get('max_holding_days', 10),
                min_holding_days=trading_params.get('min_holding_days', 1),
                exit_on_low_prob=False
            )

            print(f"[Worker {worker_id}] Running backtest with trading params: {trading_params}")
            print(f"[Worker {worker_id}] Position strategy: {position_strategy}")

            # 加载价格数据
            price_data = engine.load_price_data(
                Path('src/lstm/data'),
                start_date,
                end_date
            )

            if price_data is None:
                return {
                    'status': 'failed',
                    'error': 'Failed to load price data',
                }

            # 获取交易日列表
            trading_dates = sorted(price_data['date'].cast(str).unique().to_list())

            # 运行回测
            backtest_result = engine.run_backtest(predictions, price_data, trading_dates)

            # 计算指标
            metrics = engine.calculate_metrics()

            return {
                'status': 'success',
                'total_return': metrics.get('total_return', 0),
                'sharpe_ratio': metrics.get('sharpe_ratio', 0),
                'max_drawdown': metrics.get('max_drawdown', 0),
                'win_rate': metrics.get('win_rate', 0),
                'n_trades': metrics.get('n_trades', 0),
                'annual_return': metrics.get('annual_return', 0),
                'n_days': metrics.get('n_days', 0),
                'backtest_result': backtest_result,
                'metrics': metrics,
            }

        except Exception as e:
            import traceback
            return {
                'status': 'failed',
                'error': str(e),
                'traceback': traceback.format_exc(),
            }

    def _calculate_score(self, backtest_result: Dict) -> float:
        """
        计算综合评分

        评分指标：
        - 总收益率（40%）
        - 夏普比率（30%）
        - 最大回撤（20%，负向）
        - 胜率（10%）
        """
        if backtest_result.get('status') != 'success':
            return -999.0

        total_return = backtest_result.get('total_return', 0)
        sharpe = backtest_result.get('sharpe_ratio', 0)
        max_dd = backtest_result.get('max_drawdown', 1)
        win_rate = backtest_result.get('win_rate', 0)

        # 归一化并计算加权分数
        score = (
            total_return * 0.4 +  # 收益率权重40%
            sharpe * 0.1 * 0.3 +  # 夏普比率（归一化到0-1）权重30%
            (1 - max_dd) * 0.2 +  # 回撤（反向）权重20%
            win_rate * 0.1         # 胜率权重10%
        )

        return score


if __name__ == "__main__":
    # 测试并行执行器
    executor = ParallelExecutor(max_workers=2)
    executor.start()

    # 提交测试任务
    task_id = executor.submit_task(
        "expanding_window",
        {"weight_decay_days": 20, "weight_decay_rate": 0.95},
        {"prob_threshold": 0.70, "trailing_stop_pct": 0.05}
    )

    print(f"Submitted task: {task_id}")

    # 等待结果
    time.sleep(5)

    results = executor.get_results()
    print(f"Got {len(results)} results")

    executor.stop()
