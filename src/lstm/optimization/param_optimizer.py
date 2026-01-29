"""
参数优化器

支持网格搜索和随机搜索两种模式，用于优化训练策略参数。

用法示例:
    optimizer = ParamOptimizer(strategy='v03')
    best_params = optimizer.run_optimization(mode='grid')
"""

import sys
import json
import itertools
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass, asdict
import random

import numpy as np

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.lstm.config import (
    V03ReproConfig,
    ExpandingWindowConfig,
    EXPERIMENT_RESULT_DIR,
)


@dataclass
class TrialResult:
    """单次试验结果"""
    params: Dict[str, Any]
    metrics: Dict[str, float]
    score: float
    duration_seconds: float


class ParamOptimizer:
    """参数优化器"""

    # 默认参数搜索空间
    V03_PARAM_SPACE = {
        'train_months': [4, 5, 6, 8, 10],
        'val_months': [1, 2],
        'sample_ratio': [0.4, 0.5, 0.6, 0.7],
        'prob_threshold': [0.55, 0.60, 0.65, 0.70],
        'top_n': [8, 10, 15, 20],
    }

    EXPANDING_PARAM_SPACE = {
        'weight_decay_days': [15, 18, 20, 25, 30],
        'weight_decay_rate': [0.92, 0.94, 0.95, 0.96],
        'min_train_days': [40, 60, 90],
        'max_train_days': [300, 500, 700],
        'retrain_interval': [1, 3, 5],
        'prob_threshold': [0.65, 0.70, 0.75, 0.78],
        'top_n': [8, 10, 15, 20],
    }

    def __init__(
        self,
        strategy: str = 'v03',
        param_space: Optional[Dict[str, List]] = None,
        start_date: str = '2025-04-01',
        end_date: str = '2026-01-15',
        output_dir: Optional[Path] = None,
    ):
        """
        初始化优化器

        Args:
            strategy: 策略类型 ('v03', 'expanding', 'both')
            param_space: 自定义参数空间，None 则使用默认
            start_date: 回测开始日期
            end_date: 回测结束日期
            output_dir: 结果输出目录
        """
        self.strategy = strategy
        self.start_date = start_date
        self.end_date = end_date

        # 设置参数空间
        if param_space is not None:
            self.param_space = param_space
        elif strategy == 'v03':
            self.param_space = self.V03_PARAM_SPACE.copy()
        elif strategy == 'expanding':
            self.param_space = self.EXPANDING_PARAM_SPACE.copy()
        else:
            # both: 合并两者
            self.param_space = {}

        # 输出目录
        self.output_dir = output_dir or (EXPERIMENT_RESULT_DIR / 'optimization')
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 结果存储
        self.trials: List[TrialResult] = []
        self.best_trial: Optional[TrialResult] = None

    def _calculate_score(self, metrics: Dict[str, float]) -> float:
        """
        计算综合评分

        评分公式:
            score = total_return * 0.35 +
                    sharpe_ratio * 0.08 * 0.30 +
                    (1 - max_drawdown) * 0.25 +
                    win_rate * 0.10

        Args:
            metrics: 回测指标字典

        Returns:
            综合评分 (越高越好)
        """
        total_return = metrics.get('total_return', 0)
        sharpe_ratio = metrics.get('sharpe_ratio', 0)
        max_drawdown = metrics.get('max_drawdown', 1)
        win_rate = metrics.get('win_rate', 0)

        # 限制极端值
        total_return = max(-1, min(10, total_return))  # -100% ~ 1000%
        sharpe_ratio = max(-5, min(5, sharpe_ratio))   # -5 ~ 5
        max_drawdown = max(0, min(1, max_drawdown))    # 0 ~ 100%
        win_rate = max(0, min(1, win_rate))            # 0 ~ 100%

        score = (
            total_return * 0.35 +
            sharpe_ratio * 0.08 * 0.30 +
            (1 - max_drawdown) * 0.25 +
            win_rate * 0.10
        )

        return score

    def _generate_grid_combinations(self) -> List[Dict[str, Any]]:
        """生成网格搜索的所有参数组合"""
        keys = list(self.param_space.keys())
        values = list(self.param_space.values())

        combinations = []
        for combo in itertools.product(*values):
            param_dict = dict(zip(keys, combo))
            combinations.append(param_dict)

        return combinations

    def _generate_random_samples(self, n_trials: int) -> List[Dict[str, Any]]:
        """生成随机采样的参数组合"""
        combinations = []
        for _ in range(n_trials):
            param_dict = {}
            for key, values in self.param_space.items():
                param_dict[key] = random.choice(values)
            combinations.append(param_dict)

        return combinations

    def _run_single_trial(self, params: Dict[str, Any]) -> Tuple[Dict[str, float], float]:
        """
        运行单次试验

        Args:
            params: 参数字典

        Returns:
            (metrics, duration)
        """
        import time
        start_time = time.time()

        if self.strategy == 'v03':
            metrics = self._run_v03_trial(params)
        elif self.strategy == 'expanding':
            metrics = self._run_expanding_trial(params)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

        duration = time.time() - start_time
        return metrics, duration

    def _run_v03_trial(self, params: Dict[str, Any]) -> Dict[str, float]:
        """运行 V03Repro 策略试验"""
        from src.lstm.experiments.executors.v03_repro import V03ReproExecutor
        from src.lstm.config import V03ReproConfig

        # 创建配置
        config = V03ReproConfig.with_params(**params)

        # 运行回测
        executor = V03ReproExecutor(config)
        result = executor.run(self.start_date, self.end_date)

        # 提取指标
        if 'results' in result:
            return {
                'total_return': result['results'].get('total_return', 0),
                'annual_return': result['results'].get('annual_return', 0),
                'sharpe_ratio': result['results'].get('sharpe_ratio', 0),
                'max_drawdown': result['results'].get('max_drawdown', 1),
                'win_rate': result['results'].get('trade_win_rate', 0),
                'n_trades': result['results'].get('n_trades', 0),
            }
        else:
            return {
                'total_return': 0,
                'sharpe_ratio': 0,
                'max_drawdown': 1,
                'win_rate': 0,
                'n_trades': 0,
            }

    def _run_expanding_trial(self, params: Dict[str, Any]) -> Dict[str, float]:
        """运行 ExpandingWindow 策略试验"""
        from src.lstm.experiments.executors.expanding_window import ExpandingWindowExecutor
        from src.lstm.experiments.metrics.backtest_engine import BacktestEngine
        from src.lstm.config import ExpandingWindowConfig, TRADING_CONFIG

        # 分离训练参数和交易参数
        train_params = {k: v for k, v in params.items()
                        if k not in ['prob_threshold', 'top_n']}
        trade_params = {k: v for k, v in params.items()
                        if k in ['prob_threshold', 'top_n']}

        # 创建配置
        config = ExpandingWindowConfig.with_params(**params)

        # 运行预测
        executor = ExpandingWindowExecutor(config)
        result = executor.run(self.start_date, self.end_date)

        if not result.get('predictions'):
            return {
                'total_return': 0,
                'sharpe_ratio': 0,
                'max_drawdown': 1,
                'win_rate': 0,
                'n_trades': 0,
            }

        # 运行回测
        prob_threshold = params.get('prob_threshold', TRADING_CONFIG['prob_threshold'])
        top_n = params.get('top_n', TRADING_CONFIG['top_n'])

        engine = BacktestEngine(
            top_n=top_n,
            prob_threshold=prob_threshold,
            commission=TRADING_CONFIG['commission'],
            slippage=TRADING_CONFIG['slippage'],
            trailing_stop_pct=TRADING_CONFIG.get('trailing_stop_pct', 0.05),
            max_holding_days=TRADING_CONFIG.get('max_holding_days', 10),
            min_holding_days=TRADING_CONFIG.get('min_holding_days', 1),
        )

        # 加载价格数据
        price_data = engine.load_price_data(None, self.start_date, self.end_date)
        if price_data is None:
            return {
                'total_return': 0,
                'sharpe_ratio': 0,
                'max_drawdown': 1,
                'win_rate': 0,
                'n_trades': 0,
            }

        # 获取交易日
        trading_dates = sorted(set(p['date'] for p in result['predictions']))

        # 运行回测
        engine.run_backtest(result['predictions'], price_data, trading_dates)
        metrics = engine.calculate_metrics()

        return metrics

    def run_optimization(
        self,
        mode: str = 'grid',
        n_trials: int = 20,
        save_intermediate: bool = True,
    ) -> Dict[str, Any]:
        """
        运行参数优化

        Args:
            mode: 搜索模式 ('grid' or 'random')
            n_trials: 随机搜索时的试验次数
            save_intermediate: 是否保存中间结果

        Returns:
            优化结果字典
        """
        print("=" * 70)
        print(f"参数优化 - 策略: {self.strategy}, 模式: {mode}")
        print(f"日期范围: {self.start_date} ~ {self.end_date}")
        print("=" * 70)

        # 生成参数组合
        if mode == 'grid':
            combinations = self._generate_grid_combinations()
            print(f"网格搜索: {len(combinations)} 组参数组合")
        else:
            combinations = self._generate_random_samples(n_trials)
            print(f"随机搜索: {n_trials} 次试验")

        print(f"参数空间: {list(self.param_space.keys())}")
        print("=" * 70)

        # 运行所有试验
        self.trials = []
        self.best_trial = None

        for i, params in enumerate(combinations):
            print(f"\n[{i+1}/{len(combinations)}] 参数: {params}")

            try:
                metrics, duration = self._run_single_trial(params)
                score = self._calculate_score(metrics)

                trial = TrialResult(
                    params=params,
                    metrics=metrics,
                    score=score,
                    duration_seconds=duration,
                )
                self.trials.append(trial)

                # 更新最佳结果
                if self.best_trial is None or score > self.best_trial.score:
                    self.best_trial = trial
                    print(f"  >>> 新最佳! Score: {score:.4f}")
                    print(f"      收益: {metrics.get('total_return', 0)*100:+.2f}%, "
                          f"夏普: {metrics.get('sharpe_ratio', 0):.3f}, "
                          f"回撤: {metrics.get('max_drawdown', 0)*100:.2f}%")
                else:
                    print(f"  Score: {score:.4f} (最佳: {self.best_trial.score:.4f})")

                # 保存中间结果
                if save_intermediate and (i + 1) % 5 == 0:
                    self._save_results(f'intermediate_{i+1}')

            except Exception as e:
                print(f"  ERROR: {e}")
                continue

        # 保存最终结果
        result = self._save_results('final')

        # 打印总结
        self._print_summary()

        return result

    def _save_results(self, suffix: str = '') -> Dict[str, Any]:
        """保存优化结果"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"param_search_{self.strategy}_{timestamp}_{suffix}.json"
        filepath = self.output_dir / filename

        result = {
            'strategy': self.strategy,
            'mode': 'grid' if len(self.trials) > 20 else 'random',
            'start_date': self.start_date,
            'end_date': self.end_date,
            'param_space': self.param_space,
            'n_trials': len(self.trials),
            'trials': [
                {
                    'params': t.params,
                    'metrics': t.metrics,
                    'score': t.score,
                    'duration_seconds': t.duration_seconds,
                }
                for t in self.trials
            ],
            'best': {
                'params': self.best_trial.params if self.best_trial else None,
                'metrics': self.best_trial.metrics if self.best_trial else None,
                'score': self.best_trial.score if self.best_trial else None,
            },
            'timestamp': datetime.now().isoformat(),
        }

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        print(f"\n结果已保存: {filepath}")

        # 保存最佳参数到单独文件
        if self.best_trial and suffix == 'final':
            best_params_file = self.output_dir / f"best_params_{self.strategy}.json"
            with open(best_params_file, 'w', encoding='utf-8') as f:
                json.dump({
                    'strategy': self.strategy,
                    'params': self.best_trial.params,
                    'metrics': self.best_trial.metrics,
                    'score': self.best_trial.score,
                    'updated': datetime.now().isoformat(),
                }, f, indent=2, ensure_ascii=False)
            print(f"最佳参数: {best_params_file}")

        return result

    def _print_summary(self):
        """打印优化总结"""
        print("\n" + "=" * 70)
        print("优化完成 - 总结")
        print("=" * 70)

        if not self.trials:
            print("没有成功的试验")
            return

        # 按 score 排序
        sorted_trials = sorted(self.trials, key=lambda x: x.score, reverse=True)

        print(f"\n完成试验数: {len(self.trials)}")
        print(f"最佳 Score: {self.best_trial.score:.4f}")

        print("\n最佳参数:")
        for key, value in self.best_trial.params.items():
            print(f"  {key}: {value}")

        print("\n最佳指标:")
        for key, value in self.best_trial.metrics.items():
            if key in ['total_return', 'annual_return', 'max_drawdown', 'win_rate']:
                print(f"  {key}: {value*100:.2f}%")
            else:
                print(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")

        # Top 5 结果
        print("\n排名前5的参数组合:")
        for i, trial in enumerate(sorted_trials[:5]):
            print(f"\n  #{i+1} Score: {trial.score:.4f}")
            print(f"     参数: {trial.params}")
            print(f"     收益: {trial.metrics.get('total_return', 0)*100:+.2f}%, "
                  f"夏普: {trial.metrics.get('sharpe_ratio', 0):.3f}")

        print("\n" + "=" * 70)


def load_best_params(strategy: str, output_dir: Optional[Path] = None) -> Optional[Dict[str, Any]]:
    """
    加载策略的最佳参数

    Args:
        strategy: 策略名称 ('v03' or 'expanding')
        output_dir: 参数文件目录

    Returns:
        最佳参数字典，如果不存在返回 None
    """
    if output_dir is None:
        output_dir = EXPERIMENT_RESULT_DIR / 'optimization'

    best_params_file = output_dir / f"best_params_{strategy}.json"

    if best_params_file.exists():
        with open(best_params_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return data.get('params')

    return None
