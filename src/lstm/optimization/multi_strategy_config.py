"""
多策略配置管理器

定义6种训练方案及其参数空间
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List
import copy


@dataclass
class StrategyParamSpace:
    """策略参数空间"""
    name: str
    description: str

    # 训练参数空间
    train_params: Dict[str, List[Any]]

    # 交易参数空间
    trading_params: Dict[str, List[Any]]

    # 当前最佳参数
    best_train_params: Dict[str, Any] = field(default_factory=dict)
    best_trading_params: Dict[str, Any] = field(default_factory=dict)
    best_score: float = -999.0


# 6种训练策略配置
STRATEGY_CONFIGS = {
    "expanding_window": StrategyParamSpace(
        name="expanding_window",
        description="扩展窗口 - 累积历史数据，样本加权",

        train_params={
            "min_train_days": [60],
            "max_train_days": [500],
            "weight_decay_days": [15, 20, 25],
            "weight_decay_rate": [0.93, 0.95, 0.97],
            "retrain_interval": [1, 3, 5],
        },

        trading_params={
            "prob_threshold": [0.70, 0.73, 0.75, 0.78],
            "trailing_stop_pct": [0.03, 0.05, 0.07],
            "max_holding_days": [7, 10, 15],
            "min_holding_days": [1, 2, 3],
        },

        best_train_params={
            "min_train_days": 60,
            "max_train_days": 500,
            "weight_decay_days": 20,
            "weight_decay_rate": 0.95,
            "retrain_interval": 1,
        },

        best_trading_params={
            "prob_threshold": 0.70,
            "trailing_stop_pct": 0.05,
            "max_holding_days": 10,
            "min_holding_days": 1,
        },
    ),

    "rolling_window": StrategyParamSpace(
        name="rolling_window",
        description="滚动窗口 - 固定长度训练集",

        train_params={
            "train_days": [40, 60, 90],
            "retrain_interval": [1, 3, 5],
            "use_sample_weight": [True, False],
        },

        trading_params={
            "prob_threshold": [0.70, 0.73, 0.75],
            "trailing_stop_pct": [0.03, 0.05, 0.07],
            "max_holding_days": [7, 10, 15],
        },

        best_train_params={
            "train_days": 60,
            "retrain_interval": 3,
            "use_sample_weight": True,
        },

        best_trading_params={
            "prob_threshold": 0.70,
            "trailing_stop_pct": 0.05,
            "max_holding_days": 10,
        },
    ),

    "adaptive_weight": StrategyParamSpace(
        name="adaptive_weight",
        description="自适应权重 - 根据验证准确率调整",

        train_params={
            "min_train_days": [60],
            "base_weight_decay": [0.95, 0.97],
            "acc_threshold_high": [0.60, 0.65],
            "acc_threshold_low": [0.50, 0.55],
            "retrain_interval": [1, 3],
        },

        trading_params={
            "prob_threshold": [0.70, 0.75],
            "trailing_stop_pct": [0.05, 0.07],
            "max_holding_days": [10, 15],
        },

        best_train_params={
            "min_train_days": 60,
            "base_weight_decay": 0.95,
            "acc_threshold_high": 0.65,
            "acc_threshold_low": 0.55,
            "retrain_interval": 1,
        },

        best_trading_params={
            "prob_threshold": 0.70,
            "trailing_stop_pct": 0.05,
            "max_holding_days": 10,
        },
    ),

    "ensemble_multi_scale": StrategyParamSpace(
        name="ensemble_multi_scale",
        description="多尺度集成 - 短中长期模型组合",

        train_params={
            "window_short": [20, 30],
            "window_medium": [60, 90],
            "window_long": [120, 180],
            "ensemble_weights": [[0.3, 0.4, 0.3], [0.4, 0.3, 0.3]],
            "retrain_interval": [3, 5],
        },

        trading_params={
            "prob_threshold": [0.70, 0.75],
            "trailing_stop_pct": [0.05, 0.07],
            "max_holding_days": [10, 15],
        },

        best_train_params={
            "window_short": 30,
            "window_medium": 60,
            "window_long": 120,
            "ensemble_weights": [0.3, 0.4, 0.3],
            "retrain_interval": 5,
        },

        best_trading_params={
            "prob_threshold": 0.70,
            "trailing_stop_pct": 0.05,
            "max_holding_days": 10,
        },
    ),

    "volatility_adaptive": StrategyParamSpace(
        name="volatility_adaptive",
        description="波动率自适应 - 根据市场波动调整",

        train_params={
            "min_train_days": [60],
            "vol_lookback": [20, 30],
            "vol_high_threshold": [0.03, 0.04],
            "vol_low_threshold": [0.01, 0.015],
            "retrain_interval": [1, 3],
        },

        trading_params={
            "prob_threshold": [0.70, 0.75],
            "trailing_stop_pct": [0.05, 0.07],
            "max_holding_days": [10, 15],
            "vol_scaling": [True, False],  # 是否根据波动率调整仓位
        },

        best_train_params={
            "min_train_days": 60,
            "vol_lookback": 20,
            "vol_high_threshold": 0.03,
            "vol_low_threshold": 0.015,
            "retrain_interval": 1,
        },

        best_trading_params={
            "prob_threshold": 0.70,
            "trailing_stop_pct": 0.05,
            "max_holding_days": 10,
            "vol_scaling": True,
        },
    ),

    "momentum_enhanced": StrategyParamSpace(
        name="momentum_enhanced",
        description="动量增强 - 结合短期动量信号",

        train_params={
            "min_train_days": [60],
            "momentum_lookback": [5, 10, 20],
            "momentum_weight": [0.2, 0.3, 0.4],
            "weight_decay_rate": [0.93, 0.95],
            "retrain_interval": [1, 3],
        },

        trading_params={
            "prob_threshold": [0.70, 0.75],
            "trailing_stop_pct": [0.05, 0.07],
            "max_holding_days": [7, 10],
            "momentum_filter": [True, False],  # 是否过滤负动量股票
        },

        best_train_params={
            "min_train_days": 60,
            "momentum_lookback": 10,
            "momentum_weight": 0.3,
            "weight_decay_rate": 0.95,
            "retrain_interval": 1,
        },

        best_trading_params={
            "prob_threshold": 0.70,
            "trailing_stop_pct": 0.05,
            "max_holding_days": 10,
            "momentum_filter": True,
        },
    ),
}


class MultiStrategyManager:
    """多策略管理器"""

    def __init__(self):
        self.strategies = copy.deepcopy(STRATEGY_CONFIGS)
        self.optimization_history = {name: [] for name in self.strategies.keys()}

    def get_strategy(self, name: str) -> StrategyParamSpace:
        """获取策略配置"""
        return self.strategies[name]

    def get_all_strategies(self) -> List[str]:
        """获取所有策略名称"""
        return list(self.strategies.keys())

    def update_best_params(self, name: str, train_params: Dict,
                          trading_params: Dict, score: float):
        """更新最佳参数"""
        if score > self.strategies[name].best_score:
            self.strategies[name].best_train_params = train_params
            self.strategies[name].best_trading_params = trading_params
            self.strategies[name].best_score = score

            # 记录历史
            self.optimization_history[name].append({
                'train_params': train_params,
                'trading_params': trading_params,
                'score': score,
            })

    def get_param_combinations(self, name: str, max_combinations: int = 10):
        """
        获取参数组合（用于网格搜索）

        如果组合数过多，随机采样max_combinations个
        """
        strategy = self.strategies[name]

        # 生成训练参数组合
        train_combos = self._generate_combinations(strategy.train_params)

        # 生成交易参数组合
        trading_combos = self._generate_combinations(strategy.trading_params)

        # 限制组合数
        import random
        if len(train_combos) > max_combinations:
            train_combos = random.sample(train_combos, max_combinations)
        if len(trading_combos) > max_combinations:
            trading_combos = random.sample(trading_combos, max_combinations)

        return train_combos, trading_combos

    def _generate_combinations(self, param_space: Dict[str, List]) -> List[Dict]:
        """生成参数组合"""
        from itertools import product

        keys = list(param_space.keys())
        values = [param_space[k] for k in keys]

        combinations = []
        for combo in product(*values):
            combinations.append(dict(zip(keys, combo)))

        return combinations

    def get_random_params(self, name: str) -> tuple:
        """随机采样参数（用于随机搜索）"""
        import random
        strategy = self.strategies[name]

        train_params = {
            k: random.choice(v)
            for k, v in strategy.train_params.items()
        }

        trading_params = {
            k: random.choice(v)
            for k, v in strategy.trading_params.items()
        }

        return train_params, trading_params


if __name__ == "__main__":
    # 测试
    manager = MultiStrategyManager()

    print("=" * 60)
    print("多策略配置管理器")
    print("=" * 60)

    for name in manager.get_all_strategies():
        strategy = manager.get_strategy(name)
        print(f"\n策略: {strategy.name}")
        print(f"描述: {strategy.description}")
        print(f"训练参数空间: {len(manager._generate_combinations(strategy.train_params))} 个组合")
        print(f"交易参数空间: {len(manager._generate_combinations(strategy.trading_params))} 个组合")
