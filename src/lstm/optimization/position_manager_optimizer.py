"""
仓位管理逻辑优化器

持续迭代优化仓位管理策略
"""

import json
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime


class PositionManagerOptimizer:
    """仓位管理优化器"""

    def __init__(self, output_dir: Path = None):
        self.output_dir = output_dir or Path("src/lstm/data/results/optimization/position_mgmt")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 仓位管理策略池
        self.strategies = self._initialize_strategies()

        # 当前最佳策略
        self.best_strategy = None
        self.best_score = -999.0

        # 优化历史
        self.optimization_history = []

    def _initialize_strategies(self) -> List[Dict]:
        """初始化仓位管理策略池"""

        strategies = [
            {
                'name': 'equal_weight',
                'description': '等权重 - 平均分配资金',
                'params': {
                    'method': 'equal',
                },
                'score': 0.0,
            },

            {
                'name': 'prob_weighted',
                'description': '概率加权 - 根据预测概率分配',
                'params': {
                    'method': 'prob_weighted',
                    'min_weight': 0.5,  # 最低权重
                    'max_weight': 2.0,  # 最高权重
                },
                'score': 0.0,
            },

            {
                'name': 'kelly_criterion',
                'description': 'Kelly公式 - 根据胜率和盈亏比',
                'params': {
                    'method': 'kelly',
                    'kelly_fraction': 0.5,  # Kelly分数的一半（保守）
                    'max_position': 0.2,    # 单个仓位最大占比
                },
                'score': 0.0,
            },

            {
                'name': 'volatility_adjusted',
                'description': '波动率调整 - 低波动高仓位',
                'params': {
                    'method': 'vol_adjusted',
                    'target_vol': 0.02,     # 目标波动率2%
                    'lookback': 20,         # 波动率回溯期
                },
                'score': 0.0,
            },

            {
                'name': 'risk_parity',
                'description': '风险平价 - 均衡风险贡献',
                'params': {
                    'method': 'risk_parity',
                    'target_risk': 0.15,    # 目标组合风险
                },
                'score': 0.0,
            },

            {
                'name': 'dynamic_sizing',
                'description': '动态仓位 - 根据盈亏动态调整',
                'params': {
                    'method': 'dynamic',
                    'base_size': 1.0,
                    'win_multiplier': 1.2,   # 连胜时放大
                    'loss_multiplier': 0.8,  # 连亏时缩小
                    'lookback': 5,           # 查看最近5笔交易
                },
                'score': 0.0,
            },

            {
                'name': 'concentration_limit',
                'description': '集中度限制 - 限制单股票/行业占比',
                'params': {
                    'method': 'concentration',
                    'max_stock_pct': 0.15,   # 单股票最大15%
                    'max_sector_pct': 0.40,  # 单行业最大40%
                },
                'score': 0.0,
            },

            {
                'name': 'adaptive_leverage',
                'description': '自适应杠杆 - 根据市场状态调整',
                'params': {
                    'method': 'adaptive_leverage',
                    'base_leverage': 1.0,
                    'max_leverage': 1.5,
                    'vol_threshold': 0.025,  # 波动率阈值
                },
                'score': 0.0,
            },
        ]

        return strategies

    def get_strategy(self, name: str) -> Dict:
        """获取策略"""
        for strategy in self.strategies:
            if strategy['name'] == name:
                return strategy
        return None

    def get_all_strategies(self) -> List[str]:
        """获取所有策略名称"""
        return [s['name'] for s in self.strategies]

    def evaluate_strategy(self, strategy_name: str, backtest_result: Dict) -> float:
        """
        评估仓位管理策略

        Args:
            strategy_name: 策略名称
            backtest_result: 回测结果

        Returns:
            评分
        """
        strategy = self.get_strategy(strategy_name)
        if not strategy:
            return -999.0

        # 提取关键指标
        total_return = backtest_result.get('total_return', 0)
        sharpe = backtest_result.get('sharpe_ratio', 0)
        max_dd = backtest_result.get('max_drawdown', 1)
        win_rate = backtest_result.get('win_rate', 0)

        # 计算评分（与parallel_executor中的逻辑一致）
        score = (
            total_return * 0.4 +
            sharpe * 0.1 * 0.3 +
            (1 - max_dd) * 0.2 +
            win_rate * 0.1
        )

        # 更新策略评分
        strategy['score'] = score

        # 记录历史
        self.optimization_history.append({
            'timestamp': datetime.now().isoformat(),
            'strategy': strategy_name,
            'score': score,
            'backtest_result': backtest_result,
        })

        # 更新最佳策略
        if score > self.best_score:
            self.best_score = score
            self.best_strategy = strategy_name

        return score

    def get_next_strategy_to_test(self) -> Dict:
        """
        获取下一个要测试的策略

        策略：
        - 70%概率：基于最佳策略进行参数微调
        - 30%概率：随机选择其他策略
        """
        import random

        if random.random() < 0.7 and self.best_strategy:
            # 微调最佳策略
            best = self.get_strategy(self.best_strategy)
            return self._perturb_strategy(best)
        else:
            # 随机选择
            return random.choice(self.strategies)

    def _perturb_strategy(self, strategy: Dict) -> Dict:
        """微调策略参数"""
        import copy
        import random

        perturbed = copy.deepcopy(strategy)
        params = perturbed['params']

        # 根据策略类型进行不同的微调
        method = params.get('method')

        if method == 'prob_weighted':
            # 调整权重范围
            if random.random() < 0.5:
                params['min_weight'] = random.choice([0.3, 0.5, 0.7])
            else:
                params['max_weight'] = random.choice([1.5, 2.0, 2.5])

        elif method == 'kelly':
            # 调整Kelly分数
            params['kelly_fraction'] = random.choice([0.25, 0.5, 0.75])
            params['max_position'] = random.choice([0.15, 0.20, 0.25])

        elif method == 'vol_adjusted':
            # 调整目标波动率
            params['target_vol'] = random.choice([0.015, 0.020, 0.025])

        elif method == 'dynamic':
            # 调整动态倍数
            params['win_multiplier'] = random.choice([1.1, 1.2, 1.3])
            params['loss_multiplier'] = random.choice([0.7, 0.8, 0.9])

        return perturbed

    def apply_position_sizing(self, predictions: List[Dict], portfolio_value: float,
                             strategy_name: str) -> List[Dict]:
        """
        应用仓位管理策略

        Args:
            predictions: 预测列表 [{symbol, prob}, ...]
            portfolio_value: 当前组合价值
            strategy_name: 策略名称

        Returns:
            增加了position_size字段的预测列表
        """
        strategy = self.get_strategy(strategy_name)
        if not strategy:
            # 默认等权重
            return self._equal_weight_sizing(predictions, portfolio_value)

        method = strategy['params'].get('method')

        if method == 'equal':
            return self._equal_weight_sizing(predictions, portfolio_value)

        elif method == 'prob_weighted':
            return self._prob_weighted_sizing(predictions, portfolio_value, strategy['params'])

        elif method == 'kelly':
            return self._kelly_sizing(predictions, portfolio_value, strategy['params'])

        elif method == 'vol_adjusted':
            return self._vol_adjusted_sizing(predictions, portfolio_value, strategy['params'])

        else:
            # 默认
            return self._equal_weight_sizing(predictions, portfolio_value)

    def _equal_weight_sizing(self, predictions: List[Dict], portfolio_value: float) -> List[Dict]:
        """等权重"""
        n = len(predictions)
        if n == 0:
            return []

        size_per_stock = portfolio_value / n

        for pred in predictions:
            pred['position_size'] = size_per_stock

        return predictions

    def _prob_weighted_sizing(self, predictions: List[Dict], portfolio_value: float,
                             params: Dict) -> List[Dict]:
        """概率加权"""
        min_weight = params.get('min_weight', 0.5)
        max_weight = params.get('max_weight', 2.0)

        # 计算权重
        total_weight = 0
        for pred in predictions:
            prob = pred.get('prob', 0.5)
            # 线性映射：0.5->min_weight, 1.0->max_weight
            weight = min_weight + (max_weight - min_weight) * (prob - 0.5) / 0.5
            pred['weight'] = max(min_weight, min(weight, max_weight))
            total_weight += pred['weight']

        # 归一化并分配资金
        for pred in predictions:
            pred['position_size'] = portfolio_value * (pred['weight'] / total_weight)

        return predictions

    def _kelly_sizing(self, predictions: List[Dict], portfolio_value: float,
                     params: Dict) -> List[Dict]:
        """Kelly公式"""
        kelly_fraction = params.get('kelly_fraction', 0.5)
        max_position = params.get('max_position', 0.2)

        # 简化：假设胜率=prob，盈亏比=1.5
        for pred in predictions:
            prob = pred.get('prob', 0.5)
            win_rate = prob
            payoff_ratio = 1.5

            # Kelly公式: f = (win_rate * payoff_ratio - (1 - win_rate)) / payoff_ratio
            kelly = (win_rate * payoff_ratio - (1 - win_rate)) / payoff_ratio
            kelly = max(0, min(kelly, max_position))

            # 使用Kelly分数的一半（保守）
            size = portfolio_value * kelly * kelly_fraction
            pred['position_size'] = size

        return predictions

    def _vol_adjusted_sizing(self, predictions: List[Dict], portfolio_value: float,
                            params: Dict) -> List[Dict]:
        """波动率调整（简化版本，实际需要历史波动率数据）"""
        target_vol = params.get('target_vol', 0.02)

        # 简化：假设所有股票波动率相同，等权重
        # 实际应该根据历史波动率调整
        return self._equal_weight_sizing(predictions, portfolio_value)

    def save_state(self):
        """保存状态"""
        state_file = self.output_dir / "position_mgmt_state.json"

        with open(state_file, 'w') as f:
            json.dump({
                'best_strategy': self.best_strategy,
                'best_score': self.best_score,
                'strategies': self.strategies,
                'optimization_history': self.optimization_history,
            }, f, indent=2, default=str)

        print(f"[OK] Position management state saved: {state_file}")

    def load_state(self):
        """加载状态"""
        state_file = self.output_dir / "position_mgmt_state.json"

        if not state_file.exists():
            return False

        with open(state_file, 'r') as f:
            state = json.load(f)

        self.best_strategy = state.get('best_strategy')
        self.best_score = state.get('best_score', -999.0)
        self.strategies = state.get('strategies', self.strategies)
        self.optimization_history = state.get('optimization_history', [])

        print(f"[OK] Position management state loaded: {state_file}")
        return True


if __name__ == "__main__":
    # 测试
    optimizer = PositionManagerOptimizer()

    print("=" * 60)
    print("仓位管理优化器")
    print("=" * 60)

    print(f"\n可用策略:")
    for strategy_name in optimizer.get_all_strategies():
        strategy = optimizer.get_strategy(strategy_name)
        print(f"  - {strategy_name}: {strategy['description']}")

    print(f"\n测试仓位分配:")
    predictions = [
        {'symbol': 'SHSE.600000', 'prob': 0.75},
        {'symbol': 'SHSE.600001', 'prob': 0.65},
        {'symbol': 'SHSE.600002', 'prob': 0.80},
    ]

    result = optimizer.apply_position_sizing(predictions, 1000000, 'prob_weighted')

    for pred in result:
        print(f"  {pred['symbol']}: prob={pred['prob']:.2f}, size=${pred['position_size']:,.0f}")
