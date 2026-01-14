"""
策略执行模块 - 基于ML模型的交易策略

结合fund_max_1策略逻辑和ML预测进行交易决策
"""

import sys
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
import json

import numpy as np
import polars as pl
import torch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from pipeline.shared.config import (
    MODEL_CHECKPOINT_DIR, BACKTEST_RESULT_DIR, BACKTEST_CONFIG,
    CLEANED_DATA_DIR, FEATURE_DATA_DIR, DEVICE
)
from pipeline.shared.utils import setup_logger, timer
from pipeline.data_cleaning.features import FEATURE_COLS, STRATEGY_PARAMS, FeatureEngineer


logger = setup_logger("strategy_executor", BACKTEST_RESULT_DIR / "strategy.log")


@dataclass
class Signal:
    """交易信号"""
    symbol: str
    date: str
    signal_type: str  # "BUY" / "SELL" / "HOLD"
    ml_prob: float  # ML模型预测的上涨概率
    ml_pred: int    # ML模型预测类别 (0/1)
    price_breakout: bool  # 是否突破25日最高价
    volume_confirm: bool  # 成交量是否确认
    stop_loss_triggered: bool  # 是否触发止损
    confidence: float  # 综合置信度
    reason: str  # 信号原因


@dataclass 
class StrategyState:
    """策略状态"""
    on_position: bool = False
    entry_date: str = ""
    entry_price: float = 0.0
    max_price: float = 0.0  # 持仓期间最高价
    holding_days: int = 0
    current_profit: float = 0.0


class StrategyExecutor:
    """策略执行器 - 结合ML模型和策略规则"""
    
    def __init__(self, config: dict | None = None):
        self.config = config or BACKTEST_CONFIG
        self.device = DEVICE
        
        # 策略参数
        self.max_window = STRATEGY_PARAMS["max_window"]
        self.vol_avg_window = STRATEGY_PARAMS["vol_avg_window"]
        self.vol_ratio = STRATEGY_PARAMS["vol_ratio"]
        self.stop_loss_ratio = STRATEGY_PARAMS["stop_loss_ratio"]
        self.holding_days = STRATEGY_PARAMS["holding_days"]
        
        # ML配置
        self.min_probability = self.config.get("min_probability", 0.6)
        
        # 模型
        self.model = None
        self.X_mean = None
        self.X_std = None
        self.lookback = 60  # 回看天数
        
        # 特征工程器
        self.feature_engineer = FeatureEngineer()
        
        logger.info("="*60)
        logger.info("Strategy Executor initialized")
        logger.info(f"Max Window: {self.max_window} days")
        logger.info(f"Vol Avg Window: {self.vol_avg_window} days")
        logger.info(f"Vol Ratio Threshold: {self.vol_ratio}")
        logger.info(f"Stop Loss Ratio: {self.stop_loss_ratio}")
        logger.info(f"ML Min Probability: {self.min_probability}")
        logger.info("="*60)
    
    def load_model(self):
        """加载训练好的模型"""
        logger.info("Loading trained model...")
        
        checkpoint_path = MODEL_CHECKPOINT_DIR / "best_model.pt"
        if not checkpoint_path.exists():
            raise FileNotFoundError("No trained model found. Run training first.")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        model_config = checkpoint['config']
        
        # 加载标准化参数
        X_mean_path = MODEL_CHECKPOINT_DIR / "X_mean.npy"
        X_std_path = MODEL_CHECKPOINT_DIR / "X_std.npy"
        
        if X_mean_path.exists() and X_std_path.exists():
            self.X_mean = np.load(X_mean_path)
            self.X_std = np.load(X_std_path)
            input_size = len(self.X_mean)
        else:
            input_size = len(FEATURE_COLS)
        
        # 重建模型
        from pipeline.training.train import LSTMModel
        
        self.model = LSTMModel(
            input_size=input_size,
            hidden_size=model_config["hidden_size"],
            num_layers=model_config["num_layers"],
            dropout=model_config["dropout"],
            num_classes=model_config["num_classes"]
        )
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(self.device)
        self.model.eval()
        
        logger.info(f"Model loaded: epoch={checkpoint['epoch']}, "
                   f"val_acc={checkpoint['val_acc']:.4f}, "
                   f"input_size={input_size}")
        
        return {
            "epoch": checkpoint['epoch'],
            "val_acc": checkpoint['val_acc'],
            "input_size": input_size,
            "hidden_size": model_config["hidden_size"],
            "num_layers": model_config["num_layers"]
        }
    
    def predict(self, features: np.ndarray) -> tuple[int, float]:
        """使用模型预测
        
        Args:
            features: (seq_len, num_features) 特征数组
            
        Returns:
            (predicted_class, probability)
        """
        if self.model is None:
            raise RuntimeError("Model not loaded")
        
        # 标准化
        if self.X_mean is not None and self.X_std is not None:
            features = (features - self.X_mean) / (self.X_std + 1e-8)
        
        # 转为Tensor
        X = torch.from_numpy(features).float().unsqueeze(0).to(self.device)
        
        # 预测
        with torch.no_grad():
            outputs = self.model(X)
            probs = torch.softmax(outputs, dim=1)
            pred_class = outputs.argmax(dim=1).item()
            prob = probs[0, 1].item()  # 正类（盈利）概率
        
        return pred_class, prob
    
    def check_strategy_conditions(self, df: pl.DataFrame, idx: int) -> dict:
        """检查策略条件
        
        Args:
            df: 包含计算好特征的DataFrame
            idx: 当前行索引
            
        Returns:
            策略条件字典
        """
        row = df.row(idx, named=True)
        
        # 价格突破条件
        price_breakout = row.get("price_breakout", 0) == 1.0
        
        # 成交量确认条件
        vol_ratio = row.get("volume_ratio_10", 0)
        volume_confirm = vol_ratio >= self.vol_ratio
        
        # 止损条件
        price_vs_max = row.get("price_vs_max_25", 1.0)
        stop_loss_triggered = price_vs_max <= self.stop_loss_ratio
        
        # 开盘涨跌
        open_gap = row.get("open_gap", 0)
        open_up = open_gap > 0
        
        return {
            "price_breakout": price_breakout,
            "volume_confirm": volume_confirm,
            "stop_loss_triggered": stop_loss_triggered,
            "vol_ratio": vol_ratio,
            "price_vs_max": price_vs_max,
            "open_up": open_up,
            "open_gap": open_gap
        }
    
    def generate_signal(self, df: pl.DataFrame, idx: int, 
                       state: StrategyState) -> Signal:
        """生成交易信号
        
        Args:
            df: 特征数据
            idx: 当前索引
            state: 当前策略状态
            
        Returns:
            交易信号
        """
        row = df.row(idx, named=True)
        date = str(row["date"])
        symbol = row.get("symbol", "UNKNOWN")
        close_price = row["close"]
        
        # 检查策略条件
        conditions = self.check_strategy_conditions(df, idx)
        
        # 提取特征进行ML预测
        if idx >= self.lookback:
            features = df.select(FEATURE_COLS)[idx-self.lookback:idx].to_numpy()
            ml_pred, ml_prob = self.predict(features)
        else:
            ml_pred, ml_prob = 0, 0.5
        
        # 综合决策逻辑
        signal_type = "HOLD"
        reason = ""
        confidence = 0.0
        
        if not state.on_position:
            # 未持仓时，检查买入条件
            buy_conditions = [
                ml_prob >= self.min_probability,          # ML预测盈利概率高
                conditions["price_breakout"],             # 价格突破25日最高
                conditions["volume_confirm"],             # 成交量放大确认
            ]
            
            if all(buy_conditions):
                signal_type = "BUY"
                confidence = ml_prob
                reason = f"ML预测盈利概率{ml_prob:.1%}, 突破25日高点, 量比{conditions['vol_ratio']:.2f}"
            elif ml_prob >= 0.7 and conditions["volume_confirm"]:
                # 高概率但未突破，可考虑买入
                signal_type = "BUY"
                confidence = ml_prob * 0.8
                reason = f"ML高概率{ml_prob:.1%}, 量比{conditions['vol_ratio']:.2f}"
            else:
                reason = f"不满足买入条件: ML概率{ml_prob:.1%}"
                
        else:
            # 持仓时，检查卖出条件
            state.holding_days += 1
            current_profit = close_price / state.entry_price - 1
            state.current_profit = current_profit
            
            # 更新最高价
            if close_price > state.max_price:
                state.max_price = close_price
            
            sell_conditions = []
            
            # 止损
            if conditions["stop_loss_triggered"]:
                sell_conditions.append("触发止损")
                signal_type = "SELL"
                confidence = 0.9
                
            # 成交量萎缩
            elif conditions["vol_ratio"] < 1.0:
                sell_conditions.append("成交量萎缩")
                signal_type = "SELL"
                confidence = 0.7
                
            # 持仓到期
            elif state.holding_days >= self.holding_days:
                sell_conditions.append(f"持仓{state.holding_days}天到期")
                signal_type = "SELL"
                confidence = 0.8
                
            # ML预测转向
            elif ml_prob < 0.4:
                sell_conditions.append(f"ML预测转空{ml_prob:.1%}")
                signal_type = "SELL"
                confidence = 1 - ml_prob
            
            if sell_conditions:
                reason = ", ".join(sell_conditions)
            else:
                reason = f"继续持有: 盈利{current_profit:.1%}, 持仓{state.holding_days}天"
        
        return Signal(
            symbol=symbol,
            date=date,
            signal_type=signal_type,
            ml_prob=ml_prob,
            ml_pred=ml_pred,
            price_breakout=conditions["price_breakout"],
            volume_confirm=conditions["volume_confirm"],
            stop_loss_triggered=conditions["stop_loss_triggered"],
            confidence=confidence,
            reason=reason
        )
    
    def export_strategy(self, output_path: Path | None = None) -> dict:
        """导出策略配置和模型信息"""
        model_info = self.load_model()
        
        strategy_config = {
            "name": "ML_FundMax1_Strategy",
            "version": "1.0.0",
            "description": "基于LSTM模型和fund_max_1策略规则的混合策略",
            "created_at": datetime.now().isoformat(),
            
            "strategy_params": {
                "max_window": self.max_window,
                "vol_avg_window": self.vol_avg_window,
                "vol_ratio": self.vol_ratio,
                "stop_loss_ratio": self.stop_loss_ratio,
                "holding_days": self.holding_days,
            },
            
            "ml_config": {
                "model_type": "BiLSTM_Attention",
                "input_features": len(FEATURE_COLS),
                "feature_names": FEATURE_COLS,
                "lookback_days": self.lookback,
                "min_probability": self.min_probability,
                **model_info
            },
            
            "buy_conditions": [
                "ML预测盈利概率 >= 60%",
                "价格突破25日最高价",
                "成交量 >= 10日均量 * 1.2",
            ],
            
            "sell_conditions": [
                "价格跌破25日最高价 * 95% (止损)",
                "成交量 < 10日均量 (量能萎缩)",
                f"持仓超过{self.holding_days}天",
                "ML预测盈利概率 < 40%",
            ],
            
            "features": {
                "basic": ["ret_1", "ret_5", "ret_10", "ret_20"],
                "ma": ["ma_5_ratio", "ma_10_ratio", "ma_20_ratio", "ma_60_ratio"],
                "volatility": ["volatility_5", "volatility_10", "volatility_20"],
                "technical": ["rsi_14", "macd_dif", "macd_dea", "macd_hist", "bb_position"],
                "volume": ["volume_ratio_5", "volume_ratio_20", "turnover_ma_ratio"],
                "strategy": ["open_gap", "price_vs_max_25", "volume_ratio_10", 
                           "price_breakout", "stop_loss_signal"],
            }
        }
        
        if output_path is None:
            output_path = BACKTEST_RESULT_DIR / "strategy_config.json"
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(strategy_config, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Strategy exported to {output_path}")
        
        return strategy_config


def main():
    """导出策略"""
    executor = StrategyExecutor()
    config = executor.export_strategy()
    print(json.dumps(config, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
