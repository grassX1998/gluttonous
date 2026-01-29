"""
集成执行器

封装多模型集成的训练和回测流程。
"""

import logging
from typing import Optional, List

from ..base import BaseExecutor
from ..config import EnsembleConfig, LightGBMConfig, MLPConfig
from ..lightgbm.model import LightGBMModel
from ..mlp.model import MLPModel
from .voting import VotingEnsemble
from pipeline.data_cleaning.features import FEATURE_COLS


class EnsembleExecutor(BaseExecutor):
    """集成执行器"""

    def __init__(self, config: Optional[EnsembleConfig] = None,
                 logger: Optional[logging.Logger] = None):
        """
        初始化执行器

        Args:
            config: 集成配置，如果为 None 则使用默认配置
            logger: 日志记录器
        """
        if config is None:
            config = EnsembleConfig()

        # 添加 strategy_name 属性
        config.strategy_name = "ensemble"

        super().__init__(config, logger)

        # 特征数量
        self.input_dim = len(FEATURE_COLS)

        # 子模型配置
        self.lightgbm_config = LightGBMConfig.with_params(
            train_days=config.train_days,
            val_days=config.val_days,
            sample_ratio=config.sample_ratio,
            random_seed=config.random_seed,
        )
        self.mlp_config = MLPConfig.with_params(
            train_days=config.train_days,
            val_days=config.val_days,
            sample_ratio=config.sample_ratio,
            random_seed=config.random_seed,
        )

    def create_model(self) -> VotingEnsemble:
        """创建集成模型实例"""
        models = []

        for model_name in self.config.models:
            if model_name == 'lightgbm':
                model = LightGBMModel(
                    config=self.lightgbm_config,
                    feature_names=list(FEATURE_COLS),
                )
            elif model_name == 'mlp':
                model = MLPModel(
                    config=self.mlp_config,
                    input_dim=self.input_dim,
                )
            else:
                self.logger.warning(f"未知模型类型: {model_name}，跳过")
                continue

            models.append(model)

        return VotingEnsemble(
            models=models,
            weights=self.config.weights,
            voting=self.config.voting,
        )
