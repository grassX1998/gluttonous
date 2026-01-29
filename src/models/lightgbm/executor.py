"""
LightGBM 执行器

封装 LightGBM 模型的训练和回测流程。
"""

import logging
from typing import Optional

from ..base import BaseExecutor
from ..config import LightGBMConfig
from .model import LightGBMModel
from pipeline.data_cleaning.features import FEATURE_COLS


class LightGBMExecutor(BaseExecutor):
    """LightGBM 执行器"""

    def __init__(self, config: Optional[LightGBMConfig] = None,
                 logger: Optional[logging.Logger] = None):
        """
        初始化执行器

        Args:
            config: LightGBM 配置，如果为 None 则使用默认配置
            logger: 日志记录器
        """
        if config is None:
            config = LightGBMConfig()

        # 添加 strategy_name 属性
        config.strategy_name = "lightgbm"

        super().__init__(config, logger)

    def create_model(self) -> LightGBMModel:
        """创建 LightGBM 模型实例"""
        return LightGBMModel(
            config=self.config,
            feature_names=list(FEATURE_COLS)
        )
