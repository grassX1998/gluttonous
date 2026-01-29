"""
MLP 执行器

封装 MLP 模型的训练和回测流程。
"""

import logging
from typing import Optional

from ..base import BaseExecutor
from ..config import MLPConfig
from .model import MLPModel
from pipeline.data_cleaning.features import FEATURE_COLS


class MLPExecutor(BaseExecutor):
    """MLP 执行器"""

    def __init__(self, config: Optional[MLPConfig] = None,
                 logger: Optional[logging.Logger] = None):
        """
        初始化执行器

        Args:
            config: MLP 配置，如果为 None 则使用默认配置
            logger: 日志记录器
        """
        if config is None:
            config = MLPConfig()

        # 添加 strategy_name 属性
        config.strategy_name = "mlp"

        super().__init__(config, logger)

        # 保存输入维度
        self.input_dim = len(FEATURE_COLS)

    def create_model(self) -> MLPModel:
        """创建 MLP 模型实例"""
        return MLPModel(
            config=self.config,
            input_dim=self.input_dim,
        )
