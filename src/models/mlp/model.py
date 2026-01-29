"""
MLP 模型实现

简单的前馈神经网络，使用高 dropout 和 batch normalization 防止过拟合。
"""

from typing import Dict, Optional, List
from pathlib import Path
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from ..base import BaseModel
from ..config import MLPConfig


class SimpleMLP(nn.Module):
    """简单前馈网络（2-3层，高正则化）"""

    def __init__(self, input_dim: int, hidden_sizes: List[int] = [64, 32],
                 dropout: float = 0.5, batch_norm: bool = True):
        """
        初始化网络

        Args:
            input_dim: 输入维度
            hidden_sizes: 隐藏层大小列表
            dropout: Dropout 比例
            batch_norm: 是否使用 Batch Normalization
        """
        super().__init__()

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_sizes:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        # 输出层（二分类）
        layers.append(nn.Linear(prev_dim, 2))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        return self.net(x)


class MLPModel(BaseModel):
    """MLP 模型封装"""

    def __init__(self, config: MLPConfig, input_dim: int):
        """
        初始化模型

        Args:
            config: 模型配置
            input_dim: 输入特征维度
        """
        self.config = config
        self.input_dim = input_dim
        self.model: Optional[SimpleMLP] = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def fit(self, X: np.ndarray, y: np.ndarray,
            X_val: Optional[np.ndarray] = None,
            y_val: Optional[np.ndarray] = None) -> Dict:
        """
        训练模型

        Args:
            X: 训练特征
            y: 训练标签
            X_val: 验证特征
            y_val: 验证标签

        Returns:
            训练指标
        """
        # 创建模型
        self.model = SimpleMLP(
            input_dim=self.input_dim,
            hidden_sizes=self.config.hidden_sizes,
            dropout=self.config.dropout,
            batch_norm=self.config.batch_norm,
        ).to(self.device)

        # 创建数据集
        train_dataset = TensorDataset(
            torch.from_numpy(X).float(),
            torch.from_numpy(y).long()
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            pin_memory=(self.device.type == 'cuda'),
        )

        val_loader = None
        if X_val is not None and y_val is not None:
            val_dataset = TensorDataset(
                torch.from_numpy(X_val).float(),
                torch.from_numpy(y_val).long()
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.config.batch_size,
                pin_memory=(self.device.type == 'cuda'),
            )

        # 优化器和损失函数
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        criterion = nn.CrossEntropyLoss()

        # 学习率调度器
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=2, verbose=False
        )

        # 早停
        best_val_loss = float('inf')
        best_val_acc = 0.0
        best_state = None
        patience_counter = 0

        # 训练循环
        for epoch in range(self.config.epochs):
            # 训练阶段
            self.model.train()
            train_loss = 0.0
            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            train_loss /= len(train_loader)

            # 验证阶段
            if val_loader is not None:
                self.model.eval()
                val_loss = 0.0
                correct = 0
                total = 0

                with torch.no_grad():
                    for X_batch, y_batch in val_loader:
                        X_batch = X_batch.to(self.device)
                        y_batch = y_batch.to(self.device)

                        outputs = self.model(X_batch)
                        loss = criterion(outputs, y_batch)
                        val_loss += loss.item()

                        _, predicted = outputs.max(1)
                        total += y_batch.size(0)
                        correct += predicted.eq(y_batch).sum().item()

                val_loss /= len(val_loader)
                val_acc = correct / total if total > 0 else 0

                # 学习率调度
                scheduler.step(val_loss)

                # 早停检查
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_val_acc = val_acc
                    best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= self.config.early_stop_patience:
                        break

        # 加载最佳模型
        if best_state is not None:
            self.model.load_state_dict(best_state)

        return {
            'best_val_acc': float(best_val_acc),
            'best_val_loss': float(best_val_loss),
            'epochs_trained': epoch + 1,
        }

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        预测概率

        Args:
            X: 特征矩阵

        Returns:
            概率矩阵 (n_samples, 2)
        """
        if self.model is None:
            raise RuntimeError("模型尚未训练")

        self.model.eval()
        X_tensor = torch.from_numpy(X).float().to(self.device)

        # 批量预测
        batch_size = 10000
        all_probs = []

        with torch.no_grad():
            for i in range(0, len(X_tensor), batch_size):
                batch = X_tensor[i:i+batch_size]
                outputs = self.model(batch)
                probs = torch.softmax(outputs, dim=1)
                all_probs.append(probs.cpu().numpy())

        return np.vstack(all_probs)

    def save(self, path: str) -> None:
        """保存模型"""
        if self.model is None:
            raise RuntimeError("模型尚未训练")

        torch.save({
            'model_state': self.model.state_dict(),
            'input_dim': self.input_dim,
            'hidden_sizes': self.config.hidden_sizes,
            'dropout': self.config.dropout,
            'batch_norm': self.config.batch_norm,
        }, path)

    def load(self, path: str) -> None:
        """加载模型"""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)

        self.input_dim = checkpoint['input_dim']
        self.model = SimpleMLP(
            input_dim=self.input_dim,
            hidden_sizes=checkpoint['hidden_sizes'],
            dropout=checkpoint['dropout'],
            batch_norm=checkpoint['batch_norm'],
        ).to(self.device)

        self.model.load_state_dict(checkpoint['model_state'])
