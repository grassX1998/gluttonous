"""
LSTM 模型定义

包含两个模型：
1. SimpleLSTMModel - 简化版，用于快速实验
2. LSTMModel - 完整版，带双向LSTM和注意力机制
"""

import torch
import torch.nn as nn


class SimpleLSTMModel(nn.Module):
    """简化的LSTM模型 - 用于快速实验和回测

    这是一个轻量级模型，用于快速迭代场景。
    与 LSTMModel 相比，没有双向LSTM和注意力机制，计算更快。
    """

    def __init__(self, input_size: int, hidden_size: int = 128,
                 num_layers: int = 2, dropout: float = 0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1, :])


class LSTMModel(nn.Module):
    """优化的LSTM模型 - 支持双向和注意力机制"""

    def __init__(self, input_size: int, hidden_size: int = 128,
                 num_layers: int = 2, dropout: float = 0.3, num_classes: int = 2):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )

        # 注意力机制
        self.attention = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
            nn.Softmax(dim=1)
        )

        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.BatchNorm1d(hidden_size),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout / 2),
            nn.Linear(hidden_size // 2, num_classes)
        )

    def forward(self, x):
        # x: (batch, seq_len, features)
        lstm_out, _ = self.lstm(x)  # (batch, seq_len, hidden*2)

        # Attention
        attn_weights = self.attention(lstm_out)  # (batch, seq_len, 1)
        context = torch.sum(lstm_out * attn_weights, dim=1)  # (batch, hidden*2)

        # Classification
        out = self.classifier(context)  # (batch, num_classes)
        return out
