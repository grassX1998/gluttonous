"""
LSTM模型定义
"""

import torch
import torch.nn as nn


class LSTMClassifier(nn.Module):
    """LSTM分类模型 - 预测涨跌"""
    
    def __init__(self, input_size: int, hidden_size: int = 64, num_layers: int = 2, 
                 dropout: float = 0.3, num_classes: int = 2):
        super().__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        self.attention = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
            nn.Softmax(dim=1)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes)
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


class LSTMRegressor(nn.Module):
    """LSTM回归模型 - 预测收益率"""
    
    def __init__(self, input_size: int, hidden_size: int = 64, num_layers: int = 2, 
                 dropout: float = 0.3):
        super().__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        self.regressor = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1)
        )
    
    def forward(self, x):
        # x: (batch, seq_len, features)
        lstm_out, _ = self.lstm(x)  # (batch, seq_len, hidden*2)
        
        # 使用最后一个时间步
        last_out = lstm_out[:, -1, :]  # (batch, hidden*2)
        
        # Regression
        out = self.regressor(last_out)  # (batch, 1)
        return out.squeeze(-1)


class TransformerClassifier(nn.Module):
    """Transformer分类模型"""
    
    def __init__(self, input_size: int, d_model: int = 64, nhead: int = 4, 
                 num_layers: int = 2, dropout: float = 0.3, num_classes: int = 2):
        super().__init__()
        
        self.input_proj = nn.Linear(input_size, d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes)
        )
    
    def forward(self, x):
        # x: (batch, seq_len, features)
        x = self.input_proj(x)  # (batch, seq_len, d_model)
        x = self.transformer(x)  # (batch, seq_len, d_model)
        
        # Global average pooling
        x = x.mean(dim=1)  # (batch, d_model)
        
        out = self.classifier(x)  # (batch, num_classes)
        return out


def create_model(model_type: str, input_size: int, **kwargs) -> nn.Module:
    """创建模型"""
    if model_type == "lstm_classifier":
        return LSTMClassifier(input_size, **kwargs)
    elif model_type == "lstm_regressor":
        return LSTMRegressor(input_size, **kwargs)
    elif model_type == "transformer":
        return TransformerClassifier(input_size, **kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
