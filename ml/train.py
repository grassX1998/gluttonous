"""
模型训练脚本
"""

import os
import sys
from pathlib import Path
from datetime import datetime
import argparse

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from ml.data.dataset import build_full_dataset, create_dataloaders
from ml.features.technical import FEATURE_COLS
from ml.models.lstm import create_model


# 配置
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_DIR = Path(__file__).parent / "checkpoints"
MODEL_DIR.mkdir(exist_ok=True)


def train_epoch(model, train_loader, criterion, optimizer, device):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for X, y in train_loader:
        X, y = X.to(device), y.to(device)
        
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        total_loss += loss.item()
        
        # 计算准确率（分类任务）
        if outputs.dim() > 1:
            _, predicted = outputs.max(1)
            correct += predicted.eq(y).sum().item()
            total += y.size(0)
    
    avg_loss = total_loss / len(train_loader)
    accuracy = correct / total if total > 0 else 0
    
    return avg_loss, accuracy


def evaluate(model, data_loader, criterion, device):
    """评估模型"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for X, y in data_loader:
            X, y = X.to(device), y.to(device)
            
            outputs = model(X)
            loss = criterion(outputs, y)
            
            total_loss += loss.item()
            
            # 计算准确率（分类任务）
            if outputs.dim() > 1:
                _, predicted = outputs.max(1)
                correct += predicted.eq(y).sum().item()
                total += y.size(0)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(y.cpu().numpy())
    
    avg_loss = total_loss / len(data_loader)
    accuracy = correct / total if total > 0 else 0
    
    # 计算更多指标
    if all_preds:
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        
        # 正类预测精度
        pos_mask = all_preds == 1
        if pos_mask.sum() > 0:
            precision = (all_labels[pos_mask] == 1).mean()
        else:
            precision = 0
        
        # 正类召回率
        true_pos_mask = all_labels == 1
        if true_pos_mask.sum() > 0:
            recall = (all_preds[true_pos_mask] == 1).mean()
        else:
            recall = 0
    else:
        precision = recall = 0
    
    return avg_loss, accuracy, precision, recall


def train(config: dict):
    """完整训练流程"""
    print("=" * 60)
    print("Training Configuration:")
    for k, v in config.items():
        print(f"  {k}: {v}")
    print("=" * 60)
    
    # 构建数据集
    print("\nBuilding dataset...")
    X, y = build_full_dataset(
        start_date=config["start_date"],
        end_date=config["end_date"],
        lookback=config["lookback"],
        predict_days=config["predict_days"],
        max_stocks=config["max_stocks"]
    )
    
    # 数据标准化
    print("Normalizing features...")
    X_mean = X.mean(axis=(0, 1), keepdims=True)
    X_std = X.std(axis=(0, 1), keepdims=True) + 1e-8
    X = (X - X_mean) / X_std
    
    # 保存标准化参数
    np.save(MODEL_DIR / "X_mean.npy", X_mean)
    np.save(MODEL_DIR / "X_std.npy", X_std)
    
    # 创建数据加载器
    print("Creating dataloaders...")
    train_loader, val_loader, test_loader = create_dataloaders(
        X, y,
        train_ratio=config["train_ratio"],
        val_ratio=config["val_ratio"],
        batch_size=config["batch_size"],
        classification=config["classification"]
    )
    
    # 创建模型
    print(f"Creating model: {config['model_type']}...")
    input_size = len(FEATURE_COLS)
    model = create_model(config["model_type"], input_size, 
                         hidden_size=config["hidden_size"],
                         num_layers=config["num_layers"],
                         dropout=config["dropout"])
    model = model.to(DEVICE)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # 损失函数和优化器
    if config["classification"]:
        # 处理类别不平衡
        pos_weight = torch.tensor([(y <= 0.02).sum() / (y > 0.02).sum()]).to(DEVICE)
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.MSELoss()
    
    optimizer = Adam(model.parameters(), lr=config["learning_rate"], weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    # 训练循环
    print("\nStarting training...")
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(config["epochs"]):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, DEVICE)
        val_loss, val_acc, val_precision, val_recall = evaluate(model, val_loader, criterion, DEVICE)
        
        scheduler.step(val_loss)
        
        print(f"Epoch {epoch+1}/{config['epochs']}: "
              f"Train Loss={train_loss:.4f}, Train Acc={train_acc:.2%} | "
              f"Val Loss={val_loss:.4f}, Val Acc={val_acc:.2%}, "
              f"Precision={val_precision:.2%}, Recall={val_recall:.2%}")
        
        # 早停
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # 保存最佳模型
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'config': config,
            }, MODEL_DIR / "best_model.pt")
        else:
            patience_counter += 1
            if patience_counter >= config["early_stop_patience"]:
                print(f"Early stopping at epoch {epoch+1}")
                break
    
    # 加载最佳模型并在测试集上评估
    print("\nEvaluating on test set...")
    checkpoint = torch.load(MODEL_DIR / "best_model.pt")
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_loss, test_acc, test_precision, test_recall = evaluate(model, test_loader, criterion, DEVICE)
    print(f"Test Results: Loss={test_loss:.4f}, Acc={test_acc:.2%}, "
          f"Precision={test_precision:.2%}, Recall={test_recall:.2%}")
    
    return model


def main():
    parser = argparse.ArgumentParser(description="Train stock prediction model")
    parser.add_argument("--model", type=str, default="lstm_classifier", 
                        choices=["lstm_classifier", "lstm_regressor", "transformer"])
    parser.add_argument("--start_date", type=str, default="2025-01-01")
    parser.add_argument("--end_date", type=str, default="2025-10-31")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--max_stocks", type=int, default=200)
    
    args = parser.parse_args()
    
    config = {
        "model_type": args.model,
        "start_date": args.start_date,
        "end_date": args.end_date,
        "lookback": 20,
        "predict_days": 1,
        "max_stocks": args.max_stocks,
        "train_ratio": 0.7,
        "val_ratio": 0.15,
        "batch_size": args.batch_size,
        "hidden_size": 64,
        "num_layers": 2,
        "dropout": 0.3,
        "learning_rate": args.lr,
        "epochs": args.epochs,
        "early_stop_patience": 10,
        "classification": args.model != "lstm_regressor",
    }
    
    train(config)


if __name__ == "__main__":
    main()
