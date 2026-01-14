"""
训练模块 - Phase 3: Model Training

使用GPU加速训练，支持混合精度、梯度累积等优化技术
"""

import sys
from pathlib import Path
from datetime import datetime
import json

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from pipeline.shared.config import (
    TRAIN_DATA_DIR, MODEL_CHECKPOINT_DIR, TRAIN_CONFIG, DEVICE,
    NUM_WORKERS, PIN_MEMORY, PREFETCH_FACTOR
)
from pipeline.shared.utils import (
    setup_logger, timer, memory_monitor, get_gpu_info, clear_gpu_memory
)


logger = setup_logger("training", MODEL_CHECKPOINT_DIR / "training.log")


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


class Trainer:
    """模型训练器 - 支持GPU加速和各种优化技术"""
    
    def __init__(self, config: dict | None = None):
        self.config = config or TRAIN_CONFIG
        self.device = DEVICE
        
        # 混合精度训练
        self.use_amp = self.config["use_amp"] and torch.cuda.is_available()
        self.scaler = GradScaler() if self.use_amp else None
        
        # 训练历史
        self.history = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
            "learning_rates": []
        }
        
        # 最佳模型跟踪
        self.best_val_acc = 0.0
        self.best_epoch = 0
        self.patience_counter = 0
        
        logger.info("="*60)
        logger.info(f"Device: {self.device}")
        if torch.cuda.is_available():
            gpu_info = get_gpu_info()
            logger.info(f"GPU: {gpu_info['device_name']}")
            logger.info(f"GPU Memory: {gpu_info['total_memory_gb']:.1f}GB")
            logger.info(f"Mixed Precision: {self.use_amp}")
        logger.info("="*60)
    
    @timer
    @memory_monitor
    def load_data(self) -> tuple[DataLoader, DataLoader, DataLoader]:
        """加载训练数据 - 优化内存使用"""
        logger.info("Loading training data...")
        
        # 从保存的训练数据加载
        train_path = TRAIN_DATA_DIR / "train.npz"
        val_path = TRAIN_DATA_DIR / "val.npz"
        test_path = TRAIN_DATA_DIR / "test.npz"
        
        if not all(p.exists() for p in [train_path, val_path, test_path]):
            raise FileNotFoundError("Training data not found. Run feature engineering first.")
        
        # 加载数据（使用内存映射减少内存占用）
        train_data = np.load(train_path, mmap_mode='r')
        val_data = np.load(val_path, mmap_mode='r')
        test_data = np.load(test_path, mmap_mode='r')
        
        # 转换为Tensor（复制到内存）
        X_train = torch.from_numpy(train_data['X'][:]).float()
        y_train = torch.from_numpy(train_data['y'][:]).long()
        X_val = torch.from_numpy(val_data['X'][:]).float()
        y_val = torch.from_numpy(val_data['y'][:]).long()
        X_test = torch.from_numpy(test_data['X'][:]).float()
        y_test = torch.from_numpy(test_data['y'][:]).long()
        
        logger.info(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
        
        # 创建数据集
        train_dataset = TensorDataset(X_train, y_train)
        val_dataset = TensorDataset(X_val, y_val)
        test_dataset = TensorDataset(X_test, y_test)
        
        # 创建DataLoader - 充分利用多核CPU和内存
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config["batch_size"],
            shuffle=True,
            num_workers=NUM_WORKERS,
            pin_memory=PIN_MEMORY,
            prefetch_factor=PREFETCH_FACTOR,
            persistent_workers=True if NUM_WORKERS > 0 else False
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config["batch_size"] * 2,  # 验证时可用更大batch
            shuffle=False,
            num_workers=NUM_WORKERS // 2,
            pin_memory=PIN_MEMORY,
            prefetch_factor=PREFETCH_FACTOR,
            persistent_workers=True if NUM_WORKERS > 0 else False
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.config["batch_size"] * 2,
            shuffle=False,
            num_workers=NUM_WORKERS // 2,
            pin_memory=PIN_MEMORY
        )
        
        return train_loader, val_loader, test_loader
    
    def create_model(self, input_size: int) -> nn.Module:
        """创建模型"""
        model = LSTMModel(
            input_size=input_size,
            hidden_size=self.config["hidden_size"],
            num_layers=self.config["num_layers"],
            dropout=self.config["dropout"],
            num_classes=self.config["num_classes"]
        )
        
        model = model.to(self.device)
        
        # 统计参数量
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Model parameters: {total_params:,} (trainable: {trainable_params:,})")
        
        return model
    
    @timer
    def train_epoch(self, model: nn.Module, train_loader: DataLoader, 
                   criterion: nn.Module, optimizer: torch.optim.Optimizer) -> tuple[float, float]:
        """训练一个epoch"""
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (X, y) in enumerate(train_loader):
            X, y = X.to(self.device), y.to(self.device)
            
            optimizer.zero_grad()
            
            # 混合精度训练
            if self.use_amp:
                with autocast():
                    outputs = model(X)
                    loss = criterion(outputs, y)
                
                self.scaler.scale(loss).backward()
                
                # 梯度裁剪
                self.scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                self.scaler.step(optimizer)
                self.scaler.update()
            else:
                outputs = model(X)
                loss = criterion(outputs, y)
                loss.backward()
                
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
            
            total_loss += loss.item()
            
            # 计算准确率
            _, predicted = outputs.max(1)
            correct += predicted.eq(y).sum().item()
            total += y.size(0)
            
            # 定期输出进度
            if (batch_idx + 1) % 50 == 0:
                logger.info(f"  Batch [{batch_idx+1}/{len(train_loader)}], "
                          f"Loss: {loss.item():.4f}, "
                          f"Acc: {100.*correct/total:.2f}%")
        
        avg_loss = total_loss / len(train_loader)
        accuracy = correct / total
        
        return avg_loss, accuracy
    
    @timer
    def evaluate(self, model: nn.Module, data_loader: DataLoader, 
                criterion: nn.Module) -> tuple[float, float, dict]:
        """评估模型"""
        model.eval()
        total_loss = 0
        correct = 0
        total = 0
        all_preds = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for X, y in data_loader:
                X, y = X.to(self.device), y.to(self.device)
                
                if self.use_amp:
                    with autocast():
                        outputs = model(X)
                        loss = criterion(outputs, y)
                else:
                    outputs = model(X)
                    loss = criterion(outputs, y)
                
                total_loss += loss.item()
                
                # 计算准确率
                probs = torch.softmax(outputs, dim=1)
                _, predicted = outputs.max(1)
                correct += predicted.eq(y).sum().item()
                total += y.size(0)
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(y.cpu().numpy())
                all_probs.extend(probs[:, 1].cpu().numpy())  # 正类概率
        
        avg_loss = total_loss / len(data_loader)
        accuracy = correct / total
        
        # 计算更多指标
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)
        
        # 正类精确率和召回率
        pos_mask = all_preds == 1
        true_pos_mask = all_labels == 1
        
        precision = (np.sum(all_preds[pos_mask] == all_labels[pos_mask]) / 
                    np.sum(pos_mask)) if np.sum(pos_mask) > 0 else 0
        recall = (np.sum(all_preds[true_pos_mask] == all_labels[true_pos_mask]) / 
                 np.sum(true_pos_mask)) if np.sum(true_pos_mask) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        metrics = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "positive_ratio": np.mean(all_preds)
        }
        
        return avg_loss, accuracy, metrics
    
    @timer
    @memory_monitor
    def train(self, train_loader: DataLoader, val_loader: DataLoader, 
             test_loader: DataLoader):
        """完整训练流程"""
        logger.info("="*60)
        logger.info("Starting Training")
        logger.info("="*60)
        
        # 创建模型
        input_size = train_loader.dataset[0][0].shape[-1]
        model = self.create_model(input_size)
        
        # 计算类别权重（处理不平衡）
        if self.config.get("use_class_weight", False):
            # 统计训练集标签分布
            all_labels = []
            for _, labels in train_loader:
                all_labels.extend(labels.numpy().tolist())
            label_counts = np.bincount(all_labels)
            total = sum(label_counts)
            # 权重 = total / (n_classes * count)
            class_weights = torch.tensor([total / (2.0 * c) for c in label_counts], 
                                        dtype=torch.float32).to(self.device)
            logger.info(f"Class weights: {class_weights.cpu().numpy()}")
            criterion = nn.CrossEntropyLoss(weight=class_weights)
        else:
            criterion = nn.CrossEntropyLoss()
        
        # 优化器
        if self.config["optimizer"] == "adam":
            optimizer = Adam(model.parameters(), 
                           lr=self.config["learning_rate"],
                           weight_decay=self.config["weight_decay"])
        else:
            optimizer = AdamW(model.parameters(),
                            lr=self.config["learning_rate"],
                            weight_decay=self.config["weight_decay"])
        
        # 学习率调度器
        if self.config["scheduler"] == "reduce_on_plateau":
            scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, 
                                         patience=5, verbose=True)
        else:
            scheduler = CosineAnnealingLR(optimizer, T_max=self.config["epochs"])
        
        # 训练循环
        for epoch in range(1, self.config["epochs"] + 1):
            logger.info(f"\nEpoch {epoch}/{self.config['epochs']}")
            logger.info("-" * 60)
            
            # 训练
            train_loss, train_acc = self.train_epoch(model, train_loader, 
                                                     criterion, optimizer)
            
            # 验证
            val_loss, val_acc, val_metrics = self.evaluate(model, val_loader, criterion)
            
            # 记录历史
            self.history["train_loss"].append(train_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_loss"].append(val_loss)
            self.history["val_acc"].append(val_acc)
            self.history["learning_rates"].append(optimizer.param_groups[0]['lr'])
            
            # 输出结果
            logger.info(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            logger.info(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            logger.info(f"Val Precision: {val_metrics['precision']:.4f}, "
                       f"Recall: {val_metrics['recall']:.4f}, F1: {val_metrics['f1']:.4f}")
            logger.info(f"Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
            
            # 更新学习率
            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(val_acc)
            else:
                scheduler.step()
            
            # 保存最佳模型
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.best_epoch = epoch
                self.patience_counter = 0
                
                self.save_checkpoint(model, optimizer, epoch, val_acc, val_metrics)
                logger.info(f"[BEST] New best model saved! Val Acc: {val_acc:.4f}")
            else:
                self.patience_counter += 1
                logger.info(f"Patience: {self.patience_counter}/{self.config['patience']}")
            
            # 早停
            if self.patience_counter >= self.config["patience"]:
                logger.info(f"\nEarly stopping at epoch {epoch}")
                break
            
            # 清理GPU内存
            if torch.cuda.is_available():
                clear_gpu_memory()
        
        # 加载最佳模型并在测试集上评估
        logger.info("\n" + "="*60)
        logger.info("Evaluating best model on test set")
        logger.info("="*60)
        
        best_checkpoint = torch.load(MODEL_CHECKPOINT_DIR / "best_model.pt", weights_only=False)
        model.load_state_dict(best_checkpoint['model_state_dict'])
        
        test_loss, test_acc, test_metrics = self.evaluate(model, test_loader, criterion)
        
        logger.info(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")
        logger.info(f"Test Precision: {test_metrics['precision']:.4f}, "
                   f"Recall: {test_metrics['recall']:.4f}, F1: {test_metrics['f1']:.4f}")
        
        # 保存训练历史
        self.save_history(test_acc, test_metrics)
        
        return model
    
    def save_checkpoint(self, model: nn.Module, optimizer: torch.optim.Optimizer,
                       epoch: int, val_acc: float, metrics: dict):
        """保存模型检查点"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_acc': val_acc,
            'metrics': metrics,
            'config': self.config,
            'timestamp': datetime.now().isoformat()
        }
        
        torch.save(checkpoint, MODEL_CHECKPOINT_DIR / "best_model.pt")
        
        # 保存最近的几个checkpoint
        torch.save(checkpoint, MODEL_CHECKPOINT_DIR / f"checkpoint_epoch_{epoch}.pt")
    
    def save_history(self, test_acc: float, test_metrics: dict):
        """保存训练历史"""
        history_data = {
            "config": self.config,
            "history": self.history,
            "best_epoch": self.best_epoch,
            "best_val_acc": self.best_val_acc,
            "test_acc": test_acc,
            "test_metrics": test_metrics,
            "timestamp": datetime.now().isoformat()
        }
        
        with open(MODEL_CHECKPOINT_DIR / "training_history.json", "w") as f:
            json.dump(history_data, f, indent=2)
        
        logger.info(f"Training history saved")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Model Training Pipeline")
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--hidden_size", type=int, default=None)
    parser.add_argument("--num_layers", type=int, default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=None)
    parser.add_argument("--patience", type=int, default=None)
    
    args = parser.parse_args()
    
    # 更新配置
    config = TRAIN_CONFIG.copy()
    if args.batch_size:
        config["batch_size"] = args.batch_size
    if args.epochs:
        config["epochs"] = args.epochs
    if args.learning_rate:
        config["learning_rate"] = args.learning_rate
    if args.hidden_size:
        config["hidden_size"] = args.hidden_size
    if args.num_layers:
        config["num_layers"] = args.num_layers
    if args.dropout:
        config["dropout"] = args.dropout
    if args.weight_decay:
        config["weight_decay"] = args.weight_decay
    if args.patience:
        config["patience"] = args.patience
    
    # 创建训练器
    trainer = Trainer(config)
    
    # 加载数据
    train_loader, val_loader, test_loader = trainer.load_data()
    
    # 训练模型
    trainer.train(train_loader, val_loader, test_loader)


if __name__ == "__main__":
    main()
