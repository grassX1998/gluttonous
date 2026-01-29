"""
实验注册表管理器

管理实验的创建、记录、查询和持久化。

实验注册表存储位置: logs/experiments_registry.json

实验 ID 格式: exp_{YYYYMMDD}_{HHMMSS}_{hash6}
例如: exp_20260117_143022_a1b2c3
"""

import json
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

from pipeline.shared.config import PROJECT_ROOT


# 注册表存储位置
REGISTRY_PATH = PROJECT_ROOT / "logs" / "experiments_registry.json"


class ExperimentRegistry:
    """实验注册表管理器"""

    def __init__(self):
        """初始化注册表"""
        self._ensure_registry_exists()
        self._registry = self._load_registry()

    def _ensure_registry_exists(self):
        """确保注册表文件存在"""
        REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
        if not REGISTRY_PATH.exists():
            self._save_registry({
                "experiments": [],
                "last_updated": datetime.now().isoformat()
            })

    def _load_registry(self) -> Dict:
        """加载注册表"""
        try:
            with open(REGISTRY_PATH, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            return {"experiments": [], "last_updated": datetime.now().isoformat()}

    def _save_registry(self, data: Dict = None):
        """保存注册表"""
        if data is None:
            data = self._registry
        data["last_updated"] = datetime.now().isoformat()

        with open(REGISTRY_PATH, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    @staticmethod
    def generate_experiment_id() -> str:
        """
        生成唯一的实验 ID

        格式: exp_{YYYYMMDD}_{HHMMSS}_{hash6}

        Returns:
            实验 ID 字符串
        """
        import uuid
        now = datetime.now()
        timestamp = now.strftime("%Y%m%d_%H%M%S")

        # 使用 uuid4 确保唯一性
        hash_value = uuid.uuid4().hex[:6]

        return f"exp_{timestamp}_{hash_value}"

    def create_experiment(
        self,
        strategy: str,
        start_date: str,
        end_date: str,
        config_snapshot: Dict[str, Any]
    ) -> str:
        """
        创建新实验记录

        Args:
            strategy: 策略名称
            start_date: 开始日期
            end_date: 结束日期
            config_snapshot: 配置快照

        Returns:
            实验 ID
        """
        experiment_id = self.generate_experiment_id()
        date_str = datetime.now().strftime("%Y-%m-%d")

        # 实验结果目录
        results_path = f"logs/{date_str}/experiments/{experiment_id}/"

        experiment = {
            "id": experiment_id,
            "strategy": strategy,
            "start_date": start_date,
            "end_date": end_date,
            "config_snapshot": config_snapshot,
            "results_path": results_path,
            "metrics": None,  # 回测完成后填充
            "status": "running",
            "created_at": datetime.now().isoformat(),
            "completed_at": None
        }

        self._registry["experiments"].append(experiment)
        self._save_registry()

        return experiment_id

    def update_experiment(
        self,
        experiment_id: str,
        metrics: Dict[str, float] = None,
        status: str = None,
        extra_data: Dict = None
    ):
        """
        更新实验记录

        Args:
            experiment_id: 实验 ID
            metrics: 回测指标
            status: 状态 (running/completed/failed)
            extra_data: 额外数据
        """
        for exp in self._registry["experiments"]:
            if exp["id"] == experiment_id:
                if metrics is not None:
                    exp["metrics"] = metrics
                if status is not None:
                    exp["status"] = status
                    if status == "completed":
                        exp["completed_at"] = datetime.now().isoformat()
                if extra_data is not None:
                    for key, value in extra_data.items():
                        exp[key] = value
                break

        self._save_registry()

    def get_experiment(self, experiment_id: str) -> Optional[Dict]:
        """
        获取实验记录

        Args:
            experiment_id: 实验 ID

        Returns:
            实验记录字典，如果不存在返回 None
        """
        for exp in self._registry["experiments"]:
            if exp["id"] == experiment_id:
                return exp
        return None

    def list_experiments(
        self,
        strategy: str = None,
        status: str = None,
        limit: int = None
    ) -> List[Dict]:
        """
        列出实验记录

        Args:
            strategy: 按策略过滤
            status: 按状态过滤
            limit: 返回数量限制

        Returns:
            实验记录列表
        """
        experiments = self._registry["experiments"].copy()

        # 按创建时间倒序排列
        experiments.sort(key=lambda x: x["created_at"], reverse=True)

        # 过滤
        if strategy is not None:
            experiments = [e for e in experiments if e["strategy"] == strategy]
        if status is not None:
            experiments = [e for e in experiments if e["status"] == status]

        # 限制数量
        if limit is not None:
            experiments = experiments[:limit]

        return experiments

    def get_best_experiment(self, strategy: str, metric: str = "sharpe_ratio") -> Optional[Dict]:
        """
        获取指定策略的最佳实验

        Args:
            strategy: 策略名称
            metric: 评判指标 (sharpe_ratio/total_return/等)

        Returns:
            最佳实验记录，如果不存在返回 None
        """
        experiments = [
            e for e in self._registry["experiments"]
            if e["strategy"] == strategy
            and e["status"] == "completed"
            and e.get("metrics") is not None
        ]

        if not experiments:
            return None

        # 按指标排序
        try:
            experiments.sort(
                key=lambda x: x["metrics"].get(metric, 0),
                reverse=True
            )
            return experiments[0]
        except Exception:
            return None

    def get_experiment_results_dir(self, experiment_id: str) -> Path:
        """
        获取实验结果目录

        Args:
            experiment_id: 实验 ID

        Returns:
            结果目录路径
        """
        exp = self.get_experiment(experiment_id)
        if exp is None:
            # 如果实验不存在，返回默认路径
            date_str = datetime.now().strftime("%Y-%m-%d")
            return PROJECT_ROOT / "logs" / date_str / "experiments" / experiment_id

        return PROJECT_ROOT / exp["results_path"]

    def cleanup_old_experiments(self, days: int = 30, keep_best: int = 5):
        """
        清理旧实验记录

        Args:
            days: 保留最近 N 天的实验
            keep_best: 每个策略至少保留 N 个最佳实验
        """
        from datetime import timedelta

        cutoff_date = (datetime.now() - timedelta(days=days)).isoformat()

        # 收集每个策略的最佳实验
        best_by_strategy = {}
        for exp in self._registry["experiments"]:
            strategy = exp["strategy"]
            if strategy not in best_by_strategy:
                best_by_strategy[strategy] = []

            if exp["status"] == "completed" and exp.get("metrics"):
                best_by_strategy[strategy].append(exp)

        # 对每个策略按 sharpe_ratio 排序，保留 top N
        best_ids = set()
        for strategy, exps in best_by_strategy.items():
            exps.sort(
                key=lambda x: x["metrics"].get("sharpe_ratio", 0),
                reverse=True
            )
            for exp in exps[:keep_best]:
                best_ids.add(exp["id"])

        # 过滤实验
        new_experiments = []
        for exp in self._registry["experiments"]:
            # 保留最佳实验
            if exp["id"] in best_ids:
                new_experiments.append(exp)
                continue
            # 保留最近的实验
            if exp["created_at"] >= cutoff_date:
                new_experiments.append(exp)
                continue

        self._registry["experiments"] = new_experiments
        self._save_registry()


# 全局实例
_registry_instance = None


def get_registry() -> ExperimentRegistry:
    """获取全局注册表实例"""
    global _registry_instance
    if _registry_instance is None:
        _registry_instance = ExperimentRegistry()
    return _registry_instance
