"""
策略归档脚本

将成功的策略完整归档，包括代码、配置、结果和文档。

Usage:
    python scripts/archive_strategy.py --version v0.3 --strategy expanding_window --result_file path/to/result.json
"""

import sys
import json
import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def archive_strategy(version: str, strategy: str, result_file: Path):
    """
    归档策略的完整信息

    Args:
        version: 版本号，如 v0.3
        strategy: 策略名称，如 expanding_window
        result_file: 实验结果文件路径
    """
    print(f"\n{'='*60}")
    print(f"策略归档工具 - {version}")
    print(f"{'='*60}\n")

    # 创建归档目录
    archive_dir = PROJECT_ROOT / "archive" / version
    archive_dir.mkdir(parents=True, exist_ok=True)
    print(f"创建归档目录: {archive_dir}")

    # 创建子目录
    (archive_dir / "code").mkdir(exist_ok=True)
    (archive_dir / "config").mkdir(exist_ok=True)
    (archive_dir / "data").mkdir(exist_ok=True)
    (archive_dir / "results").mkdir(exist_ok=True)
    (archive_dir / "docs").mkdir(exist_ok=True)
    print("创建子目录: code, config, data, results, docs")

    # 1. 复制代码
    print("\n[1/7] 复制代码快照...")
    copy_code(archive_dir)

    # 2. 保存配置
    print("[2/7] 保存配置文件...")
    save_config(archive_dir, strategy)

    # 3. 复制结果
    print("[3/7] 复制实验结果...")
    copy_results(archive_dir, strategy, result_file)

    # 4. 加载结果数据
    print("[4/7] 加载结果数据...")
    result_data = load_result_data(result_file)

    # 5. 生成策略文档
    print("[5/7] 生成策略文档...")
    generate_strategy_doc(archive_dir, version, strategy, result_data)

    # 6. 生成 README
    print("[6/7] 生成 README...")
    generate_readme(archive_dir, version, strategy, result_data)

    # 7. 保存依赖清单
    print("[7/7] 保存依赖清单...")
    save_requirements(archive_dir)

    print(f"\n{'='*60}")
    print(f"✅ 策略归档完成: {archive_dir}")
    print(f"{'='*60}\n")


def copy_code(archive_dir: Path):
    """复制代码快照"""
    # 复制 src/lstm
    if (PROJECT_ROOT / "src" / "lstm").exists():
        shutil.copytree(
            PROJECT_ROOT / "src" / "lstm",
            archive_dir / "code" / "lstm",
            dirs_exist_ok=True,
            ignore=shutil.ignore_patterns("data", "__pycache__", "*.pyc")
        )
        print("  ✓ src/lstm/ 已复制")

    # 复制 pipeline
    if (PROJECT_ROOT / "pipeline").exists():
        shutil.copytree(
            PROJECT_ROOT / "pipeline",
            archive_dir / "code" / "pipeline",
            dirs_exist_ok=True,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc")
        )
        print("  ✓ pipeline/ 已复制")


def save_config(archive_dir: Path, strategy: str):
    """保存配置文件"""
    try:
        from src.lstm.config import (
            MODEL_CONFIG, TRADING_CONFIG,
            ALL_STRATEGY_CONFIGS
        )

        # 获取策略配置
        strategy_config_class = ALL_STRATEGY_CONFIGS.get(strategy)
        if strategy_config_class:
            strategy_config_obj = strategy_config_class()
            strategy_config = {
                "strategy_name": strategy_config_obj.strategy_name,
                "description": strategy_config_obj.description,
            }
            # 添加其他配置属性
            for attr in dir(strategy_config_obj):
                if not attr.startswith('_') and attr not in ['strategy_name', 'description']:
                    val = getattr(strategy_config_obj, attr)
                    if not callable(val):
                        strategy_config[attr] = val
        else:
            strategy_config = {"error": "Strategy config not found"}

        # 保存完整配置
        config_data = {
            "model": MODEL_CONFIG,
            "trading": TRADING_CONFIG,
            "strategy": strategy_config,
        }

        config_file = archive_dir / "config" / "full_config.json"
        with open(config_file, "w", encoding="utf-8") as f:
            json.dump(config_data, f, indent=2, ensure_ascii=False)

        print(f"  ✓ 配置已保存: {config_file.name}")

    except Exception as e:
        print(f"  ⚠ 配置保存失败: {e}")


def copy_results(archive_dir: Path, strategy: str, result_file: Path):
    """复制实验结果"""
    if result_file.exists():
        dest_file = archive_dir / "results" / f"{strategy}_result.json"
        shutil.copy(result_file, dest_file)
        print(f"  ✓ 结果已复制: {dest_file.name}")
    else:
        print(f"  ⚠ 结果文件不存在: {result_file}")


def load_result_data(result_file: Path) -> Dict[str, Any]:
    """加载结果数据"""
    if result_file.exists():
        with open(result_file, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def generate_strategy_doc(archive_dir: Path, version: str, strategy: str, result_data: Dict):
    """生成完整的策略文档"""
    metrics = result_data.get("metrics", {})

    strategy_doc = f"""# 策略文档 - {version}

**策略名称**: {strategy}
**归档日期**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 策略概述

{result_data.get('config', {}).get('description', '扩展窗口训练策略')}

## 性能指标

### 收益指标

- **总收益率**: {metrics.get('total_return', 0)*100:.2f}%
- **年化收益率**: {metrics.get('annual_return', 0)*100:.2f}%
- **夏普比率**: {metrics.get('sharpe_ratio', 0):.3f}

### 风险指标

- **最大回撤**: {metrics.get('max_drawdown', 0)*100:.2f}%
- **胜率**: {metrics.get('win_rate', 0)*100:.2f}%

### 交易统计

- **交易次数**: {metrics.get('n_trades', 0)}
- **交易天数**: {metrics.get('n_days', 0)}

## 模型配置

详见 `config/full_config.json`

## 复现步骤

```bash
# 1. 数据清洗
python -m pipeline.data_cleaning.clean

# 2. 特征工程
python -m pipeline.data_cleaning.features

# 3. 数据校验
python -m pipeline.data_validation.validate

# 4. 运行实验
python src/lstm/scripts/run_experiments.py \\
    --strategies {strategy} \\
    --calculate_metrics
```

## 关键发现

### 优势

- 策略收益稳定
- 回撤可控
- 交易频率合理

### 局限

- 依赖历史数据质量
- 需要定期重训练
- 计算资源消耗较大

## 改进方向

1. 优化特征工程
2. 调整样本权重策略
3. 引入更多市场因子

## 相关文档

- 完整配置：`config/full_config.json`
- 实验结果：`results/{strategy}_result.json`
- 代码快照：`code/`
"""

    doc_file = archive_dir / "STRATEGY.md"
    with open(doc_file, "w", encoding="utf-8") as f:
        f.write(strategy_doc)

    print(f"  ✓ 策略文档: {doc_file.name}")


def generate_readme(archive_dir: Path, version: str, strategy: str, result_data: Dict):
    """生成归档 README"""
    metrics = result_data.get("metrics", {})

    readme_content = f"""# 策略归档 - {version}

**策略名称**: {strategy}
**归档日期**: {datetime.now().strftime('%Y-%m-%d')}

## 性能摘要

| 指标 | 数值 |
|------|------|
| 总收益率 | {metrics.get('total_return', 0)*100:+.2f}% |
| 夏普比率 | {metrics.get('sharpe_ratio', 0):.3f} |
| 最大回撤 | {metrics.get('max_drawdown', 0)*100:.2f}% |
| 胜率 | {metrics.get('win_rate', 0)*100:.2f}% |
| 交易次数 | {metrics.get('n_trades', 0)} |

## 目录说明

- `code/`: 代码快照
  - `lstm/`: LSTM 训练框架
  - `pipeline/`: 数据处理流水线
- `config/`: 配置文件
  - `full_config.json`: 完整配置（模型+交易+策略）
- `results/`: 回测结果
  - `{strategy}_result.json`: 实验结果
- `docs/`: 相关文档
- `STRATEGY.md`: 完整策略文档
- `README.md`: 本文件

## 快速复现

详见 `STRATEGY.md` 中的复现步骤。

## 版本信息

- **归档版本**: {version}
- **框架版本**: LSTM v1.0.0
- **归档时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 相关资源

- 项目主页: `../../README.md`
- 框架文档: `../../src/lstm/README.md`
- 快速入门: `../../docs/QUICKSTART_LSTM.md`
"""

    readme_file = archive_dir / "README.md"
    with open(readme_file, "w", encoding="utf-8") as f:
        f.write(readme_content)

    print(f"  ✓ README: {readme_file.name}")


def save_requirements(archive_dir: Path):
    """保存依赖清单"""
    requirements_content = """# Python 依赖清单

## 核心依赖

polars>=0.20.0
torch>=2.0.0
numpy>=1.24.0

## 可选依赖

pandas>=2.0.0
matplotlib>=3.7.0
scikit-learn>=1.3.0

## 安装方式

```bash
pip install -r requirements.txt
```
"""

    req_file = archive_dir / "requirements.txt"
    with open(req_file, "w", encoding="utf-8") as f:
        f.write(requirements_content)

    print(f"  ✓ 依赖清单: {req_file.name}")


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(
        description="策略归档工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python scripts/archive_strategy.py --version v0.3 --strategy expanding_window \\
      --result_file src/lstm/data/results/experiments/expanding_window_20260116.json
        """
    )

    parser.add_argument("--version", required=True, help="版本号，如 v0.3")
    parser.add_argument("--strategy", required=True, help="策略名称，如 expanding_window")
    parser.add_argument("--result_file", required=True, help="实验结果文件路径")

    args = parser.parse_args()

    result_file = Path(args.result_file)
    if not result_file.is_absolute():
        result_file = PROJECT_ROOT / result_file

    archive_strategy(args.version, args.strategy, result_file)


if __name__ == "__main__":
    main()
