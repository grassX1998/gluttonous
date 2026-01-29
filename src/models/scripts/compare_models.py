"""
多模型对比脚本

运行所有模型并生成对比报告。

用法:
    python src/models/scripts/compare_models.py \
        --start_date 2025-04-01 \
        --end_date 2026-01-15
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime

# 添加项目根目录
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.config import LightGBMConfig, MLPConfig, EnsembleConfig
from src.models.lightgbm import LightGBMExecutor
from src.models.mlp import MLPExecutor
from src.models.ensemble import EnsembleExecutor
from src.models.experiments import ExperimentManager


def main():
    parser = argparse.ArgumentParser(description="多模型对比实验")
    parser.add_argument("--start_date", type=str, default="2025-04-01",
                        help="开始日期 (YYYY-MM-DD)")
    parser.add_argument("--end_date", type=str, default="2026-01-15",
                        help="结束日期 (YYYY-MM-DD)")
    parser.add_argument("--models", type=str, default="lightgbm,mlp,ensemble",
                        help="要运行的模型，逗号分隔")
    parser.add_argument("--save_results", action="store_true", default=True,
                        help="保存实验结果")
    parser.add_argument("--update_claude_md", action="store_true", default=False,
                        help="更新 CLAUDE.md")

    # 通用参数
    parser.add_argument("--train_days", type=int, default=60)
    parser.add_argument("--sample_ratio", type=float, default=0.5)
    parser.add_argument("--top_n", type=int, default=10)
    parser.add_argument("--prob_threshold", type=float, default=0.60)

    args = parser.parse_args()

    models_to_run = args.models.split(",")

    print("=" * 70)
    print("多模型对比实验")
    print("=" * 70)
    print(f"日期范围: {args.start_date} ~ {args.end_date}")
    print(f"模型列表: {models_to_run}")
    print(f"通用参数: train_days={args.train_days}, sample_ratio={args.sample_ratio}")
    print(f"交易参数: top_n={args.top_n}, prob_threshold={args.prob_threshold}")
    print("=" * 70)

    # 创建实验管理器
    manager = ExperimentManager()

    # 运行各模型
    for model_name in models_to_run:
        print(f"\n{'='*70}")
        print(f"运行模型: {model_name}")
        print(f"{'='*70}")

        if model_name == 'lightgbm':
            config = LightGBMConfig.with_params(
                train_days=args.train_days,
                sample_ratio=args.sample_ratio,
                top_n=args.top_n,
                prob_threshold=args.prob_threshold,
            )
            executor = LightGBMExecutor(config)

        elif model_name == 'mlp':
            config = MLPConfig.with_params(
                train_days=args.train_days,
                sample_ratio=args.sample_ratio,
                top_n=args.top_n,
                prob_threshold=args.prob_threshold,
            )
            executor = MLPExecutor(config)

        elif model_name == 'ensemble':
            config = EnsembleConfig.with_params(
                models=['lightgbm', 'mlp'],
                voting='soft',
                train_days=args.train_days,
                sample_ratio=args.sample_ratio,
                top_n=args.top_n,
                prob_threshold=args.prob_threshold,
            )
            executor = EnsembleExecutor(config)

        else:
            print(f"未知模型: {model_name}，跳过")
            continue

        # 运行实验
        manager.run_experiment(
            executor,
            start_date=args.start_date,
            end_date=args.end_date,
            calculate_metrics=True,
        )

    # 保存结果
    if args.save_results:
        save_path = manager.save_results()
        print(f"\n结果已保存: {save_path}")

    # 打印对比表格
    print("\n" + "=" * 70)
    print("模型对比结果")
    print("=" * 70)
    manager.print_comparison_table()

    # 更新 CLAUDE.md
    if args.update_claude_md:
        manager.update_claude_md()
        print("\nCLAUDE.md 已更新")

    # 打印详细结果
    print("\n" + "=" * 70)
    print("详细结果")
    print("=" * 70)
    for strategy_name, result in manager.results.items():
        if 'metrics' in result:
            metrics = result['metrics']
            print(f"\n{strategy_name}:")
            print(f"  总收益率:   {metrics['total_return']*100:+.2f}%")
            print(f"  年化收益:   {metrics['annual_return']*100:+.2f}%")
            print(f"  夏普比率:   {metrics['sharpe_ratio']:.3f}")
            print(f"  最大回撤:   {metrics['max_drawdown']*100:.2f}%")
            print(f"  胜率:       {metrics['win_rate']*100:.2f}%")
            print(f"  交易次数:   {metrics['n_trades']}")
            print(f"  训练次数:   {len(result['retrain_dates'])}")


if __name__ == "__main__":
    main()
