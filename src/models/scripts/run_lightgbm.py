"""
运行 LightGBM 回测实验

用法:
    python src/models/scripts/run_lightgbm.py \
        --start_date 2025-04-01 \
        --end_date 2026-01-15
"""

import sys
import argparse
from pathlib import Path

# 添加项目根目录
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.config import LightGBMConfig
from src.models.lightgbm import LightGBMExecutor
from src.models.experiments import ExperimentManager


def main():
    parser = argparse.ArgumentParser(description="运行 LightGBM 回测实验")
    parser.add_argument("--start_date", type=str, default="2025-04-01",
                        help="开始日期 (YYYY-MM-DD)")
    parser.add_argument("--end_date", type=str, default="2026-01-15",
                        help="结束日期 (YYYY-MM-DD)")
    parser.add_argument("--calculate_metrics", action="store_true", default=True,
                        help="计算回测指标")
    parser.add_argument("--save_results", action="store_true", default=True,
                        help="保存实验结果")

    # LightGBM 超参数
    parser.add_argument("--num_leaves", type=int, default=16)
    parser.add_argument("--max_depth", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=0.05)
    parser.add_argument("--n_estimators", type=int, default=100)
    parser.add_argument("--train_days", type=int, default=60)
    parser.add_argument("--sample_ratio", type=float, default=0.5)

    # 交易参数
    parser.add_argument("--top_n", type=int, default=10)
    parser.add_argument("--prob_threshold", type=float, default=0.60)

    args = parser.parse_args()

    print("=" * 60)
    print("LightGBM 回测实验")
    print("=" * 60)
    print(f"日期范围: {args.start_date} ~ {args.end_date}")
    print(f"模型参数: num_leaves={args.num_leaves}, max_depth={args.max_depth}")
    print(f"训练参数: train_days={args.train_days}, sample_ratio={args.sample_ratio}")
    print(f"交易参数: top_n={args.top_n}, prob_threshold={args.prob_threshold}")
    print("=" * 60)

    # 创建配置
    config = LightGBMConfig.with_params(
        num_leaves=args.num_leaves,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        n_estimators=args.n_estimators,
        train_days=args.train_days,
        sample_ratio=args.sample_ratio,
        top_n=args.top_n,
        prob_threshold=args.prob_threshold,
    )

    # 创建执行器
    executor = LightGBMExecutor(config)

    # 创建实验管理器
    manager = ExperimentManager()

    # 运行实验
    result = manager.run_experiment(
        executor,
        start_date=args.start_date,
        end_date=args.end_date,
        calculate_metrics=args.calculate_metrics,
    )

    # 保存结果
    if args.save_results:
        save_path = manager.save_results("lightgbm")
        print(f"\n结果已保存: {save_path}")

    # 打印结果
    if 'metrics' in result:
        print("\n" + "=" * 60)
        print("回测结果")
        print("=" * 60)
        metrics = result['metrics']
        print(f"总收益率:   {metrics['total_return']*100:+.2f}%")
        print(f"年化收益:   {metrics['annual_return']*100:+.2f}%")
        print(f"夏普比率:   {metrics['sharpe_ratio']:.3f}")
        print(f"最大回撤:   {metrics['max_drawdown']*100:.2f}%")
        print(f"胜率:       {metrics['win_rate']*100:.2f}%")
        print(f"交易次数:   {metrics['n_trades']}")
        print(f"训练次数:   {len(result['retrain_dates'])}")
        print("=" * 60)


if __name__ == "__main__":
    main()
