"""
13小时持续优化脚本

自动运行多策略训练、回测和参数优化
"""

import time
import json
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List
import random

from src.lstm.optimization.multi_strategy_config import MultiStrategyManager
from src.lstm.optimization.parallel_executor import ParallelExecutor
from src.lstm.optimization.position_manager_optimizer import PositionManagerOptimizer


class ContinuousOptimizer:
    """持续优化器"""

    def __init__(self, duration_hours: float = 13.0,
                 max_workers: int = 2,
                 save_interval_minutes: int = 30):
        """
        Args:
            duration_hours: 运行时长（小时）
            max_workers: 并行worker数
            save_interval_minutes: 保存间隔（分钟）
        """
        self.duration_hours = duration_hours
        self.max_workers = max_workers
        self.save_interval_minutes = save_interval_minutes

        self.strategy_manager = MultiStrategyManager()
        self.executor = ParallelExecutor(max_workers=max_workers)
        self.position_manager = PositionManagerOptimizer()

        self.output_dir = Path("src/lstm/data/results/optimization")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.start_time = None
        self.end_time = None
        self.iteration = 0
        self.total_tasks = 0
        self.completed_tasks = 0

        # 优化历史
        self.optimization_log = []

    def run(self):
        """运行持续优化"""
        print("=" * 80)
        print("Continuous Optimization System".center(80))
        print("=" * 80)
        print(f"Duration: {self.duration_hours} hours")
        print(f"Workers: {self.max_workers}")
        print(f"Save interval: {self.save_interval_minutes} minutes")
        print(f"Output dir: {self.output_dir}")
        print("=" * 80)

        self.start_time = datetime.now()
        self.end_time = self.start_time + timedelta(hours=self.duration_hours)

        print(f"\nStart time: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"End time:   {self.end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print("\n" + "=" * 80 + "\n")

        # 启动并行执行器
        self.executor.start()

        last_save_time = datetime.now()

        try:
            while datetime.now() < self.end_time:
                self.iteration += 1

                print(f"\n{'='*80}")
                print(f"Iteration #{self.iteration}")
                print(f"Elapsed: {self._get_elapsed_time()}")
                print(f"Remaining: {self._get_remaining_time()}")
                print(f"Completed: {self.completed_tasks}/{self.total_tasks}")
                print(f"{'='*80}\n")

                # 1. 收集已完成的结果
                self._collect_results()

                # 2. 选择策略并生成新任务
                self._generate_tasks()

                # 3. 定期保存进度
                if (datetime.now() - last_save_time).total_seconds() > self.save_interval_minutes * 60:
                    self._save_progress()
                    last_save_time = datetime.now()

                # 4. 短暂休眠，避免CPU占用过高
                time.sleep(10)

        except KeyboardInterrupt:
            print("\n\n[!] 收到中断信号，正在停止...")

        finally:
            self._finalize()

    def _collect_results(self):
        """收集完成的结果"""
        results = self.executor.get_results(timeout=0.1)

        for result in results:
            self.completed_tasks += 1

            if result.get('status') == 'completed':
                strategy_name = result['strategy']
                score = result['score']
                train_params = result['train_params']
                trading_params = result['trading_params']

                # 更新最佳参数
                self.strategy_manager.update_best_params(
                    strategy_name, train_params, trading_params, score
                )

                # 记录日志
                self.optimization_log.append({
                    'iteration': self.iteration,
                    'timestamp': datetime.now().isoformat(),
                    'strategy': strategy_name,
                    'score': score,
                    'train_params': train_params,
                    'trading_params': trading_params,
                    'backtest_metrics': result['backtest_result'],
                })

                print(f"[OK] Task completed: {result['task_id']}")
                print(f"  Strategy: {strategy_name}")
                print(f"  Score: {score:.4f}")
                print(f"  Return: {result['backtest_result'].get('total_return', 0)*100:+.2f}%")
                print(f"  Sharpe: {result['backtest_result'].get('sharpe_ratio', 0):.3f}")

            else:
                print(f"[ERROR] Task failed: {result['task_id']}")
                print(f"  Error: {result.get('error', 'Unknown error')}")

    def _generate_tasks(self):
        """生成新任务（限制队列大小，避免无限增长）"""
        # 检查队列是否有空余容量
        max_pending = self.max_workers * 2  # 最大排队数为 worker 数的2倍
        pending_count = self.executor.get_pending_count()

        if pending_count >= max_pending:
            print(f"[Queue full] Pending tasks: {pending_count}/{max_pending}, skip submitting")
            return

        # 计算还可以提交多少任务
        available_slots = max_pending - pending_count

        # 策略选择：轮询或根据历史表现智能选择
        strategies = self.strategy_manager.get_all_strategies()

        # 为每个策略生成任务（根据剩余时间调整）
        remaining_hours = (self.end_time - datetime.now()).total_seconds() / 3600

        if remaining_hours < 2:
            # 剩余时间<2小时，只测试最佳参数的微调
            num_tasks_per_strategy = 1
        elif remaining_hours < 6:
            # 剩余2-6小时，中等探索
            num_tasks_per_strategy = 1
        else:
            # 剩余时间充足，多样化探索
            num_tasks_per_strategy = 1

        tasks_submitted = 0

        for strategy_name in strategies:
            if tasks_submitted >= available_slots:
                break

            for _ in range(num_tasks_per_strategy):
                if tasks_submitted >= available_slots:
                    break

                train_params, trading_params = self._sample_params(strategy_name)

                # 选择仓位管理策略（随机或基于历史表现）
                position_strategy = self._select_position_strategy()

                task_id = self.executor.submit_task(
                    strategy_name, train_params, trading_params, position_strategy
                )

                self.total_tasks += 1
                tasks_submitted += 1
                print(f"-> Submit task: {task_id}")
                print(f"  Strategy: {strategy_name}")
                print(f"  Train params: {train_params}")
                print(f"  Trade params: {trading_params}")
                print(f"  Position strategy: {position_strategy}")

        print(f"[Queue status] Submitted: {tasks_submitted}, Pending: {self.executor.get_pending_count()}")

    def _sample_params(self, strategy_name: str) -> tuple:
        """
        智能采样参数

        策略：
        - 70%概率：基于最佳参数进行小幅扰动（exploitation）
        - 30%概率：随机探索参数空间（exploration）
        """
        strategy = self.strategy_manager.get_strategy(strategy_name)

        if random.random() < 0.7 and strategy.best_score > -999:
            # Exploitation：基于最佳参数扰动
            train_params = self._perturb_params(
                strategy.best_train_params,
                strategy.train_params
            )
            trading_params = self._perturb_params(
                strategy.best_trading_params,
                strategy.trading_params
            )
        else:
            # Exploration：随机采样
            train_params, trading_params = self.strategy_manager.get_random_params(
                strategy_name
            )

        return train_params, trading_params

    def _perturb_params(self, best_params: Dict, param_space: Dict) -> Dict:
        """扰动参数（在最佳参数附近探索）"""
        perturbed = best_params.copy()

        # 随机选择1-2个参数进行扰动
        keys_to_perturb = random.sample(
            list(param_space.keys()),
            k=min(2, len(param_space))
        )

        for key in keys_to_perturb:
            candidates = param_space[key]
            current_value = best_params.get(key)

            if current_value in candidates:
                # 选择相邻的值
                idx = candidates.index(current_value)
                if random.random() < 0.5 and idx > 0:
                    perturbed[key] = candidates[idx - 1]
                elif idx < len(candidates) - 1:
                    perturbed[key] = candidates[idx + 1]
            else:
                # 随机选择
                perturbed[key] = random.choice(candidates)

        return perturbed

    def _save_progress(self):
        """保存进度"""
        print("\n[Saving progress...]")

        # 保存优化日志
        log_file = self.output_dir / f"optimization_log_{self.start_time.strftime('%Y%m%d_%H%M%S')}.json"
        with open(log_file, 'w') as f:
            json.dump({
                'start_time': self.start_time.isoformat(),
                'current_time': datetime.now().isoformat(),
                'iteration': self.iteration,
                'total_tasks': self.total_tasks,
                'completed_tasks': self.completed_tasks,
                'optimization_log': self.optimization_log,
            }, f, indent=2, default=str)

        # 保存各策略最佳参数
        for strategy_name in self.strategy_manager.get_all_strategies():
            strategy = self.strategy_manager.get_strategy(strategy_name)

            best_params_file = self.output_dir / f"best_params_{strategy_name}.json"
            with open(best_params_file, 'w') as f:
                json.dump({
                    'strategy': strategy_name,
                    'best_score': strategy.best_score,
                    'best_train_params': strategy.best_train_params,
                    'best_trading_params': strategy.best_trading_params,
                    'optimization_history': self.strategy_manager.optimization_history[strategy_name],
                }, f, indent=2)

        print(f"[OK] Progress saved to: {self.output_dir}")

    def _finalize(self):
        """完成优化"""
        print("\n\n" + "=" * 80)
        print("Optimization Completed".center(80))
        print("=" * 80)

        # 停止执行器
        self.executor.stop()

        # 最终保存
        self._save_progress()

        # 生成总结报告
        self._generate_summary_report()

        print(f"\nTotal runtime: {self._get_elapsed_time()}")
        print(f"Completed tasks: {self.completed_tasks}/{self.total_tasks}")
        print(f"\nResults saved in: {self.output_dir}")
        print("=" * 80)

    def _generate_summary_report(self):
        """生成总结报告"""
        report_file = self.output_dir / f"summary_report_{self.start_time.strftime('%Y%m%d_%H%M%S')}.md"

        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(f"# 持续优化总结报告\n\n")
            f.write(f"**开始时间**: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"**结束时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"**运行时长**: {self._get_elapsed_time()}\n")
            f.write(f"**完成任务**: {self.completed_tasks}/{self.total_tasks}\n\n")

            f.write(f"## 各策略最佳结果\n\n")

            for strategy_name in self.strategy_manager.get_all_strategies():
                strategy = self.strategy_manager.get_strategy(strategy_name)

                f.write(f"### {strategy.name}\n\n")
                f.write(f"**描述**: {strategy.description}\n\n")
                f.write(f"**最佳评分**: {strategy.best_score:.4f}\n\n")

                f.write(f"**最佳训练参数**:\n```json\n")
                f.write(json.dumps(strategy.best_train_params, indent=2))
                f.write(f"\n```\n\n")

                f.write(f"**最佳交易参数**:\n```json\n")
                f.write(json.dumps(strategy.best_trading_params, indent=2))
                f.write(f"\n```\n\n")

                f.write("---\n\n")

        print(f"[OK] Summary report generated: {report_file}")

    def _get_elapsed_time(self) -> str:
        """获取已运行时间"""
        if self.start_time is None:
            return "0:00:00"

        elapsed = datetime.now() - self.start_time
        hours = int(elapsed.total_seconds() // 3600)
        minutes = int((elapsed.total_seconds() % 3600) // 60)
        seconds = int(elapsed.total_seconds() % 60)

        return f"{hours}:{minutes:02d}:{seconds:02d}"

    def _get_remaining_time(self) -> str:
        """获取剩余时间"""
        if self.end_time is None:
            return "未知"

        remaining = self.end_time - datetime.now()

        if remaining.total_seconds() < 0:
            return "0:00:00"

        hours = int(remaining.total_seconds() // 3600)
        minutes = int((remaining.total_seconds() % 3600) // 60)
        seconds = int(remaining.total_seconds() % 60)

        return f"{hours}:{minutes:02d}:{seconds:02d}"

    def _select_position_strategy(self) -> str:
        """
        选择仓位管理策略

        策略：
        - 70%概率：使用当前最佳仓位策略
        - 30%概率：随机尝试其他策略
        """
        import random

        all_strategies = self.position_manager.get_all_strategies()

        if random.random() < 0.7 and self.position_manager.best_strategy:
            # 使用当前最佳策略
            return self.position_manager.best_strategy
        else:
            # 随机选择策略
            return random.choice(all_strategies)


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description="13小时持续优化")
    parser.add_argument("--hours", type=float, default=13.0, help="运行时长（小时）")
    parser.add_argument("--workers", type=int, default=2, help="并行worker数")
    parser.add_argument("--save-interval", type=int, default=30, help="保存间隔（分钟）")

    args = parser.parse_args()

    optimizer = ContinuousOptimizer(
        duration_hours=args.hours,
        max_workers=args.workers,
        save_interval_minutes=args.save_interval
    )

    optimizer.run()


if __name__ == "__main__":
    main()
