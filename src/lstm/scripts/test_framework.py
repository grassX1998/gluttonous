"""
测试 LSTM 实验框架

快速验证框架是否正常工作
"""

import sys
from pathlib import Path

# 添加项目根目录
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def test_imports():
    """测试导入"""
    print("=" * 60)
    print("测试 1: 模块导入")
    print("=" * 60)

    try:
        # 配置导入
        from src.lstm.config import (
            DEVICE,
            MODEL_CONFIG,
            TRADING_CONFIG,
            ALL_STRATEGY_CONFIGS,
            ExpandingWindowConfig
        )
        print(f"[OK] 配置导入成功")
        print(f"  设备: {DEVICE}")
        print(f"  可用策略: {list(ALL_STRATEGY_CONFIGS.keys())}")

        # 模型导入
        from src.lstm.models import SimpleLSTMModel, LSTMModel
        print(f"[OK] 模型导入成功")

        # 实验框架导入
        from src.lstm.experiments import ExperimentManager, BaseStrategyExecutor
        print(f"[OK] 实验框架导入成功")

        return True

    except Exception as e:
        print(f"[ERROR] 导入失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_model_creation():
    """测试模型创建"""
    print("\n" + "=" * 60)
    print("测试 2: 模型创建")
    print("=" * 60)

    try:
        import torch
        from src.lstm.models import SimpleLSTMModel
        from pipeline.data_cleaning.features import FEATURE_COLS

        input_size = len(FEATURE_COLS)
        model = SimpleLSTMModel(input_size=input_size)

        print(f"[OK] SimpleLSTMModel 创建成功")
        print(f"  输入维度: {input_size}")
        print(f"  参数量: {sum(p.numel() for p in model.parameters()):,}")

        # 测试前向传播
        x = torch.randn(16, 1, input_size)
        y = model(x)
        print(f"[OK] 前向传播测试通过: {y.shape}")

        return True

    except Exception as e:
        print(f"[ERROR] 模型创建失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_executor_creation():
    """测试执行器创建"""
    print("\n" + "=" * 60)
    print("测试 3: 执行器创建")
    print("=" * 60)

    try:
        from src.lstm.experiments.executors.expanding_window import ExpandingWindowExecutor
        from src.lstm.config import ExpandingWindowConfig

        config = ExpandingWindowConfig()
        executor = ExpandingWindowExecutor(config)

        print(f"[OK] 执行器创建成功")
        print(f"  策略: {executor.config.strategy_name}")
        print(f"  描述: {executor.config.description}")

        return True

    except Exception as e:
        print(f"[ERROR] 执行器创建失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_manager_creation():
    """测试管理器创建"""
    print("\n" + "=" * 60)
    print("测试 4: 管理器创建")
    print("=" * 60)

    try:
        from src.lstm.experiments import ExperimentManager

        manager = ExperimentManager(strategies=["expanding_window"])

        print(f"[OK] 管理器创建成功")
        print(f"  策略数量: {len(manager.strategy_names)}")
        print(f"  输出目录: {manager.output_dir}")

        return True

    except Exception as e:
        print(f"[ERROR] 管理器创建失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """运行所有测试"""
    print("\n")
    print("#" * 60)
    print("#" + " " * 18 + "LSTM 框架测试" + " " * 18 + "#")
    print("#" * 60)
    print()

    tests = [
        ("模块导入", test_imports),
        ("模型创建", test_model_creation),
        ("执行器创建", test_executor_creation),
        ("管理器创建", test_manager_creation),
    ]

    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n[ERROR] {name}异常: {e}\n")
            results.append((name, False))

    # 总结
    print("\n" + "=" * 60)
    print("测试总结")
    print("=" * 60)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for name, result in results:
        status = "[PASS]" if result else "[FAIL]"
        print(f"{status} {name}")

    print()
    print(f"总计: {passed}/{total} 测试通过")

    if passed == total:
        print("\nSUCCESS! LSTM 框架测试全部通过")
        return 0
    else:
        print(f"\nWARNING: {total - passed} 个测试失败")
        return 1


if __name__ == "__main__":
    exit(main())
