"""
绘制回测结果图表
- 收益率曲线
- 持仓比例图
- 回撤曲线
"""
import json
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import numpy as np

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

RESULT_DIR = Path(".pipeline_data/backtest_results")
OUTPUT_DIR = Path("docs/images")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_latest_result():
    """加载最新的回测结果"""
    files = sorted(RESULT_DIR.glob("backtest_v5_*.json"), reverse=True)
    if not files:
        raise FileNotFoundError("No backtest result found")
    
    latest = files[0]
    print(f"Loading: {latest}")
    
    with open(latest, "r") as f:
        data = json.load(f)
    
    return data, latest.stem


def plot_returns_curve(daily_data: list, title_suffix: str = ""):
    """绘制收益率曲线"""
    dates = [datetime.strptime(d["date"], "%Y-%m-%d") for d in daily_data]
    cum_returns = [d["cum_return"] * 100 for d in daily_data]  # 转为百分比
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # 收益率曲线
    ax.plot(dates, cum_returns, 'b-', linewidth=1.5, label='策略累计收益')
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    
    # 标注关键点
    max_idx = np.argmax(cum_returns)
    min_idx = np.argmin(cum_returns)
    
    ax.scatter([dates[max_idx]], [cum_returns[max_idx]], color='green', s=100, zorder=5)
    ax.annotate(f'最高: {cum_returns[max_idx]:.1f}%', 
                xy=(dates[max_idx], cum_returns[max_idx]),
                xytext=(10, 10), textcoords='offset points', fontsize=10)
    
    # 设置格式
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    plt.xticks(rotation=45)
    
    ax.set_xlabel('日期', fontsize=12)
    ax.set_ylabel('累计收益率 (%)', fontsize=12)
    ax.set_title(f'Backtest V5 累计收益率曲线 {title_suffix}', fontsize=14)
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    
    # 填充正负收益区域
    ax.fill_between(dates, cum_returns, 0, 
                    where=[r >= 0 for r in cum_returns], 
                    color='green', alpha=0.1)
    ax.fill_between(dates, cum_returns, 0, 
                    where=[r < 0 for r in cum_returns], 
                    color='red', alpha=0.1)
    
    plt.tight_layout()
    output_path = OUTPUT_DIR / "backtest_v5_returns.png"
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved: {output_path}")


def plot_positions(daily_data: list, title_suffix: str = ""):
    """绘制每日持仓数量"""
    dates = [datetime.strptime(d["date"], "%Y-%m-%d") for d in daily_data]
    n_trades = [d["n_trades"] for d in daily_data]
    
    fig, ax = plt.subplots(figsize=(14, 4))
    
    ax.bar(dates, n_trades, color='steelblue', alpha=0.7, width=1.5)
    
    # 平均线
    avg_trades = np.mean(n_trades)
    ax.axhline(y=avg_trades, color='red', linestyle='--', 
               label=f'平均持仓: {avg_trades:.1f}只')
    
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    plt.xticks(rotation=45)
    
    ax.set_xlabel('日期', fontsize=12)
    ax.set_ylabel('持仓数量', fontsize=12)
    ax.set_title(f'Backtest V5 每日持仓数量 {title_suffix}', fontsize=14)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    output_path = OUTPUT_DIR / "backtest_v5_positions.png"
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved: {output_path}")


def plot_drawdown(daily_data: list, title_suffix: str = ""):
    """绘制回撤曲线"""
    dates = [datetime.strptime(d["date"], "%Y-%m-%d") for d in daily_data]
    drawdown = [-d["drawdown"] * 100 for d in daily_data]  # 转为负值百分比
    
    fig, ax = plt.subplots(figsize=(14, 4))
    
    ax.fill_between(dates, drawdown, 0, color='red', alpha=0.3)
    ax.plot(dates, drawdown, 'r-', linewidth=1)
    
    # 最大回撤标注
    min_idx = np.argmin(drawdown)
    ax.scatter([dates[min_idx]], [drawdown[min_idx]], color='darkred', s=100, zorder=5)
    ax.annotate(f'最大回撤: {-drawdown[min_idx]:.1f}%', 
                xy=(dates[min_idx], drawdown[min_idx]),
                xytext=(10, -20), textcoords='offset points', fontsize=10)
    
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    plt.xticks(rotation=45)
    
    ax.set_xlabel('日期', fontsize=12)
    ax.set_ylabel('回撤 (%)', fontsize=12)
    ax.set_title(f'Backtest V5 回撤曲线 {title_suffix}', fontsize=14)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = OUTPUT_DIR / "backtest_v5_drawdown.png"
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved: {output_path}")


def plot_combined(daily_data: list, results: dict, title_suffix: str = ""):
    """绘制组合图表"""
    dates = [datetime.strptime(d["date"], "%Y-%m-%d") for d in daily_data]
    cum_returns = [d["cum_return"] * 100 for d in daily_data]
    drawdown = [-d["drawdown"] * 100 for d in daily_data]
    n_trades = [d["n_trades"] for d in daily_data]
    
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    
    # 1. 收益率曲线
    ax1 = axes[0]
    ax1.plot(dates, cum_returns, 'b-', linewidth=1.5)
    ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax1.fill_between(dates, cum_returns, 0, 
                     where=[r >= 0 for r in cum_returns], 
                     color='green', alpha=0.1)
    ax1.fill_between(dates, cum_returns, 0, 
                     where=[r < 0 for r in cum_returns], 
                     color='red', alpha=0.1)
    ax1.set_ylabel('累计收益率 (%)', fontsize=11)
    ax1.set_title(f'Backtest V5 回测结果 | 收益: {results["total_return"]*100:+.1f}% | '
                  f'Sharpe: {results["sharpe_ratio"]:.2f} | '
                  f'最大回撤: {results["max_drawdown"]*100:.1f}%', fontsize=13)
    ax1.grid(True, alpha=0.3)
    ax1.legend(['累计收益'], loc='upper left')
    
    # 2. 回撤曲线
    ax2 = axes[1]
    ax2.fill_between(dates, drawdown, 0, color='red', alpha=0.3)
    ax2.plot(dates, drawdown, 'r-', linewidth=1)
    ax2.set_ylabel('回撤 (%)', fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    # 3. 持仓数量
    ax3 = axes[2]
    ax3.bar(dates, n_trades, color='steelblue', alpha=0.7, width=1.5)
    avg_trades = np.mean(n_trades)
    ax3.axhline(y=avg_trades, color='red', linestyle='--', alpha=0.7)
    ax3.set_ylabel('持仓数量', fontsize=11)
    ax3.set_xlabel('日期', fontsize=11)
    ax3.grid(True, alpha=0.3, axis='y')
    
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax3.xaxis.set_major_locator(mdates.MonthLocator())
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    output_path = OUTPUT_DIR / "backtest_v5_combined.png"
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved: {output_path}")


def main():
    data, filename = load_latest_result()
    
    if "daily_data" not in data:
        print("Error: No daily_data in result file. Please re-run backtest.")
        return
    
    daily_data = data["daily_data"]
    results = data["results"]
    
    print(f"\n=== Results ===")
    print(f"Total Return: {results['total_return']*100:+.2f}%")
    print(f"Sharpe Ratio: {results['sharpe_ratio']:.3f}")
    print(f"Max Drawdown: {results['max_drawdown']*100:.2f}%")
    print(f"Trading Days: {len(daily_data)}")
    print()
    
    # 绘制图表
    title_suffix = f"(Top10, Prob≥0.60)"
    plot_returns_curve(daily_data, title_suffix)
    plot_positions(daily_data, title_suffix)
    plot_drawdown(daily_data, title_suffix)
    plot_combined(daily_data, results, title_suffix)
    
    print("\nAll plots saved to docs/images/")


if __name__ == "__main__":
    main()
