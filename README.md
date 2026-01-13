# gluttonous

基于 Futu API 的 A 股/基金量化交易策略系统。

## 功能

- 实时行情订阅与监控
- 量化交易策略回测
- 交易信号生成

## 环境配置

```bash
# 创建虚拟环境
python3 -m venv .venv

# 激活虚拟环境
source .venv/bin/activate

# 安装依赖
pip install -r requirements.txt
```

## 项目结构

```
gluttonous/
├── docs/strategy/     # 策略文档
├── src/               # 核心模块
│   ├── futuapi/       # Futu API 封装
│   └── calculator/    # 技术指标计算
├── utils/             # 工具模块
└── archive/           # 归档文件 (历史脚本、策略、回测notebooks)
```

## 依赖

- Python 3.11+
- Futu API (需要本地运行 OpenD)
- Polars (数据处理)

## 文档

策略详细文档见 [docs/strategy/](docs/strategy/)
