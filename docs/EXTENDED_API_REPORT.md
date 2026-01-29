# 掘金扩展API数据可用性报告

生成时间: 2026-01-18

## 测试结论

### 免费可用的API (3个)

| API | 功能 | 最早数据 |
|-----|------|---------|
| stk_get_fundamentals_balance | 资产负债表 | 2015-12-31 |
| stk_get_fundamentals_income | 利润表 | 2015-12-31 |
| stk_get_finance_deriv | 财务衍生指标 | 2015-12-31 |

### 需要付费订阅的API (5个)

| API | 功能 |
|-----|------|
| stk_get_industry_category | 行业分类体系 |
| stk_get_symbol_industry | 股票行业映射 |
| stk_get_dividend | 分红数据 |
| stk_get_adj_factor | 复权因子 |
| stk_get_share_change | 股本变动 |

### 待确认的API (1个)

| API | 功能 | 状态 |
|-----|------|------|
| stk_get_fundamentals_cashflow | 现金流量表 | 字段名待确认 |

## 已验证的有效字段名

### 资产负债表 (stk_get_fundamentals_balance)

| 字段名 | 含义 |
|--------|------|
| fix_ast | 固定资产 |
| lt_eqy_inv | 长期股权投资 |
| mny_cptl | 货币资金 |
| invt / INVT | 存货 |

### 利润表 (stk_get_fundamentals_income)

| 字段名 | 含义 |
|--------|------|
| NET_PROF | 净利润 |
| oper_prof / OPER_PROF | 营业利润 |
| inc_tax / INC_TAX | 所得税费用 |

### 财务衍生指标 (stk_get_finance_deriv)

| 字段名 | 含义 |
|--------|------|
| ROE | 净资产收益率 |
| ROA | 总资产收益率 |
| eps_basic / EPS_BASIC | 基本每股收益 |
| BPS | 每股净资产 |

## API调用示例

### 资产负债表

```python
import gm.api

# 初始化
gm.api.set_serv_addr("192.168.31.252:7001")
gm.api.set_token("your_token")

# 获取浦发银行2024年三季报资产负债表
result = gm.api.stk_get_fundamentals_balance(
    symbol="SHSE.600000",
    fields="fix_ast,lt_eqy_inv,mny_cptl,invt",
    start_date="2024-09-30",
    end_date="2024-09-30",
    df=False,
)
# 返回: [{'symbol': 'SHSE.600000', 'pub_date': '...', 'rpt_date': '2024-09-30', ...}]
```

### 利润表

```python
result = gm.api.stk_get_fundamentals_income(
    symbol="SHSE.600000",
    fields="NET_PROF,oper_prof,inc_tax",
    start_date="2024-09-30",
    end_date="2024-09-30",
    df=False,
)
```

### 财务衍生指标

```python
result = gm.api.stk_get_finance_deriv(
    symbol="SHSE.600000",
    fields="ROE,ROA,eps_basic,BPS",
    start_date="2024-09-30",
    end_date="2024-09-30",
    df=False,
)
```

## 重要说明

1. **fields参数必填**: 必须指定具体字段名，不能为空，最多20个字段
2. **字段名大小写敏感**: 部分字段名如 `ROE`、`NET_PROF` 需要大写
3. **单股票查询**: 每次只能查询一只股票，批量查询需要循环遍历
4. **报告期日期**: 使用财报发布周期的日期：
   - 一季报: YYYY-03-31
   - 中报: YYYY-06-30
   - 三季报: YYYY-09-30
   - 年报: YYYY-12-31

## 项目集成

已在以下文件中实现财务数据采集功能：

- `pipeline/data_collection/juejin_api.py` - API封装函数
- `pipeline/data_collection/collector.py` - 数据采集器集成
- `pipeline/shared/config.py` - 目录配置

### 使用方法

```bash
# 采集指定日期数据（包含财务数据）
python -m pipeline.data_collection.collector --date 2026-01-17

# 仅采集K线数据，不采集扩展数据
python -m pipeline.data_collection.collector --date 2026-01-17 --no-extended
```

## 后续计划

1. **确认现金流量表字段名** - 需要查阅掘金官方完整字段文档
2. **评估付费数据** - 如需行业分类、分红复权等数据，需订阅掘金数据服务
3. **增加更多财务字段** - 根据策略需求扩展采集的字段

## 参考资料

- [掘金量化官方文档](https://www.myquant.cn/docs2/)
- [股票财务数据函数](https://myquant.cn/docs2/sdk/python/API介绍/股票财务数据及基础数据函数（免费）.html)
