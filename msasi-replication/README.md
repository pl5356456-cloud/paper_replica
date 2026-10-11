# 让 Agent 复现研报：MSASI 2.0 情绪指标

## 这是什么

视频《让 Agent 复现研报》里用到的代码。

内容是 Claude Code 对摩根士丹利研报《MSASI 2.0：A股市场情绪与技术信号的全新解读》（2025-10-31）中 A 股情绪指标的复现，以及随后的两轮改进研究。代码由 Agent 编写；人负责定方向和做判断（用什么数据替代、哪个结果可信、阈值怎么定）。

除了把私有依赖换成两个本地小模块（`msasi/paths.py`、`msasi/metrics.py`）并改了包名和路径之外，代码逻辑保持 Agent 当时写出来的样子，没有重构。

## 结果（如实）

**第一次复现：指标形态对上了，回测没对上。**

- 1 个月均值指标的取值范围为 7.5–87.2（研报：8–87）。
- 回测结果：21 笔交易，胜率 43%，做多每笔平均约 +1%。
- 研报给出的是：做多 15 笔，胜率 67%，累计 +91.2%。
- 结论：本复现**没有**重现研报的回测结果。

**为什么复现常常对不上：**

- 数据是替代品：北向资金净流入自 2024 年 8 月起停止披露；外资被动资金流没有免费数据源。
- 研报没有写明全部实现细节。
- 样本区间不同。
- 发表偏差。

**第一轮改进（权重）：**

- 研报的 R² 加权，去掉前视偏差后重做（每月只用当时已知的数据重新估计权重）：IC(60日) −0.025，t −0.88，与噪声无法区分。原因之一是 R² 丢掉了相关性的符号。
- 改用第一主成分（PCA）加权：IC −0.29，t ≈ −3。采用此方案。

**第二轮改进（阈值与持有期）：**

- 12 组「阈值 × 持有期」组合中，11 组优于无条件收益。
- 阈值定为 20/85（分位数），选取原则是「平坦区域、最小改动」。
- 安慰剂检验（1000 次循环平移）：固定持有 70 天并不比运气好（p = 0.24）；「情绪回到 50 分位时退出」优于运气（p = 0.035）。后者是约 7 条尝试过的规则之一，所以只能算有迹象，不算证明。

**最终规则回测（中证全指，2017-01 → 2026-09）：**

| | 年化收益 | 最大回撤 |
|---|---|---|
| 本规则 | 6.9% | −27.1% |
| 始终满仓 | 3.7% | −36.9% |

规则为：分位数 ≤ 20 → 满仓；≥ 85 → 空仓；回到 50 → 半仓。它的作用主要是降低回撤；在阴跌的慢熊里它不起作用。建议当作仓位参考，不要当作独立的择时信号。

> 说明：本仓库 `msasi/shadow.py` 中的低位阈值常量仍是 `OVERSOLD_PCT = 15.0`（高位 `OVERBOUGHT_PCT = 85.0`）。15 → 20 的调整来自之后的一次阈值敏感性检查（2026-10-05），那次改动不在这份代码里。代码按 Agent 当时写的原样保留；要复现上面的最终规则，把该常量改为 20。

## 仓库结构

```
msasi-replication/
├── README.md
├── requirements.txt
├── msasi/                  # 复现代码（Python 包）
│   ├── __init__.py
│   ├── __main__.py         # python -m msasi 入口
│   ├── cli.py              # 命令行：daily / run
│   ├── config.py           # 参数、11 个分项指标的登记表、信号阈值
│   ├── engine.py           # 研报方法：100 日滚动 min-max 归一 → R² 加权 → 20 日均值 → 极值信号
│   ├── backtest.py         # 事件回测：触发信号后持有 70 个交易日
│   ├── shadow.py           # 改进版：PCA 第一主成分权重 + 分位数仓位规则，及其回测
│   ├── signal.py           # 每日运行：算出最新一期指标并写出结果文件（同时运行 shadow）
│   ├── alerts.py           # 指标进入超买/超卖区时生成提示并记录
│   ├── paths.py            # 本地数据路径（新增，替代原项目的私有路径模块）
│   ├── metrics.py          # 绩效统计 compute_perf_stats / PerfStats（从原项目的公共模块中摘出）
│   └── data/
│       └── readers.py      # 读取 11 个分项指标的原始数据
```

## 怎么运行

需要 Python 3.9 及以上。

```bash
pip install -r requirements.txt

# 数据根目录，默认是当前目录下的 ./data
export MSASI_DATA_ROOT=/path/to/your/data

# 完整流程 + 研报规则的事件回测
python -m msasi run --backtest

# 每日运行：写出最新指标，并运行改进版（shadow）
python -m msasi daily
python -m msasi daily --date 2026-04-13

python -m msasi --help
```

结果写到 `$MSASI_DATA_ROOT/cn/equities/sentiment/msasi/`：`msasi_daily.parquet`、`msasi_signals.parquet`、`latest_weights.parquet`、`latest_components.parquet`、`sentiment_alerts.jsonl`，以及改进版的 `shadow_pca_daily / events / weights / backtest / equity.parquet`。

中间聚合结果缓存在 `$MSASI_DATA_ROOT/_internal/msasi_cache/`，`run --clear-cache` 可清除。

## 你需要自己准备的数据

**本仓库不含任何数据。没有数据，什么都跑不起来。**

代码只读本地 parquet 文件，不联网、不下载。下面是 `msasi/data/readers.py` 实际读取的路径和字段（路径相对于 `MSASI_DATA_ROOT`）。原项目的数据来自自建的采集流程，这里没有提供；你需要用自己的数据源按同样的路径和列名准备。

| 用途 | 路径 | 需要的列 |
|---|---|---|
| 个股日行情（算创业板成交额、全 A 成交额、涨停家数） | `cn/equities/prices/raw/<代码>.parquet`，每只股票一个文件，文件名以 6 位代码开头（用前缀 `300` / `301` / `688` 区分板块） | `date`, `amount`（成交额）, `pctChg`（涨跌幅，单位 %） |
| 中证全指日行情（交易日历、RSI-30、回测标的） | `cn/indices/prices/raw/allshare.parquet` | `date`, `close` |
| 沪深300 日行情（算期货基差的现货端） | `cn/indices/prices/raw/hs300.parquet` | `date`, `close` |
| 股指期货日行情（IF 成交额、主力合约基差） | `cn/derivatives/index_futures/raw/date=YYYY-MM-DD/*.parquet` | `variety`（取 `IF`）, `turnover`, `close`, `open_interest` |
| ETF 期权日统计（认购/认沽持仓比，标的 510300） | `cn/derivatives/index_options/raw/date=YYYY-MM-DD/*.parquet` | `合约标的代码`, `未平仓认购合约数`, `未平仓认沽合约数` |
| 北向成交额 | `cn/flows/northbound/clean/northbound.parquet`；没有则退回 `cn/flows/northbound/raw/northbound_turnover.parquet` | 前者：`date`, `northbound_turnover_cny`（或 `northbound_turnover`）；后者：`date`, `col_4` |
| 融资余额 | `cn/flows/financing/clean/margin.parquet`；没有则退回 `cn/flows/financing/raw/margin_info.parquet` | 前者：`date`, `margin_balance_cny`（或 `margin_financing_balance`）；后者：`日期`, `融资余额` |
| 上交所新增开户数（月度） | `cn/flows/sse_new_account/clean/sse_new_account.parquet` | `date`（月末）, `new_account_total` |

补充说明：

- 第 11 个分项「外资被动资金流」没有数据源，代码用北向成交额的 20 日均值代替。
- 中证全指文件缺失时程序直接报错；其他数据缺失时对应分项记为空值，指标仍会算出来，但已经不是同一个指标。
- 个股行情需要覆盖全部 A 股，且历史从 2013 年起（权重估计从 2014 年开始）。

## 免费查看每日更新的结果

不想自己准备数据的话，每日更新的情绪指标可以在 https://a.owlpha.app 的「情绪」页面免费查看。

## 声明

- 本仓库仅用于研究和学习，不构成任何投资建议。
- 本项目与摩根士丹利（Morgan Stanley）无关，未获其授权或认可。
- 仓库中不包含研报原文。研报仅以标题引用：Morgan Stanley，《MSASI 2.0：A股市场情绪与技术信号的全新解读》（2025-10-31）。

## License

仓库所有者尚未选定许可证。
