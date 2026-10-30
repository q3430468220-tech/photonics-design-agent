# PRISM-Min（中文说明）

**一个最小的、可归因的光子器件设计 LLM agent 循环。**
先预测 → 再仿真 → 分解损耗 → 保留或回滚 → 写日志。
一条硬规则：测量值偏离预测超过 2σ，下一轮必须修改 **model.py**（理解），不许改设计。

英文完整说明见 [README.md](README.md)。

---

## 它是什么

PRISM-Min 就是"人写 program.md、agent 改一个 Python 文件、仿真器当裁判、keep/discard"的
autoresearch 式循环，加上三个文件和两条规则：

| | 图里的 agent 循环 | PRISM-Min |
|---|---|---|
| 每次仿真的反馈 | 一个标量 FOM | FOM **加损耗分解**：基模透射、高阶模、串扰、反射、辐射，以及**实测闭合**（通量盒 ≈ 1） |
| 仿真前 | 无 | `model.py` 必须**先登记预测**（FOM ± σ），没有预测裁判拒绝运行 |
| 记忆 | 叙述式日志 | 结构化日志（预测 vs 实测、分解、几何特征）+ `model.py`（用代码表达的理解）+ `laws.md`（带证据的规律） |
| 规则一 | — | 预测偏差 > 2σ ⇒ 下一轮是**模型轮**（只能改 model.py） |
| 规则二 | — | **理解门**：model.py 必须把最近大多数运行预测在 2σ 内（且 σ 不能虚高），否则不再花仿真预算 |

agent "被允许做什么"由运行器强制执行，不靠提示词：每轮只能改一个文件，碰不到仿真设置和 FOM
定义，闭合失败的运行作废而不是计分。

## 安装

```bash
git clone <本仓库> && cd prism-min
pip install -e .            # 只依赖 numpy、matplotlib、requests
cp .env.example .env        # 填入大模型 API key（以及 LUMERICAL_ROOT）
```

不依赖任何厂商 SDK，Anthropic 和 OpenAI 兼容接口都用 `requests` 直接调用，DeepSeek / Qwen /
中转服务只需设置 `OPENAI_BASE_URL`。

## 快速开始

**1. 离线看一遍循环（不需要 key，不需要仿真器）：**

```bash
python examples/demo_scripted.py
```

**2. 建立并验证自己的实验：**

```bash
prism init experiments/my_crossing            # 从 soi_crossing 模板复制
# 编辑 experiments/my_crossing/program.md
prism check experiments/my_crossing --plot    # 构建几何、DRC、合理性检查、特征、模型预测
prism eval  experiments/my_crossing --manual  # 人工触发一次仿真，看分解表和闭合误差
```

`prism eval --manual` 这一步不要跳过。整个循环的上限就是裁判的可信度：先看初始设计的分解表和
闭合误差，确认裁判是对的，再让 agent 花预算。用 `--backend lumerical` 时这也是验证 FDTD 设置的地方。

**3. 跑 agent：**

```bash
prism run experiments/my_crossing --backend mock --iters 30       # 玩具物理，快
prism run experiments/my_crossing --backend lumerical --sims 50   # 真实 FDTD
prism status experiments/my_crossing
prism review experiments/my_crossing   # 开一个干净会话审计 laws.md 和日志
```

## 实验文件夹

| 文件 | 谁写 | 作用 |
|---|---|---|
| `program.md` | 人 | 目标、平台、接口、FOM、预算、提示。agent 只读 |
| `sim_config.py` | 人 | `CONFIG = SimConfig(...)`：波段、网格、PML、DRC 规则、FOM 阈值、预算、理解门参数。**锁定**，每次运行记录其哈希 |
| `design.py` | agent（设计轮） | `build() -> Geometry`，用 `prism_min.geometry` 的辅助函数 |
| `model.py` | agent（模型轮） | `predict(geom) -> {fom, sigma, breakdown?, notes}`，agent 的物理理解 |
| `laws.md` | agent（只追加） | 标注 `[prior]` / `[tentative]` / `[confirmed]` 的规律，带迭代号 |
| `journal.jsonl`、`journal.md` | 运行器 | 每轮一条：假设、修改、预测、实测、分解、决定 |
| `runs/iter_NNN/` | 运行器 | design.py / model.py 快照、LLM 原始回复、仿真数据、评估结果、图 |
| `best/design.py` | 运行器 | 当前最优设计（回滚时恢复到它） |

## 一次仿真返回什么

全部归一化到注入功率，逐波长：

```
wl(um)  IL(dB)  TE0     higher  xtalk   refl    rad     box
1.5300   1.069  0.7818  0.0001  0.0925  0.0002  0.1239  0.9986
```

`TE0` 是有用的透射（FOM = 波段内最差的 −10·log10）；`higher` 是直通端口中非基模的功率（输出模式
失配）；`xtalk` 是侧端口之和；`refl` 是输入端反向功率；`rad` 是离开包围通量盒的其余功率。`box`
是**实测**的总出射通量，必须等于 1 ± `closure_tol`，否则这次运行作废——测量坏了不等于设计坏了。

同样的分解思路可用于有源器件，只是代数形式不同（1/Q = 1/Q_c + 1/Q_int；VπL ∝ g/(n³ r₃₃ Γ)；
1/f² ≈ 1/f_RC² + 1/f_photon²）。这些后端尚未编写。

## 决策规则（都在 `SimConfig` 里）

- **保留**：约束满足（或在都不满足时有所改善）**且** FOM 改善超过 `improve_threshold_db`（噪声阈值）；否则恢复 `design.py`。
- **预测偏差** `|实测 − 预测| / σ > prediction_miss_sigma` ⇒ 下一轮为模型轮。σ 上限为
  `max(sigma_floor_db, sigma_cap_rel × 预测 FOM)`，模型不能靠巨大的不确定度"永远正确"。
- **理解门**：累计 `gate_min_runs` 次计分运行后，当前 model.py 必须把最近 `gate_window` 次中至少
  `gate_min_fraction` 比例预测在 2σ 内，否则强制模型轮。连续模型轮最多 `max_consecutive_model_turns` 次，防止卡死。
- **预算**：`max_sims` 次仿真；检查和模型轮不计费。

## 后端

### `mock`：交叉器件的解析玩具模型
从栅格测量几何（交叉口宽度、直通臂宽度、taper 绝热参数、端口直波导长度），返回类似衍射、串扰、失配、
反射的项，加 2% 的确定性"网格噪声"。端口直波导太短时报告未收敛，闭合失败。**它不是物理**，只是为了
让循环、守卫、闭合和记账能在任何机器上测试。玩具最优解约为 4.4 µm 抛物线 taper 交叉，≈0.06 dB。

### `lumerical`：通过 lumapi 调 FDTD
`prism_min/backends/lumerical.py` 根据 `SimConfig` 和 `Geometry` 搭完整仿真：端口波导延伸穿过 PML、
输入端模式源加反射监视器、其余端口功率加模式展开监视器、6 面通量盒做闭合、z 向场监视器出图。
设置 `LUMERICAL_ROOT`（包含 `api/python/lumapi.py` 的目录，路径必须是纯 ASCII）。

> **状态：按 lumapi 惯例编写，但在编写环境中没有 Lumerical，未实际执行。** 先用
> `prism eval <dir> --backend lumerical --manual` 验证。最可能需要改一行的是材料名和模式展开结果的键名。

## 怎样判断 PRISM-Min 是否真的更好

本仓库背后的主张是一个假设，不是结论。诚实的检验方法：

1. 在同一任务、同一后端上复现原版循环（关掉预测守卫和理解门、隐藏分解），作为基线。
2. 一次只加一个模块：分解 → 预测加 2σ 规则 → model.py 加理解门。同样仿真预算、至少 3 个随机种子，
   画"最优 FOM 随仿真次数"曲线。
3. 和随机搜索、标准优化器（CMA-ES / 贝叶斯优化）在同一参数化下比较。把预测偏差分布作为"理解"指标报告。

(1)–(3) 需要的数据都在日志里（`journal.jsonl` 可机器读取）。

## 路线图

- 在真实交叉器件上验证并加固 Lumerical 后端（场图、最优设计的细网格收敛复核）。
- 在 LLM 选定的结构内部做伴随法 / 梯度精调。
- 多保真度初筛（粗网格 / 2.5D 先行）。
- 保留若干结构不同的候选，而不是单一最优。
- 有源器件分解后端（MRM：Δn_eff/V × 重叠、1/Q 预算；TFLN MZM：VπL 各因子、带宽三项限制）。

## 许可证

MIT
