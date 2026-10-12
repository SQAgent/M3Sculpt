# M3Sculptor / m3s — Agent 备忘录

## 项目概览

力学超材料**论文检索增强生成（RAG）**助手，工作流使用 **LangGraph StateGraph**，检索、图像路径获取、性质匹配和计算适配仍复用自研代码（2026-10-12 按用户要求恢复论文流程）。

只有两轮流程，normal 计算恢复八子块镜像候选与侵蚀/膨胀遍历；拓扑优化、Abaqus、MATLAB 和全库数值检索仍已删除：

1. **第一轮**：用户输入研究主题 → 知识图谱检索论文（Neo4j TAGS + Milvus 语义 + 关键词）→ 逐篇判断相关性 → 合适则生成分析报告并询问性质；不合适自动排除并换下一篇，直到合适。
2. **第二轮**：用户给出性质目标与容差 → 分析该论文子图对应结构是否满足 → 满足则返回图片供确认；不满足则沿排序推荐下一篇论文并重复。

**没有独立后端**：界面和工作流在同一个进程内执行，只有 `src/m3s/app/server.py` 一个进程。

## 关键命令

```bash
# 安装（开发模式）
conda activate m3s
pip install -e .

# 启动（单进程，前端 + 工作流）
bash src/m3s/START.sh --port 8766 --mode test

# Lint / 格式化
make lint
make format

# 启动基础设施服务（必须 sudo）
sudo bash check.sh           # 启动 neo4j + mongod + milvus(docker)
```

## 目录结构（删减后的全部文件）

```
src/m3s/
  app/            server.py + index.html + app.js + styles.css + README.md（唯一入口）
  engine.py       NODES 表 + build_graph() + Engine/Session（LangGraph StateGraph）
  geometry.py     最大连接组件体素化、八子块镜像、六邻域侵蚀/膨胀候选
  homogenization.py Python 周期六面体有限元（对应论文 SI S8）
  routers.py      route_after_input()
  states.py       SPAgentState (TypedDict)
  messages.py     纯 dict 消息 {role, content}
  llm.py          chat() / chat_json() / embed()；chat 默认走 opencode CLI，embed 走 urllib
  progress.py     每轮进度上报（contextvars sink）
  prompts/        只剩 4 个提示词
  screening.py    论文矛盾筛除的纯文本规则 + JSON 判定归一化
  doi_workflow.py 纯逻辑：目标解析、SubImage 路径推导、最近结果比较
  computation.py  正常模式计算链路适配层
  configuration.py HunyuanConfig + Neo4j/Mongo/Milvus 配置
  db/             neo4j.py / mongo.py / milvus.py（只留实际用到的方法）
  nodes/          preprocessing.py / parameters.py / doi_property_loop.py / evaluation.py
  SubAgents/      TagsQuery.py（论文检索排序）
  client/         m3hunyuan.py（三维重建）+ utils.py（体素化）
  elastic/        coords.py / core.py / elastic_constants.py（工程常数）
  scripts/        chfem_runner.py + verify_langgraph_examples.py（少量真实局部验证）
  eval/           queries.py / recall.py / run_recall.py（离线召回率评测，不被服务流程 import）
  START.sh
```

已删除：`Tools/`（debugtool、m3prompts、notebook）、`src/MATLAB/`、`bin/`、`paper/`、`docs/`、`Example/`、`NewVersion/`、`agent.ipynb`、`M3Sculptor_graph.png`、`.langgraph_api/`、`log/`、`MongoDB_Database_Summary.md`、`start_m3spy.sh`，以及 `elastic/` 中的绘图/IO/Elastic2D-3D 类、`client/utils.py` 中的对称化与多格式导出函数、`configuration.py` 中的 Homogenization/TOP/Morphology/Configuration/MCP 配置。

## 工作流引擎

节点函数签名保持 `state -> updates`，改动节点无需改引擎。

`NODES` 共 15 个：

```
NODE_Preprocessing        理解研究主题（冻结 query 与模式）
EXTRACT_PARAMETERS       从自然语言解析明确的基体材料参数
FIND_BY_TAGS              检索候选论文并推荐一篇
RECEIVE_DOI_REQUIREMENT   解析目标值与容差
PREPARE_EVALUATION        准备结构评估
LOAD_DATABASE             test 模式读已有计算记录
IMAGE_PROCESS / RECONSTRUCT / VOXELIZE / SYMMETRIZE / MORPHOLOGY / HOMOGENIZE / ENGINEERING_CONSTANTS
                          normal 模式实时计算链路
CHECK_DOI_PROPERTY        比较目标与结果，命中或换下一篇
SELECT_RESULT             确认所选结构
```

- `build_graph()` 用 StateGraph 编译条件边，`run()` 逐节点流式同步会话；`_next_node()` 用 `routers.route_after_input` / `loop.route_doi_step` / `evaluation.route_evaluation` 决定流转，`STEP_LIMIT=10000`
- `SPAgentState.messages` 使用 `Annotated[..., operator.add]`；`merge()` 同步流式更新时消息追加、其他键覆盖
- `Engine.submit()` 在后台线程跑一轮；同会话重复提交返回 409，不排队不重发
- 会话在内存中最多 64 个（`OrderedDict` LRU），服务重启不保留历史
- 节点异常如实写入 `session.error` 并置 `status=error`，**不得**伪装成“未找到”

## 两种结果获取模式

模式由 `--mode normal|test` 或 `M3SCULPTOR_MODE` 决定，前端不提供切换；切模式要新建对话（`frozen_mode` 会拦截混用）。

- **test**：从 MongoDB `StructureProperties` 读该 DOI SubImage 对应的已有工程常数，跳过图像处理与求解器；查询不带目标值/容差条件。
- **normal**：对该 DOI 的每张去重 SubImage 逐个执行 图像处理（Qwen 编辑，`AL_API_KEY`）→ Hunyuan3D 重建（`HUNYUAN_URL`）→ 64³ 初始体素 → 八子块分别镜像 → 初始/镜像结构六邻域侵蚀和膨胀 → 逐候选 Python 周期均匀化（可显式选择 CHFEM）→ 本地 `elastic` 算工程常数，取最接近目标的结构；正常模式**禁止**读 `StructureProperties`。计算默认值为 `computation.DEFAULT_CONFIG`（E=1 GPa、v=0.3、分辨率 64、orthorhombic、Python 求解器）；允许在自然语言中明确指定基体杨氏模量和泊松比，**不恢复前端 compute_config 参数**。原始重建保留为候选，八个子块分别镜像，默认侵蚀 6 步、膨胀 8 步；形态学采用周期单元边界。完全相同的非空体素去重并保存来源，空候选记录于 manifest，不调用求解器。

前端**只有一个自然语言输入框**：没有性质下拉框、目标值/容差输入框，也没有基体材料/对称性/分辨率设置框；`POST /api/session/{id}/message` 只收 `{text}`，不要再恢复 `compute_config` 参数。

两模式共用同一容差判断：只比较 `is_stable=true` 的结果，在当前 DOI 内取绝对误差最小的图片。模量统一 GPa（支持 MPa/kPa/Pa 换算），泊松比无量纲；未给容差用半步长（泊松比 0.05、杨氏模量 0.05 GPa、剪切模量 0.025 GPa）。目标判断是确定性数值比较，不用 LLM 判命中。误差与容差在提示中带单位（模量 GPa，泊松比无单位）。

## 离线召回率评测

`src/m3s/eval/`，只在命令行手动跑，不被服务流程 import：

```bash
python -m m3s.eval.run_recall --out output/recall_eval --rounds 5 --workers 6
```

- `queries.py`：16 条短语查询（蜂窝/手性/高强等）+ 14 条长句查询，可用 `--queries file.json` 覆盖、`--expand N` 让模型补充；`TARGET_GRID` 为 3 种性质 × 目标值 × 容差共 30 个组合
- `recall.py`：`retrieve()` 复用 `TagsQueryHandler`（`top_k=0`，只取排序），`Screener` 走 opencode 判相关性并写文件缓存，`screen_rounds()` 按同样的 `SCREEN_LIMIT` 预算取前 5 篇，`check_case()` 用 `closest_row`+`within_tolerance` 判定
- **只要某轮 DOI 内存在满足容差的稳定结构就算召回**；`hit_round` 记录第几轮命中，`covered_subimages` 统计该性质累计覆盖的去重 SubImage
- 中间过程全部落盘：`per_query/qNNN.json`（推荐列表、逐篇筛除理由、每个目标组合的逐轮最近误差与图片路径）、`records.json`、`summary.json`、`cache/{rank,screen,evidence}`
- 评测固定 test 模式（读 `StructureProperties`），不触发重建或 CHFEM

论文用的按轮次指标由两个离线脚本在已缓存排序上重算，不重跑检索：

```bash
python -m m3s.eval.round_metrics --eval output/recall_eval --out output/round_metrics
python -m m3s.eval.baseline     --eval output/recall_eval --out output/round_metrics --workers 6
python -m m3s.eval.plots        --metrics output/round_metrics --out output/figures
```

- `round_metrics.py`：`PAPER_GRID` 为论文目标网格（泊松比 −1.5~1.0 步长 0.1、杨氏模量 0~0.8 GPa 步长 0.1、剪切模量 0~0.30 GPa 步长 0.05，容差半步长，共 42 个目标）。`load_population()` 把 structure_id 的 sym/ero/con 后缀归并到 SubImage 粒度，得到 2163 个 SubImage / 364 篇 DOI 的真值总体。输出成功率、分性质成功率、MRR、oracle 上界、累计可行 SubImage、中位最近误差、容差扫描与随机序基线
- `baseline.py`：随机抽 5 篇论文并用同一 `Screener` 判主题相关性，给出**联合成功率**（相关 ∧ 数值可行）
- `plots.py`：读上面两个脚本的 JSON 出图（`figure5_rounds.png/pdf` 四联图 + `figure5_relevance.png`），中文字体自动选 `Noto Sans CJK JP`，不重算指标
- **关键结论**：只看容差时随机序不弱于本方案（每篇论文约 30 个变体使数值几乎总可达），区分能力来自主题相关性——随机论文相关率仅 13.33%，联合成功率第 5 轮 36.03% vs 本方案 92.62%。写论文/加指标时不要只报数值成功率

## 论文检索与筛选

`SubAgents/TagsQuery.py`：AI 关键词解析 → Neo4j `TAGS` 标签筛选（无命中回落纯 Milvus）→ 读标题摘要 → 语义分 0.5 + 关键词分 0.5 → 排序。数值目标**不参与**论文排序。

`screening.py` + `PROMPTS["paper_relevance_screen"]`：先用标题级矛盾规则（高刚度 vs 负刚度/减振、负泊松比 vs 正泊松比、轻质 vs 全致密）快速排除，再交 LLM 判定；每轮最多筛 `SCREEN_LIMIT=20` 篇，被排除的论文会在回复开头列出理由。

## 提示词

`src/m3s/prompts/__init__.py` 只剩 4 个，均为检索流程实际调用：

- `translate_query`（中文查询转英文关键词，必须输出英文）
- `tags_keyword_parse`（结构化 JSON，不要注入中文要求）
- `tags_answer`（论文关联说明）
- `paper_relevance_screen`（结构化 JSON 判定）

不要再引入 `prompts/core.py`；FEA 参数提取、形态学、图像识别等提示词已随功能删除。

## HTTP 接口

`GET /api/config`、`POST /api/session`、`POST /api/session/{id}/message`（只收 `{text}`）、`GET /api/session/{id}/state`（`status` ∈ `idle/running/success/error`）。

默认 `--host 0.0.0.0`（或 `M3SCULPTOR_HOST`）**对局域网开放**，服务无鉴权，仅可信网络使用；只给本机用需显式 `--host 127.0.0.1`。默认**不做同源校验**，加 `--strict-origin` 才要求 `Origin` 主机名与 `Host` 一致，个别来源用 `--allow-origin` 放行。POST 要求 `Content-Type: application/json`。前端 `connect()` 连接阶段就真发一次 `POST /api/session`，被拒会立刻显示。

## 环境变量

`.env` + `python-dotenv` 加载：

| 变量 | 用途 |
|------|------|
| `M3S_CHAT_BACKEND` | `opencode`（默认）走 opencode CLI；`http` 才用 OpenAI 兼容接口 |
| `M3S_OPENCODE_BIN` / `M3S_OPENCODE_MODEL` / `M3S_OPENCODE_AGENT` / `M3S_OPENCODE_WORKDIR` | opencode CLI 路径、模型、agent、隔离工作目录 |
| `VLM_MODEL` / `VLM_API_KEY` / `VLM_URL` | 仅 `M3S_CHAT_BACKEND=http` 时的对话模型（未设置时回落 `RETRIEVAL_LLM_*`） |
| `EMBED_MODEL` / `EMBED_API_KEY` / `EMBED_URL` | 文本嵌入模型 |
| `NEO4J_*` / `MONGO_*` / `MILVUS_*` | 三个数据库 |
| `MONGO_PERFORMANCE_DATABASE` / `MONGO_PERFORMANCE_DESIGN_SPACE` | test 模式读性质用（默认 `MechanicalMetamaterialProjectDatabase` / `DS3d_6`） |
| `AL_API_KEY` / `HUNYUAN_URL` / `CHFEM_CONDA_ENV` / `M3S_WORKDIR` | normal 模式计算链路 |
| `M3S_HOMOGENIZATION_BACKEND` | `python`（默认，SI 周期有限元）或 `chfem`（GPU，需指定可用 conda 环境） |

## 基础设施依赖

- **Neo4j**、**MongoDB**（systemd 服务），**Milvus v2.4.5**（Docker）
- normal 模式还需 **Hunyuan3D API**（默认 `http://localhost:7860/generate`）；默认 Python 求解器使用 NumPy/SciPy，选择 CHFEM 时需要 **chfem**（例如 `CHFEM_CONDA_ENV=chfem`）

## 导入说明

统一使用 `m3s.` 前缀的绝对导入，源码目录 `src/m3s` 与包名一致：

```python
from m3s.engine import Engine
from m3s.states import SPAgentState
from m3s.llm import chat, chat_json, embed
from m3s.messages import ai, last_human_text
```

旧的 `M3Sculptor.*` 导入路径已失效，不要恢复。运行时需要 `src` 在 `PYTHONPATH` 中（`START.sh` 已设置）或已 `pip install -e .`。

## 重要约定

- Ruff 配置在 `pyproject.toml`：开启 E/F/I/D/UP/T201，忽略 E501/D417；`make lint` 目标是 `src/m3s`
- 定向检查常用：`conda run -n m3s python -m ruff check --select F,E7,E9 src/m3s`
- 数据库/服务失败、图片缺失、数值无效必须中断并报错，**不能**当作“未命中”，也不会自动从 normal 退回数据库
- 展示给用户的数值必须说明属于具体计算结构/变体，不代表整张图或论文报告值

## LangGraph 典型例子验证（2026-10-12）

`PYTHONPATH=src python -m pytest -q tests` 覆盖候选遍历、最近匹配、换 DOI、错误重试、模式隔离与周期 FEM 解析解。

`PYTHONPATH=src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 CHFEM_CONDA_ENV=chfem python -m m3s.scripts.verify_langgraph_examples --solver chfem --resolution 64 --steps 1 --out output/langgraph_validation_20261012/gpu_64` 从本地体素开始验证实心参考、闭孔结构和简单立方晶格。该验证不触发检索、Qwen 编辑或 Hunyuan 重建；不能当作完整端到端复现。
