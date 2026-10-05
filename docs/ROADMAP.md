# 7 周路线图：多模态个人事务助理

> 一个产品载体，七周把它养大。每周加一个真实能力，取舍是真实的，
> 最后手里的东西能画成一张架构图，而不是七份互不相干的笔记。

## 0. 一句话目标

做一个**有文件系统权限、能看图的个人事务助理**：管待办和日历，查天气汇率，
能读写/搜索本地文件，会拆解多步目标，记得住你的偏好，按需检索你的资料，
需要时拉起写作小组。

## 1. 产品设定

载体：**会看的助理**。多模态不是硬塞的，而是主线。

- 拍发票/小票 -> 记账 + 日历事件（视觉输入 -> 结构化 -> 工具）
- 拍白板/手写笔记 -> 拆成待办
- 拍菜单/餐厅 -> "这家有素食" -> 写进长期偏好
- 截航班/酒店图 -> 直接进入行程规划
- 语音备忘（暂缓，成本高）-> 转文字 -> 进检索

一张照片从输入走到记忆、再走到待办，把前面所有周串成一条链。

## 2. 技术栈

一以贯之：**Python + LangGraph + FastMCP(MCP) + ChromaDB/Milvus + FastAPI + Docker**

| 层 | 选型 | 说明 |
| --- | --- | --- |
| 语言 | Python 3.10+ | 本机 3.10.11 |
| Agent 编排 | LangGraph | StateGraph / Node / Edge / 条件分支 / checkpoint |
| 工具协议 | MCP | 本机 `mcp 2.2.0`，`FastMCP` 已改名 `MCPServer` |
| 向量库 | ChromaDB 起步 -> Milvus | 从第一天就写 `VectorStore` 接口，换库只改适配器 |
| 服务 | FastAPI + uvicorn | 异步接口 |
| 部署 | Docker + 云函数 | 云函数用 ASGI 入口 |
| 模型 | OpenAI 兼容 | 复用 W1 的 `openai_compat` + `ReplayTransport` |

## 3. 横切面（每周都做，别留到最后）

- **离线优先**：`ReplayTransport` + `FakeModel` 是默认开发模式，真模型只在产出
  简历数字时调用。
- **故障注入**：超时/参数错/瞎编工具名用 `FaultInjectingTransport` 覆盖，
  不拿真 API 撞 429。
- **评测**：复用 W1 harness，每周往 `eval/tasks.jsonl` 加任务。每个改动都有数字。
- **可观测**：W1 的 `observer` 升级为带 `trace_id` 的结构化 trace。
- **省 token**：见第 5 节，作为硬约束执行。

## 4. 每周路线

### W1 已完成
Agent loop：retry/backoff、context compaction、tool dispatch、13 题评测。
位置：`agentloop/`、`eval/`、`tests/`、`demo.py`。

### W2 工具体系 + FastAPI + M1 多模态输入
- **目标**：5 个工具（天气/汇率/待办/日历/搜索），Pydantic v2 生成 schema，
  一类错误一个 `ErrorKind`，异步 FastAPI 接口；`Message.content` 支持 content blocks。
- **交付**：`agentkit/`（`schema.py` / `errors.py` / `tools/` / `dispatch.py` /
  `nested.py` / `service.py`）、`POST /agent/run` 异步接口、工具单测。
- **验收**：模型瞎编工具名 -> transcript 收到可纠正错误而非 500；工具超时 ->
  降级返回不拖垮 run；参数错 -> 结构化定位；图片以 block 形式进消息。
- **面试锚点**：schema 是给模型的契约；错误是数据、喂回模型自纠；
  多模态要早加，因为它改的是 `Message` 类型不是加个工具。

### W3 文件系统 MCP Server + M2 出图工具
- **目标**：read / write / list / 搜内容，stdio 冒烟，再挂 FastAPI；图表用
  matplotlib/PIL 离线渲染成工具。
- **交付**：`mcp_server/`（`fs.py` / `server.py` / `cloud.py`）、路径沙箱、
  云函数 ASGI 入口。
- **验收**：客户端真能列目录；`../../etc` 路径穿越被拒；图片工具返回确定产物。
- **面试锚点**：MCP 解决工具复用/宿主解耦；为什么必须做路径沙箱；
  出图这类"输出多模态"只是加工具，不动架构。

### W3.5 检索器 + Naive RAG（基线）
- **目标**：切片 -> 索引 -> 检索 -> 生成，跑测评出基线。
- **交付**：`retrieval/`（`chunking.py` / `index.py` / `generation.py`）、
  基线报告。
- **验收**：hit rate / MRR / 答案准确率有数字；多跳和无答案题**故意做不到**。
- **面试锚点**：**检索器复用，naive 留基线，agentic 才是交付物**。
  谁把 naive 当终点，谁就做成"agent 包 RAG"而不是 agentic RAG。

### W4 规划 Agent（LangGraph，已完成）
- **目标**："规划周末旅行"：搜目的地 -> 查天气 -> 比价 -> 生成行程，每步结果决定下一步。
- **交付**：`graph/`（`state.py` / `nodes.py` / `build.py` / `checkpoint.py`）。
- **验收**：中途中断能从 checkpoint 恢复。
- **面试锚点**：checkpoint 的价值是恢复、人工审批、human-in-the-loop。

### W5 三层记忆（已完成）
- **目标**：工作记忆（当轮 scratchpad）、会话记忆、长期记忆
  （向量存语义 + 结构化表存偏好/历史决策）。
- **交付**：`memory/`（`working.py` / `session.py` / `longterm.py` / `store.py`）。
- **验收**：跨会话记住偏好；会话记忆直接复用 W4 checkpoint，不重复造。
- **面试锚点**：直接答"会话很长 Prompt 爆了怎么办"——分层 + 摘要 +
  检索式记忆 + 滑窗，而不是硬塞。

### W6 Agentic RAG（已完成）
- **目标**：不做固定管道。把 `retrieve` 封成**工具**，agent 自己决定何时检索、
  检索几次、结果不够怎么换 query。
- **交付**：`retrieve` 工具 + 与 W3.5 基线的对比报告。
- **验收**：同一批任务，agentic vs naive 的成功率/工具调用/成本三项数字齐全。
- **面试锚点**：现在考的是"何时检索"，不是"怎么搭"；
  W5 的长期记忆向量库就是 W6 的 retriever，一次建设两处复用。

### W7 多 Agent：只做"单 vs 多"对比实验（已完成）
- **目标**：Researcher / Writer / Reviewer + Supervisor 的**最小**版本，
  用数字证明什么时候该上多 Agent、什么时候是浪费。
- **交付**：对比报告（token 成本 / 成功率 / 通信开销）。
- **验收**：能用自测数据说明多 Agent 的盈亏平衡点。
- **面试锚点**：Anthropic 报告 + 通信成本论证，别只会说"多 Agent 更强"。
  **降级为实验而不是搭大团队**，因为报告自己就说了别滥用，这周的主题就是验证它。

## 5. 省 token 策略（硬约束）

优先级排序：

- **P0 离线开发成默认**：replay + 故障注入，真调用只留产出数字。
  把每个功能跑上百遍去调的循环，一分钱不花。
- **P0 图看一次就塌缩成文本**：多轮里历史中的图片 block 不反复重发，
  首轮看完塌缩成 caption/OCR 文本。这也是 `_summarize` 要修的真实 bug。
- **P1 prompt caching**：稳定内容放前面（system -> tool schema -> 固定历史），
  易变内容放后面（用户新消息），保证缓存前缀命中。
- **P1 compaction 提前触发**：别等顶到窗口；工具/检索只回片段，需要细节再 `read`。
- **P1 分层模型**：路由/选工具用便宜模型，最终合成用贵模型。
- **P2 多 Agent 换手传摘要**：不传整条转录，否则成本线性相乘。

放大因子是真正的敌人：每轮重发全部历史。W1 基线是 **2.92x**，
账单是最终大小的约三倍。所有优化都指向"让历史更短、更稳"。

## 6. 模型与语料选型

### 按角色配模型，不是选一个

| 角色 | 选型要点 | 候选方向 |
| --- | --- | --- |
| 路由/选工具 | 快、便宜、tool calling 稳 | 小模型或便宜档 |
| 主力合成 | 中文、长文、性价比 | 中档 |
| 视觉 VLM | 图像理解、OCR、PDF | 视觉档 |
| Embedding | 多语言、可本地跑 | 专用，**不是**生成模型 |
| Rerank | 中文、轻量 | BGE-reranker |
| 本地兜底 | 隐私、离线 | 开源权重 |

选型维度（按 agent 重要性排序）：**tool calling 准确率 > JSON mode 稳定性 >
长上下文是否退化 > 并发/限流 > 价格 > prompt caching**。
价格与上下文规格变动快，以官方 pricing 页为准，不背数字。

### 默认方案（中文、省钱、多模态）

- 主力生成：DeepSeek 或 Qwen
- 视觉：Qwen-VL 或 GLM-4V
- Embedding：BGE-M3（中文首选、开源、可本地跑 -> 边际成本零）
- Rerank：BGE-reranker
- 语料：自己的笔记 + 论文 PDF，100~300 条起步

### 用 W1 harness 做模型 A/B

同一批任务换 `--model` 跑一遍，比 pass rate / 工具调用次数 / billed tokens /
幻觉率。"我凭什么选这个模型" -> "我拿自己的评测集测过"。
`negative` 分类正好测幻觉率，这是选 agent 模型最该看的指标。

### RAG 语料从哪来

- **自己的数据（推荐起步）**：笔记（Obsidian/Notion）、邮件导出、PDF 论文、
  自己的 repo。独家、隐私安全、有真实查询需求。
- **公开数据**：维基 dump、arXiv、政府开放数据、法规、年报、官方文档。
- **构造语料**：为评测专门造，小而精、答案可控（W1 的 `CORPUS` 即此模式）。

判断标准：**语料必须有正确答案可验证**。没有标准答案就做不出评测，
只能"感觉还行"。

坑：别一上来几百万文档（100~1000 条足够，质量才是考点）；
语料要覆盖 W1 四类题（多跳要实体关联、无答案题要确实没答案）；
中文分块和 PDF 解析是质量上限（解析烂则后面全烂）。

## 7. 目录结构（目标）

```
agentloop/           # W1：最小 loop，已完成
agentkit/            # W2：工具、schema、错误、FastAPI 服务
  tools/             #     天气 汇率 待办 日历 搜索
retrieval/           # W3.5/W6：切片、索引、生成、检索器
mcp_server/          # W3：文件系统 MCP Server + 云函数入口
graph/               # W4：LangGraph 状态机
memory/              # W5：三层记忆
eval/                # 评测集与报告（已完成，持续加任务）
docs/                # 路线图与文档
tests/               # 离线测试
```

## 8. 默认决策记录（已确认）

1. **模态**：视觉优先，语音暂不做。
2. **M3 多模态检索**：先 OCR 到文本空间；视觉检索（CLIP/ColPali）留到有把握再上。
3. **周数**：7 周，W7 做"单 vs 多 Agent"对比实验，不搭大团队。
4. **出图**：图表级（matplotlib/PIL 离线渲染），不出照片。

## 9. 进度

- W1 已完成。
- W2 已完成（`agentkit/`）：Pydantic schema、5 工具、ErrorKind 分类、
  异步 dispatch 与并发调用、FastAPI 接口、content blocks、图片塌缩。
  48 项新测试，全量 114 项离线通过。入口 `python scripts/demo_w2.py`。
- W3 已完成（`mcp_server/` + `agentkit/tools/chart.py`）：沙箱文件系统
  （read/write/list/search + 六类穿越攻击全部拦截）、`MCPServer` 的 FastMCP
  式薄 adapter 与错误翻译、云函数 ASGI 入口（挂载路径与 allowed_hosts 两个
  必设项）、M2 出图工具。68 项新测试，全量 182 项离线通过。
  入口 `python scripts/demo_w3.py`。
- W3.5 已完成（`retrieval/`）：切片、BM25 索引（藏在 `VectorStore` 协议后，
  为 W6 换 Chroma/Milvus 留口）、Naive RAG 管道与基线报告。
  基线数字：hit 0.90 / mrr 0.90 / answer 0.92；multi_hop 命中 0.67
  （multi-01 缺 `context-window`，是 W6 要修的具体案例）。
  21 项新测试，全量 245 项离线通过。入口 `python scripts/rag_baseline.py`。

- W4 已完成（`graph/`）：LangGraph StateGraph（node / edge / 条件分支 / checkpoint）、
  复用 W2 工具的薄节点层、纯函数决策层（`plan.py`）、静态断点式人工审批 +
  `InMemorySaver` 恢复。26 项新测试，全量 271 项离线通过。入口 `python scripts/demo_w4.py`。

- W5 已完成（`memory/`）：工作记忆（token 预算 + 显式淘汰）、会话记忆（滑窗 + 滚动摘要 +
  被淘汰轮次的检索）、长期记忆（结构化偏好/决策 + 语义召回，JSON 持久化）。
  会话记忆由 `session_from_checkpoint` 直接水合自 W4 的 checkpoint，不另存一份历史。
  30 项新测试，全量 301 项离线通过。入口 `python scripts/demo_w5.py`。

- W6 已完成（`retrieval/tools.py` + `agentic.py` + `compare.py`）：`retrieve` 注册成
  W2 真工具（schema/超时/错误分类全复用）、检索决策层（是否检索 / 够不够 / 换查询 /
  多轮累积证据）、相关反馈式查询扩展、naive vs agentic 对比报告（含成本列与 bottleneck 列）。
  结果：检索命中 0.90 -> 1.00、coverage 0.95 -> 1.00（multi_hop 0.83 -> 1.00），
  成本检索次数 1.00 -> 1.92、token 3921 -> 4274；答案准确率 0.92 不变，因为离线抽取式
  生成器无法跨篇组合——`multi-01` 的 coverage 已 0.50 -> 1.00，bottleneck 已从检索转到生成。
  34 项新测试，全量 335 项离线通过。入口 `python scripts/rag_compare.py`。

- W7 已完成（`agents/`）：最小 Supervisor 团队（Researcher -> Writer -> Reviewer），
  `Handoff` / `Mailbox` 把通信开销独立计量（按 hop 与边分别记账），有界修订回路
  （复核缺失项以 key 回传再填，不重新检索），`SingleAgent` 是最强诚实基线（同证据、
  同写作容量，只少一次复核）。对比报告用单一变量（brief 条目数）算出盈亏平衡点：
  **4 条以上才回本**（4 条：coverage 0.75 -> 1.00，代价 5.66x token）；3 条以内团队付
  ~2.9x 成本、零收益。通信占团队账单 30-42%，P2 规则实测：交接传摘要 vs 全转录，
  factor 从 1.4x 扩到 3.0x（摘要随 brief 增长，转录随 hop 累积）。
  32 项新测试，全量 367 项离线通过。入口 `python scripts/rag_agents.py`。

## 10. 风险与取舍

- **LangGraph API 变动快**：node/edge 写薄，业务逻辑别长在框架里。
- **MCP 2.x 迁移**：经典 `from mcp.server.fastmcp import FastMCP` 会报错，
  本机需用 `mcp.server.mcpserver.MCPServer`，包一层薄 adapter 保持 FastMCP 风格。
- **网络受限**：embedding / whisper / CLIP 权重与 `langgraph`/`chromadb`
  安装都可能需要批准；检索器第一期用 TF-IDF/BM25 离线起步，不被卡住。
- **图像 token 很贵**：一张 1024x1024 图约 1000+ token。这倒逼 compaction 升级，
  是个真实的优化故事。
- **多 Agent 成本**：W7 用实验量化，不盲目扩大。