# 给 agent 的上下文

这个仓库是一个 7 周 agent 学习项目。**两台机器交替开发**（台式机 + 笔记本），
靠 GitHub 同步。你可能是其中一台上的 agent，接手别人刚做完的工作。

## 第一件事：先读状态，再动手

```powershell
git pull --rebase origin main
git log --oneline -8
```

**`git log` 是倒序的：越靠上越新。** 这一点必须说清楚，因为已经出过一次真实事故：
另一台机器上的 agent 看到 W3.5 的提交排在 W3 上面，误判成「W3.5 插队到 W3 前面，
所以 W3 还没做」，而事实恰好相反。

判断"某一步做了没有"，**不要看提交的顺序或标题，去看代码在不在**：

```powershell
# 例：W3 做了没？
Test-Path mcp_server            # True = 做了
```

进度真源在 `docs/ROADMAP.md` 的「进度」一节。**每次完成一步都要更新它。**

## 当前进度

| 阶段 | 状态 | 代码位置 | 入口 |
| --- | --- | --- | --- |
| W1 最小 agent loop | ✅ | `agentloop/` | `python demo.py` |
| W2 工具 + FastAPI + 多模态输入 | ✅ | `agentkit/` | `python scripts/demo_w2.py` |
| W3 MCP 文件系统 + 出图 | ✅ | `mcp_server/` + `agentkit/tools/chart.py` | `python scripts/demo_w3.py` |
| W3.5 Naive RAG 基线 | ✅ | `retrieval/` | `python scripts/rag_baseline.py` |
| W4 LangGraph 规划 Agent | ✅ | `graph/` | `python scripts/demo_w4.py` |
| W5 三层记忆 | ✅ | `memory/` | `python scripts/demo_w5.py` |
| W6 Agentic RAG | ✅ | `retrieval/`（`tools.py` + `agentic.py` + `compare.py`） | `python scripts/rag_compare.py` |
| W7 多 Agent 对比实验 | ✅ | `agents/` | `python scripts/rag_agents.py` |

完整计划见 `docs/ROADMAP.md`，协作纪律见 `docs/WORKFLOW.md`。

## 怎么验证这一步真的做完了

```powershell
python -m unittest discover -s tests -t .      # 应该 367 项全过
python scripts\rag_baseline.py                 # W3.5 基线数字，应该能复现
python scripts\check_docs.py                   # 文档与代码是否漂移
python scripts\demo_w4.py                      # W4 规划 Agent，看分支与 checkpoint
python scripts\demo_w5.py                      # W5 三层记忆，看跨会话与 checkpoint 桥接
python scripts\rag_compare.py                  # W6 naive vs agentic，数字对比
python scripts\rag_agents.py                   # W7 单 vs 多，盈亏平衡点
```

**测试必须能离线跑通、不需要 API key。** 这是本仓库的硬约束：所有测试用
`FakeModel` / `HeuristicModel` / replay transport，不碰网络、不花钱。真模型只在
"要产出简历数字"时才调用。

**文档漂移是可检查的，不要靠自觉。** `scripts/check_docs.py` 会核对三件事：
当前测试总数（README / 本文件 / ROADMAP）、文档承诺的入口文件是否存在、
两张进度表是否一致。它已作为 `tests/test_docs.py` 进套件，所以**测试数一变，
文档不改就会红**。这个检查器自己抓到的第一个问题就是它引起的 244 -> 245 漂移。

## 环境

- Windows + PowerShell。**`apply_patch` 在这台机器上会被 PowerShell 拆坏参数**
  （多行 `*** End Patch` 传不进去），改用 `.NET` 写入：
  ```powershell
  [IO.File]::WriteAllText((Join-Path $PWD $rel), $text, (New-Object Text.UTF8Encoding($false)))
  ```
  PowerShell **不支持 heredoc**（`<<'EOF'` 会语法错误），所以提交信息要写临时文件再
  `git commit -F`。
- **中文不要走 `Set-Content -Encoding ASCII`**：会变成 `?`。要么用 `[IO.File]::WriteAllText`
  （上面这行），要么把脚本存成 UTF-8 再跑。PowerShell 里内联多行 Python 同样会被转义搞坏。
- Python 解释器：`C:\Users\xuweiyang\AppData\Local\Programs\Python\Python310\python.exe`
  （`py` 启动器当前不工作，用绝对路径）
- **网络受限**。`pip install` 和模型 API 都可能需要用户批准。检索器故意做成
  BM25 离线可用，就是为了不让网络卡住进度。
- 依赖见 `requirements.txt`。`agentloop/` 是纯标准库，不需要任何依赖。

## 代码约定

- 每个模块顶部 docstring 讲清**为什么这么设计、有什么取舍**，不是复述代码。
  这是本仓库最重要的一条约定。
- 不要为了不相关的问题顺手改代码。修 bug 要单独说明。
- 新文件用 UTF-8 无 BOM、LF 换行。
- 提交信息第一行说"做了什么"，正文说"为什么、取舍是什么"。

## 一步一步来

用户的要求是：**做一步就 commit + push 一次**，开工前先 pull。不要攒成一大坨。
没推送的提交不算备份。**代码动了，文档必须同步动**——这一步和 commit 同等重要。

## 已踩过的坑（别重复踩）

- MCP 2.x 把 `FastMCP` 改名成 `MCPServer`，经典 import 直接报错。adapter 在
  `mcp_server/adapter.py`。
- Python 3.10 的 `builtins.TimeoutError` 和 `asyncio.TimeoutError` **是不同类型**，
  只 catch 一个会漏掉 await 超时。
- MCP SDK 只有抛它自己的 `ToolError` 才算"预期失败"；抛其他异常，消息会被丢弃，
  模型只看到 `Error executing tool <name>`。
- 挂载 MCP 内层路由到 `/`，否则端点是 `/mcp/mcp`。
- 云函数必须传 `allowed_hosts`，否则默认的 localhost 白名单会让所有真实请求 421。
- 沙箱必须**先 resolve 再比较**；`startswith` 会被根目录内的 symlink 绕过。
- **本机是 Python 3.10，LangGraph 的 `interrupt()` 在 async 下不可用**：它要读
  runnable config，而那靠 3.11 的 task context 传播，3.10 上每个 async run 都报
  `Called get_config outside of a runnable context`。W4 的人工审批闸门改用
  `compile(interrupt_before=["approve"])` 静态断点，配 `update_state` 后
  `ainvoke(None, config)` 恢复。升到 3.11 才能用回 `interrupt()` 的写法。
- **LangGraph 里 async 节点必须用 `ainvoke`/`astream`**，混用 sync `invoke` 会报
  "No synchronous function provided"。W4 全程 async。
- **docstring 写清楚一件事：`check_docs.py` 绝不能跑整套测试。** 第一版它跑了，
  而 `tests/test_docs.py` 又调它，于是无限递归，python 进程淹了机器。现在它用
  AST 解析数测试函数，不执行。
