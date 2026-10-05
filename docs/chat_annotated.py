"""异步的多轮（turn）循环，以及一个用于离线驱动的确定性模型。

整体形状和 W1 阶段的 runtime 一样 —— 调用模型、派发工具、把结果追加进消息、重复 —— 
但作为「服务」，这里有两个强制性的改动：

- 同一轮里的每个工具调用是并发等待的，而不是顺序执行。
  三个互不相关的查询应该只花一次往返的时间，而不是三次。
- 整轮（run）有一个总截止时间。W1 只给每个工具单独限时；这里给整个运行限时，
  因为调用方是一个 HTTP 请求，背后有客户端在等。

``HeuristicModel`` 不是 mock（模拟对象）。它是一个真实的、确定性的策略：
根据任务文本挑选工具，从而让服务和它的测试在没有 API key、没有网络的情况下也能跑；
同时真正的模型可以原封不动地替换进来，不用改这个循环。
"""

from __future__ import annotations

import json          # 下面 HeuristicModel 没用，但同模块其他函数序列化时用得到
import re            # 关键词路由用的正则
import time          # perf_counter 计时，算 elapsed_ms
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol, Sequence

from agentloop.llm import ToolCall   # W1 里定义的「模型请求调用某个工具」的数据结构

from .dispatch import DispatchResult, ToolInvoker  # 工具执行器 + 每次执行的结果
from .messages import ChatMessage, estimate_message_tokens  # 支持图片的消息 + token 估算

# 观察者回调：事件名 + 一堆字段。用于日志/埋点，不参与核心逻辑。
Observer = Callable[[str, dict[str, Any]], None]


class AsyncModel(Protocol):
    """循环所依赖的异步契约。只有一个方法，和 W1 的模型接口一样。

    Protocol 是「结构化类型」：任何实现了 ``acomplete`` 的类都算，
    不需要显式继承。所以 HeuristicModel、真实的 OpenAI 客户端可以互换。
    """

    async def acomplete(
        self, messages: Sequence[ChatMessage], tools: Sequence[dict[str, Any]]
    ) -> ChatMessage:
        # 传进完整对话历史 + 可用工具的 JSON schema，返回模型下一条消息
        # （可能带 tool_calls，也可能是最终回答）
        ...


@dataclass
class TurnResult:
    """一次运行里，调用方或测试判断结果所需要的全部信息。"""

    messages: list[ChatMessage]              # 完整的对话记录（含 system/user/assistant/tool）
    answer: str                              # 最终回答文本；被截断时为空字符串
    dispatch: list[DispatchResult] = field(default_factory=list)  # 每次工具调用的结果
    turns: int = 0                           # 实际循环了几轮
    truncated: bool = False                  # 是否因为超时/达到上限而被截断
    elapsed_ms: float = 0.0                  # 总耗时（毫秒）

    @property
    def tokens(self) -> int:
        """整段对话的 token 估算总量（含工具输出）。"""
        return sum(estimate_message_tokens(m) for m in self.messages)

    @property
    def failed_calls(self) -> list[DispatchResult]:
        """所有失败的工具调用，方便测试断言「错了几次、错在哪」。"""
        return [d for d in self.dispatch if not d.ok]


class ToolCallingAgent:
    """不断循环：直到模型不再请求工具、直接给出回答为止。

    这就是「agent 主循环」：模型 -> 工具 -> 结果塞回消息 -> 再问模型。
    """

    def __init__(
        self,
        model: AsyncModel,          # 谁来产生下一步（真实模型或 HeuristicModel）
        invoker: ToolInvoker,       # 真正执行工具的人（负责校验、超时、分类错误）
        *,
        system_prompt: str = "You are a helpful assistant. Use tools when they help.",
        max_turns: int = 8,         # 最多循环几轮，防止模型无限调工具
        run_timeout: float = 60.0,  # 整个 run 的总时限（秒）
        observer: Observer | None = None,  # 可选的埋点/日志回调
    ) -> None:
        self.model = model
        self.invoker = invoker
        self.system_prompt = system_prompt
        self.max_turns = max_turns
        self.run_timeout = run_timeout
        self.observer = observer

    async def run(self, task: str, *, history: Sequence[ChatMessage] | None = None) -> TurnResult:
        """执行任务，返回一次完整的运行结果。这是唯一的公开入口。"""
        started = time.perf_counter()   # 单调时钟，用来算总耗时和判断超时
        messages: list[ChatMessage] = []

        # 1) 组装初始对话：system（人设） + 历史（可选的、之前几轮） + 本次 user 任务
        if self.system_prompt:
            messages.append(ChatMessage(role="system", content=self.system_prompt))
        messages.extend(history or [])   # history 是外部传入的上下文，不改动调用方的列表
        messages.append(ChatMessage(role="user", content=task))

        dispatch: list[DispatchResult] = []
        turns = 0
        truncated = False

        # 2) 主循环：每轮先问模型，模型要么给工具调用，要么给最终答案
        while turns < self.max_turns:
            turns += 1

            # 每轮开始都检查总时限：超时就直接退出循环（带 truncated 标记）
            if time.perf_counter() - started > self.run_timeout:
                truncated = True
                self._emit("run_timeout", turns=turns)
                break

            # 把完整历史 + 当前已注册工具的 schema 交给模型
            reply = await self.model.acomplete(list(messages), self.invoker.registry.schemas())
            messages.append(reply)

            # 模型没有请求工具 -> 它给出了最终回答，run 结束
            if not reply.tool_calls:
                # 关键优化：把历史里的图片「折叠」成图片说明文字。
                # 否则后续每一轮都会重新发送原始图片，token 成本爆炸。
                messages = [m.collapse_images() for m in messages]
                elapsed = (time.perf_counter() - started) * 1000
                self._emit("answer", turns=turns, elapsed_ms=round(elapsed, 3))
                return TurnResult(messages, reply.text(), dispatch, turns, False, elapsed)

            # 模型请求了工具：并发执行同一轮里的所有调用（见 _invoke_all）
            results = await self._invoke_all(reply.tool_calls)

            # 把每个执行结果追加成 role="tool" 的消息，并带上 tool_call_id，
            # 模型才能把「结果」和「它刚才的请求」对应起来。
            for result in results:
                dispatch.append(result)
                messages.append(
                    ChatMessage(role="tool", content=result.output, tool_call_id=result.call_id)
                )

        # 3) 走出循环只可能是两种情况：达到 max_turns，或 run_timeout 后 break
        if turns >= self.max_turns:
            self._emit("max_turns", turns=turns)
        elapsed = (time.perf_counter() - started) * 1000
        # 被截断时 answer 为空；调用方通过 truncated/turns 判断发生了什么
        return TurnResult(messages, "", dispatch, turns, truncated, elapsed)

    async def _invoke_all(self, calls: Sequence[ToolCall]) -> list[DispatchResult]:
        """并发执行一轮里的全部工具调用，并保持结果顺序与 calls 一致。

        asyncio.gather 会同时启动所有协程，返回顺序与传入顺序相同，
        所以这里 ``list(...)`` 之后可以直接按位置对应回每条 tool_call。
        某个工具超时/报错不会取消其它工具，因为 ToolInvoker.invoke 内部
        会捕获异常并返回失败的 DispatchResult。
        """
        import asyncio   # 局部导入：只有这个方法需要它

        return list(
            await asyncio.gather(
                *(self.invoker.invoke(c.name, c.arguments, call_id=c.id) for c in calls)
            )
        )

    def _emit(self, event: str, **fields: Any) -> None:
        """把事件交给 observer（如果有）。没有 observer 时是零成本空操作。"""
        if self.observer is not None:
            self.observer(event, fields)


class HeuristicModel:
    """确定性的替身模型，但它是真的会去调用工具的。

    写它出来，是为了让服务在没有 API key 的情况下也能被端到端验证。
    它是个关键词路由器，不是语言模型，这里也明确这样标注：它的职责是
    验证整条「管道」（schema 定义、派发、错误处理、超时），
    而真正的模型可以在 ``AsyncModel`` 这个接口后面直接换上，循环一行都不用改。
    """

    # 规则表：(正则, 工具名, 默认参数)。按顺序匹配，先命中先返回。
    _RULES: tuple[tuple[str, str, dict[str, Any]], ...] = (
        (r"天气|weather|forecast|气温", "weather", {}),
        (r"汇率|换算|convert|currency|美元|欧元", "convert_currency", {}),
        (r"待办|todo|任务清单|记一下|提醒我", "todo", {}),
        (r"日历|日程|安排|calendar|会议", "calendar", {}),
        (r"搜|查一下|notes|检索|search", "search", {}),
    )

    def __init__(self) -> None:
        self.calls = 0   # 记录自己被调用了几次，用来区分「第一次」和「第二次」

    async def acomplete(
        self, messages: Sequence[ChatMessage], tools: Sequence[dict[str, Any]]
    ) -> ChatMessage:
        self.calls += 1

        # 第二次被调用时，说明工具已经执行完、结果就在消息里了。
        # 这时候不再继续发工具请求（否则会无限循环），而是总结一句。
        if self.calls > 1:
            last = next((m for m in reversed(messages) if m.role == "tool"), None)  # 最近的工具结果
            body = last.text() if last else ""
            return ChatMessage(role="assistant", content=f"Done. Tool said: {body[:300]}")

        # 第一次调用：从最近的 user 消息里拿任务文本
        task = next((m.text() for m in reversed(messages) if m.role == "user"), "")

        # 按规则表做关键词匹配，命中就发一个工具调用
        for pattern, name, defaults in self._RULES:
            if re.search(pattern, task, re.IGNORECASE):
                return ChatMessage(
                    role="assistant",
                    # 固定 call_id="c1"，参数由 _args 兜底生成，保证一定是合法调用
                    tool_calls=[ToolCall("c1", name, self._args(name, task, defaults))],
                )

        # 一个都没命中：直接给一句回答（不带 tool_calls，循环就此结束）
        return ChatMessage(role="assistant", content=f"Nothing to look up for: {task}")

    @staticmethod
    def _args(name: str, task: str, defaults: dict[str, Any]) -> dict[str, Any]:
        """尽力生成参数，让这个路由器产出「合法」的工具调用（能通过 schema 校验）。"""
        if name == "weather":
            # 先匹配已知城市，再兜底，避免把问题里的第一个词（比如 "what"）
            # 误当成地名。
            for city in ("Beijing", "Shanghai", "Tokyo", "Paris", "Lisbon", "Sydney"):
                if city.lower() in task.lower():
                    return {"city": city, "days": 1}
            return {"city": "Shanghai", "days": 1}

        if name == "convert_currency":
            # 从文本里抓第一个数字作为金额，抓不到就用 100.0
            amount = re.search(r"\d+(?:\.\d+)?", task)
            return {
                "amount": float(amount.group(0)) if amount else 100.0,
                "source": "USD",
                "target": "CNY",
            }

        if name == "todo":
            # 把整句当待办内容，截断到 80 字符；空的就给个默认文案
            return {"action": "add", "text": task.strip()[:80] or "new task"}

        if name == "calendar":
            return {"action": "list"}   # 日历工具用一个固定动作

        # 兜底是 search 之类的检索工具
        return {"query": task.strip() or "help", "k": 3}


__all__ = ["AsyncModel", "HeuristicModel", "ToolCallingAgent", "TurnResult"]
