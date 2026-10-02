"""A minimal, dependency-free agent runtime.

The point of this package is to make the agent loop itself explicit:
message handling, tool dispatch, retries with backoff, and context
compaction. Every piece is small enough to read in one sitting.
"""

from .context import CompactionResult, compact, estimate_tokens, message_tokens
from .corpus import CORPUS
from .eval import Task, TaskResult, grade, load_tasks
from .llm import FakeModel, FlakyModel, LLMError, Message, Model, ToolCall
from .runtime import Agent, AgentResult, ToolResult
from .tools import Tool, ToolError, ToolRegistry, build_default_registry, rank_entries

__all__ = [
    "Agent",
    "AgentResult",
    "CORPUS",
    "CompactionResult",
    "FakeModel",
    "FlakyModel",
    "LLMError",
    "Message",
    "Model",
    "Task",
    "TaskResult",
    "Tool",
    "ToolCall",
    "ToolError",
    "ToolRegistry",
    "ToolResult",
    "build_default_registry",
    "compact",
    "estimate_tokens",
    "grade",
    "load_tasks",
    "message_tokens",
    "rank_entries",
]