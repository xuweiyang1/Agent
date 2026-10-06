"""The local deployment: a chat interface with tools, plus the travel chain.

Two modes, one page:

- **Chat** (default): a real conversational assistant with tools. You can
  ask anything, manage todos, check the weather, create calendar events.
- **Travel chain**: the original nine-step pipeline (perceive -> memory ->
  plan -> retrieve -> todos -> calendar -> chart -> persist -> govern),
  kept under `/plan` for demos and benchmarking.

The chat agent uses the same tool registry and storage as the chain, so a
todo created in chat shows up in `/state` and vice versa.
"""

from __future__ import annotations

import json
import os
from datetime import date
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse

from agentkit.chat import ToolCallingAgent
from agentkit.dispatch import ToolInvoker
from agentkit.messages import ChatMessage
from agentkit.openai_model import AsyncOpenAICompatibleModel
from agentkit.tools import build_registry

from . import DEFAULT_REQUEST, run_chain

# ---------------------------------------------------------------------------
# Chat page
# ---------------------------------------------------------------------------

CHAT_PAGE = """<!doctype html>
<html lang="zh"><head><meta charset="utf-8">
<title>local assistant</title>
<style>
 * { box-sizing: border-box; }
 body { font: 15px/1.5 system-ui, sans-serif; margin: 0; padding: 0; background: #fafafa; }
 .layout { display: flex; height: 100vh; max-width: 1200px; margin: 0 auto; }
 .sidebar { width: 280px; background: #fff; border-right: 1px solid #e5e5e5; padding: 1.5rem; overflow-y: auto; }
 .chat { flex: 1; display: flex; flex-direction: column; background: #fff; }
 .chat-header { padding: 1rem 1.5rem; border-bottom: 1px solid #e5e5e5; font-weight: 600; }
 .messages { flex: 1; padding: 1.5rem; overflow-y: auto; }
 .msg { margin-bottom: 1rem; max-width: 80%; }
 .msg.user { margin-left: auto; text-align: right; }
 .msg .bubble { display: inline-block; padding: .6rem .9rem; border-radius: 12px; text-align: left; }
 .msg.user .bubble { background: #007AFF; color: #fff; }
 .msg.assistant .bubble { background: #f0f0f0; color: #333; }
 .input-bar { padding: 1rem 1.5rem; border-top: 1px solid #e5e5e5; display: flex; gap: .5rem; }
 .input-bar input { flex: 1; padding: .6rem .8rem; border: 1px solid #ddd; border-radius: 8px; font-size: 15px; }
 .input-bar button { padding: .6rem 1.2rem; background: #007AFF; color: #fff; border: none; border-radius: 8px; font-weight: 600; cursor: pointer; }
 .sidebar h3 { margin-top: 0; font-size: 14px; color: #666; text-transform: uppercase; letter-spacing: .05em; }
 .sidebar ul { list-style: none; padding: 0; margin: 0 0 1.5rem; }
 .sidebar li { padding: .3rem 0; font-size: 14px; }
 .sidebar a { color: #007AFF; text-decoration: none; font-size: 14px; }
 .tool-calls { font-size: 12px; color: #999; margin-top: .3rem; }
</style></head><body>
<div class="layout">
  <div class="sidebar">
    <h3>Tools</h3>
    <ul>
      <li>🌤 weather — 查天气</li>
      <li>💱 currency — 汇率换算</li>
      <li>📝 todo — 待办管理</li>
      <li>📅 calendar — 日历事件</li>
      <li>🔍 search — 知识库搜索</li>
      <li>📊 chart — 生成图表</li>
    </ul>
    <h3>Views</h3>
    <ul>
      <li><a href="/state" target="_blank">查看待办和日历</a></li>
      <li><a href="/plan">旅行规划（9步 chain）</a></li>
    </ul>
    <h3>Session</h3>
    <ul>
      <li id="token-count">tokens: 0</li>
    </ul>
  </div>
  <div class="chat">
    <div class="chat-header">💬 Assistant</div>
    <div class="messages" id="messages">
      <div class="msg assistant">
        <div class="bubble">你好！我是你的本地助手。可以聊天、查天气、管待办、建日历事件。有什么需要帮忙的？</div>
      </div>
    </div>
    <div class="input-bar">
      <input id="input" placeholder="输入消息..." autofocus>
      <button id="send">发送</button>
    </div>
  </div>
</div>
<script>
const messagesEl = document.getElementById('messages');
const inputEl = document.getElementById('input');
const sendBtn = document.getElementById('send');
const tokenEl = document.getElementById('token-count');

let history = [];

function addMsg(role, text) {
  const div = document.createElement('div');
  div.className = 'msg ' + role;
  const bubble = document.createElement('div');
  bubble.className = 'bubble';
  bubble.textContent = text;
  div.appendChild(bubble);
  messagesEl.appendChild(div);
  messagesEl.scrollTop = messagesEl.scrollHeight;
}

async function send() {
  const text = inputEl.value.trim();
  if (!text) return;
  inputEl.value = '';
  addMsg('user', text);

  try {
    const resp = await fetch('/chat', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({message: text, history: history}),
    });
    const data = await resp.json();
    addMsg('assistant', data.answer);
    history = data.history;
    tokenEl.textContent = 'tokens: ' + data.tokens;
  } catch (e) {
    addMsg('assistant', 'error: ' + e.message);
  }
}

sendBtn.addEventListener('click', send);
inputEl.addEventListener('keydown', e => { if (e.key === 'Enter') send(); });
</script>
</body></html>
"""

# ---------------------------------------------------------------------------
# Travel chain page (kept as-is)
# ---------------------------------------------------------------------------

PLAN_PAGE = """<!doctype html>
<html lang="zh"><head><meta charset="utf-8">
<title>travel chain · local assistant</title>
<style>
 body {{ font: 15px/1.5 system-ui, sans-serif; max-width: 44rem; margin: 3rem auto; padding: 0 1rem; }}
 label {{ display:block; margin:.6rem 0 .15rem; font-weight:600; }}
 input, textarea {{ width:100%; padding:.45rem; box-sizing:border-box; }}
 button {{ margin-top:1rem; padding:.6rem 1.1rem; font-weight:600; }}
 pre {{ background:#f5f5f5; padding:1rem; overflow:auto; }}
 .row {{ display:flex; gap:1rem; }} .row > div {{ flex:1; }}
 a {{ color:#007AFF; }}
</style></head><body>
<h1>travel chain</h1>
<p><a href="/">← 返回聊天助手</a></p>
<p>这是 9 步全链路 demo：感知 -> 记忆 -> 规划 -> 检索 -> 待办 -> 日历 -> 图表 -> 持久化 -> 治理。</p>
<form method="post" action="/plan" enctype="multipart/form-data">
  <label for="request">Request (intent)</label>
  <input id="request" name="request" value="{request}">
  <div class="row">
    <div><label for="destination">Destination</label>
      <input id="destination" name="destination" value="{destination}"></div>
    <div><label for="budget">Budget</label>
      <input id="budget" name="budget" value="{budget}"></div>
  </div>
  <label for="note">Note</label>
  <input id="note" name="note" value="{note}">
  <div class="row">
    <div><label for="today">Today (YYYY-MM-DD)</label>
      <input id="today" name="today" value="{today}"></div>
    <div><label for="nights">Nights</label>
      <input id="nights" name="nights" value="2"></div>
  </div>
  <label for="photo">Photo (optional; needs a vision model)</label>
  <input id="photo" name="photo" type="file" accept="image/*">
  <button type="submit">Plan</button>
</form>
{result}
</body></html>
"""

RESULT = """<h2>Result</h2><pre>{body}</pre>"""


def _blank_plan_page(**overrides: str) -> str:
    fields = {
        "request": DEFAULT_REQUEST,
        "destination": "",
        "budget": "",
        "note": "",
        "today": date.today().isoformat(),
        **overrides,
    }
    return PLAN_PAGE.format(result="", **fields)


def _plan_page_with_result(run: Any, fields: dict[str, str]) -> str:
    body = run.render()
    if run.answer:
        body += f"\n\nanswer: {run.answer}"
    return PLAN_PAGE.format(result=RESULT.format(body=body), **fields)


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

def create_app(
    *,
    state_dir: str | Path,
    model: Any | None = None,
) -> FastAPI:
    """Build the local app. ``model`` enables the chat and photo paths."""
    state = Path(state_dir)
    state.mkdir(parents=True, exist_ok=True)
    runs_dir = state / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    app = FastAPI(title="local-assistant", version="0.2.0")
    app.state.state_dir = state
    app.state.model = model

    # Build the chat agent once, reuse across requests.
    chat_agent: ToolCallingAgent | None = None
    if model is not None:
        registry = build_registry(
            with_todos=True,
            with_calendar=True,
            with_weather=True,
            with_fx=True,
            with_search=True,
            with_chart=True,
        )
        invoker = ToolInvoker(registry, default_timeout=30.0)
        chat_agent = ToolCallingAgent(
            model,
            invoker,
            system_prompt=(
                "You are a helpful personal assistant running locally. "
                "Use tools when they help. Be concise and friendly. "
                "Confirm actions like creating todos or calendar events clearly."
            ),
            max_turns=8,
        )

    async def _run_travel_chain(
        *,
        request: str,
        fields: dict[str, str],
        today: date,
        nights: int,
    ) -> Any:
        run = await run_chain(
            request=request or DEFAULT_REQUEST,
            today=today,
            nights=nights,
            perceived_override=fields,
            state_dir=state,
        )
        serial = run.to_dict()
        target = runs_dir / f"{today.isoformat()}-{len(list(runs_dir.glob('*.json'))):03d}.json"
        target.write_text(
            json.dumps(serial, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        return run

    # -- health -----------------------------------------------------------

    @app.get("/healthz")
    async def healthz() -> dict[str, Any]:
        return {
            "ok": True,
            "service": "local-assistant",
            "chat_enabled": chat_agent is not None,
            "vision": type(model).__name__ if model is not None else None,
        }

    # -- chat (default) ----------------------------------------------------

    @app.get("/", response_class=HTMLResponse)
    async def home() -> str:
        if chat_agent is None:
            # No model configured — show the plan page instead.
            return _blank_plan_page()
        return CHAT_PAGE

    @app.post("/chat")
    async def chat(request: dict[str, Any]) -> JSONResponse:
        if chat_agent is None:
            raise HTTPException(status_code=503, detail="no model configured")

        message = request.get("message", "")
        raw_history = request.get("history", [])
        if not message:
            raise HTTPException(status_code=400, detail="message is required")

        try:
            history_list = [
                ChatMessage(role=m.get("role", "user"), content=m.get("content", ""))
                for m in raw_history
            ]
        except Exception:
            history_list = []

        result = await chat_agent.run(message, history=history_list)

        # Return history as plain dicts for the frontend.
        history_out = [
            {"role": m.role, "content": m.text()}
            for m in result.messages
            if m.role in ("user", "assistant")
        ]

        return JSONResponse({
            "answer": result.answer,
            "history": history_out,
            "tokens": model.usage.total if hasattr(model, "usage") else 0,
            "tool_calls": [d.name for d in result.dispatch],
        })

    # -- travel chain -----------------------------------------------------

    @app.get("/plan", response_class=HTMLResponse)
    async def plan_form() -> str:
        return _blank_plan_page()

    @app.post("/plan", response_class=HTMLResponse)
    async def plan(
        request: str = Form(DEFAULT_REQUEST),
        destination: str = Form(""),
        budget: str = Form(""),
        note: str = Form(""),
        today: str = Form(""),
        nights: int = Form(2),
        photo: UploadFile | None = None,
    ) -> Any:
        fields: dict[str, str] = {
            key: value
            for key, value in (
                ("destination", destination),
                ("budget", budget),
                ("note", note),
            )
            if value.strip()
        }
        if photo is not None and photo.filename:
            if model is None:
                raise HTTPException(
                    status_code=400,
                    detail="a photo was attached but no vision model is configured; "
                    "set ASSISTANT_API_KEY or type the fields",
                )
            from .vision import read_fields

            suffix = Path(photo.filename).suffix or ".png"
            target = state / f"upload{suffix}"
            target.write_bytes(await photo.read())
            for key, value in (await read_fields(target, model=model)).items():
                fields.setdefault(key, value)

        chosen = _parse_day(today) if today.strip() else date.today()
        run = await _run_travel_chain(request=request, fields=fields, today=chosen, nights=nights)
        return _plan_page_with_result(
            run,
            {
                "request": request,
                "destination": fields.get("destination", ""),
                "budget": fields.get("budget", ""),
                "note": fields.get("note", ""),
                "today": chosen.isoformat(),
            },
        )

    # -- state -------------------------------------------------------------

    @app.get("/state")
    async def show_state() -> JSONResponse:
        return JSONResponse(
            {
                "todos": _read(state / "todos.json", "items"),
                "events": _read(state / "calendar.json", "events"),
            }
        )

    return app


def _parse_day(raw: str) -> date:
    try:
        return date.fromisoformat(raw.strip())
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"expected YYYY-MM-DD, got {raw!r}") from exc


def _read(path: Path, key: str) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []
    if isinstance(payload, dict):
        value = payload.get(key, [])
    else:
        value = payload
    return value if isinstance(value, list) else []


def build_model_from_env() -> Any | None:
    """An async model from environment variables, or ``None`` when no key is set.

    Defaults to Bailian (DashScope) qwen-plus, overridable by env vars:

    - ``ASSISTANT_API_KEY_ENV`` (default ``DASHSCOPE_API_KEY``)
    - ``ASSISTANT_MODEL`` (default ``qwen-plus``)
    - ``ASSISTANT_BASE_URL`` (default ``https://dashscope.aliyuncs.com/compatible-mode/v1``)
    """
    env_var = os.environ.get("ASSISTANT_API_KEY_ENV", "DASHSCOPE_API_KEY")
    key = os.environ.get(env_var)
    if not key:
        return None
    from agentkit.openai_model import AsyncOpenAICompatibleModel

    return AsyncOpenAICompatibleModel(
        key,
        model=os.environ.get("ASSISTANT_MODEL", "qwen-plus"),
        base_url=os.environ.get(
            "ASSISTANT_BASE_URL",
            "https://dashscope.aliyuncs.com/compatible-mode/v1",
        ),
    )


__all__ = ["build_model_from_env", "create_app"]
