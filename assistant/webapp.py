"""The local deployment: a chat interface with memory, plus the travel chain.

Two modes, one page:

- **Chat** (default): a conversational assistant with tools. You can ask
  anything, manage todos, keep notes, check the weather, and it *remembers
  you* across sessions.
- **Travel chain**: the original nine-step pipeline (perceive -> memory ->
  plan -> retrieve -> todos -> calendar -> chart -> persist -> govern),
  kept under ``/plan`` for demos and benchmarking.

Three things this layer owns, and each fixes a real complaint about the first
version:

- **The transcript is the server's, not the browser's.** The first version
  kept history in a JavaScript variable, so a reload -- let alone a restart --
  wiped it. Here every turn is appended to ``state_dir/chat.json`` and the
  page is rendered from it, which is what "come back tomorrow" requires. The
  file is plain JSON so a human can read or repair it.
- **W5's long-term memory is actually wired in.** The first version had no
  way to write a preference, so nothing was ever remembered. The chat agent
  now gets a ``memory`` tool to store preferences/decisions/facts, and every
  turn is prefixed with ``LongTermMemory.context_for`` so a stored preference
  is a constraint on the next answer rather than a diary entry. Todos,
  events, notes and memory all live under the same ``state_dir`` and survive
  a restart.
- **The page is presentable.** One page, one composer, tool-call chips, a
  memory/token readout, and it renders the stored transcript on load instead
  of starting blank.

Storage stays files under one directory. A SQLite table would be better for
many users and strictly worse here, where the honest requirement is "one
person, survives a restart" and a readable file is a feature. Auth and a real
database are stage two; the chain and the tools stay put, which is the whole
argument for doing it in this order.
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
from agentkit.tools import build_registry

from memory import build_memory

from . import DEFAULT_REQUEST, run_chain

# How many stored turns are replayed to the model as history. Older turns are
# still in the file (and shown in the page); this only bounds the prompt, which
# is the cost decision this whole project is about. W5's session layer is the
# richer answer and is exercised by the chain; the chat page keeps the simple
# bounded window on purpose so the deployment stays one readable file.
HISTORY_TURNS = 20

# The assistant's standing instructions. Memory is appended per turn rather
# than baked in here, because it changes between turns and a constant prompt
# that claims knowledge it lacks is worse than none.
BASE_SYSTEM_PROMPT = (
    "You are a warm, capable personal assistant running locally. "
    "Use tools when they help. Be concise and friendly, and reply in the "
    "user's language.\n"
    "When the user tells you a lasting preference, a standing decision, or a "
    "fact about themselves, store it with the memory tool so you still know "
    "it in the next session. Do not store transient small talk."
)

# ---------------------------------------------------------------------------
# Chat page
# ---------------------------------------------------------------------------

CHAT_PAGE = """<!doctype html>
<html lang="zh"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>local assistant</title>
<style>
  :root {
    --accent: #4f46e5; --accent-soft: #eef2ff; --ink: #1f2333;
    --muted: #6b7280; --line: #e6e8ef; --bg: #f6f7fb; --card: #ffffff;
    --user: #4f46e5; --assistant: #f2f3f7;
  }
  * { box-sizing: border-box; }
  html, body { height: 100%; margin: 0; }
  body { font: 15px/1.6 -apple-system, "Segoe UI", "PingFang SC", "Microsoft YaHei", system-ui, sans-serif;
         color: var(--ink); background: var(--bg); }
  .layout { display: flex; height: 100vh; max-width: 1180px; margin: 0 auto;
            background: var(--card); box-shadow: 0 1px 40px rgba(20,20,50,.06); }
  /* sidebar */
  .sidebar { width: 288px; flex: 0 0 288px; background: #fbfbfd; border-right: 1px solid var(--line);
             padding: 1.5rem 1.25rem; overflow-y: auto; }
  .brand { display: flex; align-items: center; gap: .6rem; font-weight: 700; font-size: 17px;
           margin-bottom: 1.5rem; }
  .brand .dot { width: 30px; height: 30px; border-radius: 9px;
                background: linear-gradient(135deg, #6366f1, #8b5cf6);
                display: grid; place-items: center; color: #fff; font-size: 15px; }
  .sidebar h3 { font-size: 11px; letter-spacing: .09em; text-transform: uppercase;
                color: var(--muted); margin: 1.4rem 0 .5rem; font-weight: 700; }
  .sidebar ul { list-style: none; padding: 0; margin: 0; }
  .sidebar li { padding: .32rem 0; font-size: 13.5px; color: #4b5563; display: flex; gap: .55rem; }
  .sidebar a { color: var(--accent); text-decoration: none; font-size: 13.5px; }
  .sidebar a:hover { text-decoration: underline; }
  .pill { display: inline-flex; align-items: center; gap: .4rem; background: #fff;
          border: 1px solid var(--line); border-radius: 999px; padding: .3rem .7rem;
          font-size: 12.5px; color: var(--muted); margin: .15rem .25rem .15rem 0; }
  .pill b { color: var(--ink); font-weight: 600; }
  /* chat */
  .chat { flex: 1; display: flex; flex-direction: column; min-width: 0; }
  .chat-header { padding: 1rem 1.6rem; border-bottom: 1px solid var(--line);
                 display: flex; align-items: center; justify-content: space-between; }
  .chat-header .title { font-weight: 650; }
  .chat-header .sub { font-size: 12.5px; color: var(--muted); }
  .messages { flex: 1; overflow-y: auto; padding: 1.6rem; }
  .msg { display: flex; gap: .7rem; margin-bottom: 1.1rem; max-width: 82%; }
  .msg.user { margin-left: auto; flex-direction: row-reverse; }
  .avatar { width: 30px; height: 30px; border-radius: 50%; flex: 0 0 30px; display: grid;
            place-items: center; font-size: 14px; color: #fff;
            background: linear-gradient(135deg, #6366f1, #8b5cf6); }
  .msg.user .avatar { background: #dfe3ff; color: #4338ca; }
  .bubble { background: var(--assistant); border-radius: 14px; padding: .7rem 1rem;
            white-space: pre-wrap; word-break: break-word; }
  .msg.user .bubble { background: var(--user); color: #fff; border-bottom-right-radius: 4px; }
  .msg.assistant .bubble { border-bottom-left-radius: 4px; }
  .chips { margin-top: .45rem; display: flex; flex-wrap: wrap; gap: .35rem; }
  .chip { font-size: 11.5px; color: #4338ca; background: var(--accent-soft);
          border-radius: 999px; padding: .12rem .55rem; }
  .sys { text-align: center; color: var(--muted); font-size: 12.5px; margin: 1rem 0; }
  .empty { text-align: center; color: var(--muted); margin-top: 22vh; }
  .empty h2 { color: var(--ink); margin-bottom: .4rem; }
  /* composer */
  .input-bar { padding: .9rem 1.5rem 1.2rem; border-top: 1px solid var(--line);
               display: flex; gap: .6rem; }
  .input-bar input { flex: 1; padding: .75rem .95rem; border: 1px solid var(--line);
                     border-radius: 12px; font-size: 15px; outline: none; background: #fafafe; }
  .input-bar input:focus { border-color: var(--accent); background: #fff;
                           box-shadow: 0 0 0 3px rgba(79,70,229,.12); }
  .input-bar button { padding: .75rem 1.35rem; background: var(--accent); color: #fff;
                      border: none; border-radius: 12px; font-weight: 650; cursor: pointer; }
  .input-bar button:disabled { opacity: .55; cursor: default; }
  .ghost { background: #fff; border: 1px solid var(--line); color: var(--muted);
           border-radius: 9px; padding: .35rem .7rem; font-size: 12.5px; cursor: pointer; }
  .ghost:hover { color: var(--accent); border-color: var(--accent); }
  .typing span { display: inline-block; width: 6px; height: 6px; margin-right: 3px;
                 border-radius: 50%; background: #b9bccb; animation: b 1.2s infinite; }
  .typing span:nth-child(2) { animation-delay: .15s; }
  .typing span:nth-child(3) { animation-delay: .3s; }
  @keyframes b { 0%, 60%, 100% { transform: translateY(0); } 30% { transform: translateY(-4px); } }
</style></head><body>
<div class="layout">
  <aside class="sidebar">
    <div class="brand"><span class="dot">✦</span> local assistant</div>

    <h3>Tools</h3>
    <ul>__TOOL_LIST__</ul>

    <h3>Views</h3>
    <ul>
      <li><a href="/state" target="_blank">待办 / 日历 / 笔记（JSON）</a></li>
      <li><a href="/plan">旅行规划（9 步 chain）</a></li>
    </ul>

    <h3>Memory</h3>
    <div>__MEMORY_PILLS__</div>

    <h3>Session</h3>
    <div>
      <span class="pill">tokens <b id="token-count">__TOKENS__</b></span>
      <span class="pill">turns <b id="turn-count">__TURNS__</b></span>
    </div>
    <div style="margin-top:.7rem"><button class="ghost" id="clear">清空对话</button></div>
  </aside>

  <main class="chat">
    <div class="chat-header">
      <div class="title">💬 Assistant</div>
      <div class="sub" id="status">就绪</div>
    </div>
    <div class="messages" id="messages">__MESSAGES__</div>
    <div class="input-bar">
      <input id="input" placeholder="输入消息，回车发送…" autofocus autocomplete="off">
      <button id="send">发送</button>
    </div>
  </main>
</div>
<script>
const messagesEl = document.getElementById('messages');
const inputEl = document.getElementById('input');
const sendBtn = document.getElementById('send');
const tokenEl = document.getElementById('token-count');
const turnEl = document.getElementById('turn-count');
const statusEl = document.getElementById('status');
const clearBtn = document.getElementById('clear');

function bubble(role, text) {
  const msg = document.createElement('div');
  msg.className = 'msg ' + role;
  const avatar = document.createElement('div');
  avatar.className = 'avatar';
  avatar.textContent = role === 'user' ? '你' : '✦';
  const body = document.createElement('div');
  const b = document.createElement('div');
  b.className = 'bubble';
  b.textContent = text;
  body.appendChild(b);
  msg.appendChild(avatar);
  msg.appendChild(body);
  messagesEl.appendChild(msg);
  messagesEl.scrollTop = messagesEl.scrollHeight;
  return body;
}

function chipRow(parent, tools) {
  if (!tools || !tools.length) return;
  const row = document.createElement('div');
  row.className = 'chips';
  tools.forEach(t => {
    const c = document.createElement('span');
    c.className = 'chip';
    c.textContent = t;
    row.appendChild(c);
  });
  parent.appendChild(row);
}

async function send() {
  const text = inputEl.value.trim();
  if (!text) return;
  inputEl.value = '';
  sendBtn.disabled = true;
  statusEl.textContent = '思考中…';
  bubble('user', text);

  const pending = bubble('assistant', '');
  pending.querySelector('.bubble');
  const b = pending.lastChild.firstChild;
  b.className = 'bubble typing';
  b.innerHTML = '<span></span><span></span><span></span>';

  try {
    const resp = await fetch('/chat', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({message: text}),
    });
    const data = await resp.json();
    if (!resp.ok) throw new Error(data.detail || 'request failed');
    b.className = 'bubble';
    b.textContent = data.answer || '(no answer)';
    chipRow(pending, data.tools);
    tokenEl.textContent = data.tokens;
    turnEl.textContent = data.turns;
    statusEl.textContent = '就绪';
  } catch (e) {
    b.className = 'bubble';
    b.textContent = '出错了：' + e.message;
    statusEl.textContent = '出错';
  } finally {
    sendBtn.disabled = false;
    messagesEl.scrollTop = messagesEl.scrollHeight;
  }
}

sendBtn.addEventListener('click', send);
inputEl.addEventListener('keydown', e => { if (e.key === 'Enter') send(); });
clearBtn.addEventListener('click', async () => {
  if (!confirm('清空这次对话？（长期记忆和待办不受影响）')) return;
  await fetch('/chat/clear', {method: 'POST'});
  location.reload();
});
</script>
</body></html>
"""

# The tool list is data, not markup, so the sidebar cannot drift from the
# registry the model is actually given.
TOOL_ICONS: dict[str, str] = {
    "weather": "🌤",
    "convert_currency": "💱",
    "todo": "📝",
    "calendar": "📅",
    "search": "🔍",
    "render_chart": "📊",
    "list_files": "📁",
    "read_file": "📄",
    "write_file": "✏️",
    "search_files": "🔎",
    "memory": "🧠",
    "note": "🗒",
}


def _tool_list_html(names: list[str]) -> str:
    items = []
    for name in names:
        icon = TOOL_ICONS.get(name, "•")
        items.append(f'<li><span>{icon}</span><span>{name}</span></li>')
    return "".join(items) or "<li>（无）</li>"


def _escape(text: str) -> str:
    return (
        text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    )


def _message_html(turns: list[dict[str, Any]]) -> str:
    if not turns:
        return (
            '<div class="empty"><h2>你好 👋</h2>'
            "<div>我是本地助手。可以聊天、查天气、管待办、记笔记，还会记住你的偏好。</div></div>"
        )
    blocks: list[str] = []
    for turn in turns:
        role = "user" if turn.get("role") == "user" else "assistant"
        avatar = "你" if role == "user" else "✦"
        text = _escape(str(turn.get("text", "")))
        chips = ""
        tools = turn.get("tools") or []
        if tools:
            chips = '<div class="chips">' + "".join(
                f'<span class="chip">{_escape(str(t))}</span>' for t in tools
            ) + "</div>"
        blocks.append(
            f'<div class="msg {role}"><div class="avatar">{avatar}</div>'
            f'<div><div class="bubble">{text}</div>{chips}</div></div>'
        )
    return "".join(blocks)


# ---------------------------------------------------------------------------
# Travel chain page (kept as-is)
# ---------------------------------------------------------------------------

PLAN_PAGE = """<!doctype html>
<html lang="zh"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>travel chain · local assistant</title>
<style>
 :root {{ --accent: #4f46e5; --line: #e6e8ef; --muted: #6b7280; }}
 body {{ font: 15px/1.6 -apple-system, "Segoe UI", "PingFang SC", system-ui, sans-serif;
         max-width: 46rem; margin: 3rem auto; padding: 0 1rem; color: #1f2333; }}
 h1 {{ margin-bottom: .3rem; }}
 label {{ display:block; margin:.7rem 0 .2rem; font-weight:600; font-size: 14px; }}
 input, textarea {{ width:100%; padding:.55rem .7rem; box-sizing:border-box;
                    border:1px solid #ddd; border-radius:9px; font-size:14px; }}
 input:focus {{ outline:none; border-color: var(--accent); box-shadow: 0 0 0 3px rgba(79,70,229,.12); }}
 button {{ margin-top:1.1rem; padding:.65rem 1.3rem; font-weight:650; color:#fff;
           background: var(--accent); border:none; border-radius:10px; cursor:pointer; }}
 pre {{ background:#f6f7fb; padding:1rem; border-radius:10px; overflow:auto; }}
 .row {{ display:flex; gap:1rem; }} .row > div {{ flex:1; }}
 a {{ color: var(--accent); }}
</style></head><body>
<h1>travel chain</h1>
<p><a href="/">← 返回聊天助手</a></p>
<p>这是 9 步全链路 demo：感知 → 记忆 → 规划 → 检索 → 待办 → 日历 → 图表 → 持久化 → 治理。</p>
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

    app = FastAPI(title="local-assistant", version="0.3.0")
    app.state.state_dir = state
    app.state.model = model

    # Long-term memory is built even without a model, so ``/state`` can report
    # it and a restart demonstrates persistence rather than a fresh start.
    longterm, _session, _working = build_memory(path=state / "memory.json")

    chat_agent: ToolCallingAgent | None = None
    registry = None
    if model is not None:
        from agentkit.tools.calendar import CalendarService
        from agentkit.tools.notes import NoteService
        from agentkit.tools.todo import TodoService

        workspace = os.environ.get("ASSISTANT_WORKSPACE", str(Path.home()))
        # The services are passed in explicitly so a chat-made todo or event
        # lands in ``state_dir`` beside the chain's -- the first version left
        # the registry's defaults in memory, so /state never saw them.
        registry = build_registry(
            with_todos=True,
            with_calendar=True,
            with_weather=True,
            with_fx=True,
            with_search=True,
            with_chart=True,
            with_files=True,
            with_memory=True,
            with_notes=True,
            chart_output_dir=str(state / "charts"),
            workspace=workspace,
            todo_service=TodoService(path=state / "todos.json"),
            calendar_service=CalendarService(path=state / "calendar.json"),
            longterm=longterm,
            note_service=NoteService(path=state / "notes.json"),
        )
        invoker = ToolInvoker(registry, default_timeout=30.0)
        chat_agent = ToolCallingAgent(model, invoker, max_turns=8)
    app.state.longterm = longterm
    app.state.registry = registry

    # -- transcript on disk ------------------------------------------------

    transcript_path = state / "chat.json"

    def _load_turns() -> list[dict[str, Any]]:
        if not transcript_path.is_file():
            return []
        try:
            payload = json.loads(transcript_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return []
        turns = payload.get("turns", payload if isinstance(payload, list) else [])
        return turns if isinstance(turns, list) else []

    def _save_turns(turns: list[dict[str, Any]]) -> None:
        payload = {"turns": turns, "updated": date.today().isoformat()}
        transcript_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )

    # -- health -----------------------------------------------------------

    @app.get("/healthz")
    async def healthz() -> dict[str, Any]:
        return {
            "ok": True,
            "service": "local-assistant",
            "chat_enabled": chat_agent is not None,
            "tools": registry.names() if registry is not None else [],
            "memories": len(longterm),
            "vision": type(model).__name__ if model is not None else None,
        }

    # -- chat (default) ----------------------------------------------------

    def _render_chat() -> str:
        turns = _load_turns()
        names = registry.names() if registry is not None else []
        prefs = longterm.preferences()
        pills = [
            f'<span class="pill">偏好 <b>{len(prefs)}</b></span>',
            f'<span class="pill">决策 <b>{len(longterm.decisions())}</b></span>',
            f'<span class="pill">共 <b>{len(longterm)}</b> 条</span>',
        ]
        pages = (
            CHAT_PAGE.replace("__TOOL_LIST__", _tool_list_html(names))
            .replace("__MEMORY_PILLS__", "".join(pills))
            .replace("__MESSAGES__", _message_html(turns))
            .replace("__TOKENS__", "0")
            .replace("__TURNS__", str(len(turns)))
        )
        return pages

    @app.get("/", response_class=HTMLResponse)
    async def home() -> str:
        if chat_agent is None:
            # No model configured -- show the plan page instead.
            return _blank_plan_page()
        return _render_chat()

    @app.post("/chat")
    async def chat(request: dict[str, Any]) -> JSONResponse:
        if chat_agent is None:
            raise HTTPException(status_code=503, detail="no model configured")

        message = str(request.get("message", "")).strip()
        if not message:
            raise HTTPException(status_code=400, detail="message is required")

        turns = _load_turns()
        history = [
            ChatMessage(role=turn["role"], content=str(turn.get("text", "")))
            for turn in turns[-HISTORY_TURNS:]
            if turn.get("role") in ("user", "assistant") and str(turn.get("text", "")).strip()
        ]

        # Memory is consulted *before* the answer and injected as a constraint,
        # which is the difference between remembering and displaying.
        memory_block = longterm.context_for(message)
        system_prompt = BASE_SYSTEM_PROMPT
        if memory_block:
            system_prompt = f"{BASE_SYSTEM_PROMPT}\n\n{memory_block}"
        chat_agent.system_prompt = system_prompt

        result = await chat_agent.run(message, history=history)

        tool_names = [dispatch.name for dispatch in result.dispatch if dispatch.ok]
        turns.append({"role": "user", "text": message})
        turns.append({"role": "assistant", "text": result.answer, "tools": tool_names})
        _save_turns(turns)

        return JSONResponse({
            "answer": result.answer,
            "tools": tool_names,
            "tokens": model.usage.total if hasattr(model, "usage") else 0,
            "turns": len(turns),
            "memories": len(longterm),
        })

    @app.post("/chat/clear")
    async def chat_clear() -> dict[str, Any]:
        """Clear the transcript. Long-term memory and todos are untouched."""
        _save_turns([])
        return {"ok": True}

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

    # -- state -------------------------------------------------------------

    @app.get("/state")
    async def show_state() -> JSONResponse:
        return JSONResponse(
            {
                "todos": _read(state / "todos.json", "items"),
                "events": _read(state / "calendar.json", "events"),
                "notes": _read(state / "notes.json", "notes"),
                "memories": [
                    {"id": record.id, "kind": record.kind, "text": record.text}
                    for record in longterm.structured.all()
                ],
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
        enable_search=True,
    )


__all__ = ["build_model_from_env", "create_app"]