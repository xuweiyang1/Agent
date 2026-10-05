"""The local deployment: a browser form in front of the whole chain.

Everything so far is a CLI demo, which proves the chain works but is not
something you would use on a Tuesday. This module is the smallest thing that
makes it usable at home: one page, a form, and the same ``run_chain`` call the
demo makes. No accounts, no database, no cloud -- the deliberate scope of
stage one is "a single person on a LAN".

Three decisions worth naming, because each was a fork:

- **State lives in files under one directory.** ``state_dir`` holds
  ``todos.json``, ``calendar.json`` and a run log. A SQLite table would be
  better for many users and strictly worse here, where the honest requirement
  is "survive a restart" and a human being able to read the file is a feature.
- **The vision model is optional and additive.** With ``DEEPSEEK_API_KEY`` (or
  a compatible gateway) set, ``POST /plan/photo`` reads a real image; without
  it the form is still fully usable by typing the fields. A local deployment
  that refuses to start without a paid key is a worse assistant than one that
  degrades.
- **The chain is untouched.** This layer only decides where the *fields* come
  from (a form, or a model reading a photo) and where the *results* go (a run
  log). ``run_chain`` gained ``perceived_override`` and ``state_dir`` for this;
  it did not gain branches.

Stage two (hosting) replaces this file's storage and adds auth. The chain and
the tools stay put, which is the whole argument for doing it in this order.
"""

from __future__ import annotations

import json
import os
from datetime import date
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse

from . import DEFAULT_REQUEST, run_chain

PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>local assistant</title>
<style>
 body {{ font: 15px/1.5 system-ui, sans-serif; max-width: 44rem; margin: 3rem auto; padding: 0 1rem; }}
 label {{ display:block; margin:.6rem 0 .15rem; font-weight:600; }}
 input, textarea {{ width:100%; padding:.45rem; box-sizing:border-box; }}
 button {{ margin-top:1rem; padding:.6rem 1.1rem; font-weight:600; }}
 pre {{ background:#f5f5f5; padding:1rem; overflow:auto; }}
 .row {{ display:flex; gap:1rem; }} .row > div {{ flex:1; }}
</style></head><body>
<h1>local assistant</h1>
<p>Fill the facts below, or attach a photo and let a vision model read them.
The request is intent only; the facts decide the plan.</p>
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
  <label for="photo">Photo (optional; needs an API key to read)</label>
  <input id="photo" name="photo" type="file" accept="image/*">
  <button type="submit">Plan</button>
</form>
<form method="get" action="/state" style="margin-top:1.5rem">
  <button type="submit">Show saved todos and events</button>
</form>
{result}
</body></html>
"""

RESULT = """<h2>Result</h2><pre>{body}</pre>"""


def _blank_page(**overrides: str) -> str:
    fields = {
        "request": DEFAULT_REQUEST,
        "destination": "",
        "budget": "",
        "note": "",
        "today": date.today().isoformat(),
        **overrides,
    }
    return PAGE.format(result="", **fields)


def _page_with_result(run: Any, fields: dict[str, str]) -> str:
    body = run.render()
    if run.answer:
        body += f"\n\nanswer: {run.answer}"
    return PAGE.format(result=RESULT.format(body=body), **fields)


def create_app(
    *,
    state_dir: str | Path,
    model: Any | None = None,
) -> FastAPI:
    """Build the local app. ``model`` enables the photo path when provided."""
    state = Path(state_dir)
    state.mkdir(parents=True, exist_ok=True)
    runs_dir = state / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    app = FastAPI(title="local-assistant", version="0.1.0")
    app.state.state_dir = state
    app.state.model = model

    async def _run(
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

    @app.get("/healthz")
    async def healthz() -> dict[str, Any]:
        return {
            "ok": True,
            "service": "local-assistant",
            "vision": type(model).__name__ if model is not None else None,
        }

    @app.get("/", response_class=HTMLResponse)
    async def home() -> str:
        return _blank_page()

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
                    "set DEEPSEEK_API_KEY or type the fields",
                )
            from .vision import read_fields

            suffix = Path(photo.filename).suffix or ".png"
            target = state / f"upload{suffix}"
            target.write_bytes(await photo.read())
            # Typed fields win over the photo: a person correcting the model is
            # the one signal here we trust more than the model.
            for key, value in (await read_fields(target, model=model)).items():
                fields.setdefault(key, value)

        chosen = _parse_day(today) if today.strip() else date.today()
        run = await _run(request=request, fields=fields, today=chosen, nights=nights)
        return _page_with_result(
            run,
            {
                "request": request,
                "destination": fields.get("destination", ""),
                "budget": fields.get("budget", ""),
                "note": fields.get("note", ""),
                "today": chosen.isoformat(),
            },
        )

    @app.get("/state")
    async def show_state() -> JSONResponse:
        """Read the persisted stores directly, without running the chain."""
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
    """A vision-capable async model, or ``None`` when no key is configured.

    Keeping this out of ``create_app`` means the app can be built (and tested)
    with no environment at all, which is the same rule the rest of the repo
    follows: importing and constructing must never require a key.
    """
    env_var = os.environ.get("ASSISTANT_API_KEY_ENV", "DEEPSEEK_API_KEY")
    key = os.environ.get(env_var)
    if not key:
        return None
    from agentkit.openai_model import AsyncOpenAICompatibleModel

    return AsyncOpenAICompatibleModel(
        key,
        model=os.environ.get("ASSISTANT_MODEL", "deepseek-chat"),
        base_url=os.environ.get("ASSISTANT_BASE_URL", "https://api.deepseek.com"),
    )


__all__ = ["build_model_from_env", "create_app"]