"""Serve the local assistant in a browser.

    python scripts/serve_local.py --offline
    python scripts/serve_local.py --port 9000 --state .localstate

No API key is needed to start: the form works by typing the fields. Pass
``--offline`` to open the complete chat UI with the deterministic local
heuristic model (no network, no key). Without it, the chat UI requires a
provider key; setting one (DEEPSEEK_API_KEY by default, or ASSISTANT_* to
point at another gateway) also enables the real vision input.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from assistant.webapp import build_model_from_env, create_app


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Local assistant web UI")
    parser.add_argument("--host", default="127.0.0.1", help="127.0.0.1 keeps it off the LAN by default")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--state", default=".localstate", help="where todos, events and runs are kept")
    parser.add_argument(
        "--offline",
        action="store_true",
        help="use the deterministic local model so chat works without an API key",
    )
    options = parser.parse_args(argv)

    if options.offline:
        from agentkit.chat import HeuristicModel

        model = HeuristicModel()
        model_label = "HeuristicModel (offline)"
    else:
        model = build_model_from_env()
        model_label = type(model).__name__ if model else "(none -- use --offline or set an API key)"
    app = create_app(state_dir=options.state, model=model)

    import uvicorn

    print(f"local assistant on http://{options.host}:{options.port}")
    print(f"model: {model_label}")
    uvicorn.run(app, host=options.host, port=options.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
