"""A chart tool: output multimodality, at the cost of one registry entry.

This is the contrast that makes M1 interesting. Supporting images on the
*input* side changed ``Message``, and that change rippled through token
estimation, compaction and wire encoding. Producing an image on the *output*
side changes nothing structural at all: it is a tool, with a schema, a
timeout, and a classified error path, exactly like the other six. Same
registry, same dispatcher, same transcript.

The one trap this file is built around: a tool result travels back through
the transcript as text, and a PNG encoded as a data URL is on the order of a
hundred thousand characters. Returning that by default would undo every
token saving in ``messages.py``. So the default return is small -- a path and
a digest -- and the inline bytes are opt-in for callers that specifically
want them (an HTTP response body, for instance) rather than for the model.

Rendering runs on the ``Agg`` backend, which is why it works offline and in a
container: no display, no window system. matplotlib is imported lazily so a
service can import ``agentkit`` without paying for a plotting library it may
never use.
"""

from __future__ import annotations

import base64
import hashlib
import io
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic import Field, model_validator

from ..errors import ErrorKind, ToolCallError
from ..registry import ToolRegistry
from ..schema import ToolArgs

CHART_KINDS = ("bar", "line", "pie", "scatter")
MAX_POINTS = 500


class ChartArgs(ToolArgs):
    kind: str = Field(..., description=f"Chart type, one of: {', '.join(CHART_KINDS)}.")
    labels: list[str] = Field(..., min_length=1, description="Category labels, one per value.")
    values: list[float] = Field(..., min_length=1, description="Numeric values, same length as labels.")
    title: str = Field("", description="Chart title.")
    x_label: str = Field("", description="Axis label for the x axis.")
    y_label: str = Field("", description="Axis label for the y axis.")

    @model_validator(mode="after")
    def _check_shape(self) -> "ChartArgs":
        """Validate the pair, not the halves.

        ``min_length`` catches an empty list; only a model-level check
        catches labels and values disagreeing, which is the mistake a model
        actually makes. Reporting it here means the error arrives before any
        rendering work happens.
        """
        if self.kind not in CHART_KINDS:
            raise ValueError(f"kind must be one of: {', '.join(CHART_KINDS)}")
        if len(self.labels) != len(self.values):
            raise ValueError(
                f"labels ({len(self.labels)}) and values ({len(self.values)}) "
                "must be the same length"
            )
        if len(self.values) > MAX_POINTS:
            raise ValueError(f"too many points: {len(self.values)} (max {MAX_POINTS})")
        return self


@dataclass
class ChartService:
    """Renders a chart to a PNG file, with inline bytes available on request.

    ``output_dir`` defaults to a directory under the system temp path so the
    tool works with no configuration, and so a test can inject its own and
    assert on real files. Filenames come from a digest of the chart's inputs,
    which means rendering the same chart twice reuses one file instead of
    littering the directory.
    """

    output_dir: Path | None = None
    _renders: int = field(default=0, init=False)

    def resolve_output_dir(self) -> Path:
        base = self.output_dir or Path(tempfile.gettempdir()) / "agentkit-charts"
        resolved = Path(base).expanduser().resolve()
        resolved.mkdir(parents=True, exist_ok=True)
        return resolved

    def render(self, args: ChartArgs, *, inline: bool = False) -> dict[str, Any]:
        self._renders += 1
        png = self._png(args)

        target = self.resolve_output_dir() / f"{self._digest(args)}.png"
        target.write_bytes(png)

        result: dict[str, Any] = {
            "kind": args.kind,
            "points": len(args.values),
            "bytes": len(png),
            "path": str(target),
            "sha256": hashlib.sha256(png).hexdigest()[:16],
            # Deliberately not a data URL: see the module docstring. A model
            # reads this result, and its useful content is "a chart exists at
            # this path", which is a handful of tokens.
            "note": "PNG written to path; pass inline=true to also receive the bytes.",
        }
        if inline:
            result["data_url"] = "data:image/png;base64," + base64.b64encode(png).decode("ascii")
        return result

    def _digest(self, args: ChartArgs) -> str:
        """A stable filename from the chart's inputs.

        Hashing the *inputs* rather than the PNG bytes is intentional: PNG
        encoding is not guaranteed byte-identical across matplotlib versions,
        but the same data should still land on the same filename.
        """
        import json

        payload = json.dumps(
            {
                "kind": args.kind,
                "labels": args.labels,
                "values": args.values,
                "title": args.title,
                "x_label": args.x_label,
                "y_label": args.y_label,
            },
            sort_keys=True,
            ensure_ascii=False,
        )
        return "chart-" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]

    def _png(self, args: ChartArgs) -> bytes:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt

        fig = None
        try:
            fig = self._figure(args, plt)
            buffer = io.BytesIO()
            fig.savefig(buffer, format="png", dpi=110, bbox_inches="tight")
            return buffer.getvalue()
        except ToolCallError:
            raise
        except Exception as exc:  # noqa: BLE001 - reclassified below
            raise ToolCallError(
                f"could not render chart: {exc}",
                kind=ErrorKind.INTERNAL,
                details={"kind": args.kind, "points": len(args.values)},
            ) from exc
        finally:
            # Closing matters in a long-lived service: matplotlib keeps every
            # open figure in a global registry, so leaking them is a slow
            # memory leak that only appears under load.
            if fig is not None:
                plt.close(fig)

    @staticmethod
    def _figure(args: ChartArgs, plt: Any):
        fig, ax = plt.subplots(figsize=(7, 4.5), layout="constrained")
        if args.kind == "bar":
            ax.bar(args.labels, args.values)
        elif args.kind == "line":
            ax.plot(args.labels, args.values, marker="o")
        elif args.kind == "scatter":
            ax.scatter(args.labels, args.values)
        else:
            ax.pie(args.values, labels=args.labels, autopct="%1.1f%%")

        if args.kind != "pie":
            if args.x_label:
                ax.set_xlabel(args.x_label)
            if args.y_label:
                ax.set_ylabel(args.y_label)
            if len(args.labels) > 1:
                for tick in ax.get_xticklabels():
                    tick.set_rotation(30)
                    tick.set_ha("right")
        if args.title:
            ax.set_title(args.title)
        return fig


def register(
    registry: ToolRegistry,
    service: ChartService | None = None,
) -> ChartService:
    """Attach the ``render_chart`` tool."""
    svc = service or ChartService()

    @registry.tool(
        "render_chart",
        "Render a chart (bar, line, pie or scatter) as a PNG and return its "
        "path. Use it to show a comparison or a trend instead of describing "
        "it in prose.",
        args_model=ChartArgs,
        timeout=15.0,
    )
    def render_chart(
        kind: str,
        labels: list[str],
        values: list[float],
        title: str = "",
        x_label: str = "",
        y_label: str = "",
        inline: bool = False,
    ) -> dict[str, Any]:
        args = ChartArgs(
            kind=kind,
            labels=labels,
            values=values,
            title=title,
            x_label=x_label,
            y_label=y_label,
        )
        return svc.render(args, inline=inline)

    return svc


__all__ = ["CHART_KINDS", "MAX_POINTS", "ChartArgs", "ChartService", "register"]
