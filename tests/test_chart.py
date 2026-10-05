"""Tests for the chart tool (M2).

The design claim these pin down is a token-cost one: rendering an image must
not put the image into the transcript. A PNG as a data URL is on the order of
a hundred thousand characters, so "the default result is small" is a
behaviour worth asserting, not a detail worth trusting.
"""

from __future__ import annotations

import asyncio
import tempfile
import unittest
from pathlib import Path

from agentkit import ErrorKind, ToolInvoker, build_registry
from agentkit.tools import chart


def run(coro):
    return asyncio.run(coro)


class ChartServiceTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.output_dir = Path(self._tmp.name)
        self.service = chart.ChartService(output_dir=self.output_dir)

    def tearDown(self):
        self._tmp.cleanup()

    def _args(self, **overrides):
        base = {
            "kind": "bar",
            "labels": ["Mon", "Tue", "Wed"],
            "values": [3.0, 5.0, 2.0],
            "title": "Tasks done",
        }
        base.update(overrides)
        return chart.ChartArgs(**base)

    def test_renders_a_real_png_file(self):
        result = self.service.render(self._args())
        path = Path(result["path"])
        self.assertTrue(path.is_file())
        # PNG magic number, so this is a genuine image and not an empty file.
        self.assertEqual(path.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")
        self.assertEqual(result["bytes"], path.stat().st_size)

    def test_default_result_is_small_enough_for_a_transcript(self):
        """The whole point: the model gets a path, not a hundred thousand chars."""
        result = self.service.render(self._args())
        self.assertNotIn("data_url", result)
        self.assertLess(len(str(result)), 500)

    def test_inline_bytes_are_opt_in(self):
        result = self.service.render(self._args(), inline=True)
        self.assertTrue(result["data_url"].startswith("data:image/png;base64,"))
        self.assertGreater(len(result["data_url"]), 1000)

    def test_the_same_inputs_reuse_the_same_filename(self):
        first = self.service.render(self._args())
        second = self.service.render(self._args())
        self.assertEqual(first["path"], second["path"])
        self.assertEqual(len(list(self.output_dir.glob("*.png"))), 1)

    def test_different_inputs_get_different_filenames(self):
        self.service.render(self._args())
        self.service.render(self._args(values=[1.0, 1.0, 1.0]))
        self.assertEqual(len(list(self.output_dir.glob("*.png"))), 2)

    def test_filenames_are_digest_derived_so_no_path_is_model_controlled(self):
        """No user-supplied filename means no traversal vector to defend."""
        result = self.service.render(self._args())
        name = Path(result["path"]).name
        self.assertTrue(name.startswith("chart-"))
        self.assertEqual(len(name), len("chart-") + 12 + len(".png"))
        self.assertTrue(name.endswith(".png"))

    def test_figures_do_not_leak_between_renders(self):
        """matplotlib keeps open figures globally; leaking them is a slow leak."""
        import matplotlib.pyplot as plt

        for _ in range(5):
            self.service.render(self._args())
        self.assertEqual(plt.get_fignums(), [])

    def test_output_dir_is_created_on_demand(self):
        nested = self.output_dir / "deep" / "nested"
        service = chart.ChartService(output_dir=nested)
        result = service.render(self._args())
        self.assertTrue(Path(result["path"]).is_file())
        self.assertTrue(nested.is_dir())

    def test_every_chart_kind_renders(self):
        for kind in chart.CHART_KINDS:
            with self.subTest(kind=kind):
                result = self.service.render(self._args(kind=kind))
                self.assertGreater(result["bytes"], 1000)


class ChartValidationTests(unittest.TestCase):
    def test_mismatched_label_and_value_counts_are_rejected(self):
        with self.assertRaises(ValueError):
            chart.ChartArgs(kind="bar", labels=["a", "b"], values=[1.0])

    def test_unknown_kind_is_rejected(self):
        with self.assertRaises(ValueError):
            chart.ChartArgs(kind="donut", labels=["a"], values=[1.0])

    def test_too_many_points_is_rejected(self):
        with self.assertRaises(ValueError):
            chart.ChartArgs(
                kind="line",
                labels=[str(i) for i in range(chart.MAX_POINTS + 1)],
                values=[1.0] * (chart.MAX_POINTS + 1),
            )


class ChartDispatchTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.invoker = ToolInvoker(
            build_registry(chart_output_dir=self._tmp.name)
        )

    def tearDown(self):
        self._tmp.cleanup()

    def test_reachable_through_the_registry(self):
        result = run(
            self.invoker.invoke(
                "render_chart",
                {"kind": "line", "labels": ["a", "b"], "values": [1.0, 2.0]},
            )
        )
        self.assertTrue(result.ok)
        self.assertTrue(Path(result.value["path"]).is_file())

    def test_bad_arguments_are_classified_like_every_other_tool(self):
        result = run(
            self.invoker.invoke(
                "render_chart",
                {"kind": "bar", "labels": ["a", "b"], "values": [1.0]},
            )
        )
        self.assertFalse(result.ok)
        self.assertEqual(result.failure.kind, ErrorKind.BAD_ARGUMENTS)

    def test_the_tool_is_advertised_with_its_enum_hint(self):
        schema = self.invoker.registry.get("render_chart").parameters()
        self.assertIn("bar", schema["properties"]["kind"]["description"])
        self.assertEqual(
            sorted(schema["required"]), ["kind", "labels", "values"]
        )


if __name__ == "__main__":
    unittest.main()
