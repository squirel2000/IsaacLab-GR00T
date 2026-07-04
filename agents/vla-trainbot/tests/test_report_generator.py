"""Unit tests for report_generator pure builders."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "harness"))
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "agents" / "tools" / "common"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "harness" / "core"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "harness" / "stages"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "harness" / "web"))

import report_generator as rg

STATE = {
    "current": "DONE",
    "profile": "n1d7",
    "data": {
        "gpu_id": 1,
        "output_dir": "/data/exp/N1_7_stage1",
        "zip_size": 123_456_789,
        "zip_local": r"D:\tmp\N1_7_stage1.zip",
        "deploy_host": "asus@192.168.32.185",
        "deploy_remote": "/home/asus/Gits/IsaacLab-GR00T/artifacts/checkpoints/gr00t/N1_7_stage1",
    },
    "stages": {
        "TRAINING": {"status": "done", "attempts": 1,
                     "started_at": "2026-06-08T10:00:00", "finished_at": "2026-06-08T12:30:00"},
        "DEPLOYING": {"status": "done", "attempts": 2},
    },
}
CONFIG = {"active_profile": "n1d7",
          "profiles": {"n1d7": {"dataset_path": "/data/ds", "output_dir": "/data/exp/N1_7_stage1"}}}
METRICS = [{"step": 1000, "total": 100000, "loss": 0.5, "lr": 1e-4},
           {"step": 2000, "total": 100000, "loss": 0.3, "lr": 1e-4}]


class ContextTests(unittest.TestCase):
    def test_build_context(self):
        ctx = rg.build_context(STATE, METRICS, CONFIG)
        self.assertEqual(ctx["run_name"], "N1_7_stage1")
        self.assertEqual(ctx["profile"], "n1d7")
        self.assertEqual(ctx["gpu_id"], 1)
        self.assertAlmostEqual(ctx["final_loss"], 0.3)
        self.assertEqual(ctx["num_points"], 2)
        self.assertEqual(ctx["train_duration"], "2h 30m")
        self.assertAlmostEqual(ctx["zip_size_mb"], 123.456789)
        self.assertEqual(ctx["dataset"], "/data/ds")
        # stages cover the full ORDER, in order
        self.assertEqual(ctx["stages"][0][0], "IDLE")
        self.assertEqual([s[0] for s in ctx["stages"]][2], "TRAINING")


class HtmlTests(unittest.TestCase):
    def setUp(self):
        self.ctx = rg.build_context(STATE, METRICS, CONFIG)

    def test_chartjs_inlined_when_src_given(self):
        out = rg.build_html(self.ctx, METRICS, chart_js_src="window.Chart=1;/*lib*/")
        self.assertIn('id="lossChart"', out)
        self.assertIn("window.Chart=1;", out)        # library inlined
        self.assertIn("1000", out)                    # loss data inlined
        self.assertNotIn("<svg", out)
        self.assertIn('asus@192.168.32.185', out)     # deploy host shown
        self.assertIn("data-theme", out)              # dark/light toggle present

    def test_svg_fallback_when_no_src(self):
        out = rg.build_html(self.ctx, METRICS, chart_js_src=None)
        self.assertIn("<svg", out)
        self.assertNotIn('id="lossChart"', out)
        self.assertIn("polyline", out)

    def test_empty_metrics_svg_message(self):
        ctx = rg.build_context(STATE, [], CONFIG)
        out = rg.build_html(ctx, [], chart_js_src=None)
        self.assertIn("No training metrics", out)


class SvgTests(unittest.TestCase):
    def test_polyline_has_points(self):
        svg = rg._svg_line_chart([0, 50, 100], [1.0, 0.5, 0.2])
        self.assertIn("polyline", svg)
        self.assertIn("points=", svg)

    def test_empty(self):
        self.assertIn("No training metrics", rg._svg_line_chart([], []))


if __name__ == "__main__":
    unittest.main()
