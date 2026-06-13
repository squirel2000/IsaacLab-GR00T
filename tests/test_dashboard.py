"""Unit tests for dashboard.build_status + pipeline_progress merge."""
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pipeline"))

import dashboard as dash
import pipeline_progress as pp
from pipeline_state import ORDER


class ProgressTests(unittest.TestCase):
    def setUp(self):
        self._orig = pp.PROGRESS_PATH
        pp.PROGRESS_PATH = Path(tempfile.mkdtemp()) / "progress.json"
        self.addCleanup(lambda: setattr(pp, "PROGRESS_PATH", self._orig))

    def test_update_and_read_merge(self):
        pp.update("download", done=5, total=10, pct=50.0)
        d = pp.read()
        self.assertEqual(d["phase"], "download")
        self.assertEqual(d["download"]["pct"], 50.0)
        pp.update("download", rate_mbps=88)            # merge into same phase
        d = pp.read()
        self.assertEqual(d["download"]["done"], 5)      # preserved
        self.assertEqual(d["download"]["rate_mbps"], 88)

    def test_phase_switch_keeps_prior(self):
        pp.update("download", done=10)
        pp.update("deploy", done=1)
        d = pp.read()
        self.assertEqual(d["phase"], "deploy")
        self.assertIn("download", d)                    # prior phase retained

    def test_read_missing_is_empty(self):
        pp.PROGRESS_PATH = Path(tempfile.mkdtemp()) / "nope.json"
        self.assertEqual(pp.read(), {})


class BuildStatusTests(unittest.TestCase):
    STATE = {"current": "DOWNLOADING", "profile": "n1d7",
             "data": {"gpu_id": 1, "zip_name": "x.zip", "runid": "r", "zip_size": 26_000_000_000},
             "stages": {"TRAINING": {"status": "done", "attempts": 1}}}
    PROGRESS = {"phase": "download", "updated_at": "2026-06-09T09:30:00",
                "download": {"done": 5, "total": 10, "pct": 50.0, "rate_mbps": 88}}
    METRICS = [{"step": 10, "loss": 0.5}, {"step": 20, "loss": 0.3}]

    def test_shape(self):
        out = dash.build_status(self.STATE, self.PROGRESS, self.METRICS,
                                partial_bytes=5, wifi="CJ86GJI4_5G")
        self.assertEqual(out["current"], "DOWNLOADING")
        self.assertEqual([s["name"] for s in out["stages"]], [s.value for s in ORDER])
        tr = next(s for s in out["stages"] if s["name"] == "TRAINING")
        self.assertEqual(tr["status"], "done")
        pend = next(s for s in out["stages"] if s["name"] == "REPORTING")
        self.assertEqual(pend["status"], "pending")     # default for unstarted stages
        self.assertEqual(out["loss"]["steps"], [10, 20])
        self.assertEqual(out["partial_download_bytes"], 5)
        self.assertEqual(out["wifi"], "CJ86GJI4_5G")
        self.assertEqual(out["progress"]["download"]["pct"], 50.0)

    def test_series_split_by_metric(self):
        metrics = [
            {"step": 10, "loss": 0.5, "lr": 1e-4, "grad_norm": 2.0},
            {"step": 20, "loss": 0.3, "lr": 9e-5, "grad_norm": 1.5},
            {"step": 20, "eval_loss": 0.4},
        ]
        out = dash.build_status({"current": "TRAINING"}, {}, metrics)
        s = out["series"]
        self.assertEqual(s["train_loss"]["vals"], [0.5, 0.3])
        self.assertEqual(s["eval_loss"]["vals"], [0.4])
        self.assertEqual(s["eval_loss"]["steps"], [20])
        self.assertEqual(s["grad_norm"]["steps"], [10, 20])
        self.assertEqual(s["lr"]["vals"], [1e-4, 9e-5])

    def test_done_marks_terminal_stage(self):
        out = dash.build_status({"current": "DONE", "stages": {}}, {}, [])
        done = next(s for s in out["stages"] if s["name"] == "DONE")
        self.assertEqual(done["status"], "done")     # terminal chip shows green

    def test_downsample_caps_points(self):
        big = list(range(5000))
        out = dash.build_status({"current": "TRAINING"}, {},
                                [{"step": i, "loss": 1.0} for i in big])
        self.assertLessEqual(len(out["loss"]["steps"]), dash._MAX_POINTS + 1)
        self.assertEqual(out["loss"]["steps"][-1], 4999)   # last point kept


if __name__ == "__main__":
    unittest.main()
