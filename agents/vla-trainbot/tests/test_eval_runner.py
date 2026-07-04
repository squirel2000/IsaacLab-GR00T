import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "agents" / "tools" / "common"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "harness" / "core"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "harness" / "stages"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "harness" / "web"))

import eval_runner


class TestParseEvalLog(unittest.TestCase):
    def test_counts_success_and_rate(self):
        log = (
            "Episode 0 finished after 120 steps (Success: True, Terminated: tensor([True]), Truncated: tensor([False]))\n"
            "some unrelated line\n"
            "Episode 1 finished after 400 steps (Success: False, Terminated: tensor([False]), Truncated: tensor([True]))\n"
            "Episode 2 finished after 90 steps (Success: True, Terminated: tensor([True]), Truncated: tensor([False]))\n"
        )
        r = eval_runner.parse_eval_log(log)
        self.assertEqual(r["n"], 3)
        self.assertEqual(r["success"], 2)
        self.assertAlmostEqual(r["rate"], 2 / 3)

    def test_empty_log(self):
        self.assertEqual(eval_runner.parse_eval_log(""), {"n": 0, "success": 0, "rate": 0.0})


class TestBuildEvalConfig(unittest.TestCase):
    def test_one_run_points_at_checkpoint(self):
        import yaml
        cfg = {"eval": {"target_episodes": 50,
                        "backend_config": "scripts/eval/configs/gr00t_n17_openarm_o6.json"}}
        spec = yaml.safe_load(eval_runner.build_eval_config(cfg, "/data/ckpt/gr00t/myrun", "myrun"))
        self.assertEqual(spec["defaults"]["target"], 50)
        self.assertTrue(spec["defaults"]["headless"])
        self.assertEqual(len(spec["runs"]), 1)
        self.assertEqual(spec["runs"][0]["checkpoint"], "/data/ckpt/gr00t/myrun")
        self.assertEqual(spec["runs"][0]["tag"], "myrun")
        self.assertEqual(spec["runs"][0]["config"], cfg["eval"]["backend_config"])


class TestSkipWhenDisabled(unittest.TestCase):
    def test_run_noops_and_records_skip(self):
        rec = {}

        class FakeState:
            profile = "n1d7"
            def get(self, *a, **k):
                return None
            def record_outputs(self, **kw):
                rec.update(kw)

        # eval.enabled false -> must return immediately (no network) and record the skip.
        eval_runner.run({"eval": {"enabled": False}}, FakeState())
        self.assertTrue(rec.get("eval_skipped"))


if __name__ == "__main__":
    unittest.main()
