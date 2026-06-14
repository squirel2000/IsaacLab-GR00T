"""Unit tests for training_monitor pure logic: log parsing + cfg building."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pipeline"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts" / "common"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts" / "pipeline"))

import training_monitor as tm


class ExtractMetricsTests(unittest.TestCase):
    def test_loss_stamped_with_latest_step(self):
        text = (
            "Some banner line\n"
            "  5%|#         | 5000/100000 [01:23<26:00, 3.5it/s]\n"
            "{'loss': 0.4213, 'learning_rate': 9.8e-05, 'epoch': 0.5}\n"
        )
        recs, step, total = tm.extract_metrics(text)
        self.assertEqual(step, 5000)
        self.assertEqual(total, 100000)
        self.assertEqual(len(recs), 1)
        self.assertEqual(recs[0]["step"], 5000)
        self.assertEqual(recs[0]["total"], 100000)
        self.assertAlmostEqual(recs[0]["loss"], 0.4213)
        self.assertAlmostEqual(recs[0]["lr"], 9.8e-05)

    def test_carry_step_across_chunks(self):
        recs1, step1, total1 = tm.extract_metrics("7000/100000 [..]\n")
        # next chunk has a loss but no fresh progress line -> uses carried step
        recs2, step2, total2 = tm.extract_metrics(
            "{'loss': 0.1, 'learning_rate': 1e-5}\n", step1, total1)
        self.assertEqual(recs2[0]["step"], 7000)
        self.assertEqual(recs2[0]["total"], 100000)

    def test_loss_without_lr(self):
        recs, _, _ = tm.extract_metrics("{'loss': 1.5}\n")
        self.assertEqual(recs[0]["loss"], 1.5)
        self.assertIsNone(recs[0]["lr"])

    def test_no_metrics(self):
        recs, step, total = tm.extract_metrics("nothing useful here\n")
        self.assertEqual(recs, [])
        self.assertEqual((step, total), (0, 0))


class GradEvalTests(unittest.TestCase):
    def test_grad_norm_and_eval_loss_split(self):
        text = ("5000/300000 [..]\n"
                "{'loss': 0.5, 'grad_norm': 1.23, 'learning_rate': 9e-05}\n"
                "{'eval_loss': 0.42, 'eval_runtime': 12.0}\n")
        recs, _, _ = tm.extract_metrics(text)
        train = [r for r in recs if "loss" in r][0]
        self.assertAlmostEqual(train["grad_norm"], 1.23)
        self.assertAlmostEqual(train["lr"], 9e-05)
        ev = [r for r in recs if "eval_loss" in r][0]
        self.assertAlmostEqual(ev["eval_loss"], 0.42)
        self.assertNotIn("loss", ev)              # eval record carries no training loss


class ComputeEtaTests(unittest.TestCase):
    def test_session_rate(self):
        # baseline t=1000 step=135000; now t=1100 step=135200 total=300000 -> rate 2/s
        el, eta = tm.compute_eta(1000, 135000, 135200, 300000, 1100)
        self.assertEqual(el, 100)                 # session elapsed
        self.assertEqual(eta, 82400)              # (300000-135200)/2

    def test_no_eta_until_progress(self):
        self.assertEqual(tm.compute_eta(1000, 135000, 135000, 300000, 1050), (50, None))

    def test_no_baseline(self):
        self.assertEqual(tm.compute_eta(None, 0, 0, 0, 0), (None, None))

    def test_train_total_filters_eval_bars(self):
        text = ("135000/300000 [..]\n"            # training bar
                "  50%| 4/8 [..]\n"                # eval bar (total 8) -> must be ignored
                "{'loss': 0.1, 'learning_rate': 1e-5}\n")
        recs, step, total = tm.extract_metrics(text, train_total=300000)
        self.assertEqual((step, total), (135000, 300000))   # eval bar didn't hijack the step
        self.assertEqual(recs[0]["step"], 135000)


class BuildCfgTests(unittest.TestCase):
    CONFIG = {
        "local": {"download_dir": r"D:\tmp\ckpt"},
        "pegasus": {"state_root": "/data/VLA/tingying/pegasus_runs"},
        "training": {"monitor_interval_sec": 300},
    }
    PROFILE = {
        "train_cwd": "/data/VLA/.../Isaac-GR00T_n1d7",
        "env_activate": "true",
        "dataset_path": "/data/ds",
        "output_dir": "/data/exp/run/N1_7_stage1",
        "max_steps": 100000,
        "train_cmd_template": "CUDA_VISIBLE_DEVICES={gpu} uv run python launch.py "
                              "--dataset-path {dataset_path} --output-dir {output_dir} "
                              "--gpu-id {gpu} --max-steps {max_steps}",
    }

    def test_builds_expected_cfg(self):
        cfg = tm.build_finetune_cfg(self.CONFIG, self.PROFILE, gpu_id=1)
        self.assertEqual(cfg["conda_activate"], "true")        # uv: no-op activate
        self.assertEqual(cfg["run_name"], "N1_7_stage1")
        self.assertEqual(cfg["zip_name"], "N1_7_stage1.zip")
        self.assertEqual(cfg["keep_checkpoint"], "checkpoint-100000")
        self.assertIn("checkpoint-100000", cfg["cleanup_cmd"])
        self.assertEqual(cfg["zip_parent"], "/data/exp/run")
        self.assertEqual(cfg["poll"], 300)
        self.assertEqual(cfg["local_dir"], r"D:\tmp\ckpt")
        # GPU injected into the command (both positions):
        self.assertIn("CUDA_VISIBLE_DEVICES=1", cfg["train_cmd"])
        self.assertIn("--gpu-id 1", cfg["train_cmd"])
        self.assertIn("--max-steps 100000", cfg["train_cmd"])


class InfoFromStateTests(unittest.TestCase):
    class FakeState:
        def __init__(self, d):
            self._d = d

        def get(self, k, default=None):
            return self._d.get(k, default)

    def test_reconstructs_from_runid(self):
        cfg = {"state_root": "/data/VLA/tingying/pegasus_runs", "zip_name": "N1_7.zip"}
        st = self.FakeState({"runid": "N1_7_x_20260609", "zip_name": "N1_7.zip"})
        info = tm.info_from_state(cfg, st)
        self.assertEqual(info["runid"], "N1_7_x_20260609")
        self.assertEqual(info["state_abs"],
                         "/data/VLA/tingying/pegasus_runs/N1_7_x_20260609")
        self.assertEqual(info["state_rel"],
                         "VLA/tingying/pegasus_runs/N1_7_x_20260609")  # /data/ stripped
        self.assertEqual(info["zip_name"], "N1_7.zip")

    def test_zip_name_falls_back_to_cfg(self):
        cfg = {"state_root": "/data/x", "zip_name": "fromcfg.zip"}
        info = tm.info_from_state(cfg, self.FakeState({"runid": "r1"}))
        self.assertEqual(info["zip_name"], "fromcfg.zip")
        self.assertEqual(info["state_abs"], "/data/x/r1")


if __name__ == "__main__":
    unittest.main()
