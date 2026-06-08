"""Unit tests for training_monitor pure logic: log parsing + cfg building."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

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


if __name__ == "__main__":
    unittest.main()
