"""Unit tests for pipeline_config: deep merge, profile resolution, command building."""
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "harness"))
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "agents" / "tools" / "common"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "harness" / "core"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "harness" / "stages"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "harness" / "web"))

import pipeline_config as pc


class DeepMergeTests(unittest.TestCase):
    def test_nested_override_keeps_siblings(self):
        base = {"training": {"a": 1, "b": 2}, "wifi": {"x": "cj"}}
        override = {"training": {"b": 99}}
        out = pc.deep_merge(base, override)
        self.assertEqual(out["training"]["b"], 99)   # overridden
        self.assertEqual(out["training"]["a"], 1)    # sibling preserved
        self.assertEqual(out["wifi"]["x"], "cj")     # untouched branch preserved

    def test_does_not_mutate_base(self):
        base = {"training": {"a": 1}}
        pc.deep_merge(base, {"training": {"a": 2}})
        self.assertEqual(base["training"]["a"], 1)


class LoadConfigTests(unittest.TestCase):
    def _write(self, text):
        f = tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False, encoding="utf-8")
        f.write(text)
        f.close()
        self.addCleanup(lambda: Path(f.name).unlink(missing_ok=True))
        return f.name

    def test_defaults_filled_and_user_wins(self):
        path = self._write(
            "training:\n  gpu_idle_threshold_util: 5\n"
            "active_profile: n1d5\n"
        )
        cfg = pc.load_config(path)
        self.assertEqual(cfg["training"]["gpu_idle_threshold_util"], 5)      # user override
        self.assertEqual(cfg["training"]["monitor_interval_sec"], 300)        # default filled
        self.assertEqual(cfg["wifi"]["external"], "cj")                       # default branch
        self.assertEqual(cfg["active_profile"], "n1d5")

    def test_missing_file_errors(self):
        with self.assertRaises(SystemExit):
            pc.load_config(tempfile.gettempdir() + "/does-not-exist-9z.yaml")


class ResolveProfileTests(unittest.TestCase):
    CFG = {"active_profile": "n1d7",
           "profiles": {"n1d5": {"train_cwd": "/a"}, "n1d7": {"train_cwd": "/b"}}}

    def test_default_uses_active_profile(self):
        name, prof = pc.resolve_profile(self.CFG)
        self.assertEqual(name, "n1d7")
        self.assertEqual(prof["train_cwd"], "/b")

    def test_cli_override(self):
        name, prof = pc.resolve_profile(self.CFG, cli_profile="n1d5")
        self.assertEqual(name, "n1d5")
        self.assertEqual(prof["train_cwd"], "/a")

    def test_unknown_profile_errors(self):
        with self.assertRaises(SystemExit):
            pc.resolve_profile(self.CFG, cli_profile="nope")


class BuildTrainCmdTests(unittest.TestCase):
    PROFILE = {
        "dataset_path": "/data/ds",
        "output_dir": "/data/out",
        "max_steps": 100000,
        "train_cmd_template": (
            "CUDA_VISIBLE_DEVICES={gpu} uv run python launch.py\n"
            "  --dataset-path {dataset_path} --output-dir {output_dir}\n"
            "  --gpu-id {gpu} --max-steps {max_steps}"
        ),
    }

    def test_substitutes_all_placeholders(self):
        cmd = pc.build_train_cmd(self.PROFILE, gpu_id=1)
        self.assertIn("CUDA_VISIBLE_DEVICES=1", cmd)
        self.assertIn("--gpu-id 1", cmd)              # gpu replaced in BOTH positions
        self.assertIn("--dataset-path /data/ds", cmd)
        self.assertIn("--output-dir /data/out", cmd)
        self.assertIn("--max-steps 100000", cmd)
        self.assertNotIn("{", cmd)                    # no leftover placeholders
        self.assertNotIn("\n", cmd)                   # folded to one line

    def test_missing_template_errors(self):
        with self.assertRaises(SystemExit):
            pc.build_train_cmd({"dataset_path": "x", "output_dir": "y", "max_steps": 1}, 0)


if __name__ == "__main__":
    unittest.main()
