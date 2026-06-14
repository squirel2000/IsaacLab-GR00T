"""Unit tests for gpu_monitor pure parsing/selection logic."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pipeline"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts" / "common"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts" / "pipeline" / "core"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts" / "pipeline" / "stages"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts" / "pipeline" / "web"))

import gpu_monitor as gm


class ParseTests(unittest.TestCase):
    def test_parses_standard_csv(self):
        text = "0, 5, 1024\n1, 97, 71000\n"
        gpus = gm.parse_nvidia_smi(text)
        self.assertEqual(len(gpus), 2)
        self.assertEqual(gpus[0], {"index": 0, "util": 5, "mem_gb": 1024 / 1024})
        self.assertEqual(gpus[1]["index"], 1)
        self.assertAlmostEqual(gpus[1]["mem_gb"], 71000 / 1024)

    def test_ignores_blank_and_garbage_lines(self):
        text = "\nindex, utilization.gpu, memory.used\n0, 0, 12\n"  # header-ish line skipped
        gpus = gm.parse_nvidia_smi(text)
        self.assertEqual([g["index"] for g in gpus], [0])

    def test_na_fields_treated_as_busy(self):
        gpus = gm.parse_nvidia_smi("0, [N/A], [N/A]\n")
        self.assertEqual(gpus[0]["util"], 100)
        self.assertGreater(gpus[0]["mem_gb"], 1000)


class PickTests(unittest.TestCase):
    def test_picks_lowest_idle_index(self):
        gpus = [{"index": 0, "util": 3, "mem_gb": 0.5},
                {"index": 1, "util": 1, "mem_gb": 0.2}]
        self.assertEqual(gm.pick_idle_gpu(gpus, 10, 5.0), 0)

    def test_skips_busy_util(self):
        gpus = [{"index": 0, "util": 80, "mem_gb": 0.5},
                {"index": 1, "util": 2, "mem_gb": 0.2}]
        self.assertEqual(gm.pick_idle_gpu(gpus, 10, 5.0), 1)

    def test_skips_busy_memory(self):
        gpus = [{"index": 0, "util": 2, "mem_gb": 60.0},   # idle util but loaded memory
                {"index": 1, "util": 2, "mem_gb": 0.2}]
        self.assertEqual(gm.pick_idle_gpu(gpus, 10, 5.0), 1)

    def test_all_busy_returns_none(self):
        gpus = [{"index": 0, "util": 90, "mem_gb": 70},
                {"index": 1, "util": 95, "mem_gb": 70}]
        self.assertIsNone(gm.pick_idle_gpu(gpus, 10, 5.0))

    def test_priority_prefers_gpu1_when_both_idle(self):
        gpus = [{"index": 0, "util": 2, "mem_gb": 0.2},
                {"index": 1, "util": 2, "mem_gb": 0.2}]
        self.assertEqual(gm.pick_idle_gpu(gpus, 10, 5.0, priority=[1, 0]), 1)

    def test_priority_falls_back_when_preferred_busy(self):
        gpus = [{"index": 0, "util": 2, "mem_gb": 0.2},
                {"index": 1, "util": 90, "mem_gb": 70}]   # GPU1 busy -> fall back to GPU0
        self.assertEqual(gm.pick_idle_gpu(gpus, 10, 5.0, priority=[1, 0]), 0)


if __name__ == "__main__":
    unittest.main()
