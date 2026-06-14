"""Unit tests for net_util ping argument construction (both platforms)."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pipeline"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts" / "common"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts" / "pipeline"))

import net_util


class PingArgsTests(unittest.TestCase):
    def setUp(self):
        self._orig = net_util._IS_WINDOWS
        self.addCleanup(lambda: setattr(net_util, "_IS_WINDOWS", self._orig))

    def test_windows_args(self):
        net_util._IS_WINDOWS = True
        self.assertEqual(net_util._ping_args("1.2.3.4", 1, 1000),
                         ["ping", "-n", "1", "-w", "1000", "1.2.3.4"])

    def test_posix_args_convert_ms_to_sec(self):
        net_util._IS_WINDOWS = False
        self.assertEqual(net_util._ping_args("8.8.8.8", 2, 2000),
                         ["ping", "-c", "2", "-W", "2", "8.8.8.8"])

    def test_posix_timeout_floor_is_one_second(self):
        net_util._IS_WINDOWS = False
        args = net_util._ping_args("h", 1, 200)        # 0.2s rounds up to 1s floor
        self.assertEqual(args[args.index("-W") + 1], "1")


if __name__ == "__main__":
    unittest.main()
