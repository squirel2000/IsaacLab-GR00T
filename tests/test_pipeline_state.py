"""Unit tests for the pipeline state machine."""
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pipeline"))

from pipeline_state import PipelineState, Stage, next_stage, ORDER


class StateTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp()) / "pipeline_state.json"

    def fresh(self):
        return PipelineState.load(self.tmp)

    def test_fresh_state_is_idle(self):
        s = self.fresh()
        self.assertIs(s.current, Stage.IDLE)
        self.assertFalse(self.tmp.exists())   # nothing written until save/begin

    def test_begin_increments_attempts_and_sets_in_progress(self):
        s = self.fresh()
        s.begin(Stage.GPU_WAIT)
        self.assertEqual(s.attempts(Stage.GPU_WAIT), 1)
        self.assertEqual(s.status(Stage.GPU_WAIT), "in_progress")
        s.begin(Stage.GPU_WAIT)               # retry
        self.assertEqual(s.attempts(Stage.GPU_WAIT), 2)

    def test_complete_advances_current(self):
        s = self.fresh()
        s.begin(Stage.GPU_WAIT)
        nxt = s.complete(Stage.GPU_WAIT)
        self.assertIs(nxt, Stage.TRAINING)
        self.assertIs(s.current, Stage.TRAINING)
        self.assertEqual(s.status(Stage.GPU_WAIT), "done")

    def test_fail_records_error_and_keeps_current(self):
        s = self.fresh()
        s.begin(Stage.TRAINING)
        s.fail(Stage.TRAINING, "boom")
        self.assertIs(s.current, Stage.TRAINING)
        self.assertEqual(s.status(Stage.TRAINING), "failed")
        self.assertEqual(s.stages["TRAINING"]["error"], "boom")

    def test_resume_round_trip(self):
        s = self.fresh()
        s.begin(Stage.DOWNLOADING)
        s.record_output("zip_local", r"D:\tmp\run.zip")
        # simulate a restart: load a brand new object from the same file
        s2 = PipelineState.load(self.tmp)
        self.assertIs(s2.current, Stage.DOWNLOADING)
        self.assertEqual(s2.get("zip_local"), r"D:\tmp\run.zip")
        self.assertEqual(s2.attempts(Stage.DOWNLOADING), 1)

    def test_reset_clears_progress(self):
        s = self.fresh()
        s.begin(Stage.TRAINING)
        s.record_output("runid", "abc")
        s.reset()
        self.assertIs(s.current, Stage.IDLE)
        self.assertEqual(s.stages, {})
        self.assertIsNone(s.get("runid"))

    def test_full_advance_sequence(self):
        s = self.fresh()
        stage = Stage.IDLE
        visited = [stage]
        for _ in range(len(ORDER) + 2):       # over-iterate; DONE must be a fixed point
            stage = next_stage(stage)
            visited.append(stage)
        self.assertEqual(visited[:len(ORDER)], ORDER)
        self.assertIs(next_stage(Stage.DONE), Stage.DONE)

    def test_profile_persists(self):
        s = self.fresh()
        s.set_profile("n1d7")
        self.assertEqual(PipelineState.load(self.tmp).profile, "n1d7")


if __name__ == "__main__":
    unittest.main()
