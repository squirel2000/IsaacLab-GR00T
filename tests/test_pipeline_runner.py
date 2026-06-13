"""Orchestration tests for pipeline_runner: ordering, resume, no-repeat.

Stage handlers and the network are mocked, so this exercises the state-machine driving
logic without any live Pegasus/asus-4090/Wi-Fi access.
"""
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pipeline"))

import pipeline_runner as pr
from pipeline_state import PipelineState, Stage

CONFIG = {"wifi": {"external": "cj", "local": "omap",
                   "internet_check_ip": "8.8.8.8", "net_ready_timeout_sec": 60}}


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp()) / "pipeline_state.json"
        self.calls = []           # ordered (stage) handler invocations
        # network is a no-op that always "succeeds"
        for fn in ("switch_wifi", "ensure_reachable"):
            p = mock.patch.object(pr.net_util, fn, lambda *a, **k: True)
            p.start(); self.addCleanup(p.stop)
        p = mock.patch.object(pr.net_util, "wait_for_host", lambda *a, **k: True)
        p.start(); self.addCleanup(p.stop)

    def _handlers(self, failing=None):
        """Build a HANDLERS dict recording calls; ``failing`` stage raises once."""
        self._failed_once = set()

        def make(stage):
            def fn(config, state):
                self.calls.append(stage)
                if failing == stage and stage not in self._failed_once:
                    self._failed_once.add(stage)
                    raise RuntimeError("boom")
            return fn
        return {s: make(s) for s in pr.HANDLERS}

    def test_happy_path_order(self):
        with mock.patch.object(pr, "HANDLERS", self._handlers()):
            state = PipelineState.load(self.tmp)
            pr.run_pipeline(CONFIG, state, "n1d7")
        self.assertEqual(self.calls, [Stage.GPU_WAIT, Stage.TRAINING, Stage.DOWNLOADING,
                                      Stage.DEPLOYING, Stage.REPORTING])
        self.assertIs(state.current, Stage.DONE)
        self.assertTrue(state.get("finalized"))

    def test_resume_after_failure_no_repeat(self):
        handlers = self._handlers(failing=Stage.TRAINING)
        # first run: fails at TRAINING
        with mock.patch.object(pr, "HANDLERS", handlers):
            state = PipelineState.load(self.tmp)
            with self.assertRaises(SystemExit):
                pr.run_pipeline(CONFIG, state, "n1d7")
        self.assertIs(state.current, Stage.TRAINING)
        self.assertEqual(state.status(Stage.TRAINING), "failed")
        self.assertEqual(self.calls, [Stage.GPU_WAIT, Stage.TRAINING])

        # resume: reload from disk (simulating a restart), TRAINING now succeeds
        self.calls.clear()
        with mock.patch.object(pr, "HANDLERS", handlers):
            state2 = PipelineState.load(self.tmp)
            pr.run_pipeline(CONFIG, state2, "n1d7")
        # GPU_WAIT is NOT re-run; resumes at TRAINING through to the end
        self.assertEqual(self.calls, [Stage.TRAINING, Stage.DOWNLOADING,
                                      Stage.DEPLOYING, Stage.REPORTING])
        self.assertIs(state2.current, Stage.DONE)
        self.assertEqual(state2.attempts(Stage.TRAINING), 2)   # 1 failed + 1 success
        self.assertEqual(state2.attempts(Stage.GPU_WAIT), 1)   # ran exactly once

    def test_idle_auto_advances(self):
        with mock.patch.object(pr, "HANDLERS", self._handlers()):
            state = PipelineState.load(self.tmp)
            self.assertIs(state.current, Stage.IDLE)
            pr.run_pipeline(CONFIG, state, "n1d7")
        # IDLE produced no handler call but the run still completed
        self.assertNotIn(Stage.IDLE, self.calls)
        self.assertIs(state.current, Stage.DONE)


if __name__ == "__main__":
    unittest.main()
