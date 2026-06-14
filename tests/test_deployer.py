"""Unit tests for deployer.connect_ssh retry logic (paramiko mocked)."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pipeline"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts" / "common"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts" / "pipeline"))

import deployer


class FakeClient:
    def __init__(self, should_fail):
        self.should_fail = should_fail
        self.connected = False
        self.closed = False

    def connect(self, **kwargs):
        if self.should_fail:
            raise ConnectionError("refused")
        self.connected = True

    def close(self):
        self.closed = True


def make_factory(fail_seq):
    it = iter(fail_seq)
    created = []

    def factory():
        c = FakeClient(next(it))
        created.append(c)
        return c

    return factory, created


class ConnectSshTests(unittest.TestCase):
    def test_succeeds_after_retries(self):
        factory, created = make_factory([True, True, False])
        client = deployer.connect_ssh("h", "u", "pw", retry_max=3, interval=0,
                                      client_factory=factory)
        self.assertTrue(client.connected)
        self.assertEqual(len(created), 3)          # two failed, third succeeded
        self.assertTrue(created[0].closed)         # failed clients are closed

    def test_raises_after_exhausting_retries(self):
        factory, created = make_factory([True, True, True])
        with self.assertRaises(RuntimeError):
            deployer.connect_ssh("h", "u", "pw", retry_max=3, interval=0,
                                 client_factory=factory)
        self.assertEqual(len(created), 3)

    def test_first_try_success(self):
        factory, created = make_factory([False])
        client = deployer.connect_ssh("h", "u", "pw", retry_max=3, interval=0,
                                      client_factory=factory)
        self.assertTrue(client.connected)
        self.assertEqual(len(created), 1)


if __name__ == "__main__":
    unittest.main()
