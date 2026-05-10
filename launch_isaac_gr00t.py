#!/usr/bin/env python3
"""Compatibility wrapper for launching Isaac GR00T via launch_isaac_policy.py."""

import subprocess
import sys
from pathlib import Path


def main():
    root = Path(__file__).resolve().parent
    args = ["--save-video" if arg == "--save-img" else arg for arg in sys.argv[1:]]
    subprocess.run([sys.executable, str(root / "launch_isaac_policy.py"), "--policy", "gr00t", *args], check=True)


if __name__ == "__main__":
    main()
