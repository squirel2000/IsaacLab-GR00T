#!/usr/bin/env python3
"""Single entry point for the IsaacLab-GR00T N1.7 fine-tune → deploy automation.

One CLI drives the whole flow by reusing one implementation per behavior (nothing is
reimplemented): `run/status/stop` share ``pipeline_runner``'s dispatcher, `dashboard` calls
``dashboard.serve`` (scripts/pipeline/web), and `finetune` delegates to ``run_finetune.main``
(scripts/common/run_finetune.py — the standalone low-level tool).

    python gr00t_pipeline.py run                 # resume (or start) the full pipeline
    python gr00t_pipeline.py run --reset         # wipe state and run from the beginning
    python gr00t_pipeline.py run --profile n1d5  # pick a training profile (default: config)
    python gr00t_pipeline.py status              # print pipeline state and exit
    python gr00t_pipeline.py stop                # terminate the detached training run
    python gr00t_pipeline.py dashboard           # read-only live web dashboard
    python gr00t_pipeline.py finetune [run|monitor|watch|download|status|stop|selftest]
                                                 # fine-tune-only tool (run_finetune.py CONFIG block)

Every subcommand accepts --config PATH (default ./config.yaml). The `finetune` passthrough
uses run_finetune.py's own CONFIG block, NOT config.yaml — it's the manual low-level tool.
"""
import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent
_PIPE = ROOT / "scripts" / "pipeline"
sys.path.insert(0, str(ROOT))                            # project_paths (workspace helper)
sys.path.insert(0, str(ROOT / "scripts" / "common"))     # pegasus / run_finetune / wifi_switch
for _sub in ("core", "stages", "web"):                   # pipeline_* + stage modules + dashboard
    sys.path.insert(0, str(_PIPE / _sub))


def main() -> None:
    ap = argparse.ArgumentParser(prog="gr00t_pipeline.py", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="resume (or start) the full pipeline")
    p_run.add_argument("--reset", action="store_true", help="wipe state and run from the start")
    p_run.add_argument("--profile", choices=["n1d5", "n1d7"], help="training profile")
    p_run.add_argument("--config", help="path to config.yaml (default ./config.yaml)")

    for name, helptext in (("status", "print pipeline state and exit"),
                           ("stop", "terminate the detached training run and exit")):
        sp = sub.add_parser(name, help=helptext)
        sp.add_argument("--config", help="path to config.yaml (default ./config.yaml)")

    p_dash = sub.add_parser("dashboard", help="read-only live web dashboard")
    p_dash.add_argument("--host", default="0.0.0.0", help="bind host (default 0.0.0.0 for LAN)")
    p_dash.add_argument("--port", type=int, default=8770)
    p_dash.add_argument("--config", help="path to config.yaml (default ./config.yaml)")

    p_ft = sub.add_parser("finetune", help="fine-tune-only tool (delegates to run_finetune.py)")
    p_ft.add_argument("ft_cmd", nargs="?", default="run",
                      choices=["run", "start", "monitor", "watch", "download",
                               "status", "stop", "selftest"],
                      help="run_finetune action (default: run)")
    p_ft.add_argument("--insecure", action="store_true", help="skip TLS verification")

    args = ap.parse_args()

    if args.command == "dashboard":
        import dashboard
        dashboard.serve(args.host, args.port, args.config)
        return

    if args.command == "finetune":
        import run_finetune
        # run_finetune.main() reads sys.argv; rebuild it for the delegated call.
        sys.argv = ["run_finetune.py", args.ft_cmd] + (["--insecure"] if args.insecure else [])
        run_finetune.main()
        return

    # run / status / stop -> the pipeline runner's shared dispatcher (run_cli)
    import pipeline_runner as pr
    ns = SimpleNamespace(config=args.config, profile=getattr(args, "profile", None),
                         reset=getattr(args, "reset", False),
                         status=(args.command == "status"),
                         stop=(args.command == "stop"),
                         resume=(args.command == "run"))
    try:
        pr.run_cli(ns)
    except SystemExit as e:
        if e.code not in (0, None):
            sys.exit(e.code if isinstance(e.code, int) else 1)


if __name__ == "__main__":
    main()
