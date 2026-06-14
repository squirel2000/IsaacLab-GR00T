#!/usr/bin/env python3
"""End-to-end automation orchestrator for IsaacLab-GR00T fine-tuning.

Walks a persistent state machine
    IDLE -> GPU_WAIT -> TRAINING -> DOWNLOADING -> DEPLOYING -> REPORTING -> DONE
calling one module per stage and persisting progress after each, so a restart resumes
from where it left off (training keeps running on Pegasus regardless of local drops).

Wi-Fi is managed at stage boundaries: the external network (CJ86GJI4_5G) reaches Pegasus
for GPU_WAIT/TRAINING/DOWNLOADING; the deployer switches to the LAN (OMAP-Motion-5G) to
reach asus-4090; finalize switches back to external and confirms internet.

Usage (PowerShell):
    python pipeline_runner.py --reset            # start fresh from IDLE
    python pipeline_runner.py --resume           # continue after a disconnect (default)
    python pipeline_runner.py --status           # show current state, do nothing
    python pipeline_runner.py --profile n1d5     # pick the N1.5 (conda) profile
"""
from __future__ import annotations

import argparse
import os
import sys

import pegasus as pg
import pipeline_config as pc
import pipeline_progress as pp
import net_util
import gpu_monitor
import training_monitor
import downloader
import deployer
import report_generator
from pipeline_state import PipelineState, Stage
from pipeline_logging import get_logger

log = get_logger("pipeline")

# Stages that need the external network (to reach Pegasus). DEPLOYING switches to the LAN
# itself inside deployer.run; REPORTING is purely local.
_PEGASUS_STAGES = {Stage.GPU_WAIT, Stage.TRAINING, Stage.DOWNLOADING}

HANDLERS = {
    Stage.GPU_WAIT: gpu_monitor.run,
    Stage.TRAINING: training_monitor.run,
    Stage.DOWNLOADING: downloader.run,
    Stage.DEPLOYING: deployer.run,
    Stage.REPORTING: report_generator.generate,
}


def finalize(config: dict, state: PipelineState) -> None:
    """收尾: return to the external (internet) Wi-Fi after the LAN deploy, confirm online."""
    if state.get("finalized"):
        return
    if net_util.ensure_external(config):         # ping-confirmed; drops LAN if needed
        log.info("Internet confirmed; back on the external network.")
    else:
        log.warning("Could not confirm internet after deploy; switch Wi-Fi to %s manually.",
                    config["wifi"]["external"])
    state.record_output("finalized", True)


def run_pipeline(config: dict, state: PipelineState, profile_name: str) -> None:
    """Drive the state machine to DONE, resuming from state.current."""
    state.set_profile(profile_name)
    log.info("Pipeline start: profile=%s, resuming at %s", profile_name, state.current.value)

    while state.current is not Stage.DONE:
        stage = state.current
        if stage is Stage.IDLE:                      # no work; just advance
            state.complete(stage)
            continue

        if stage in _PEGASUS_STAGES:
            net_util.ensure_reachable(config)    # switch to external only if internet is down

        log.info("=== Stage %s (attempt %d) ===", stage.value, state.attempts(stage) + 1)
        state.begin(stage)
        try:
            HANDLERS[stage](config, state)
        except SystemExit:
            raise
        except BaseException as e:                   # noqa: BLE001 - record any failure
            state.fail(stage, f"{type(e).__name__}: {e}")
            log.error("Stage %s failed: %s", stage.value, e)
            raise SystemExit(f"pipeline halted at {stage.value}: {e}\n"
                             f"  fix the cause, then re-run with --resume.")
        state.complete(stage)
        log.info("Stage %s done -> next: %s", stage.value, state.current.value)

    finalize(config, state)
    _print_summary(state)


def _print_summary(state: PipelineState) -> None:
    d = state.data
    print("\n" + "=" * 60)
    print("  PIPELINE COMPLETE")
    print("=" * 60)
    print(f"  profile     : {state.profile}")
    print(f"  GPU used    : index {d.get('gpu_id')}")
    print(f"  checkpoint  : {d.get('zip_local')}")
    print(f"  deployed to : {d.get('deploy_host')}:{d.get('deploy_remote')}")
    print(f"  report      : {d.get('report_path')}")
    print("=" * 60)


def show_status(state: PipelineState) -> None:
    """Print the current stage + per-stage record + key outputs (no side effects)."""
    print(f"current stage : {state.current.value}")
    print(f"profile       : {state.profile}")
    for name, rec in state.stages.items():
        line = f"  {name:<12} {rec.get('status','pending'):<12} attempts={rec.get('attempts',0)}"
        if rec.get("error"):
            line += f"  error={rec['error']}"
        print(line)
    d = state.data
    if d:
        print("outputs:")
        for k in ("gpu_id", "runid", "zip_local", "zip_size", "deploy_remote", "report_path"):
            if k in d:
                print(f"  {k} = {d[k]}")


# --------------------------------------------------------------------------- #
#  Reusable commands (shared by this module's main() and gr00t_pipeline.py)
# --------------------------------------------------------------------------- #
def bootstrap(config_path=None, profile=None):
    """Load config (+ make the Pegasus password authoritative) and the saved state."""
    config = pc.load_config(config_path)
    pw = (config.get("pegasus") or {}).get("password")
    if pw:
        os.environ["PEGASUS_PASSWORD"] = pw
        pg.PEGASUS_PASSWORD = pw
    profile_name, _ = pc.resolve_profile(config, profile)
    return config, PipelineState.load(), profile_name


def cmd_stop(config, state) -> None:
    """Stop the detached run and mark the state (so --status is honest)."""
    print(training_monitor.stop_run(config, state))
    state.stop(state.current)
    print(f"  state: {state.current.value} marked 'stopped'. --resume to continue from the "
          f"last checkpoint, or --reset to start over.")


def cmd_run(config, state, profile_name, reset=False) -> None:
    """Reset (optional) then drive the pipeline; restore connectivity on failure."""
    if reset:
        state.reset()
        pp.clear()
        log.info("State reset to IDLE.")
    try:
        run_pipeline(config, state, profile_name)
    except SystemExit as e:
        log.error(str(e))
        try:
            net_util.ensure_reachable(config)    # only switches if actually offline
        except Exception:                        # noqa: BLE001
            pass
        raise


def add_run_args(ap: argparse.ArgumentParser) -> None:
    """Attach the run/resume/reset/status/stop + profile/config args (shared with the CLI)."""
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--resume", action="store_true", help="continue from saved state (default)")
    mode.add_argument("--reset", action="store_true", help="reset to IDLE and run from the start")
    mode.add_argument("--status", action="store_true", help="print current state and exit")
    mode.add_argument("--stop", action="store_true", help="terminate the detached run and exit")
    ap.add_argument("--profile", choices=["n1d5", "n1d7"],
                    help="training profile (default: config active_profile)")
    ap.add_argument("--config", help="path to config.yaml (default: ./config.yaml)")


def run_cli(args) -> None:
    """Dispatch a parsed run-args namespace (used by main() and gr00t_pipeline.py 'run')."""
    config, state, profile_name = bootstrap(getattr(args, "config", None),
                                            getattr(args, "profile", None))
    if getattr(args, "status", False):
        return show_status(state)
    if getattr(args, "stop", False):
        return cmd_stop(config, state)
    cmd_run(config, state, profile_name, reset=getattr(args, "reset", False))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    add_run_args(ap)
    try:
        run_cli(ap.parse_args())
    except SystemExit as e:
        if e.code not in (0, None):
            sys.exit(e.code if isinstance(e.code, int) else 1)


if __name__ == "__main__":
    main()
