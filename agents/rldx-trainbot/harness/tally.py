#!/usr/bin/env python3
"""Authoritative Phase-0 LIBERO tally, computed from simulation_results.csv.

Why not video filenames: rollout_policy.py targets int(n_episodes/n_envs) episodes per
env but only breaks once EVERY env reaches that target, so faster envs overshoot and
write extra videos (55 files for a 50-episode target). Filename counting therefore
inflates the denominator and is not deduplicated. save_results_to_csv() instead writes
one row per (env_idx, episode_idx) with an explicit int(success), replacing any prior
row for the same key -- that is the real per-episode record.

Also reports per-suite rates, because the sweep runs hardest-suite-first: a single
rolling average is biased downward until the easy suites land.

Emits JSON on stdout. Usage: tally.py [output_root]  (defaults to the Phase-0 repro dir).

Promoted from tmp/tally_phase0.py (openspec task 2.4) -- see
openspec/changes/add-rldx1-cansorting-eval/tasks.md task 2.6 for the regression check
this feeds (confirm it still reports the same 97.47% suite mean / per-suite numbers
that tmp/tally_phase0.py measured before promotion).
"""
import csv, glob, json, os, sys

ROOT = sys.argv[1] if len(sys.argv) > 1 else \
    "/data/VLA/tingying/RLDX-1/output_final/libero/phase0_repro"

# suite -> (episodes per task, published target, label)
SUITES = {
    "libero_spatial": (20, 98.6, "Spatial"),
    "libero_object":  (20, 98.6, "Object"),
    "libero_goal":    (20, 98.6, "Goal"),
    "libero_10":      (50, 95.3, "Long"),
}
TRUTHY = {"1", "1.0", "true", "True", "TRUE"}


def tally_task(d):
    p = os.path.join(d, "simulation_results.csv")
    if not os.path.exists(p):
        return 0, 0, 0
    try:
        rows = list(csv.DictReader(open(p)))
    except Exception:
        return 0, 0, 0
    seen, succ = set(), 0
    for r in rows:
        key = (r.get("env_idx"), r.get("episode_idx"))
        if key in seen:          # defensive: csv should already be deduped
            continue
        seen.add(key)
        if str(r.get("success", "")).strip() in TRUTHY:
            succ += 1
    return len(seen), succ, len(rows) - len(seen)


def main() -> int:
    out = {"root": ROOT, "suites": {}, "short_tasks": [], "dupes": 0}
    g_ep = g_ok = 0
    short_of_target = 0

    for suite, (nep, target, label) in SUITES.items():
        ep = ok = 0
        tasks_complete = 0
        dirs = sorted(d for d in glob.glob(os.path.join(ROOT, suite, "*")) if os.path.isdir(d))
        for d in dirs:
            n, s, dup = tally_task(d)
            out["dupes"] += dup
            ep += n
            ok += s
            if n >= nep:
                tasks_complete += 1
            elif n > 0:
                out["short_tasks"].append(
                    {"suite": suite, "task": os.path.basename(d)[:60], "have": n, "want": nep})
                short_of_target += nep - n
        out["suites"][suite] = {
            "label": label, "episodes": ep, "success": ok,
            "expected": 10 * nep, "tasks_seen": len(dirs), "tasks_complete": tasks_complete,
            "rate": round(100.0 * ok / ep, 2) if ep else None,
            "target": target,
            "delta": round(100.0 * ok / ep - target, 2) if ep else None,
        }
        g_ep += ep
        g_ok += ok

    # GATE METRIC: the mean over the four suites, each weighted equally (task-weighted).
    #
    # This is how LIBERO is conventionally reported (4 suites x 10 tasks), and it is what
    # the 97.4 / 97.8 figures refer to. An episode-weighted total is NOT comparable here
    # because we run 50 episodes/task on libero_10 and 20 on the rest, so the hardest suite
    # takes 500 of 1100 episodes (45%) and drags the total ~1.4pp below the suite mean.
    # Gating on the episode-weighted number would manufacture a false MISS. It's still
    # reported below, as a secondary progress signal only -- never as the gate.
    complete = [v for v in out["suites"].values() if v["tasks_complete"] == 10 and v["rate"] is not None]
    all_complete = len(complete) == len(SUITES)
    suite_mean = round(sum(v["rate"] for v in complete) / len(complete), 2) if complete else None

    out["overall"] = {
        # gate (primary, task-weighted)
        "suite_mean": suite_mean,
        "suite_mean_is_final": all_complete,
        "suites_complete": len(complete),
        "readme_target": 97.4,
        "paper_target": 97.8,
        "delta_readme": round(suite_mean - 97.4, 2) if all_complete and suite_mean is not None else None,
        "delta_paper": round(suite_mean - 97.8, 2) if all_complete and suite_mean is not None else None,
        "gate": (None if not all_complete else ("PASS" if suite_mean >= 96.4 else "MISS")),
        # secondary / progress only (episode-weighted)
        "episodes": g_ep,
        "success": g_ok,
        "expected": 1100,
        "episode_weighted_rate": round(100.0 * g_ok / g_ep, 2) if g_ep else None,
        "progress": round(100.0 * g_ep / 1100, 1),
        "episodes_short_of_target": short_of_target,
    }

    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
