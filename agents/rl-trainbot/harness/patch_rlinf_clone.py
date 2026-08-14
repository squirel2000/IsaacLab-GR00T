#!/usr/bin/env python3
"""Idempotently apply the RLinf-clone patches this project needs to run GR00T-N1.7
PPO on the Pegasus container (cgroup pids.max=2048, no Docker, headless EGL).

Run ON Pegasus against a fresh RLinf clone:
    python patch_rlinf_clone.py /data/VLA/tingying/RLinf

Each patch is guarded so re-running is a no-op. A .bak is written on first edit.
See agents/rl-trainbot/harness/README.md and openspec/changes/add-rlinf-gr00t-n17-libero-rl/ for why.

Patches:
  1. cluster.py  local ray.init honors env RLINF_RAY_NUM_CPUS (cap Ray's idle-worker
     pool; default fans out to all 64 cores -> ~1900 threads -> pthread_create EAGAIN).
  2. cluster.py  include_dashboard=False on the local ray.init (dashboard *Head procs
     add ~500 threads and blow the pid cap during the init burst).
  3. cluster.py  signal_handler wraps list_actors in try/except (with the dashboard
     off, the Ray state API is unavailable; keep the shutdown path from masking the
     real worker error).
  4. libero/venv.py  add ReconfigureDummyEnv (in-process vec env). RLinf's spawn-
     subprocess env workers crash in mujoco.MjModel.from_xml_string ("unknown
     exception"); creating the env in-process (proven to work) avoids it.
  5. libero/libero_env.py  use ReconfigureDummyEnv when RLINF_LIBERO_INPROCESS=1.
  6. config/libero_spatial_ppo_gr00t_n1d7.yaml  pin component_placement to one GPU
     (RLinf ignores CUDA_VISIBLE_DEVICES and its 'all' spreads across both GPUs).
"""
import sys
import pathlib

GPU_INDEX = "1"  # single-GPU pin (cooperative use; change for a different card)


def patch(path, edits, label):
    p = pathlib.Path(path)
    if not p.exists():
        print(f"  [SKIP] {label}: {path} not found")
        return
    s = orig = p.read_text(encoding="utf-8")
    for old, new, guard in edits:
        if guard in s:
            print(f"  [ok]  {label}: already applied ({guard[:40]}...)")
            continue
        if old not in s:
            print(f"  [WARN] {label}: anchor not found — upstream changed? ({old[:40]}...)")
            continue
        s = s.replace(old, new, 1)
        print(f"  [DONE] {label}")
    if s != orig:
        bak = p.with_suffix(p.suffix + ".bak")
        if not bak.exists():
            bak.write_text(orig, encoding="utf-8")
        p.write_text(s, encoding="utf-8")


def main():
    root = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "/data/VLA/tingying/RLinf")
    print(f"Patching RLinf clone at: {root}")

    # 1-3: cluster.py
    patch(
        root / "rlinf/scheduler/cluster/cluster.py",
        [
            # 1 + 2: local (fallback) ray.init — num_cpus + include_dashboard=False
            (
                '''            ray_init_kwargs = {
                "logging_level": Cluster.LOGGING_LEVEL,
                "namespace": Cluster.NAMESPACE,
            }
            if self._ray_code_sync_fragment is not None:''',
                '''            ray_init_kwargs = {
                "logging_level": Cluster.LOGGING_LEVEL,
                "namespace": Cluster.NAMESPACE,
                "include_dashboard": False,
            }
            import os as _os
            _ncpu = _os.environ.get("RLINF_RAY_NUM_CPUS")
            if _ncpu:
                ray_init_kwargs["num_cpus"] = int(_ncpu)
            if self._ray_code_sync_fragment is not None:''',
                "RLINF_RAY_NUM_CPUS",
            ),
            # 3: signal_handler list_actors non-fatal
            (
                '''            with without_http_proxies():
                alive_actors = list_actors(
                    filters=[
                        ("STATE", "=", "ALIVE"),
                        ("RAY_NAMESPACE", "=", Cluster.NAMESPACE),
                    ]
                )
            for actor_state in alive_actors:
                actor = ray.get_actor(actor_state.name)
                ray.kill(actor, no_restart=True)''',
                '''            try:
                with without_http_proxies():
                    alive_actors = list_actors(
                        filters=[
                            ("STATE", "=", "ALIVE"),
                            ("RAY_NAMESPACE", "=", Cluster.NAMESPACE),
                        ]
                    )
                for actor_state in alive_actors:
                    actor = ray.get_actor(actor_state.name)
                    ray.kill(actor, no_restart=True)
            except Exception as _e:
                print("[signal_handler] list_actors/kill skipped:", _e)''',
                "list_actors/kill skipped",
            ),
        ],
        "cluster.py (num_cpus + dashboard + sighandler)",
    )

    # 4: libero/venv.py — append in-process ReconfigureDummyEnv
    venv_add = '''

# RLINF_LIBERO_INPROCESS_WORKAROUND
from rlinf.envs.venv.venv import DummyEnvWorker as _DummyEnvWorker, DummyVectorEnv as _DummyVectorEnv


class ReconfigureDummyEnvWorker(_DummyEnvWorker):
    def reconfigure_env_fn(self, env_fn_param):
        try:
            self.env.close()
        except Exception:
            pass
        data = dict(env_fn_param)
        seed = data.pop("seed", None)
        self.env = OffScreenRenderEnv(**data)
        if seed is not None:
            self.env.seed(seed)
        return None


class ReconfigureDummyEnv(_DummyVectorEnv):
    def __init__(self, env_fns, **kwargs):
        BaseVectorEnv.__init__(self, env_fns, ReconfigureDummyEnvWorker, **kwargs)

    def reconfigure_env_fns(self, env_fns, id=None):
        self._assert_is_not_closed()
        id = self._wrap_id(id)
        if self.is_async:
            self._assert_id(id)
        for j, i in enumerate(id):
            self.workers[i].reconfigure_env_fn(env_fns[j])
'''
    vp = root / "rlinf/envs/libero/venv.py"
    if vp.exists():
        s = vp.read_text(encoding="utf-8")
        if "RLINF_LIBERO_INPROCESS_WORKAROUND" in s:
            print("  [ok]  venv.py: already applied")
        else:
            if not vp.with_suffix(".py.bak").exists():
                vp.with_suffix(".py.bak").write_text(s, encoding="utf-8")
            vp.write_text(s + venv_add, encoding="utf-8")
            print("  [DONE] venv.py: appended ReconfigureDummyEnv")

    # 5: libero_env.py — conditional in-process selection
    patch(
        root / "rlinf/envs/libero/libero_env.py",
        [
            (
                "        env_fns = self.get_env_fns()\n        self.env = ReconfigureSubprocEnv(env_fns)",
                '''        env_fns = self.get_env_fns()
        import os as _os
        if _os.environ.get("RLINF_LIBERO_INPROCESS") == "1":
            from rlinf.envs.libero.venv import ReconfigureDummyEnv as _RDE
            self.env = _RDE(env_fns)
        else:
            self.env = ReconfigureSubprocEnv(env_fns)''',
                "RLINF_LIBERO_INPROCESS",
            ),
        ],
        "libero_env.py (in-process selection)",
    )

    # 6: base configs — pin component_placement to one GPU (N1.5/N1.6/N1.7 all use
    # the same "actor,env,rollout: all" key in their libero_*_ppo_gr00t*.yaml)
    cfg_dir = root / "examples/embodiment/config"
    if cfg_dir.exists():
        for cfg in sorted(cfg_dir.glob("libero_*_ppo_gr00t*.yaml")):
            patch(
                cfg,
                [
                    (
                        "actor,env,rollout: all",
                        f'actor,env,rollout: "{GPU_INDEX}"',
                        f'actor,env,rollout: "{GPU_INDEX}"',
                    ),
                ],
                f"config {cfg.name} (single-GPU placement)",
            )
    print("Done.")


if __name__ == "__main__":
    main()
