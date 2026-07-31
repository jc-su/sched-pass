"""sitecustomize.py -- zero-edit bootstrap for the SGLang plugin (opt-in),
LAZY by hard-won necessity.

Root-caused 2026-07-07 (bisect ladder in ROADMAP.md): importing torch at
interpreter start -- even with the plugin doing NOTHING else -- corrupts
Qwen-3B generation under SGLang (fluent, wrong-content text). The launcher
configures torch-affecting environment before ITS torch import; an earlier
import freezes different defaults. So this file must import NOTHING heavy.

Design: install a meta-path finder that watches for the HOST importing
`sglang.srt.managers.scheduler`. When that import completes (torch et al.
already imported by sglang, in sglang's intended order), run the plugin's
register() -- which arms the JIT env (pre_arm_env) and registers the
run_batch hooks. This is still ahead of any FlashInfer import (ModelRunner
construction), so the workspace/bake env lands in time.

Guards: SCHED_SGLANG=1 opt-in; SCHED_SITE_OFF=1 excludes compiler processes
(the nvcc shim -- set by serve_sglang_armed.sh at the compile entrypoints).
"""
import importlib.abc
import importlib.machinery
import os
import sys

_TARGET = "sglang.srt.managers.scheduler"


def _armable() -> bool:
    if os.environ.get("SCHED_SITE_OFF") == "1":
        return False
    try:
        if "nvcc_clang_shim" in " ".join(sys.argv[:2]):
            return False
    except Exception:
        pass
    return True


class _LazySchedFinder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    """Wrap the target module's loader so our callback runs AFTER the host
    finishes importing it. Imports nothing until that moment."""

    def __init__(self):
        self._busy = False

    def find_spec(self, name, path, target=None):
        if name != _TARGET or self._busy:
            return None
        self._busy = True
        try:
            spec = importlib.util.find_spec(name)
        finally:
            self._busy = False
        if spec is None or spec.loader is None:
            return None
        orig_exec = spec.loader.exec_module

        def exec_module(module, _orig=orig_exec):
            _orig(module)
            _register_after_host_import()

        # bind a per-spec loader wrapper without mutating the shared loader
        import types
        loader = types.SimpleNamespace(
            create_module=getattr(spec.loader, "create_module",
                                  lambda s: None),
            exec_module=exec_module)
        spec.loader = loader
        return spec


def _register_after_host_import():
    if os.environ.get("SCHED_SITE_REGNOOP") == "1":
        print("[sched] bisect: finder fired, register SKIPPED (pid %d)"
              % os.getpid())
        return
    try:
        sys.meta_path[:] = [f for f in sys.meta_path
                            if not isinstance(f, _LazySchedFinder)]
        import sched_sglang_plugin
        sched_sglang_plugin.register()
        print("[sched] SGLang plugin registered post-import (pid %d)"
              % os.getpid())
    except Exception as exc:  # never break the host process
        print(f"[sched] plugin bootstrap skipped: {exc!r}")


if os.environ.get("SCHED_SGLANG") == "1" and _armable():
    sys.meta_path.insert(0, _LazySchedFinder())
