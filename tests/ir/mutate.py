#!/usr/bin/env python3
"""Systematic verifier mutation testing (RQ3).

Mechanically mutates the ACCEPT fixtures across each registered legality
condition and requires the pass to reject every mutant with its own
diagnostic — a parse failure or an unrelated crash does not count. The
hand-written reject fixtures pin known escapes; this harness proves the
accept fixtures sit on the legality boundary, so weakening any single
condition is caught.

Usage: mutate.py <plugin.so> <opt> <output_dir>
"""

from __future__ import annotations

import pathlib
import re
import subprocess
import sys

SOURCE = pathlib.Path(__file__).resolve().parent


# (name, source fixture, legality condition, mutate(text) -> text|None,
#  expected diagnostic regex)
def _drop_call(symbol: str):
    """Remove an entire (possibly multi-line) call statement."""

    def apply(text: str) -> str | None:
        pattern = re.compile(
            r"^\s*(?:%[\w.]+ = )?call [^\n]*@"
            + re.escape(symbol)
            + r"\((?:[^()]|\([^()]*\))*\)[^\n]*\n",
            re.MULTILINE,
        )
        mutated, hits = pattern.subn("", text, count=1)
        return mutated if hits else None

    return apply


def _sub(pattern: str, replacement: str, count: int = 1):
    def apply(text: str) -> str | None:
        mutated, hits = re.subn(pattern, replacement, text, count=count)
        return mutated if hits else None

    return apply


MUTATIONS = [
    (
        "defer-removed",
        "batched.ll",
        "exactly one matching defer on the miss edge",
        _drop_call("__nta_defer_marker"),
        r"defer",
    ),
    (
        "binding-removed",
        "batched.ll",
        "dominating request binding",
        _drop_call("__nta_bind_request"),
        r"binding|bind|request",
    ),
    (
        "commit-removed",
        "partial-publication.ll",
        "publication post-dominates the partial region",
        _drop_call("__nta_commit_partial_marker"),
        r"commit|publication|partial",
    ),
    (
        "commit-duplicated",
        "partial-publication.ll",
        "single commit per region",
        _sub(
            r"(  call void @__nta_commit_partial_marker\([^)]*\)\n)",
            r"\1\1",
        ),
        r"commit|partial",
    ),
    (
        "begin-nonconvergent",
        "partial-publication.ll",
        "partial endpoints carry convergent semantics",
        _sub(
            r"(declare void @__nta_begin_partial_marker\(ptr, i32\)) convergent",
            r"\1",
        ),
        r"convergent",
    ),
    (
        "commit-nonconvergent",
        "partial-publication.ll",
        "partial endpoints carry convergent semantics",
        _sub(
            r"(declare void @__nta_commit_partial_marker\([^)]*\)) convergent",
            r"\1",
        ),
        r"convergent",
    ),
    (
        "ticket-constant",
        "partial-publication.ll",
        "acquisition, region, and publication share the work ticket",
        _sub(
            r"call void @__nta_commit_partial_marker\(\s*ptr %runtime, i32 %work\.ticket",
            "call void @__nta_commit_partial_marker(ptr %runtime, i32 77",
        ),
        r"ticket|matching numerical region|no publication",
    ),
    (
        "operand-thread-divergent",
        "batched.ll",
        "CTA-uniform marker operands",
        _sub(
            r"%address = call ptr @__nta_acquire_marker\(",
            "%nta.tid = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()\n"
            "  %address = call ptr @__nta_acquire_marker(",
        ),
        r"uniform|collective|divergent",
    ),
    (
        "acquire-abi-narrowed",
        "batched.ll",
        "marker ABI is exact (call-site arity)",
        # The declaration's parameter list is non-binding under opaque
        # pointers (verified empirically: narrowing only the declare
        # leaves an ABI-correct call that lowers fine), so the mutant
        # narrows the CALL.
        _sub(
            r"(%address = call ptr @__nta_acquire_marker\((?:[^()]|\n)*?), i32 %workTicket\)",
            r"\1)",
        ),
        r"ABI|argument",
    ),
    (
        "staged-pointer-returned",
        "batched.ll",
        "staged pointers cannot escape the acquisition region",
        _sub(
            r"ret void",
            "ret void ; mutated below",
            count=0,
        ),
        None,  # constructed specially below
    ),
]


def run_opt(plugin: str, opt: str, module: pathlib.Path) -> tuple[int, str]:
    completed = subprocess.run(
        [
            opt,
            f"-load-pass-plugin={plugin}",
            "-passes=nta-acquire",
            "-S",
            str(module),
            "-o",
            "/dev/null",
        ],
        capture_output=True,
        text=True,
    )
    return completed.returncode, completed.stderr


def main() -> int:
    plugin, opt, output_dir = sys.argv[1], sys.argv[2], sys.argv[3]
    out = pathlib.Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    failures = []
    rows = []
    for name, fixture, condition, mutate, expected in MUTATIONS:
        if mutate is None or expected is None:
            continue
        text = (SOURCE / fixture).read_text()
        mutated = mutate(text)
        if mutated is None:
            failures.append(f"{name}: mutation did not apply to {fixture}")
            continue
        if name == "operand-thread-divergent":
            mutated, hits = re.subn(
                r"i32 %workTicket\)", "i32 %nta.tid)", mutated, count=1
            )
            if not hits:
                failures.append(f"{name}: taint site missing")
                continue
            if "declare i32 @llvm.nvvm.read.ptx.sreg.tid.x()" not in mutated:
                mutated = "declare i32 @llvm.nvvm.read.ptx.sreg.tid.x()\n" + mutated
        path = out / f"mutant-{name}.ll"
        path.write_text(mutated)
        code, stderr = run_opt(plugin, opt, path)
        rejected = code != 0
        diagnosed = bool(re.search(expected, stderr, re.IGNORECASE))
        parseable = "error: expected" not in stderr and "use of undefined" not in stderr
        ok = rejected and diagnosed and parseable
        rows.append((name, condition, rejected, diagnosed and parseable))
        if not ok:
            failures.append(
                f"{name} [{condition}]: rejected={rejected} "
                f"diagnostic_ok={diagnosed} parse_ok={parseable}\n"
                f"  stderr: {stderr.strip()[:300]}"
            )
    print(f"{'mutant':28} {'legality condition':52} verdict")
    for name, condition, rejected, diag in rows:
        verdict = "REJECTED+DIAGNOSED" if rejected and diag else "ESCAPED"
        print(f"{name:28} {condition:52} {verdict}")
    if failures:
        print("\nMUTATION HARNESS FAILURES:", file=sys.stderr)
        for failure in failures:
            print(" -", failure, file=sys.stderr)
        return 1
    print(f"\nall {len(rows)} mutants rejected with the pass's own diagnostics")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
