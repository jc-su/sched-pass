#!/usr/bin/env python3
"""Repository-level engineering gate.

This gate checks properties that unit tests alone cannot establish: production
source must not contain unfinished implementation markers, workflow code must
remain outside the implementation boundary, the Python/native API version must
agree, and the five supported resource contracts must remain present. Hardware
qualification is intentionally separate; a missing NVMe or CXL endpoint is an
evaluation skip, not a reason for this source audit to lie.
"""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
import re
import subprocess
import sys


_IMPLEMENTATION_ROOTS = (
    "include",
    "lib",
    "runtime",
    "kernel",
    "python/nta_runtime",
    "tools",
)
_WORKFLOW_ROOTS = ("experiments", "benchmarks", "scripts", "tests")
_ARTIFACT_WORKFLOW_ROOTS = ("experiments", "benchmarks", "scripts")
_REQUIRED_BOUNDARIES = _IMPLEMENTATION_ROOTS + _WORKFLOW_ROOTS
_PYTHON_SUFFIXES = {".py"}
_SOURCE_SUFFIXES = {".c", ".cc", ".cpp", ".cu", ".h", ".cuh", ".py"}
_UNFINISHED = re.compile(
    r"\b(?:TODO|FIXME|XXX|HACK)\b|NotImplementedError|raise\s+NotImplemented"
)
_REQUIRED_RESOURCE_KINDS = {
    "HBM",
    "HOST_MAPPED",
    "HOST_STAGED",
    "NVME",
    "CXL_DAX",
}


def _files(root: Path, relative_roots: tuple[str, ...]):
    for relative_root in relative_roots:
        base = root / relative_root
        if not base.exists():
            continue
        yield from (
            path
            for path in base.rglob("*")
            if path.is_file()
            and "__pycache__" not in path.parts
            and path.suffix in _SOURCE_SUFFIXES
        )


def _unfinished_findings(root: Path) -> list[str]:
    findings: list[str] = []
    # Keep the audit implementation out of its own marker scan: the regular
    # expression necessarily contains the marker names it is looking for.
    scanned_roots = _IMPLEMENTATION_ROOTS + _WORKFLOW_ROOTS
    for path in _files(root, scanned_roots):
        if path == Path(__file__).resolve():
            continue
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except UnicodeDecodeError:
            continue
        for line_number, line in enumerate(lines, 1):
            if _UNFINISHED.search(line):
                findings.append(
                    f"{path.relative_to(root)}:{line_number}: {line.strip()}"
                )
        if path.suffix != ".py":
            continue
        try:
            tree = ast.parse("\n".join(lines), filename=str(path))
        except SyntaxError as error:
            findings.append(f"{path.relative_to(root)}: syntax error: {error}")
            continue
        protocol_functions = {
            child
            for class_node in ast.walk(tree)
            if isinstance(class_node, ast.ClassDef)
            and any(
                isinstance(base, ast.Name) and base.id == "Protocol"
                for base in class_node.bases
            )
            for child in ast.walk(class_node)
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if node in protocol_functions:
                continue
            body = list(node.body)
            if (
                body
                and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)
            ):
                body = body[1:]
            if len(body) == 1 and (
                isinstance(body[0], ast.Pass)
                or (
                    isinstance(body[0], ast.Expr)
                    and isinstance(body[0].value, ast.Constant)
                    and body[0].value.value is Ellipsis
                )
            ):
                findings.append(
                    f"{path.relative_to(root)}:{node.lineno}: empty function {node.name}"
                )
    return findings


def _python_syntax_findings(root: Path) -> list[str]:
    findings: list[str] = []
    for path in _files(root, _IMPLEMENTATION_ROOTS + _WORKFLOW_ROOTS):
        if path.suffix not in _PYTHON_SUFFIXES:
            continue
        try:
            ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except (SyntaxError, UnicodeDecodeError) as error:
            findings.append(f"{path.relative_to(root)}: syntax error: {error}")
    return findings


def _boundary_findings(root: Path) -> list[str]:
    findings: list[str] = []
    missing = [name for name in _REQUIRED_BOUNDARIES if not (root / name).is_dir()]
    if missing:
        findings.append("missing source boundaries: " + ", ".join(missing))
    manifest_path = root / "experiments/artifact-manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        findings.append(f"artifact manifest cannot be read: {error}")
    else:
        implementation_roots = set(manifest.get("implementation_roots", ()))
        if implementation_roots != set(_IMPLEMENTATION_ROOTS):
            findings.append(
                "artifact manifest implementation roots disagree with source audit"
            )
        workflow_roots = set(manifest.get("experiment_roots", ()))
        if workflow_roots != set(_ARTIFACT_WORKFLOW_ROOTS):
            findings.append(
                "artifact manifest experiment roots disagree with source audit"
            )

    # Production Python may be consumed by workflows, but the dependency
    # direction must not be reversed. This catches an experiment helper or
    # benchmark implementation leaking into the runtime library.
    workflow_names = {name for name in _WORKFLOW_ROOTS}
    for path in _files(root, _IMPLEMENTATION_ROOTS):
        if path.suffix != ".py":
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except (SyntaxError, UnicodeDecodeError):
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                modules = [node.module]
            else:
                continue
            for module in modules:
                root_name = module.split(".", 1)[0]
                if root_name in workflow_names:
                    findings.append(
                        f"{path.relative_to(root)}: implementation imports workflow module {module}"
                    )
    return findings


def _api_versions(root: Path) -> dict[str, int]:
    native = (root / "include/nta/RuntimeC.h").read_text(encoding="utf-8")
    python = (root / "python/nta_runtime/runtime.py").read_text(encoding="utf-8")
    native_match = re.search(r"NTA_RUNTIME_C_API_VERSION\s+(\d+)U", native)
    python_match = re.search(r"^API_VERSION\s*=\s*(\d+)$", python, re.MULTILINE)
    if native_match is None or python_match is None:
        raise RuntimeError("could not locate native/Python runtime API versions")
    versions = {
        "native": int(native_match.group(1)),
        "python": int(python_match.group(1)),
    }
    if len(set(versions.values())) != 1:
        raise RuntimeError(f"native/Python API versions diverge: {versions}")
    return versions


def _module_assignment(path: Path, name: str) -> ast.expr:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == name
            for target in node.targets
        ):
            return node.value
    raise RuntimeError(f"{path.name} has no module-level {name}")


def _string_literals(node: ast.expr) -> set[str]:
    if isinstance(node, ast.Call) and len(node.args) == 1:
        return _string_literals(node.args[0])
    if isinstance(node, ast.Dict):
        elements: list[ast.expr | None] = list(node.keys)
    elif isinstance(node, (ast.Tuple, ast.List, ast.Set)):
        elements = list(node.elts)
    else:
        raise RuntimeError("expression is not a literal string collection")
    values = {
        element.value
        for element in elements
        if isinstance(element, ast.Constant) and isinstance(element.value, str)
    }
    if len(values) != len(elements):
        raise RuntimeError("expression contains a non-string literal")
    return values


def _flashinfer_versions(root: Path) -> set[str]:
    """Return the FlashInfer releases both integration layers agree on.

    The runtime schedule extractor reads a version-specific PlanInfo layout,
    while the overlay preparer owns that version's kernel source hashes. An
    instrumented launch needs both, so a release named by only one of them is
    either a runtime claim nothing can instrument or an overlay nothing can
    schedule. Keep the two sets identical.
    """
    schedule = _string_literals(
        _module_assignment(
            root / "python/nta_runtime/flashinfer_schedule.py", "SUPPORTED_VERSIONS"
        )
    )
    overlay = _string_literals(
        _module_assignment(
            root / "tools/flashinfer/prepare_overlay.py", "SOURCE_PROFILES"
        )
    )
    if schedule != overlay:
        raise RuntimeError(
            "FlashInfer schedule and overlay versions diverge: "
            f"schedule={sorted(schedule)} overlay={sorted(overlay)}"
        )
    return schedule


def _resource_kinds(root: Path) -> set[str]:
    source = (root / "python/nta_runtime/resource_contract.py").read_text(
        encoding="utf-8"
    )
    enum_block = re.search(
        r"class\s+ResourceKind\(.*?\):(?P<body>.*?)(?:\n\n|\nclass\s+)",
        source,
        re.DOTALL,
    )
    if enum_block is None:
        raise RuntimeError("ResourceKind enum is missing")
    return set(
        re.findall(r"^\s+([A-Z][A-Z0-9_]*)\s*=", enum_block.group("body"), re.MULTILINE)
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    args = parser.parse_args()
    root = args.root.resolve()
    findings = _unfinished_findings(root)
    findings.extend(_python_syntax_findings(root))
    findings.extend(_boundary_findings(root))
    versions = _api_versions(root)
    flashinfer_versions = _flashinfer_versions(root)
    resource_kinds = _resource_kinds(root)
    missing = sorted(_REQUIRED_RESOURCE_KINDS - resource_kinds)
    if missing:
        findings.append("missing resource contracts: " + ", ".join(missing))
    diff = subprocess.run(
        ["git", "-C", str(root), "diff", "--check"],
        check=False,
        capture_output=True,
        text=True,
    )
    if diff.returncode:
        findings.append("git diff --check failed: " + diff.stdout.strip())
    if findings:
        for finding in findings:
            print(f"project-audit: FAIL: {finding}", file=sys.stderr)
        return 1
    print(
        "project-audit=pass "
        f"api_version={versions['native']} "
        f"flashinfer_versions={','.join(sorted(flashinfer_versions))} "
        f"resource_kinds={','.join(sorted(resource_kinds))} "
        "source_boundaries=pass"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
