"""Framework-neutral tenant policy parsing.

Tenant identity is deployment metadata, not something the runtime guesses from
batch position or prompt contents.  Framework adapters may provide an
explicit callback; the prefix mapper here is the small process-start adapter
used by integrations whose upstream request object has no tenant field.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
import os


TenantMapper = Callable[[str], int]
_UINT32_MAX = (1 << 32) - 1
_UINT64_MAX = (1 << 64) - 1


def tenant_budget_specs(
    environ: Mapping[str, str] | None = None,
) -> tuple[tuple[int, int], ...]:
    """Parse ``NTA_TENANT_BUDGETS=id:bytes,...`` once at startup."""
    values = os.environ if environ is None else environ
    raw = values.get("NTA_TENANT_BUDGETS", "").strip()
    if not raw:
        return ()
    specs: list[tuple[int, int]] = []
    seen: set[int] = set()
    for item in raw.split(","):
        fields = tuple(field.strip() for field in item.split(":") if field.strip())
        if len(fields) != 2:
            raise ValueError("NTA_TENANT_BUDGETS entries must be id:bytes")
        try:
            tenant_id, max_bytes = int(fields[0]), int(fields[1])
        except ValueError as error:
            raise ValueError("NTA_TENANT_BUDGETS contains an invalid value") from error
        if (
            tenant_id < 0
            or tenant_id > _UINT32_MAX
            or max_bytes < 0
            or max_bytes > _UINT64_MAX
        ):
            raise ValueError("NTA_TENANT_BUDGETS contains an invalid value")
        if tenant_id in seen:
            raise ValueError("NTA_TENANT_BUDGETS repeats a tenant")
        seen.add(tenant_id)
        specs.append((tenant_id, max_bytes))
    return tuple(specs)


def tenant_isolation_required(specs: tuple[tuple[int, int], ...]) -> bool:
    """Return whether byte-credit accounting is a hard execution constraint."""

    return any(max_bytes != _UINT64_MAX for _, max_bytes in specs)


def tenant_mapper_from_environment(
    environ: Mapping[str, str] | None = None,
) -> TenantMapper | None:
    """Build an explicit request-prefix-to-tenant mapper.

    ``NTA_TENANT_REQUEST_PREFIXES`` uses ``tenant_id:prefix`` entries, for
    example ``7:team-a/,11:team-b/``.  Unmatched requests remain tenant 0.
    Overlapping prefixes for different tenants are rejected at startup so a
    request can never be assigned nondeterministically.
    """
    values = os.environ if environ is None else environ
    raw = values.get("NTA_TENANT_REQUEST_PREFIXES", "").strip()
    if not raw:
        return None
    rules: list[tuple[int, str]] = []
    tenants: set[int] = set()
    prefixes: set[str] = set()
    for item in raw.split(","):
        fields = tuple(field.strip() for field in item.split(":", 1))
        if len(fields) != 2 or not fields[0] or not fields[1]:
            raise ValueError(
                "NTA_TENANT_REQUEST_PREFIXES entries must be tenant_id:prefix"
            )
        try:
            tenant_id = int(fields[0])
        except ValueError as error:
            raise ValueError(
                "NTA_TENANT_REQUEST_PREFIXES contains an invalid tenant"
            ) from error
        prefix = fields[1]
        if tenant_id < 0 or tenant_id > _UINT32_MAX or not prefix:
            raise ValueError("NTA_TENANT_REQUEST_PREFIXES contains an invalid value")
        if tenant_id in tenants or prefix in prefixes:
            raise ValueError("NTA_TENANT_REQUEST_PREFIXES repeats a tenant or prefix")
        if any(
            existing_prefix.startswith(prefix) or prefix.startswith(existing_prefix)
            for _, existing_prefix in rules
        ):
            raise ValueError(
                "NTA_TENANT_REQUEST_PREFIXES contains overlapping prefixes"
            )
        tenants.add(tenant_id)
        prefixes.add(prefix)
        rules.append((tenant_id, prefix))

    def map_request(request_id: str) -> int:
        matches = [
            tenant_id for tenant_id, prefix in rules if request_id.startswith(prefix)
        ]
        if len(set(matches)) > 1:
            raise RuntimeError(
                f"request ID {request_id!r} matches multiple tenant prefixes"
            )
        return matches[0] if matches else 0

    return map_request
