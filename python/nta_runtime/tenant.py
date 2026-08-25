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


def tenant_budget_specs(
    environ: Mapping[str, str] | None = None,
) -> tuple[tuple[int, int, int], ...]:
    """Parse ``NTA_TENANT_BUDGETS=id:bytes[:weight],...`` once at startup."""
    values = os.environ if environ is None else environ
    raw = values.get("NTA_TENANT_BUDGETS", "").strip()
    if not raw:
        return ()
    specs: list[tuple[int, int, int]] = []
    seen: set[int] = set()
    for item in raw.split(","):
        fields = tuple(field.strip() for field in item.split(":") if field.strip())
        if len(fields) not in (2, 3):
            raise ValueError("NTA_TENANT_BUDGETS entries must be id:bytes[:weight]")
        try:
            tenant_id, max_bytes = int(fields[0]), int(fields[1])
            weight = 1 if len(fields) == 2 else int(fields[2])
        except ValueError as error:
            raise ValueError("NTA_TENANT_BUDGETS contains an invalid value") from error
        if tenant_id < 0 or max_bytes < 0 or weight <= 0:
            raise ValueError("NTA_TENANT_BUDGETS contains an invalid value")
        if tenant_id in seen:
            raise ValueError("NTA_TENANT_BUDGETS repeats a tenant")
        seen.add(tenant_id)
        specs.append((tenant_id, max_bytes, weight))
    return tuple(specs)


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
        if tenant_id < 0 or not prefix:
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
        matches = [tenant_id for tenant_id, prefix in rules if request_id.startswith(prefix)]
        if len(set(matches)) > 1:
            raise RuntimeError(
                f"request ID {request_id!r} matches multiple tenant prefixes"
            )
        return matches[0] if matches else 0

    return map_request
