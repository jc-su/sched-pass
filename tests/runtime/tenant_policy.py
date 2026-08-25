#!/usr/bin/env python3
"""Validate the shared tenant policy parser used by both engines."""

from nta_runtime.tenant import tenant_budget_specs, tenant_mapper_from_environment


def main() -> None:
    environ = {
        "NTA_TENANT_BUDGETS": "7:1048576:3,11:2097152",
        "NTA_TENANT_REQUEST_PREFIXES": "7:team-a/,11:team-b/",
    }
    assert tenant_budget_specs(environ) == ((7, 1048576, 3), (11, 2097152, 1))
    mapper = tenant_mapper_from_environment(environ)
    assert mapper is not None
    assert mapper("team-a/request-0") == 7
    assert mapper("team-b/request-1") == 11
    assert mapper("unclassified/request-2") == 0
    try:
        tenant_mapper_from_environment(
            {"NTA_TENANT_REQUEST_PREFIXES": "1:team/,2:team/sub/"}
        )
    except ValueError:
        pass
    else:
        raise AssertionError("duplicate tenant-prefix policy was accepted")
    print("tenant-policy=pass")


if __name__ == "__main__":
    main()

