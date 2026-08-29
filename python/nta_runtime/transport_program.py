"""Framework-neutral ownership boundary for runtime transport phases."""

from __future__ import annotations

import os
import pathlib

from nta_runtime.runtime import (
    JitPhaseProgram,
    OperatorAccessProof,
    OperatorCapability,
    OperatorCoordinateMap,
    OperatorDemandBinding,
    OperatorFamily,
    OperatorForm,
    OperatorIdentityBinding,
    OperatorInstrumentation,
    OperatorPartialState,
    OperatorPlanFlag,
    OperatorReduction,
)


def load_activated_transport_program() -> tuple[JitPhaseProgram, pathlib.Path, str]:
    """Load the exact runtime-owned transport artifact selected at activation."""

    configured = os.environ.get("NTA_TRANSPORT_PROGRAM")
    if not configured:
        raise RuntimeError(
            "NTA_TRANSPORT_PROGRAM is missing; activate the complete NTA "
            "runtime environment before constructing an engine backend"
        )
    path = pathlib.Path(configured).resolve()
    if not path.is_file():
        raise RuntimeError(f"NTA transport phase program does not exist: {path}")
    expected_digest = os.environ.get("NTA_TRANSPORT_PROGRAM_SHA256", "").strip()
    if not expected_digest:
        raise RuntimeError(
            "NTA_TRANSPORT_PROGRAM_SHA256 is missing; the transport module "
            "must come from the content-checked activation environment"
        )
    program = JitPhaseProgram(path, expected_sha256=expected_digest)
    try:
        program.operator_contract.require(
            family=OperatorFamily.GENERIC,
            form=OperatorForm.INCREMENTAL,
            capabilities=(
                OperatorCapability.OBJECT_DEPENDENCIES
                | OperatorCapability.FINITE_DEFERRAL
                | OperatorCapability.PARTIAL_PUBLICATION
                | OperatorCapability.GRAPH_REPLAY
            ),
            instrumentation=OperatorInstrumentation.TIER_OWNERSHIP,
            identity_binding=OperatorIdentityBinding.NONE,
            demand_binding=OperatorDemandBinding.NONE,
            access_proof=OperatorAccessProof.NONE,
            tier_mask=(1 << 6) - 1,
        )
        program.operator_plan.require(
            family=OperatorFamily.GENERIC,
            forms=(OperatorForm.INCREMENTAL,),
            coordinate_map=OperatorCoordinateMap.UNSPECIFIED,
            partial_state=OperatorPartialState.NONE,
            reduction=OperatorReduction.NONE,
            flags=(
                OperatorPlanFlag.FIXED_CAPACITY
                | OperatorPlanFlag.GRAPH_STABLE
                | OperatorPlanFlag.EXTERNAL_WAVE_SOURCES
                | OperatorPlanFlag.GENERATION_BOUND
            ),
        )
    except BaseException:
        program.close()
        raise
    return program, path, expected_digest.lower()


__all__ = ["load_activated_transport_program"]
