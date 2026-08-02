"""Compatibility imports for the installable FlashInfer schedule frontend."""

from nta_runtime.flashinfer_schedule import (
    SUPPORTED_VERSION,
    Schedule,
    decode_schedule,
    paged_prefill_schedule,
    require_supported_version,
)

__all__ = [
    "SUPPORTED_VERSION",
    "Schedule",
    "decode_schedule",
    "paged_prefill_schedule",
    "require_supported_version",
]
