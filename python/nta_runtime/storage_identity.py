"""Framework-neutral canonical identities for immutable tier payloads."""

from __future__ import annotations

import base64
from dataclasses import dataclass
import re


SGLANG_PAGE_SCHEME = "sglang-page-v1"
VLLM_BLOCK_SCHEME = "vllm-block-v1"


@dataclass(frozen=True)
class StorageIdentity:
    """Typed opaque identity serialized into a tier catalog key."""

    scheme: str
    payload: bytes

    def __post_init__(self) -> None:
        if re.fullmatch(r"[a-z][a-z0-9-]{0,63}", self.scheme) is None:
            raise ValueError("storage identity scheme is invalid")
        if not isinstance(self.payload, bytes) or not self.payload:
            raise ValueError("storage identity payload must be non-empty bytes")

    @property
    def catalog_key(self) -> str:
        encoded = base64.urlsafe_b64encode(self.payload).decode("ascii").rstrip("=")
        key = f"{self.scheme}:{encoded}"
        if len(key.encode("ascii")) > 4096:
            raise ValueError("canonical storage identity exceeds the catalog limit")
        return key

    @classmethod
    def from_text(cls, scheme: str, value: str) -> "StorageIdentity":
        if not isinstance(value, str) or not value:
            raise ValueError("text storage identity must be non-empty")
        return cls(scheme, value.encode("utf-8"))


def sglang_storage_key(value: str) -> str:
    return StorageIdentity.from_text(SGLANG_PAGE_SCHEME, value).catalog_key


def vllm_storage_key(value: bytes) -> str:
    return StorageIdentity(VLLM_BLOCK_SCHEME, value).catalog_key
