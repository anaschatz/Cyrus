"""Narrow injected runtime seam for encrypted saved memory.

The store owns SQLite state but never constructs or selects a platform key
provider.  Production startup may inject this immutable wrapper after it has
qualified the provider independently; non-memory stores pass no runtime and
perform no native key operations.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from .memory_crypto import MemoryEnvelopeCipherV1

_INSTALL_ID = re.compile(r"^[0-9a-f]{64}$")


@runtime_checkable
class MemoryCryptoRuntime(Protocol):
    """Qualified key-domain identity plus its envelope cipher."""

    install_id: str
    cipher: MemoryEnvelopeCipherV1


@dataclass(frozen=True, slots=True)
class InjectedMemoryCryptoRuntime:
    """Concrete injection wrapper; it does not create or mutate keys."""

    install_id: str
    cipher: MemoryEnvelopeCipherV1

    def __post_init__(self) -> None:
        if (
            not isinstance(self.install_id, str)
            or _INSTALL_ID.fullmatch(self.install_id) is None
        ):
            raise ValueError("memory provider install id is invalid")
        if not isinstance(self.cipher, MemoryEnvelopeCipherV1):
            raise TypeError("memory envelope cipher is invalid")


class ProviderBoundMemoryCryptoRuntime:
    """Lazy production wrapper whose installation id appears at bootstrap."""

    __slots__ = ("cipher",)

    def __init__(self, cipher: MemoryEnvelopeCipherV1) -> None:
        if not isinstance(cipher, MemoryEnvelopeCipherV1):
            raise TypeError("memory envelope cipher is invalid")
        self.cipher = cipher

    @property
    def install_id(self) -> str:
        # The platform provider owns the random installation identity.  This
        # property intentionally fails closed until the store's journaled
        # prepare path bootstraps (or reopens) that provider.
        return self.cipher.provider.install_id


__all__ = [
    "InjectedMemoryCryptoRuntime",
    "MemoryCryptoRuntime",
    "ProviderBoundMemoryCryptoRuntime",
]
