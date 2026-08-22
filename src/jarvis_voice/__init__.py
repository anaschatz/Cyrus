"""CYRUS Voice runtime with legacy ``jarvis_voice`` protocol namespaces."""

from .protocol import Envelope, PROTOCOL_VERSION

__all__ = ["Envelope", "PROTOCOL_VERSION"]
__version__ = "0.8.0"
