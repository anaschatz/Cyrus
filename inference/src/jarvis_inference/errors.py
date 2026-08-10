from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class ApplianceError(Exception):
    status: int
    code: str
    message: str
    parameter: str | None = None

    def public(self) -> dict[str, object]:
        error: dict[str, object] = {
            "message": self.message,
            "type": "invalid_request_error" if self.status < 500 else "server_error",
            "code": self.code,
        }
        if self.parameter is not None:
            error["param"] = self.parameter
        return {"error": error}


def invalid(code: str, message: str, parameter: str | None = None) -> ApplianceError:
    return ApplianceError(400, code, message, parameter)


def unsupported(parameter: str) -> ApplianceError:
    return invalid("unsupported_parameter", "The parameter is not supported by this endpoint.", parameter)


def backend_error(message: str = "The managed inference backend failed.") -> ApplianceError:
    return ApplianceError(502, "backend_protocol_error", message)

