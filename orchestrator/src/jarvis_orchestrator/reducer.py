"""Pure task transition reducer."""

from __future__ import annotations

from .domain import InvalidStateError, TaskState, TERMINAL_STATES


ALLOWED_TRANSITIONS: dict[TaskState, frozenset[TaskState]] = {
    TaskState.ADMITTED: frozenset({TaskState.ROUTING}),
    TaskState.ROUTING: frozenset({TaskState.LOADING_CONTEXT}),
    TaskState.LOADING_CONTEXT: frozenset({TaskState.EXECUTING}),
    TaskState.EXECUTING: frozenset(
        {TaskState.AWAITING_APPROVAL, TaskState.RESPONDING}
    ),
    TaskState.AWAITING_APPROVAL: frozenset(
        {TaskState.EXECUTING, TaskState.DECISION_WAITING_RECONNECT}
    ),
    TaskState.DECISION_WAITING_RECONNECT: frozenset({TaskState.EXECUTING}),
    TaskState.RESPONDING: frozenset({TaskState.COMMITTING}),
    TaskState.COMMITTING: frozenset({TaskState.COMPLETED}),
    TaskState.COMPLETED: frozenset(),
    TaskState.CANCELLED: frozenset(),
    TaskState.INTERRUPTED: frozenset(),
    TaskState.BUDGET_EXHAUSTED: frozenset(),
    TaskState.FAILED: frozenset(),
}


def require_transition(current: TaskState, target: TaskState) -> None:
    if target in TERMINAL_STATES and current not in TERMINAL_STATES:
        return
    if target not in ALLOWED_TRANSITIONS[current]:
        raise InvalidStateError()
