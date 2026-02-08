from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any


EVENT_TYPES: tuple[str, ...] = (
    "run.created",
    "run.started",
    "run.stage",
    "run.token",
    "run.snapshot",
    "run.artifact",
    "run.metric",
    "run.cancel.requested",
    "run.cancelled",
    "run.completed",
    "run.failed",
    "run.replay.requested",
    "run.replayed",
)

TERMINAL_STATUSES: set[str] = {"completed", "failed", "cancelled"}


@dataclass(slots=True)
class RunEvent:
    run_id: str
    seq: int
    event_type: str
    stage: str
    status: str
    payload: dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: utc_now_iso())

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)



def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")



def normalize_event_type(value: Any) -> str:
    text = str(value or "").strip()
    if text:
        return text
    return "run.metric"



def make_event(
    *,
    run_id: str,
    seq: int,
    event_type: str,
    stage: str = "runtime",
    status: str = "info",
    payload: dict[str, Any] | None = None,
) -> RunEvent:
    normalized_payload = payload if isinstance(payload, dict) else {}
    return RunEvent(
        run_id=str(run_id),
        seq=int(seq),
        event_type=normalize_event_type(event_type),
        stage=str(stage or "runtime"),
        status=str(status or "info"),
        payload=normalized_payload,
    )



def event_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "required": ["run_id", "seq", "event_type", "stage", "status", "timestamp", "payload"],
        "properties": {
            "run_id": {"type": "string"},
            "seq": {"type": "integer", "minimum": 1},
            "event_type": {"type": "string", "enum": list(EVENT_TYPES)},
            "stage": {"type": "string"},
            "status": {"type": "string"},
            "timestamp": {"type": "string"},
            "payload": {"type": "object"},
        },
        "additionalProperties": False,
    }
