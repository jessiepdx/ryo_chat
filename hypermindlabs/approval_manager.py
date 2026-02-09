from __future__ import annotations

import copy
import json
from pathlib import Path
import threading
import time
from datetime import datetime, timedelta, timezone
from typing import Any
from uuid import uuid4


DEFAULT_APPROVAL_STORE_PATH = Path(__file__).resolve().parent.parent / "db" / "tool_approvals.json"
_TERMINAL_STATUSES = {"approved", "denied", "expired", "cancelled"}
_DECISION_ALIASES = {
    "approve": "approved",
    "approved": "approved",
    "allow": "approved",
    "deny": "denied",
    "denied": "denied",
    "reject": "denied",
}


class ApprovalValidationError(ValueError):
    """Raised when approval payloads or transitions are invalid."""


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _now_iso() -> str:
    return _now().isoformat(timespec="seconds")


def _parse_iso(value: Any) -> datetime | None:
    text = str(value if value is not None else "").strip()
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _as_text(value: Any, fallback: str = "") -> str:
    text = str(value if value is not None else "").strip()
    return text if text else fallback


def _as_int(value: Any, fallback: int | None = None) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return fallback


def _coerce_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return copy.deepcopy(value)
    return {}


class ApprovalManager:
    """File-backed queue for human approval decisions on risky tool calls."""

    def __init__(self, storage_path: str | Path = DEFAULT_APPROVAL_STORE_PATH):
        self._storage_path = Path(storage_path)
        self._lock = threading.RLock()
        self._ensure_store()

    def _ensure_store(self) -> None:
        self._storage_path.parent.mkdir(parents=True, exist_ok=True)
        if self._storage_path.exists():
            return
        payload = {
            "schema": "ryo.tool_approval_queue.v1",
            "updated_at": _now_iso(),
            "requests": {},
        }
        self._storage_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def _load(self) -> dict[str, Any]:
        self._ensure_store()
        try:
            raw = json.loads(self._storage_path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            raw = {}
        payload = _coerce_dict(raw)
        requests = payload.get("requests")
        if not isinstance(requests, dict):
            requests = {}
        return {
            "schema": "ryo.tool_approval_queue.v1",
            "updated_at": _as_text(payload.get("updated_at")),
            "requests": requests,
        }

    def _save(self, payload: dict[str, Any]) -> None:
        payload["updated_at"] = _now_iso()
        self._storage_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def _normalize_record(self, request_id: str, raw_record: dict[str, Any]) -> dict[str, Any]:
        status = _as_text(raw_record.get("status"), "pending").lower()
        if status not in {"pending", "approved", "denied", "expired", "cancelled"}:
            status = "pending"
        return {
            "request_id": request_id,
            "run_id": _as_text(raw_record.get("run_id")),
            "tool_name": _as_text(raw_record.get("tool_name")),
            "status": status,
            "reason": _as_text(raw_record.get("reason")),
            "requested_by_member_id": _as_int(raw_record.get("requested_by_member_id")),
            "run_owner_member_id": _as_int(raw_record.get("run_owner_member_id")),
            "requested_at": _as_text(raw_record.get("requested_at"), _now_iso()),
            "expires_at": _as_text(raw_record.get("expires_at"), _now_iso()),
            "decided_at": _as_text(raw_record.get("decided_at")) or None,
            "decided_by_member_id": _as_int(raw_record.get("decided_by_member_id")),
            "tool_args": _coerce_dict(raw_record.get("tool_args")),
            "meta": _coerce_dict(raw_record.get("meta")),
        }

    def _expire_if_needed(self, record: dict[str, Any]) -> tuple[dict[str, Any], bool]:
        normalized = _coerce_dict(record)
        if _as_text(normalized.get("status"), "pending") != "pending":
            return normalized, False
        expires_at = _parse_iso(normalized.get("expires_at"))
        if expires_at is None or _now() < expires_at:
            return normalized, False
        normalized["status"] = "expired"
        normalized["reason"] = _as_text(normalized.get("reason"), "Approval request expired.")
        normalized["decided_at"] = _now_iso()
        return normalized, True

    def request_approval(
        self,
        *,
        run_id: str,
        tool_name: str,
        tool_args: dict[str, Any] | None = None,
        requested_by_member_id: int | None = None,
        run_owner_member_id: int | None = None,
        reason: str = "",
        timeout_seconds: float = 45.0,
        meta: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        clean_run_id = _as_text(run_id)
        clean_tool = _as_text(tool_name)
        if not clean_run_id:
            raise ApprovalValidationError("run_id is required.")
        if not clean_tool:
            raise ApprovalValidationError("tool_name is required.")

        timeout = max(1.0, float(timeout_seconds))
        request_id = f"apr-{uuid4().hex[:12]}"
        record = {
            "request_id": request_id,
            "run_id": clean_run_id,
            "tool_name": clean_tool,
            "status": "pending",
            "reason": _as_text(reason),
            "requested_by_member_id": _as_int(requested_by_member_id),
            "run_owner_member_id": _as_int(run_owner_member_id),
            "requested_at": _now_iso(),
            "expires_at": (_now() + timedelta(seconds=timeout)).isoformat(timespec="seconds"),
            "decided_at": None,
            "decided_by_member_id": None,
            "tool_args": _coerce_dict(tool_args),
            "meta": _coerce_dict(meta),
        }

        with self._lock:
            payload = self._load()
            requests = payload.get("requests")
            if not isinstance(requests, dict):
                requests = {}
            requests[request_id] = record
            payload["requests"] = requests
            self._save(payload)
        return copy.deepcopy(record)

    def list_requests(
        self,
        *,
        status: str | None = None,
        run_id: str | None = None,
        member_id: int | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        normalized_status = _as_text(status).lower() if status else None
        clean_run_id = _as_text(run_id)
        member_filter = _as_int(member_id)
        max_rows = max(1, int(limit))

        with self._lock:
            payload = self._load()
            requests = payload.get("requests")
            if not isinstance(requests, dict):
                return []

            changed = False
            output: list[dict[str, Any]] = []
            for request_id, raw_record in requests.items():
                record = self._normalize_record(request_id, _coerce_dict(raw_record))
                record, expired = self._expire_if_needed(record)
                if expired:
                    requests[request_id] = record
                    changed = True

                if normalized_status and _as_text(record.get("status")).lower() != normalized_status:
                    continue
                if clean_run_id and _as_text(record.get("run_id")) != clean_run_id:
                    continue
                if member_filter is not None:
                    owner_id = _as_int(record.get("run_owner_member_id"))
                    requester_id = _as_int(record.get("requested_by_member_id"))
                    if owner_id != member_filter and requester_id != member_filter:
                        continue
                output.append(record)

            if changed:
                payload["requests"] = requests
                self._save(payload)

        output.sort(key=lambda item: _as_text(item.get("requested_at")), reverse=True)
        return copy.deepcopy(output[:max_rows])

    def get_request(self, request_id: str) -> dict[str, Any] | None:
        clean_id = _as_text(request_id)
        if not clean_id:
            return None
        with self._lock:
            payload = self._load()
            requests = payload.get("requests")
            if not isinstance(requests, dict):
                return None
            raw_record = _coerce_dict(requests.get(clean_id))
            if not raw_record:
                return None
            record = self._normalize_record(clean_id, raw_record)
            record, expired = self._expire_if_needed(record)
            if expired:
                requests[clean_id] = record
                payload["requests"] = requests
                self._save(payload)
            return copy.deepcopy(record)

    def decide_request(
        self,
        request_id: str,
        *,
        decision: str,
        actor_member_id: int | None = None,
        reason: str = "",
        meta: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        clean_id = _as_text(request_id)
        if not clean_id:
            raise ApprovalValidationError("request_id is required.")
        decision_key = _as_text(decision).lower()
        normalized_decision = _DECISION_ALIASES.get(decision_key)
        if normalized_decision is None:
            raise ApprovalValidationError("decision must be one of approve/deny.")

        with self._lock:
            payload = self._load()
            requests = payload.get("requests")
            if not isinstance(requests, dict):
                raise ApprovalValidationError("Approval request store is unavailable.")
            raw_record = _coerce_dict(requests.get(clean_id))
            if not raw_record:
                raise ApprovalValidationError("Approval request not found.")

            record = self._normalize_record(clean_id, raw_record)
            record, expired = self._expire_if_needed(record)
            if expired:
                requests[clean_id] = record
                payload["requests"] = requests
                self._save(payload)
                raise ApprovalValidationError("Approval request already expired.")

            current_status = _as_text(record.get("status"), "pending")
            if current_status in _TERMINAL_STATUSES:
                raise ApprovalValidationError(f"Approval request is already {current_status}.")

            record["status"] = normalized_decision
            record["decided_at"] = _now_iso()
            record["decided_by_member_id"] = _as_int(actor_member_id)
            if _as_text(reason):
                record["reason"] = _as_text(reason)
            if isinstance(meta, dict) and meta:
                merged_meta = _coerce_dict(record.get("meta"))
                merged_meta.update(_coerce_dict(meta))
                record["meta"] = merged_meta

            requests[clean_id] = record
            payload["requests"] = requests
            self._save(payload)
            return copy.deepcopy(record)

    def wait_for_decision(
        self,
        request_id: str,
        *,
        timeout_seconds: float | None = None,
        poll_interval_seconds: float = 0.25,
    ) -> dict[str, Any]:
        clean_id = _as_text(request_id)
        if not clean_id:
            raise ApprovalValidationError("request_id is required.")

        started = _now()
        max_wait = max(0.0, float(timeout_seconds)) if timeout_seconds is not None else None
        poll_delay = max(0.05, float(poll_interval_seconds))

        while True:
            record = self.get_request(clean_id)
            if record is None:
                raise ApprovalValidationError("Approval request not found.")
            status = _as_text(record.get("status"), "pending")
            if status in _TERMINAL_STATUSES:
                return record

            if max_wait is not None and (_now() - started).total_seconds() >= max_wait:
                try:
                    return self.decide_request(
                        clean_id,
                        decision="deny",
                        reason="Approval wait timeout exceeded.",
                    )
                except ApprovalValidationError:
                    refreshed = self.get_request(clean_id)
                    if refreshed is not None:
                        return refreshed
                    raise

            time.sleep(poll_delay)
