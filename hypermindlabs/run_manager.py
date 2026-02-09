from __future__ import annotations

import asyncio
import inspect
import json
import logging
import threading
import uuid
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable

from hypermindlabs.agent_definitions import runtime_options_from_agent_definition
from hypermindlabs.run_events import TERMINAL_STATUSES, make_event
from hypermindlabs.run_mode_handlers import normalize_run_mode
from hypermindlabs.state_snapshot_store import build_replay_state_plan


logger = logging.getLogger(__name__)


class RunCancelledError(RuntimeError):
    """Raised when a running job was cancelled by user action."""


@dataclass(slots=True)
class RunRecord:
    run_id: str
    member_id: int
    mode: str
    status: str
    request: dict[str, Any] = field(default_factory=dict)
    resolved_config: dict[str, Any] = field(default_factory=dict)
    response: Any = None
    error: str | None = None
    parent_run_id: str | None = None
    lineage: dict[str, Any] = field(default_factory=dict)
    cancel_requested: bool = False
    created_at: str = field(default_factory=lambda: _now_iso())
    updated_at: str = field(default_factory=lambda: _now_iso())

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)



def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")



def _json_default(value: Any) -> str:
    if isinstance(value, datetime):
        return value.isoformat(timespec="seconds")
    return str(value)



def _coerce_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return deepcopy(value)
    return {}



def _coerce_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return deepcopy(value)
    return []


class RunManager:
    """In-memory run lifecycle manager with optional PostgreSQL persistence hooks."""

    def __init__(
        self,
        execute_fn: Callable[["RunManager", str], Any] | None = None,
        *,
        enable_db: bool = False,
    ):
        self._execute_fn = execute_fn
        self._enable_db = bool(enable_db)
        self._lock = threading.RLock()

        self._runs: dict[str, RunRecord] = {}
        self._events: dict[str, list[dict[str, Any]]] = {}
        self._snapshots: dict[str, list[dict[str, Any]]] = {}
        self._artifacts: dict[str, list[dict[str, Any]]] = {}
        self._threads: dict[str, threading.Thread] = {}

        self._db_conninfo: str | None = None
        self._db_available: bool = False
        self._db_error_count: int = 0

        if self._enable_db:
            self._init_db()

    def _init_db(self) -> None:
        try:
            from hypermindlabs.utils import ConfigManager, execute_migration
            import psycopg

            config = ConfigManager()
            conninfo = str(config._instance.db_conninfo or "").strip()
            if not conninfo:
                logger.warning("RunManager DB persistence disabled: empty db conninfo.")
                return

            with psycopg.connect(conninfo=conninfo) as connection:
                cursor = connection.cursor()
                for migration in (
                    "080_runs.sql",
                    "081_run_events.sql",
                    "082_run_state_snapshots.sql",
                    "083_run_artifacts.sql",
                ):
                    execute_migration(cursor, migration)
                connection.commit()
                cursor.close()

            self._db_conninfo = conninfo
            self._db_available = True
            logger.info("RunManager persistence schema ready.")
        except Exception as error:  # noqa: BLE001
            self._db_available = False
            logger.warning("RunManager DB init failed, continuing with in-memory store: %s", error)

    def _db_call(self, callback: Callable[[Any], None]) -> None:
        if not self._db_available or not self._db_conninfo:
            return

        try:
            import psycopg

            with psycopg.connect(conninfo=self._db_conninfo) as connection:
                cursor = connection.cursor()
                callback(cursor)
                connection.commit()
                cursor.close()
        except Exception as error:  # noqa: BLE001
            self._db_error_count += 1
            logger.warning("RunManager DB operation failed: %s", error)
            if self._db_error_count >= 5:
                logger.warning("RunManager DB persistence disabled after repeated failures.")
                self._db_available = False

    def _persist_run(self, run: RunRecord) -> None:
        payload = run.to_dict()

        def _callback(cursor: Any) -> None:
            cursor.execute(
                """
                INSERT INTO runs (
                    run_id,
                    parent_run_id,
                    member_id,
                    mode,
                    status,
                    request_json,
                    resolved_config_json,
                    response_json,
                    error_text,
                    lineage_json,
                    cancel_requested,
                    created_at,
                    updated_at
                )
                VALUES (
                    %s, %s, %s, %s, %s,
                    %s::jsonb,
                    %s::jsonb,
                    %s::jsonb,
                    %s,
                    %s::jsonb,
                    %s,
                    %s,
                    %s
                )
                ON CONFLICT (run_id) DO UPDATE SET
                    parent_run_id = EXCLUDED.parent_run_id,
                    member_id = EXCLUDED.member_id,
                    mode = EXCLUDED.mode,
                    status = EXCLUDED.status,
                    request_json = EXCLUDED.request_json,
                    resolved_config_json = EXCLUDED.resolved_config_json,
                    response_json = EXCLUDED.response_json,
                    error_text = EXCLUDED.error_text,
                    lineage_json = EXCLUDED.lineage_json,
                    cancel_requested = EXCLUDED.cancel_requested,
                    updated_at = EXCLUDED.updated_at
                """,
                (
                    payload["run_id"],
                    payload.get("parent_run_id"),
                    payload["member_id"],
                    payload["mode"],
                    payload["status"],
                    json.dumps(payload.get("request") or {}, default=_json_default),
                    json.dumps(payload.get("resolved_config") or {}, default=_json_default),
                    json.dumps(payload.get("response"), default=_json_default),
                    payload.get("error"),
                    json.dumps(payload.get("lineage") or {}, default=_json_default),
                    bool(payload.get("cancel_requested")),
                    payload["created_at"],
                    payload["updated_at"],
                ),
            )

        self._db_call(_callback)

    def _persist_event(self, event: dict[str, Any]) -> None:
        def _callback(cursor: Any) -> None:
            cursor.execute(
                """
                INSERT INTO run_events (
                    run_id,
                    seq,
                    event_type,
                    stage,
                    status,
                    payload_json,
                    event_timestamp
                )
                VALUES (%s, %s, %s, %s, %s, %s::jsonb, %s)
                ON CONFLICT (run_id, seq) DO NOTHING
                """,
                (
                    event["run_id"],
                    event["seq"],
                    event["event_type"],
                    event["stage"],
                    event["status"],
                    json.dumps(event.get("payload") or {}, default=_json_default),
                    event["timestamp"],
                ),
            )

        self._db_call(_callback)

    def _persist_snapshot(self, snapshot: dict[str, Any]) -> None:
        def _callback(cursor: Any) -> None:
            cursor.execute(
                """
                INSERT INTO run_state_snapshots (
                    run_id,
                    step_seq,
                    stage,
                    snapshot_json,
                    snapshot_timestamp
                )
                VALUES (%s, %s, %s, %s::jsonb, %s)
                """,
                (
                    snapshot["run_id"],
                    snapshot["step_seq"],
                    snapshot.get("stage"),
                    json.dumps(snapshot.get("state") or {}, default=_json_default),
                    snapshot["timestamp"],
                ),
            )

        self._db_call(_callback)

    def _persist_artifact(self, artifact: dict[str, Any]) -> None:
        def _callback(cursor: Any) -> None:
            cursor.execute(
                """
                INSERT INTO run_artifacts (
                    run_id,
                    step_seq,
                    artifact_type,
                    artifact_name,
                    artifact_json,
                    artifact_timestamp
                )
                VALUES (%s, %s, %s, %s, %s::jsonb, %s)
                """,
                (
                    artifact["run_id"],
                    artifact.get("step_seq"),
                    artifact.get("artifact_type"),
                    artifact.get("artifact_name"),
                    json.dumps(artifact.get("artifact") or {}, default=_json_default),
                    artifact["timestamp"],
                ),
            )

        self._db_call(_callback)

    def _next_seq(self, run_id: str) -> int:
        with self._lock:
            events = self._events.setdefault(run_id, [])
            return len(events) + 1

    def append_event(
        self,
        run_id: str,
        *,
        event_type: str,
        stage: str = "runtime",
        status: str = "info",
        payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        with self._lock:
            if run_id not in self._runs:
                raise KeyError(f"Unknown run id: {run_id}")

            seq = self._next_seq(run_id)
            event = make_event(
                run_id=run_id,
                seq=seq,
                event_type=event_type,
                stage=stage,
                status=status,
                payload=payload,
            ).to_dict()
            self._events.setdefault(run_id, []).append(event)
            run = self._runs[run_id]
            run.updated_at = _now_iso()
            self._persist_event(event)
            self._persist_run(run)
            return deepcopy(event)

    def append_snapshot(
        self,
        run_id: str,
        *,
        step_seq: int,
        stage: str,
        state: dict[str, Any],
    ) -> dict[str, Any]:
        snapshot = {
            "run_id": run_id,
            "step_seq": int(step_seq),
            "stage": str(stage),
            "state": _coerce_dict(state),
            "timestamp": _now_iso(),
        }
        with self._lock:
            if run_id not in self._runs:
                raise KeyError(f"Unknown run id: {run_id}")
            self._snapshots.setdefault(run_id, []).append(snapshot)
        self._persist_snapshot(snapshot)
        return deepcopy(snapshot)

    def append_artifact(
        self,
        run_id: str,
        *,
        artifact_type: str,
        artifact_name: str,
        artifact: dict[str, Any] | None = None,
        step_seq: int | None = None,
    ) -> dict[str, Any]:
        artifact_record = {
            "run_id": run_id,
            "step_seq": int(step_seq) if isinstance(step_seq, int) else None,
            "artifact_type": str(artifact_type),
            "artifact_name": str(artifact_name),
            "artifact": _coerce_dict(artifact),
            "timestamp": _now_iso(),
        }
        with self._lock:
            if run_id not in self._runs:
                raise KeyError(f"Unknown run id: {run_id}")
            self._artifacts.setdefault(run_id, []).append(artifact_record)
        self._persist_artifact(artifact_record)
        return deepcopy(artifact_record)

    def _update_run(self, run_id: str, **fields: Any) -> dict[str, Any]:
        with self._lock:
            run = self._runs.get(run_id)
            if run is None:
                raise KeyError(f"Unknown run id: {run_id}")
            for key, value in fields.items():
                if hasattr(run, key):
                    setattr(run, key, value)
            run.updated_at = _now_iso()
            self._persist_run(run)
            return deepcopy(run.to_dict())

    def _mark_completed(self, run_id: str, result: Any) -> dict[str, Any]:
        run = self._update_run(run_id, status="completed", response=result, error=None)
        response_preview = ""
        if isinstance(result, dict):
            response_preview = str(result.get("response") or result.get("summary") or "")
        elif isinstance(result, str):
            response_preview = result

        self.append_event(
            run_id,
            event_type="run.completed",
            stage="lifecycle",
            status="completed",
            payload={
                "result": result,
                "response_preview": response_preview[:2000],
            },
        )
        return run

    def _mark_failed(self, run_id: str, error: Exception) -> dict[str, Any]:
        run = self._update_run(run_id, status="failed", error=str(error))
        self.append_event(
            run_id,
            event_type="run.failed",
            stage="lifecycle",
            status="error",
            payload={"error": str(error)},
        )
        return run

    def _mark_cancelled(self, run_id: str, reason: str = "cancel_requested") -> dict[str, Any]:
        run = self._update_run(run_id, status="cancelled", error=reason, cancel_requested=True)
        self.append_event(
            run_id,
            event_type="run.cancelled",
            stage="lifecycle",
            status="cancelled",
            payload={"reason": reason},
        )
        return run

    def is_cancel_requested(self, run_id: str) -> bool:
        with self._lock:
            run = self._runs.get(run_id)
            return bool(run.cancel_requested) if run else False

    def _assert_not_cancelled(self, run_id: str) -> None:
        if self.is_cancel_requested(run_id):
            raise RunCancelledError("cancel_requested")

    def create_run(
        self,
        *,
        member_id: int,
        mode: str,
        request_payload: dict[str, Any] | None = None,
        parent_run_id: str | None = None,
        lineage: dict[str, Any] | None = None,
        auto_start: bool = True,
    ) -> dict[str, Any]:
        normalized_mode = normalize_run_mode(mode)
        payload = _coerce_dict(request_payload)
        run_id = str(uuid.uuid4())
        resolved_config = {
            "mode": normalized_mode,
            "options": _coerce_dict(payload.get("options")),
            "agent_definition": _coerce_dict(payload.get("agent_definition")),
            "created_for": "web_ui",
        }
        record = RunRecord(
            run_id=run_id,
            member_id=int(member_id),
            mode=normalized_mode,
            status="queued",
            request=payload,
            resolved_config=resolved_config,
            parent_run_id=parent_run_id,
            lineage=_coerce_dict(lineage),
        )

        with self._lock:
            self._runs[run_id] = record
            self._events.setdefault(run_id, [])
            self._snapshots.setdefault(run_id, [])
            self._artifacts.setdefault(run_id, [])
            self._persist_run(record)

        self.append_event(
            run_id,
            event_type="run.created",
            stage="lifecycle",
            status="queued",
            payload={
                "mode": normalized_mode,
                "parent_run_id": parent_run_id,
                "lineage": _coerce_dict(lineage),
            },
        )

        if auto_start:
            self.start_run(run_id)

        return self.get_run(run_id)

    def list_runs(
        self,
        *,
        limit: int = 25,
        status: str | None = None,
        member_id: int | None = None,
    ) -> list[dict[str, Any]]:
        with self._lock:
            runs = [run.to_dict() for run in self._runs.values()]

        if status:
            runs = [run for run in runs if run.get("status") == status]
        if member_id is not None:
            runs = [run for run in runs if int(run.get("member_id", -1)) == int(member_id)]

        runs.sort(key=lambda item: str(item.get("created_at", "")), reverse=True)
        return runs[: max(1, int(limit))]

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        with self._lock:
            run = self._runs.get(run_id)
            return deepcopy(run.to_dict()) if run else None

    def get_events(self, run_id: str, *, after_seq: int = 0, limit: int = 250) -> list[dict[str, Any]]:
        with self._lock:
            events = self._events.get(run_id, [])
            filtered = [event for event in events if int(event.get("seq", 0)) > int(after_seq)]
            return deepcopy(filtered[: max(1, int(limit))])

    def get_snapshots(self, run_id: str, *, limit: int = 100) -> list[dict[str, Any]]:
        with self._lock:
            snapshots = self._snapshots.get(run_id, [])
            return deepcopy(snapshots[-max(1, int(limit)) :])

    def get_artifacts(self, run_id: str, *, limit: int = 100) -> list[dict[str, Any]]:
        with self._lock:
            artifacts = self._artifacts.get(run_id, [])
            return deepcopy(artifacts[-max(1, int(limit)) :])

    def start_run(self, run_id: str) -> dict[str, Any]:
        with self._lock:
            run = self._runs.get(run_id)
            if run is None:
                raise KeyError(f"Unknown run id: {run_id}")
            if run.status == "running":
                return deepcopy(run.to_dict())
            if run.status in TERMINAL_STATUSES:
                raise ValueError(f"Cannot start run in terminal state: {run.status}")
            if run_id in self._threads and self._threads[run_id].is_alive():
                return deepcopy(run.to_dict())

            thread = threading.Thread(target=self._run_worker, args=(run_id,), daemon=True)
            self._threads[run_id] = thread
            thread.start()
            return deepcopy(run.to_dict())

    def _run_worker(self, run_id: str) -> None:
        try:
            self._update_run(run_id, status="running")
            self.append_event(
                run_id,
                event_type="run.started",
                stage="lifecycle",
                status="running",
                payload={},
            )

            if callable(self._execute_fn):
                result = self._execute_fn(self, run_id)
            else:
                result = self._execute_default(run_id)

            if inspect.isawaitable(result):
                result = asyncio.run(result)

            run = self.get_run(run_id)
            if run is None:
                return
            if run.get("status") in TERMINAL_STATUSES:
                return
            if self.is_cancel_requested(run_id):
                self._mark_cancelled(run_id)
                return
            self._mark_completed(run_id, result)
        except RunCancelledError:
            self._mark_cancelled(run_id)
        except Exception as error:  # noqa: BLE001
            logger.exception("Run worker failed for %s", run_id)
            self._mark_failed(run_id, error)
        finally:
            with self._lock:
                self._threads.pop(run_id, None)

    def cancel_run(self, run_id: str) -> dict[str, Any]:
        run = self.get_run(run_id)
        if run is None:
            raise KeyError(f"Unknown run id: {run_id}")

        self._update_run(run_id, cancel_requested=True)
        self.append_event(
            run_id,
            event_type="run.cancel.requested",
            stage="lifecycle",
            status="cancelled",
            payload={},
        )

        latest = self.get_run(run_id)
        if latest and latest.get("status") == "queued":
            self._mark_cancelled(run_id)
        return self.get_run(run_id) or run

    def replay_run(
        self,
        run_id: str,
        *,
        replay_from_seq: int | None = None,
        state_overrides: dict[str, Any] | None = None,
        auto_start: bool = True,
    ) -> dict[str, Any]:
        source = self.get_run(run_id)
        if source is None:
            raise KeyError(f"Unknown run id: {run_id}")

        request_payload = _coerce_dict(source.get("request"))
        override_payload = _coerce_dict(state_overrides)
        request_payload["source_run_id"] = run_id
        if isinstance(replay_from_seq, int) and replay_from_seq > 0:
            request_payload["replay_from_seq"] = replay_from_seq
        if override_payload:
            request_payload["state_overrides"] = deepcopy(override_payload)

        self.append_event(
            run_id,
            event_type="run.replay.requested",
            stage="lifecycle",
            status="info",
            payload={
                "replay_from_seq": replay_from_seq,
                "has_state_overrides": bool(override_payload),
                "state_override_keys": sorted(override_payload.keys()),
            },
        )

        replay_run = self.create_run(
            member_id=int(source.get("member_id", 0)),
            mode="replay",
            request_payload=request_payload,
            parent_run_id=run_id,
            lineage={
                "type": "replay",
                "from_run_id": run_id,
                "replay_from_seq": replay_from_seq,
                "has_state_overrides": bool(override_payload),
                "state_override_keys": sorted(override_payload.keys()),
            },
            auto_start=auto_start,
        )

        self.append_event(
            run_id,
            event_type="run.replayed",
            stage="lifecycle",
            status="info",
            payload={"child_run_id": replay_run["run_id"]},
        )

        return replay_run

    def resume_run(self, run_id: str, *, auto_start: bool = True) -> dict[str, Any]:
        source = self.get_run(run_id)
        if source is None:
            raise KeyError(f"Unknown run id: {run_id}")

        resumed = self.create_run(
            member_id=int(source.get("member_id", 0)),
            mode=str(source.get("mode") or "chat"),
            request_payload=_coerce_dict(source.get("request")),
            parent_run_id=run_id,
            lineage={"type": "resume", "from_run_id": run_id},
            auto_start=auto_start,
        )
        self.append_event(
            run_id,
            event_type="run.replayed",
            stage="lifecycle",
            status="info",
            payload={"resume_run_id": resumed["run_id"]},
        )
        return resumed

    async def _run_single_orchestration(
        self,
        run_id: str,
        *,
        member_id: int,
        message: str,
        context: dict[str, Any],
        options: dict[str, Any],
        stage_prefix: str = "",
    ) -> dict[str, Any]:
        if not str(message or "").strip():
            raise ValueError("message is required")

        prefix = f"{stage_prefix}." if stage_prefix else ""

        async def stage_callback(event: dict[str, Any]) -> None:
            self._assert_not_cancelled(run_id)
            payload = _coerce_dict(event)
            stage_name = f"{prefix}{payload.get('stage', 'runtime')}"
            event_type = str(payload.get("event_type") or "run.stage")
            status = str(payload.get("status") or "running")
            event_record = self.append_event(
                run_id,
                event_type=event_type,
                stage=stage_name,
                status=status,
                payload=payload,
            )
            self.append_snapshot(
                run_id,
                step_seq=int(event_record.get("seq", 0)),
                stage=stage_name,
                state={
                    "mode": self.get_run(run_id).get("mode") if self.get_run(run_id) else "unknown",
                    "detail": payload.get("detail"),
                    "meta": payload.get("meta", {}),
                },
            )

        merged_options = _coerce_dict(options)
        merged_options["stage_callback"] = stage_callback
        existing_context = _coerce_dict(merged_options.get("run_context"))
        existing_context.setdefault("run_id", run_id)
        existing_context.setdefault("member_id", int(member_id))
        merged_options["run_context"] = existing_context

        from hypermindlabs.agents import ConversationOrchestrator

        orchestrator = ConversationOrchestrator(
            message,
            int(member_id),
            context,
            messageID=None,
            options=merged_options,
        )

        original_stream = orchestrator.streamingResponse

        def stream_callback(streamingChunk: str) -> None:
            self._assert_not_cancelled(run_id)
            chunk_text = str(streamingChunk or "")
            if chunk_text:
                self.append_event(
                    run_id,
                    event_type="run.token",
                    stage=f"{prefix}response.stream",
                    status="running",
                    payload={"chunk": chunk_text[:2000]},
                )
            try:
                original_stream(streamingChunk)
            except Exception:  # noqa: BLE001
                return

        orchestrator.streamingResponse = stream_callback

        response = await orchestrator.runAgents()
        stats = _coerce_dict(orchestrator.stats)

        self.append_event(
            run_id,
            event_type="run.metric",
            stage=f"{prefix}response.metrics",
            status="info",
            payload={"stats": stats},
        )

        return {
            "response": response,
            "stats": stats,
            "message_id": getattr(orchestrator, "messageID", None),
            "prompt_history_id": getattr(orchestrator, "promptHistoryID", None),
        }

    def _base_context_for_run(self, run: dict[str, Any]) -> dict[str, Any]:
        payload = _coerce_dict(run.get("request"))
        context = _coerce_dict(payload.get("context"))
        if not context:
            context = {
                "community_id": None,
                "chat_host_id": int(run.get("member_id", 0)),
                "platform": "web",
                "topic_id": None,
                "chat_type": "member",
                "message_timestamp": _now_iso(),
            }
        context.setdefault("platform", "web")
        context.setdefault("chat_type", "member")
        context.setdefault("chat_host_id", int(run.get("member_id", 0)))
        return context

    def _resolved_options(self, run: dict[str, Any]) -> dict[str, Any]:
        payload = _coerce_dict(run.get("request"))
        explicit_options = _coerce_dict(payload.get("options"))
        agent_definition = _coerce_dict(payload.get("agent_definition"))
        definition_options = runtime_options_from_agent_definition(agent_definition) if agent_definition else {}
        # Agent definitions provide defaults; explicit run options still take precedence.
        merged = _coerce_dict(definition_options)
        merged.update(explicit_options)
        return merged

    async def _execute_chat(self, run: dict[str, Any]) -> dict[str, Any]:
        payload = _coerce_dict(run.get("request"))
        message = str(payload.get("message") or payload.get("prompt") or "").strip()
        context = self._base_context_for_run(run)
        options = self._resolved_options(run)

        output = await self._run_single_orchestration(
            run["run_id"],
            member_id=int(run["member_id"]),
            message=message,
            context=context,
            options=options,
        )
        self.append_artifact(
            run["run_id"],
            artifact_type="response",
            artifact_name="final_response",
            artifact={"text": output.get("response", "")},
        )
        return output

    async def _execute_workflow(self, run: dict[str, Any]) -> dict[str, Any]:
        payload = _coerce_dict(run.get("request"))
        steps = _coerce_list(payload.get("workflow_steps"))
        if not steps:
            return await self._execute_chat(run)

        options = self._resolved_options(run)
        context = self._base_context_for_run(run)
        previous_output = str(payload.get("message") or payload.get("prompt") or "").strip()
        workflow_outputs: list[dict[str, Any]] = []

        for index, step in enumerate(steps, start=1):
            self._assert_not_cancelled(run["run_id"])
            step_data = _coerce_dict(step)
            step_id = str(step_data.get("id") or f"step_{index}").strip()
            step_prompt = str(step_data.get("prompt") or previous_output).strip()

            self.append_event(
                run["run_id"],
                event_type="run.stage",
                stage=f"workflow.{step_id}.start",
                status="running",
                payload={"step_index": index, "step_id": step_id},
            )

            step_output = await self._run_single_orchestration(
                run["run_id"],
                member_id=int(run["member_id"]),
                message=step_prompt,
                context=context,
                options=options,
                stage_prefix=f"workflow.{step_id}",
            )

            workflow_outputs.append(
                {
                    "step_index": index,
                    "step_id": step_id,
                    "input": step_prompt,
                    "output": step_output.get("response"),
                    "stats": step_output.get("stats", {}),
                }
            )
            previous_output = str(step_output.get("response") or previous_output)

            self.append_event(
                run["run_id"],
                event_type="run.stage",
                stage=f"workflow.{step_id}.complete",
                status="running",
                payload={"step_index": index, "step_id": step_id},
            )

        self.append_artifact(
            run["run_id"],
            artifact_type="workflow",
            artifact_name="workflow_outputs",
            artifact={"steps": workflow_outputs},
        )
        return {
            "response": previous_output,
            "workflow_outputs": workflow_outputs,
        }

    async def _execute_batch(self, run: dict[str, Any]) -> dict[str, Any]:
        payload = _coerce_dict(run.get("request"))
        batch_inputs = [str(item).strip() for item in _coerce_list(payload.get("batch_inputs")) if str(item).strip()]
        if not batch_inputs:
            raise ValueError("batch_inputs is required for batch mode")

        context = self._base_context_for_run(run)
        options = self._resolved_options(run)
        results: list[dict[str, Any]] = []

        for index, message in enumerate(batch_inputs, start=1):
            self._assert_not_cancelled(run["run_id"])
            stage_prefix = f"batch.item_{index}"
            self.append_event(
                run["run_id"],
                event_type="run.stage",
                stage=f"{stage_prefix}.start",
                status="running",
                payload={"item_index": index},
            )
            try:
                output = await self._run_single_orchestration(
                    run["run_id"],
                    member_id=int(run["member_id"]),
                    message=message,
                    context=context,
                    options=options,
                    stage_prefix=stage_prefix,
                )
                results.append(
                    {
                        "item_index": index,
                        "input": message,
                        "response": output.get("response"),
                        "status": "completed",
                    }
                )
            except RunCancelledError:
                raise
            except Exception as error:  # noqa: BLE001
                results.append(
                    {
                        "item_index": index,
                        "input": message,
                        "response": None,
                        "status": "failed",
                        "error": str(error),
                    }
                )
                self.append_event(
                    run["run_id"],
                    event_type="run.stage",
                    stage=f"{stage_prefix}.failed",
                    status="error",
                    payload={"item_index": index, "error": str(error)},
                )

            self.append_event(
                run["run_id"],
                event_type="run.stage",
                stage=f"{stage_prefix}.complete",
                status="running",
                payload={"item_index": index},
            )

        completed = [item for item in results if item.get("status") == "completed"]
        summary_text = "\n".join(
            [f"[{item['item_index']}] {str(item.get('response') or '')}" for item in completed]
        )
        summary_text = summary_text.strip() or "Batch finished with no successful responses."

        self.append_artifact(
            run["run_id"],
            artifact_type="batch",
            artifact_name="batch_results",
            artifact={"results": results},
        )

        return {
            "response": summary_text,
            "batch_results": results,
        }

    async def _execute_compare(self, run: dict[str, Any]) -> dict[str, Any]:
        payload = _coerce_dict(run.get("request"))
        message = str(payload.get("message") or payload.get("prompt") or "").strip()
        compare_models = [str(item).strip() for item in _coerce_list(payload.get("compare_models")) if str(item).strip()]
        if not message:
            raise ValueError("message is required for compare mode")
        if len(compare_models) < 2:
            raise ValueError("compare_models must include at least two model entries")

        context = self._base_context_for_run(run)
        base_options = self._resolved_options(run)
        compare_results: list[dict[str, Any]] = []

        for model_name in compare_models:
            self._assert_not_cancelled(run["run_id"])
            safe_model = model_name.replace("/", "_").replace(":", "_")
            stage_prefix = f"compare.{safe_model}"
            options = _coerce_dict(base_options)
            options["model_requested"] = model_name

            self.append_event(
                run["run_id"],
                event_type="run.stage",
                stage=f"{stage_prefix}.start",
                status="running",
                payload={"model": model_name},
            )

            try:
                output = await self._run_single_orchestration(
                    run["run_id"],
                    member_id=int(run["member_id"]),
                    message=message,
                    context=context,
                    options=options,
                    stage_prefix=stage_prefix,
                )
                compare_results.append(
                    {
                        "model": model_name,
                        "response": output.get("response"),
                        "status": "completed",
                        "stats": output.get("stats", {}),
                    }
                )
            except RunCancelledError:
                raise
            except Exception as error:  # noqa: BLE001
                compare_results.append(
                    {
                        "model": model_name,
                        "response": None,
                        "status": "failed",
                        "error": str(error),
                    }
                )
                self.append_event(
                    run["run_id"],
                    event_type="run.stage",
                    stage=f"{stage_prefix}.failed",
                    status="error",
                    payload={"model": model_name, "error": str(error)},
                )

            self.append_event(
                run["run_id"],
                event_type="run.stage",
                stage=f"{stage_prefix}.complete",
                status="running",
                payload={"model": model_name},
            )

        self.append_artifact(
            run["run_id"],
            artifact_type="compare",
            artifact_name="compare_results",
            artifact={"results": compare_results},
        )

        preview_text = "\n\n".join(
            [
                f"[{item['model']}]\n{str(item.get('response') or item.get('error') or '')}"
                for item in compare_results
            ]
        )

        return {
            "response": preview_text,
            "compare_results": compare_results,
        }

    async def _execute_replay(self, run: dict[str, Any]) -> dict[str, Any]:
        payload = _coerce_dict(run.get("request"))
        source_run_id = str(payload.get("source_run_id") or "").strip()
        source_run = self.get_run(source_run_id) if source_run_id else None

        if source_run is not None:
            source_request = _coerce_dict(source_run.get("request"))
            payload.setdefault("message", source_request.get("message"))
            payload.setdefault("prompt", source_request.get("prompt"))
            source_options = _coerce_dict(source_request.get("options"))
            merged_options = _coerce_dict(payload.get("options"))
            for key, value in source_options.items():
                merged_options.setdefault(key, value)
            payload["options"] = merged_options

        replay_from_seq_raw = payload.get("replay_from_seq")
        try:
            replay_from_seq = int(replay_from_seq_raw) if replay_from_seq_raw is not None else None
        except (TypeError, ValueError):
            replay_from_seq = None
        state_overrides = _coerce_dict(payload.get("state_overrides"))
        source_snapshots = self.get_snapshots(source_run_id, limit=1000) if source_run_id else []
        replay_plan = build_replay_state_plan(source_snapshots, replay_from_seq, state_overrides)
        selected_snapshot_seq = replay_plan.get("selected_snapshot_seq")
        merged_state = _coerce_dict(replay_plan.get("merged_state"))
        override_keys = replay_plan.get("override_keys") if isinstance(replay_plan.get("override_keys"), list) else []

        context_payload = _coerce_dict(payload.get("context"))
        replay_context = _coerce_dict(context_payload.get("replay_context"))
        replay_context.update(
            {
                "source_run_id": source_run_id or None,
                "replay_from_seq": replay_from_seq,
                "selected_snapshot_seq": selected_snapshot_seq,
                "state_override_keys": override_keys,
                "has_state_overrides": bool(state_overrides),
            }
        )
        if merged_state:
            replay_context["state"] = merged_state
        context_payload["replay_context"] = replay_context
        payload["context"] = context_payload

        self.append_event(
            run["run_id"],
            event_type="run.stage",
            stage="replay.prepare",
            status="running",
            payload={
                "source_run_id": source_run_id or None,
                "replay_from_seq": replay_from_seq,
                "selected_snapshot_seq": selected_snapshot_seq,
                "state_override_keys": override_keys,
                "has_state_overrides": bool(state_overrides),
            },
        )
        self.append_artifact(
            run["run_id"],
            artifact_type="replay",
            artifact_name="replay_state_plan",
            artifact={
                "source_run_id": source_run_id or None,
                "replay_from_seq": replay_from_seq,
                "selected_snapshot_seq": selected_snapshot_seq,
                "state_override_keys": override_keys,
                "merged_state": merged_state,
            },
        )

        run_copy = deepcopy(run)
        run_copy["request"] = payload
        result = await self._execute_chat(run_copy)
        result["replay_source_run_id"] = source_run_id or None
        result["replay_from_seq"] = replay_from_seq
        result["selected_snapshot_seq"] = selected_snapshot_seq
        result["state_override_keys"] = override_keys
        return result

    def _execute_default(self, run_id: str) -> Any:
        run = self.get_run(run_id)
        if run is None:
            raise KeyError(f"Unknown run id: {run_id}")

        mode = normalize_run_mode(run.get("mode"))
        if mode == "chat":
            return self._execute_chat(run)
        if mode == "workflow":
            return self._execute_workflow(run)
        if mode == "batch":
            return self._execute_batch(run)
        if mode == "compare":
            return self._execute_compare(run)
        if mode == "replay":
            return self._execute_replay(run)
        raise ValueError(f"Unsupported run mode: {mode}")

    def run_payload(self, run_id: str) -> dict[str, Any]:
        run = self.get_run(run_id)
        if run is None:
            raise KeyError(f"Unknown run id: {run_id}")
        return {
            "run": run,
            "events": self.get_events(run_id, after_seq=0, limit=1000),
            "snapshots": self.get_snapshots(run_id, limit=500),
            "artifacts": self.get_artifacts(run_id, limit=500),
        }

    def metrics_summary(self) -> dict[str, Any]:
        runs = self.list_runs(limit=10000)
        status_counts: dict[str, int] = {}
        durations: list[float] = []

        for run in runs:
            status = str(run.get("status") or "unknown")
            status_counts[status] = status_counts.get(status, 0) + 1
            try:
                created = datetime.fromisoformat(str(run.get("created_at")))
                updated = datetime.fromisoformat(str(run.get("updated_at")))
                durations.append(max(0.0, (updated - created).total_seconds()))
            except Exception:  # noqa: BLE001
                continue

        avg_duration = (sum(durations) / len(durations)) if durations else 0.0

        return {
            "total_runs": len(runs),
            "status_counts": status_counts,
            "average_run_seconds": round(avg_duration, 3),
            "active_runs": [run["run_id"] for run in runs if run.get("status") == "running"],
        }
