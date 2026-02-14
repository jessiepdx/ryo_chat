from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from hypermindlabs.utils import ConfigManager, DocumentManager


class DocumentIngestionJobStore:
    """Persistence and control plane wrapper for document ingestion jobs."""

    def __init__(
        self,
        *,
        config_manager: Any | None = None,
        document_manager: Any | None = None,
    ):
        if config_manager is None:
            from hypermindlabs.utils import ConfigManager

            config_manager = ConfigManager()
        if document_manager is None:
            from hypermindlabs.utils import DocumentManager

            document_manager = DocumentManager()
        self._config = config_manager
        self._documents = document_manager

    def _runtime_int(self, path: str, default: int) -> int:
        try:
            value = int(self._config.runtimeInt(path, default))
        except Exception:  # noqa: BLE001
            value = int(default)
        return value

    def _runtime_float(self, path: str, default: float) -> float:
        try:
            value = float(self._config.runtimeFloat(path, default))
        except Exception:  # noqa: BLE001
            value = float(default)
        return value

    def _runtime_str(self, path: str, default: str) -> str:
        try:
            value = self._config.runtimeValue(path, default)
        except Exception:  # noqa: BLE001
            value = default
        text = str(value if value is not None else default).strip()
        return text if text else str(default)

    def default_pipeline_version(self) -> str:
        return self._runtime_str("documents.ingestion_pipeline_version", "v1").lower()

    def enqueue_job(
        self,
        *,
        scope_payload: dict[str, Any],
        document_source_id: int,
        document_version_id: int,
        storage_object_id: int | None = None,
        actor_member_id: int | None = None,
        actor_roles: list[str] | tuple[str, ...] | None = None,
        pipeline_version: str | None = None,
        priority: int | None = None,
        max_attempts: int | None = None,
        record_metadata: dict[str, Any] | None = None,
    ) -> dict | None:
        version = str(pipeline_version or self.default_pipeline_version()).strip().lower() or "v1"
        payload = {
            "schema_version": 1,
            "scope": dict(scope_payload.get("scope", {})),
            "document_source_id": int(document_source_id),
            "document_version_id": int(document_version_id),
            "storage_object_id": storage_object_id,
            "pipeline_version": version,
            "idempotency_key": f"{int(document_source_id)}:{int(document_version_id)}:{version}",
            "job_status": "queued",
            "priority": self._runtime_int("documents.ingestion_default_priority", 100) if priority is None else int(priority),
            "max_attempts": (
                self._runtime_int("documents.ingestion_job_max_attempts", 3)
                if max_attempts is None
                else int(max_attempts)
            ),
            "record_metadata": dict(record_metadata) if isinstance(record_metadata, dict) else {},
        }
        return self._documents.enqueueDocumentIngestionJob(
            payload,
            actor_member_id=actor_member_id,
            actor_roles=actor_roles,
        )

    def list_jobs(
        self,
        scope_payload: dict[str, Any],
        *,
        statuses: list[str] | tuple[str, ...] | None = None,
        document_source_id: int | None = None,
        actor_member_id: int | None = None,
        actor_roles: list[str] | tuple[str, ...] | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        return self._documents.getDocumentIngestionJobs(
            scope_payload,
            statuses=statuses,
            document_source_id=document_source_id,
            actor_member_id=actor_member_id,
            actor_roles=actor_roles,
            limit=limit,
        )

    def get_job(
        self,
        ingestion_job_id: int,
        scope_payload: dict[str, Any],
        *,
        actor_member_id: int | None = None,
        actor_roles: list[str] | tuple[str, ...] | None = None,
    ) -> dict[str, Any] | None:
        return self._documents.getDocumentIngestionJobByID(
            ingestion_job_id,
            scope_payload,
            actor_member_id=actor_member_id,
            actor_roles=actor_roles,
        )

    def cancel_job(
        self,
        ingestion_job_id: int,
        scope_payload: dict[str, Any],
        *,
        actor_member_id: int | None = None,
        actor_roles: list[str] | tuple[str, ...] | None = None,
        reason: str = "",
    ) -> dict[str, Any] | None:
        return self._documents.cancelDocumentIngestionJob(
            ingestion_job_id,
            scope_payload,
            actor_member_id=actor_member_id,
            actor_roles=actor_roles,
            reason=reason,
        )

    def requeue_job(
        self,
        ingestion_job_id: int,
        scope_payload: dict[str, Any],
        *,
        actor_member_id: int | None = None,
        actor_roles: list[str] | tuple[str, ...] | None = None,
    ) -> dict[str, Any] | None:
        return self._documents.requeueDocumentIngestionJob(
            ingestion_job_id,
            scope_payload,
            actor_member_id=actor_member_id,
            actor_roles=actor_roles,
        )

    def list_attempts(
        self,
        ingestion_job_id: int,
        scope_payload: dict[str, Any],
        *,
        actor_member_id: int | None = None,
        actor_roles: list[str] | tuple[str, ...] | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        return self._documents.getDocumentIngestionAttempts(
            scope_payload,
            ingestion_job_id=ingestion_job_id,
            actor_member_id=actor_member_id,
            actor_roles=actor_roles,
            limit=limit,
        )

    def lease_next_job(self, *, worker_id: str) -> dict[str, Any] | None:
        lease_seconds = self._runtime_int("documents.ingestion_worker_lease_seconds", 45)
        return self._documents.leaseNextDocumentIngestionJob(
            worker_id=worker_id,
            lease_seconds=lease_seconds,
        )

    def heartbeat_job(self, ingestion_job_id: int, *, worker_id: str) -> dict[str, Any] | None:
        lease_seconds = self._runtime_int("documents.ingestion_worker_lease_seconds", 45)
        return self._documents.heartbeatDocumentIngestionJob(
            ingestion_job_id,
            worker_id=worker_id,
            lease_seconds=lease_seconds,
        )

    def mark_job_running(self, ingestion_job_id: int, *, worker_id: str) -> dict[str, Any] | None:
        return self._documents.markDocumentIngestionJobRunning(
            ingestion_job_id,
            worker_id=worker_id,
        )

    def start_attempt(self, job_record: dict[str, Any], *, worker_id: str) -> dict[str, Any] | None:
        return self._documents.createDocumentIngestionAttempt(
            job_record,
            worker_id=worker_id,
            attempt_status="running",
        )

    def finish_attempt(
        self,
        ingestion_attempt_id: int,
        *,
        attempt_status: str,
        error_message: str = "",
        error_context: dict[str, Any] | None = None,
        stage_events: list[dict[str, Any]] | None = None,
        duration_ms: int | None = None,
    ) -> dict[str, Any] | None:
        return self._documents.finishDocumentIngestionAttempt(
            ingestion_attempt_id,
            attempt_status=attempt_status,
            error_message=error_message,
            error_context=error_context,
            stage_events=stage_events,
            duration_ms=duration_ms,
        )

    def complete_job(
        self,
        ingestion_job_id: int,
        *,
        worker_id: str,
        metadata_patch: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        return self._documents.completeDocumentIngestionJob(
            ingestion_job_id,
            worker_id=worker_id,
            metadata_patch=metadata_patch,
        )

    def fail_job(
        self,
        ingestion_job_id: int,
        *,
        worker_id: str,
        error_message: str,
        error_context: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        retry_base = self._runtime_int("documents.ingestion_retry_base_seconds", 5)
        retry_max = self._runtime_int("documents.ingestion_retry_max_seconds", 300)
        return self._documents.failDocumentIngestionJob(
            ingestion_job_id,
            worker_id=worker_id,
            error_message=error_message,
            error_context=error_context,
            retry_base_seconds=retry_base,
            retry_max_seconds=retry_max,
        )

    def worker_poll_seconds(self) -> float:
        return max(0.05, self._runtime_float("documents.ingestion_worker_poll_seconds", 1.0))

    def worker_heartbeat_seconds(self) -> float:
        return max(0.5, self._runtime_float("documents.ingestion_worker_heartbeat_seconds", 10.0))

    def worker_enabled(self) -> bool:
        value = self._runtime_str("documents.ingestion_worker_enabled", "false").lower()
        return value in {"1", "true", "yes", "on"}
