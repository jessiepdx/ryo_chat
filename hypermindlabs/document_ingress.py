from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

from hypermindlabs.document_storage import (
    DocumentStorageError,
    DocumentStorageLimitError,
    LocalDocumentStorage,
)

if TYPE_CHECKING:
    from hypermindlabs.document_ingestion_jobs import DocumentIngestionJobStore
    from hypermindlabs.utils import ConfigManager


class DocumentIngressError(RuntimeError):
    """Raised when document ingress operations fail."""


class DocumentIngressService:
    """Coordinates scope-aware upload streaming and lifecycle state persistence."""

    def __init__(
        self,
        *,
        config_manager: Any | None = None,
        document_manager: Any | None = None,
        storage_backend: LocalDocumentStorage | None = None,
        ingestion_job_store: Any | None = None,
    ):
        if config_manager is None:
            from hypermindlabs.utils import ConfigManager

            config_manager = ConfigManager()
        if document_manager is None:
            from hypermindlabs.utils import DocumentManager

            document_manager = DocumentManager()

        self._config = config_manager
        self._documents = document_manager
        self._storage = storage_backend if storage_backend is not None else self._build_storage_backend()
        self._jobs = self._build_ingestion_job_store(ingestion_job_store)

    def _build_ingestion_job_store(self, value: Any | None) -> DocumentIngestionJobStore | None:
        if value is not None and hasattr(value, "enqueue_job"):
            return value
        if not hasattr(self._documents, "enqueueDocumentIngestionJob"):
            return None
        try:
            from hypermindlabs.document_ingestion_jobs import DocumentIngestionJobStore

            return DocumentIngestionJobStore(
                config_manager=self._config,
                document_manager=self._documents,
            )
        except Exception:  # noqa: BLE001
            return None

    def _runtime_int(self, path: str, default: int) -> int:
        try:
            value = int(self._config.runtimeInt(path, default))
        except Exception:  # noqa: BLE001
            value = int(default)
        return value

    def _runtime_str(self, path: str, default: str) -> str:
        try:
            value = self._config.runtimeValue(path, default)
        except Exception:  # noqa: BLE001
            value = default
        text = str(value if value is not None else default).strip()
        return text if text else str(default)

    def _build_storage_backend(self) -> LocalDocumentStorage:
        raw_path = self._runtime_str("documents.storage_local_path", "db/document_objects")
        storage_path = Path(raw_path)
        if not storage_path.is_absolute():
            storage_path = Path(__file__).resolve().parent.parent / storage_path
        max_bytes = max(1024 * 1024, self._runtime_int("documents.upload_max_bytes", 100 * 1024 * 1024))
        chunk_bytes = max(16 * 1024, self._runtime_int("documents.upload_chunk_bytes", 1024 * 1024))
        allow_empty = bool(self._runtime_int("documents.allow_empty_uploads", 0))
        return LocalDocumentStorage(
            storage_path,
            max_file_bytes=max_bytes,
            chunk_size_bytes=chunk_bytes,
            allow_empty_files=allow_empty,
        )

    def _retention_until(self) -> datetime:
        retention_days = max(1, self._runtime_int("documents.retention_default_days", 365))
        return datetime.now(timezone.utc) + timedelta(days=retention_days)

    def ingest_upload(
        self,
        *,
        stream: Any,
        filename: str,
        scope_payload: dict[str, Any],
        actor_member_id: int,
        actor_roles: list[str] | tuple[str, ...] | None = None,
        source_external_id: str = "",
        source_metadata: dict[str, Any] | None = None,
        mime_hint: str | None = None,
    ) -> dict[str, Any]:
        metadata = dict(source_metadata) if isinstance(source_metadata, dict) else {}
        try:
            persisted = self._storage.store_stream(
                stream,
                filename=filename,
                mime_hint=mime_hint,
            )
        except DocumentStorageLimitError as error:
            raise DocumentIngressError(str(error)) from error
        except DocumentStorageError as error:
            raise DocumentIngressError(str(error)) from error

        scope = {"scope": dict(scope_payload.get("scope", {}))}
        dedupe_source = self._documents.findLatestDocumentSourceByDigest(
            scope,
            persisted.file_sha256,
            actor_member_id=actor_member_id,
            actor_roles=actor_roles,
            source_name=filename,
        )

        dedupe_status = "new"
        if isinstance(dedupe_source, dict):
            source_record = dedupe_source
            source_id = int(source_record.get("document_source_id", 0) or 0)
            if source_id <= 0:
                raise DocumentIngressError("Duplicate source lookup returned an invalid source id.")
            version_number = self._documents.nextDocumentVersionNumber(
                source_id,
                scope,
                actor_member_id=actor_member_id,
                actor_roles=actor_roles,
            )
            dedupe_status = "duplicate_digest"
        else:
            source_payload = {
                "schema_version": 1,
                "scope": scope.get("scope", {}),
                "source_external_id": str(source_external_id or "").strip() or None,
                "source_name": str(filename or "").strip(),
                "source_mime": persisted.file_mime,
                "source_sha256": persisted.file_sha256,
                "source_size_bytes": persisted.file_size_bytes,
                "source_uri": self._storage.storage_uri_for_key(persisted.storage_key),
                "source_state": "received",
                "source_metadata": {
                    **metadata,
                    "storage": {
                        "backend": persisted.storage_backend,
                        "storage_key": persisted.storage_key,
                    },
                    "ingress": {
                        "dedupe_status": "new",
                        "uploaded_at": persisted.created_at,
                    },
                },
            }
            source_record = self._documents.createDocumentSource(
                source_payload,
                actor_member_id=actor_member_id,
                actor_roles=actor_roles,
            )
            if not isinstance(source_record, dict):
                raise DocumentIngressError("Failed to create document source.")
            source_id = int(source_record.get("document_source_id", 0) or 0)
            if source_id <= 0:
                raise DocumentIngressError("Document source insert did not return a valid source id.")
            version_number = 1

        storage_object = self._documents.createDocumentStorageObject(
            {
                "schema_version": 1,
                "scope": scope.get("scope", {}),
                "document_source_id": source_id,
                "storage_backend": persisted.storage_backend,
                "storage_key": persisted.storage_key,
                "storage_path": persisted.absolute_path,
                "object_state": "received",
                "file_name": str(filename or "").strip(),
                "file_mime": persisted.file_mime,
                "file_sha256": persisted.file_sha256,
                "file_size_bytes": persisted.file_size_bytes,
                "dedupe_status": dedupe_status,
                "retention_until": self._retention_until().isoformat(),
                "record_metadata": {
                    "uploaded_at": persisted.created_at,
                    "deduped_existing": bool(persisted.deduped_existing),
                },
            },
            actor_member_id=actor_member_id,
            actor_roles=actor_roles,
        )
        if not isinstance(storage_object, dict):
            raise DocumentIngressError("Failed to create document storage object.")

        version_record = self._documents.createDocumentVersion(
            {
                "schema_version": 1,
                "scope": scope.get("scope", {}),
                "document_source_id": source_id,
                "version_number": max(1, int(version_number)),
                "parser_status": "queued",
                "parser_name": "",
                "parser_version": "",
                "source_sha256": persisted.file_sha256,
                "parse_artifact": {
                    "schema_version": 1,
                    "scope": scope.get("scope", {}),
                    "parser_name": "",
                    "parser_version": "",
                    "parse_mode": "queued",
                    "status": "queued",
                    "artifact": {},
                    "warnings": [],
                    "errors": [],
                },
                "record_metadata": {
                    "ingress_state": "queued",
                    "storage_key": persisted.storage_key,
                },
            },
            actor_member_id=actor_member_id,
            actor_roles=actor_roles,
        )
        if not isinstance(version_record, dict):
            raise DocumentIngressError("Failed to create document version.")

        source_state = self._documents.updateDocumentSourceState(
            source_id,
            "queued",
            scope,
            actor_member_id=actor_member_id,
            actor_roles=actor_roles,
            metadata_patch={"ingress": {"queued_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}},
        )
        storage_state = self._documents.updateDocumentStorageObjectState(
            int(storage_object.get("storage_object_id", 0)),
            "queued",
            scope,
            actor_member_id=actor_member_id,
            actor_roles=actor_roles,
        )
        queued_job = self.enqueue_ingestion_job(
            scope_payload=scope,
            document_source_id=source_id,
            document_version_id=int(version_record.get("document_version_id", 0) or 0),
            storage_object_id=int(storage_object.get("storage_object_id", 0) or 0),
            actor_member_id=actor_member_id,
            actor_roles=actor_roles,
            record_metadata={
                "ingress_state": "queued",
                "dedupe_status": dedupe_status,
                "storage_key": persisted.storage_key,
            },
        )
        if not isinstance(queued_job, dict):
            queued_job = {
                "status": "queued",
                "document_source_id": source_id,
                "document_version_id": int(version_record.get("document_version_id", 0) or 0),
            }

        return {
            "status": "ok",
            "dedupe_status": dedupe_status,
            "source": source_state if isinstance(source_state, dict) else source_record,
            "version": version_record,
            "storage_object": storage_state if isinstance(storage_state, dict) else storage_object,
            "file": persisted.to_dict(),
            "job": queued_job,
        }

    def enqueue_ingestion_job(
        self,
        *,
        scope_payload: dict[str, Any],
        document_source_id: int,
        document_version_id: int,
        storage_object_id: int | None = None,
        actor_member_id: int,
        actor_roles: list[str] | tuple[str, ...] | None = None,
        pipeline_version: str | None = None,
        priority: int | None = None,
        max_attempts: int | None = None,
        record_metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        if self._jobs is None or not hasattr(self._jobs, "enqueue_job"):
            return None
        try:
            return self._jobs.enqueue_job(
                scope_payload=scope_payload,
                document_source_id=int(document_source_id),
                document_version_id=int(document_version_id),
                storage_object_id=storage_object_id,
                actor_member_id=actor_member_id,
                actor_roles=actor_roles,
                pipeline_version=pipeline_version,
                priority=priority,
                max_attempts=max_attempts,
                record_metadata=record_metadata,
            )
        except Exception as error:  # noqa: BLE001
            raise DocumentIngressError(f"Failed to enqueue ingestion job: {error}") from error

    def list_ingestion_jobs(
        self,
        scope_payload: dict[str, Any],
        *,
        statuses: list[str] | tuple[str, ...] | None = None,
        document_source_id: int | None = None,
        actor_member_id: int,
        actor_roles: list[str] | tuple[str, ...] | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        if self._jobs is None or not hasattr(self._jobs, "list_jobs"):
            return []
        return self._jobs.list_jobs(
            scope_payload,
            statuses=statuses,
            document_source_id=document_source_id,
            actor_member_id=actor_member_id,
            actor_roles=actor_roles,
            limit=limit,
        )

    def get_ingestion_job(
        self,
        ingestion_job_id: int,
        scope_payload: dict[str, Any],
        *,
        actor_member_id: int,
        actor_roles: list[str] | tuple[str, ...] | None = None,
    ) -> dict[str, Any] | None:
        if self._jobs is None or not hasattr(self._jobs, "get_job"):
            return None
        return self._jobs.get_job(
            ingestion_job_id,
            scope_payload,
            actor_member_id=actor_member_id,
            actor_roles=actor_roles,
        )

    def cancel_ingestion_job(
        self,
        ingestion_job_id: int,
        scope_payload: dict[str, Any],
        *,
        actor_member_id: int,
        actor_roles: list[str] | tuple[str, ...] | None = None,
        reason: str = "",
    ) -> dict[str, Any] | None:
        if self._jobs is None or not hasattr(self._jobs, "cancel_job"):
            return None
        return self._jobs.cancel_job(
            ingestion_job_id,
            scope_payload,
            actor_member_id=actor_member_id,
            actor_roles=actor_roles,
            reason=reason,
        )

    def requeue_ingestion_job(
        self,
        ingestion_job_id: int,
        scope_payload: dict[str, Any],
        *,
        actor_member_id: int,
        actor_roles: list[str] | tuple[str, ...] | None = None,
    ) -> dict[str, Any] | None:
        if self._jobs is None or not hasattr(self._jobs, "requeue_job"):
            return None
        return self._jobs.requeue_job(
            ingestion_job_id,
            scope_payload,
            actor_member_id=actor_member_id,
            actor_roles=actor_roles,
        )

    def list_ingestion_attempts(
        self,
        ingestion_job_id: int,
        scope_payload: dict[str, Any],
        *,
        actor_member_id: int,
        actor_roles: list[str] | tuple[str, ...] | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        if self._jobs is None or not hasattr(self._jobs, "list_attempts"):
            return []
        return self._jobs.list_attempts(
            ingestion_job_id,
            scope_payload,
            actor_member_id=actor_member_id,
            actor_roles=actor_roles,
            limit=limit,
        )

    def set_source_state(
        self,
        document_source_id: int,
        state: str,
        scope_payload: dict[str, Any],
        *,
        actor_member_id: int,
        actor_roles: list[str] | tuple[str, ...] | None = None,
        reason: str = "",
    ) -> dict[str, Any] | None:
        metadata_patch = {"lifecycle_reason": str(reason).strip()} if str(reason).strip() else None
        return self._documents.updateDocumentSourceState(
            int(document_source_id),
            state,
            scope_payload,
            actor_member_id=actor_member_id,
            actor_roles=actor_roles,
            metadata_patch=metadata_patch,
        )

    def soft_delete_source(
        self,
        document_source_id: int,
        scope_payload: dict[str, Any],
        *,
        actor_member_id: int,
        actor_roles: list[str] | tuple[str, ...] | None = None,
        reason: str = "",
    ) -> dict[str, Any] | None:
        source = self.set_source_state(
            document_source_id,
            "deleted",
            scope_payload,
            actor_member_id=actor_member_id,
            actor_roles=actor_roles,
            reason=reason or "soft_delete",
        )
        objects = self._documents.getDocumentStorageObjects(
            scope_payload,
            document_source_id=int(document_source_id),
            actor_member_id=actor_member_id,
            actor_roles=actor_roles,
            limit=500,
        )
        for row in objects:
            storage_object_id = int(row.get("storage_object_id", 0) or 0)
            if storage_object_id <= 0:
                continue
            self._documents.updateDocumentStorageObjectState(
                storage_object_id,
                "deleted",
                scope_payload,
                actor_member_id=actor_member_id,
                actor_roles=actor_roles,
            )
        return source

    def restore_soft_deleted_source(
        self,
        document_source_id: int,
        scope_payload: dict[str, Any],
        *,
        actor_member_id: int,
        actor_roles: list[str] | tuple[str, ...] | None = None,
    ) -> dict[str, Any] | None:
        source = self.set_source_state(
            document_source_id,
            "archived",
            scope_payload,
            actor_member_id=actor_member_id,
            actor_roles=actor_roles,
            reason="restore_soft_delete",
        )
        objects = self._documents.getDocumentStorageObjects(
            scope_payload,
            document_source_id=int(document_source_id),
            actor_member_id=actor_member_id,
            actor_roles=actor_roles,
            limit=500,
        )
        for row in objects:
            storage_object_id = int(row.get("storage_object_id", 0) or 0)
            if storage_object_id <= 0:
                continue
            if str(row.get("object_state") or "").strip().lower() == "deleted":
                self._documents.updateDocumentStorageObjectState(
                    storage_object_id,
                    "archived",
                    scope_payload,
                    actor_member_id=actor_member_id,
                    actor_roles=actor_roles,
                )
        return source
