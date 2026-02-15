from __future__ import annotations

import os
from pathlib import Path
import threading
import time
import traceback
from datetime import datetime, timezone
from typing import Any, Callable

from hypermindlabs.document_ingestion_jobs import DocumentIngestionJobStore
from hypermindlabs.document_parser.contracts import (
    build_document_parse_artifact_patch,
    canonical_status_to_document_state,
)
from hypermindlabs.document_parser.router import (
    DocumentParserExecutionError,
    DocumentParserRouter,
    DocumentParserRoutingError,
)
from hypermindlabs.document_chunker import (
    build_chunk_artifact_summary,
    build_document_chunks,
)
from hypermindlabs.document_tree_builder import (
    build_canonical_document_tree,
    build_tree_artifact_summary,
)


IngestionJobProcessor = Callable[[dict[str, Any], dict[str, Any]], dict[str, Any] | None]


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class DocumentIngestionWorkerCancelled(RuntimeError):
    """Raised when a leased ingestion job is cancelled or lease ownership is lost."""


class DocumentIngestionWorker:
    """Runs the asynchronous document ingestion loop with lease + heartbeat semantics."""

    def __init__(
        self,
        *,
        job_store: Any | None = None,
        document_manager: Any | None = None,
        config_manager: Any | None = None,
        parser_router: Any | None = None,
        processor: IngestionJobProcessor | None = None,
        worker_id: str | None = None,
    ):
        if config_manager is None:
            try:
                from hypermindlabs.utils import ConfigManager

                config_manager = ConfigManager()
            except Exception:  # noqa: BLE001
                config_manager = None
        if document_manager is None:
            from hypermindlabs.utils import DocumentManager

            document_manager = DocumentManager()
        if job_store is None:
            job_store = DocumentIngestionJobStore(
                config_manager=config_manager,
                document_manager=document_manager,
            )

        self._config = config_manager
        self._jobs = job_store
        self._documents = document_manager
        self._parser_router = parser_router if hasattr(parser_router, "parse_document") else None
        self._processor = processor if callable(processor) else self._default_process_job
        self._worker_id = str(worker_id or "").strip() or f"ingestion-{os.getpid()}-{int(time.time())}"
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._thread_lock = threading.Lock()

    @property
    def worker_id(self) -> str:
        return self._worker_id

    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> bool:
        with self._thread_lock:
            if self._thread is not None and self._thread.is_alive():
                return False
            self._stop_event.clear()
            self._thread = threading.Thread(
                target=self.run_forever,
                name=f"ryo-document-ingestion-{self._worker_id}",
                daemon=True,
            )
            self._thread.start()
            return True

    def stop(self, timeout: float = 5.0) -> bool:
        self._stop_event.set()
        with self._thread_lock:
            worker_thread = self._thread
        if worker_thread is not None and worker_thread.is_alive():
            worker_thread.join(timeout=max(0.1, float(timeout)))
        return not self.is_running()

    def run_forever(self) -> None:
        while not self._stop_event.is_set():
            processed = self.run_once()
            if processed:
                continue
            time.sleep(self._worker_poll_seconds())

    def run_once(self) -> bool:
        leased_job = self._jobs.lease_next_job(worker_id=self._worker_id)
        if not isinstance(leased_job, dict):
            return False
        self._process_job(leased_job)
        return True

    def _default_process_job(self, job_record: dict[str, Any], scope_payload: dict[str, Any]) -> dict[str, Any]:
        pipeline_version = str(job_record.get("pipeline_version") or "v1").strip() or "v1"
        parser_router = self._get_parser_router()
        parse_profile = self._build_parser_profile(job_record, scope_payload=scope_payload)
        canonical_output = parser_router.parse_document(parse_profile)
        parse_status = canonical_status_to_document_state(str(canonical_output.get("status") or "failed"))
        if parse_status != "parsed":
            raise DocumentParserExecutionError(
                "Parser router returned a failed canonical parse output.",
                attempts=[{"status": "failed_canonical_output"}],
            )
        version_id = self._safe_positive_int(job_record.get("document_version_id"), 0)
        tree_payload = build_canonical_document_tree(
            canonical_output,
            document_version_id=version_id,
        )
        tree_summary = build_tree_artifact_summary(tree_payload)
        chunk_payload = build_document_chunks(
            tree_payload=tree_payload,
            canonical_output=canonical_output,
            document_version_id=version_id,
            config_manager=self._config,
        )
        chunk_summary = build_chunk_artifact_summary(chunk_payload)
        parse_artifact_patch = build_document_parse_artifact_patch(canonical_output)
        artifact = parse_artifact_patch.get("artifact")
        if not isinstance(artifact, dict):
            artifact = {}
            parse_artifact_patch["artifact"] = artifact
        artifact["tree"] = tree_summary
        artifact["chunks"] = chunk_summary
        provenance = dict(canonical_output.get("provenance") or {})
        selected_adapter = str(provenance.get("selected_adapter") or "unknown-parser").strip() or "unknown-parser"
        selected_version = str(provenance.get("selected_adapter_version") or "").strip() or pipeline_version
        route_chain = [str(item).strip() for item in provenance.get("adapter_chain", []) if str(item).strip()]
        metadata_common = {
            "processed_at": _utc_now_iso(),
            "worker_id": self._worker_id,
            "pipeline_version": pipeline_version,
            "selected_adapter": selected_adapter,
            "selected_adapter_version": selected_version,
            "adapter_chain": route_chain,
            "confidence": provenance.get("confidence"),
            "cost": provenance.get("cost"),
            "duration_ms": provenance.get("duration_ms"),
            "tree_node_count": tree_summary.get("node_count"),
            "tree_edge_count": tree_summary.get("edge_count"),
            "tree_repaired": bool(tree_summary.get("integrity", {}).get("was_repaired")),
            "chunk_count": chunk_summary.get("chunk_count"),
            "chunk_truncated": bool(chunk_summary.get("truncated")),
        }
        return {
            "parser_name": selected_adapter,
            "parser_version": selected_version,
            "parse_artifact_patch": parse_artifact_patch,
            "tree_payload": tree_payload,
            "chunk_payload": chunk_payload,
            "job_metadata_patch": {
                **metadata_common,
                "route_debug": dict(provenance.get("route_debug") or {}),
                "profile_summary": dict(provenance.get("profile_summary") or {}),
                "tree": tree_summary,
                "chunks": chunk_summary,
            },
            "source_metadata_patch": {
                "ingestion": {
                    **metadata_common,
                    "status": "parsed",
                }
            },
            "storage_metadata_patch": {
                "ingestion": {
                    **metadata_common,
                    "status": "parsed",
                }
            },
            "version_metadata_patch": {
                "ingestion": {
                    **metadata_common,
                    "status": "parsed",
                },
                "tree": tree_summary,
                "chunks": chunk_summary,
            },
        }

    def _worker_poll_seconds(self) -> float:
        try:
            seconds = float(self._jobs.worker_poll_seconds())
        except Exception:  # noqa: BLE001
            seconds = 1.0
        return max(0.05, seconds)

    def _worker_heartbeat_seconds(self) -> float:
        try:
            seconds = float(self._jobs.worker_heartbeat_seconds())
        except Exception:  # noqa: BLE001
            seconds = 10.0
        return max(0.5, seconds)

    def _job_scope_payload(self, job_record: dict[str, Any]) -> dict[str, Any]:
        owner_member_id = self._safe_positive_int(job_record.get("owner_member_id"))
        chat_host_id = self._safe_positive_int(job_record.get("chat_host_id"))
        chat_type = str(job_record.get("chat_type") or "member").strip().lower() or "member"
        platform = str(job_record.get("platform") or "web").strip().lower() or "web"
        return {
            "scope": {
                "owner_member_id": owner_member_id,
                "chat_host_id": chat_host_id,
                "chat_type": chat_type,
                "community_id": self._safe_optional_int(job_record.get("community_id")),
                "topic_id": self._safe_optional_int(job_record.get("topic_id")),
                "platform": platform,
            }
        }

    def _get_parser_router(self) -> DocumentParserRouter:
        if self._parser_router is None:
            self._parser_router = DocumentParserRouter(config_manager=self._config)
        return self._parser_router

    def _build_parser_profile(self, job_record: dict[str, Any], *, scope_payload: dict[str, Any]) -> dict[str, Any]:
        source_id = self._safe_positive_int(job_record.get("document_source_id"), 0)
        if source_id <= 0:
            raise DocumentParserRoutingError("Missing document_source_id for parser routing.")
        owner_member_id = self._safe_positive_int(job_record.get("owner_member_id"), 0)
        if owner_member_id <= 0:
            raise DocumentParserRoutingError("Missing owner_member_id for parser routing.")
        storage_object_id = self._safe_positive_int(job_record.get("storage_object_id"), 0)
        storage_rows = self._documents.getDocumentStorageObjects(
            scope_payload,
            document_source_id=source_id,
            actor_member_id=owner_member_id,
            actor_roles=["owner"],
            limit=100,
        )
        if not isinstance(storage_rows, list) or not storage_rows:
            raise FileNotFoundError("No storage objects found for ingestion job.")
        storage_row: dict[str, Any] | None = None
        if storage_object_id > 0:
            for row in storage_rows:
                row_id = self._safe_positive_int((row or {}).get("storage_object_id"), 0)
                if row_id == storage_object_id:
                    storage_row = dict(row)
                    break
        if storage_row is None:
            storage_row = dict(storage_rows[0])
        file_path = str(storage_row.get("storage_path") or "").strip()
        if not file_path:
            raise FileNotFoundError("Storage object is missing storage_path required for parser routing.")
        file_disk_path = Path(file_path)
        if not file_disk_path.exists():
            raise FileNotFoundError(f"Storage path does not exist: {file_disk_path}")
        file_name = str(storage_row.get("file_name") or "").strip() or str(file_disk_path.name)
        file_extension = str(Path(file_name).suffix or "").strip().lower()
        record_metadata = dict(storage_row.get("record_metadata") or {}) if isinstance(storage_row.get("record_metadata"), dict) else {}
        job_metadata = dict(job_record.get("record_metadata") or {}) if isinstance(job_record.get("record_metadata"), dict) else {}
        return {
            "document_source_id": source_id,
            "document_version_id": self._safe_positive_int(job_record.get("document_version_id"), 0),
            "storage_object_id": self._safe_positive_int(storage_row.get("storage_object_id"), 0) or None,
            "file_name": file_name,
            "file_mime": str(storage_row.get("file_mime") or "").strip().lower(),
            "file_extension": file_extension,
            "file_size_bytes": self._safe_positive_int(storage_row.get("file_size_bytes"), 0),
            "file_path": str(file_disk_path),
            "file_sha256": str(storage_row.get("file_sha256") or "").strip().lower(),
            "storage_backend": str(storage_row.get("storage_backend") or "").strip().lower(),
            "storage_key": str(storage_row.get("storage_key") or "").strip(),
            "scope": dict(scope_payload.get("scope", {})),
            "record_metadata": {
                "storage_object": record_metadata,
                "ingestion_job": job_metadata,
            },
        }

    def _safe_positive_int(self, value: Any, default: int = 1) -> int:
        fallback = int(default)
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            parsed = fallback
        if parsed <= 0:
            parsed = fallback
        if parsed <= 0:
            return 0
        return parsed

    def _safe_optional_int(self, value: Any) -> int | None:
        if value is None or str(value).strip() == "":
            return None
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            return None
        if parsed < 0:
            return None
        return parsed

    def _dict_patch(self, value: Any) -> dict[str, Any]:
        return dict(value) if isinstance(value, dict) else {}

    def _event(
        self,
        event_type: str,
        *,
        status: str = "info",
        payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return {
            "event_type": str(event_type),
            "status": str(status),
            "timestamp": _utc_now_iso(),
            "payload": self._dict_patch(payload),
        }

    def _heartbeat_loop(
        self,
        ingestion_job_id: int,
        *,
        stop_event: threading.Event,
        lease_lost: threading.Event,
    ) -> None:
        interval = self._worker_heartbeat_seconds()
        while not stop_event.wait(interval):
            heartbeat_record = self._jobs.heartbeat_job(
                ingestion_job_id,
                worker_id=self._worker_id,
            )
            if not isinstance(heartbeat_record, dict):
                lease_lost.set()
                break

    def _process_job(self, leased_job: dict[str, Any]) -> None:
        ingestion_job_id = self._safe_positive_int(leased_job.get("ingestion_job_id"), 0)
        if ingestion_job_id <= 0:
            return

        stage_events: list[dict[str, Any]] = [
            self._event(
                "ingestion.job.leased",
                payload={
                    "ingestion_job_id": ingestion_job_id,
                    "worker_id": self._worker_id,
                },
            )
        ]
        running_job = self._jobs.mark_job_running(ingestion_job_id, worker_id=self._worker_id)
        if not isinstance(running_job, dict):
            return

        stage_events.append(
            self._event(
                "ingestion.job.running",
                payload={
                    "ingestion_job_id": ingestion_job_id,
                    "attempt_count": running_job.get("attempt_count"),
                    "worker_id": self._worker_id,
                },
            )
        )

        attempt_record = self._jobs.start_attempt(running_job, worker_id=self._worker_id)
        ingestion_attempt_id = 0
        if isinstance(attempt_record, dict):
            ingestion_attempt_id = self._safe_positive_int(attempt_record.get("ingestion_attempt_id"), 0)

        started = time.monotonic()
        heartbeat_stop = threading.Event()
        lease_lost = threading.Event()
        heartbeat_thread = threading.Thread(
            target=self._heartbeat_loop,
            kwargs={
                "ingestion_job_id": ingestion_job_id,
                "stop_event": heartbeat_stop,
                "lease_lost": lease_lost,
            },
            name=f"ryo-document-ingestion-heartbeat-{ingestion_job_id}",
            daemon=True,
        )
        heartbeat_thread.start()

        try:
            if self._stop_event.is_set():
                raise DocumentIngestionWorkerCancelled("worker_stop_requested")

            scope_payload = self._job_scope_payload(running_job)
            processed = self._processor(running_job, scope_payload)
            processed_payload = self._dict_patch(processed)
            if lease_lost.is_set():
                raise DocumentIngestionWorkerCancelled("lease_lost_before_completion")

            self._mark_related_success(running_job, processed_payload, scope_payload=scope_payload)

            complete_payload = self._dict_patch(processed_payload.get("job_metadata_patch"))
            if not complete_payload:
                complete_payload = {
                    "processed_at": _utc_now_iso(),
                    "worker_id": self._worker_id,
                }
            complete_record = self._jobs.complete_job(
                ingestion_job_id,
                worker_id=self._worker_id,
                metadata_patch=complete_payload,
            )
            if not isinstance(complete_record, dict):
                raise DocumentIngestionWorkerCancelled("lease_lost_before_commit")

            stage_events.append(
                self._event(
                    "ingestion.job.completed",
                    status="ok",
                    payload={
                        "ingestion_job_id": ingestion_job_id,
                        "attempt_count": complete_record.get("attempt_count"),
                    },
                )
            )
            if ingestion_attempt_id > 0:
                self._jobs.finish_attempt(
                    ingestion_attempt_id,
                    attempt_status="succeeded",
                    stage_events=stage_events,
                    duration_ms=self._duration_ms(started),
                )
            return

        except DocumentIngestionWorkerCancelled as error:
            stage_events.append(
                self._event(
                    "ingestion.job.cancelled",
                    status="warn",
                    payload={
                        "ingestion_job_id": ingestion_job_id,
                        "reason": str(error),
                    },
                )
            )
            if ingestion_attempt_id > 0:
                self._jobs.finish_attempt(
                    ingestion_attempt_id,
                    attempt_status="cancelled",
                    error_message=str(error),
                    stage_events=stage_events,
                    duration_ms=self._duration_ms(started),
                )
            return

        except Exception as error:  # noqa: BLE001
            error_text = str(error).strip() or "ingestion_failure"
            error_context = {
                "error_type": type(error).__name__,
                "traceback": traceback.format_exc(limit=20),
            }
            failed_record = self._jobs.fail_job(
                ingestion_job_id,
                worker_id=self._worker_id,
                error_message=error_text,
                error_context=error_context,
            )
            failed_status = str((failed_record or {}).get("job_status") or "failed").strip().lower()
            if failed_status == "retry_wait":
                stage_events.append(
                    self._event(
                        "ingestion.job.retry",
                        status="warn",
                        payload={
                            "ingestion_job_id": ingestion_job_id,
                            "error": error_text,
                        },
                    )
                )
            elif failed_status == "dead_letter":
                stage_events.append(
                    self._event(
                        "ingestion.job.dead_letter",
                        status="error",
                        payload={
                            "ingestion_job_id": ingestion_job_id,
                            "error": error_text,
                            "dead_letter_reason": (failed_record or {}).get("dead_letter_reason"),
                        },
                    )
                )
                try:
                    self._mark_related_failure(
                        running_job,
                        error_text=error_text,
                        scope_payload=self._job_scope_payload(running_job),
                    )
                except Exception as related_error:  # noqa: BLE001
                    error_context["related_state_error"] = str(related_error)
            else:
                stage_events.append(
                    self._event(
                        "ingestion.job.failed",
                        status="error",
                        payload={
                            "ingestion_job_id": ingestion_job_id,
                            "error": error_text,
                        },
                    )
                )
            if ingestion_attempt_id > 0:
                self._jobs.finish_attempt(
                    ingestion_attempt_id,
                    attempt_status="failed",
                    error_message=error_text,
                    error_context=error_context,
                    stage_events=stage_events,
                    duration_ms=self._duration_ms(started),
                )
            return

        finally:
            heartbeat_stop.set()
            heartbeat_thread.join(timeout=1.0)

    def _duration_ms(self, started: float) -> int:
        return max(0, int((time.monotonic() - started) * 1000))

    def _mark_related_success(
        self,
        job_record: dict[str, Any],
        processed_payload: dict[str, Any],
        *,
        scope_payload: dict[str, Any],
    ) -> None:
        source_id = self._safe_positive_int(job_record.get("document_source_id"), 0)
        version_id = self._safe_positive_int(job_record.get("document_version_id"), 0)
        storage_object_id = self._safe_positive_int(job_record.get("storage_object_id"), 0)
        actor_member_id = self._safe_positive_int(job_record.get("owner_member_id"), 1)
        pipeline_version = str(job_record.get("pipeline_version") or "v1").strip() or "v1"
        parser_name = str(processed_payload.get("parser_name") or "ingestion-worker").strip() or "ingestion-worker"
        parser_version = str(processed_payload.get("parser_version") or pipeline_version).strip() or pipeline_version

        parse_artifact_patch = self._dict_patch(processed_payload.get("parse_artifact_patch"))
        if not parse_artifact_patch:
            parse_artifact_patch = {
                "status": "parsed",
                "parse_mode": "queue_stub",
                "artifact": {
                    "worker_id": self._worker_id,
                    "pipeline_version": pipeline_version,
                },
                "errors": [],
                "warnings": [],
            }

        version_metadata_patch = self._dict_patch(processed_payload.get("version_metadata_patch"))
        if "ingestion" not in version_metadata_patch:
            version_metadata_patch["ingestion"] = {
                "status": "completed",
                "worker_id": self._worker_id,
                "completed_at": _utc_now_iso(),
                "pipeline_version": pipeline_version,
            }

        tree_payload = self._dict_patch(processed_payload.get("tree_payload"))
        if tree_payload:
            tree_nodes = list(tree_payload.get("nodes") or [])
            tree_edges = list(tree_payload.get("edges") or [])
            tree_write = self._documents.replaceDocumentVersionTree(
                version_id,
                scope_payload=scope_payload,
                nodes=tree_nodes,
                edges=tree_edges,
                actor_member_id=actor_member_id,
                actor_roles=["owner"],
            )
            if not isinstance(tree_write, dict):
                raise RuntimeError("Failed to persist document tree nodes and edges.")
            tree_metadata = version_metadata_patch.get("tree")
            if not isinstance(tree_metadata, dict):
                tree_metadata = {}
                version_metadata_patch["tree"] = tree_metadata
            tree_metadata["node_count"] = int(tree_write.get("node_count", len(tree_nodes)) or 0)
            tree_metadata["edge_count"] = int(tree_write.get("edge_count", len(tree_edges)) or 0)
            tree_metadata["persisted_at"] = _utc_now_iso()

        chunk_payload = self._dict_patch(processed_payload.get("chunk_payload"))
        if chunk_payload:
            chunk_rows = list(chunk_payload.get("chunks") or [])
            chunk_write = self._documents.replaceDocumentVersionChunks(
                version_id,
                scope_payload=scope_payload,
                chunks=chunk_rows,
                actor_member_id=actor_member_id,
                actor_roles=["owner"],
            )
            if not isinstance(chunk_write, dict):
                raise RuntimeError("Failed to persist document chunks.")
            chunk_metadata = version_metadata_patch.get("chunks")
            if not isinstance(chunk_metadata, dict):
                chunk_metadata = {}
                version_metadata_patch["chunks"] = chunk_metadata
            chunk_metadata["chunk_count"] = int(chunk_write.get("chunk_count", len(chunk_rows)) or 0)
            chunk_metadata["persisted_at"] = _utc_now_iso()

        self._documents.updateDocumentVersionParserStatus(
            version_id,
            "parsed",
            scope_payload,
            parser_name=parser_name,
            parser_version=parser_version,
            parse_artifact_patch=parse_artifact_patch,
            record_metadata_patch=version_metadata_patch,
            actor_member_id=actor_member_id,
            actor_roles=["owner"],
        )

        source_metadata_patch = self._dict_patch(processed_payload.get("source_metadata_patch"))
        if "ingestion" not in source_metadata_patch:
            source_metadata_patch["ingestion"] = {
                "status": "parsed",
                "worker_id": self._worker_id,
                "completed_at": _utc_now_iso(),
                "pipeline_version": pipeline_version,
            }
        self._documents.updateDocumentSourceState(
            source_id,
            "parsed",
            scope_payload,
            actor_member_id=actor_member_id,
            actor_roles=["owner"],
            metadata_patch=source_metadata_patch,
        )
        if storage_object_id > 0:
            self._documents.updateDocumentStorageObjectState(
                storage_object_id,
                "parsed",
                scope_payload,
                actor_member_id=actor_member_id,
                actor_roles=["owner"],
            )

    def _mark_related_failure(
        self,
        job_record: dict[str, Any],
        *,
        error_text: str,
        scope_payload: dict[str, Any],
    ) -> None:
        source_id = self._safe_positive_int(job_record.get("document_source_id"), 0)
        version_id = self._safe_positive_int(job_record.get("document_version_id"), 0)
        storage_object_id = self._safe_positive_int(job_record.get("storage_object_id"), 0)
        actor_member_id = self._safe_positive_int(job_record.get("owner_member_id"), 1)
        pipeline_version = str(job_record.get("pipeline_version") or "v1").strip() or "v1"

        self._documents.updateDocumentVersionParserStatus(
            version_id,
            "failed",
            scope_payload,
            parser_name="ingestion-worker",
            parser_version=pipeline_version,
            parse_artifact_patch={
                "status": "failed",
                "parse_mode": "queue_stub",
                "errors": [
                    {
                        "code": "ingestion_failure",
                        "message": error_text,
                    }
                ],
            },
            record_metadata_patch={
                "ingestion": {
                    "status": "failed",
                    "worker_id": self._worker_id,
                    "failed_at": _utc_now_iso(),
                    "pipeline_version": pipeline_version,
                    "error": error_text,
                }
            },
            actor_member_id=actor_member_id,
            actor_roles=["owner"],
        )
        self._documents.updateDocumentSourceState(
            source_id,
            "failed",
            scope_payload,
            actor_member_id=actor_member_id,
            actor_roles=["owner"],
            metadata_patch={
                "ingestion": {
                    "status": "failed",
                    "worker_id": self._worker_id,
                    "failed_at": _utc_now_iso(),
                    "pipeline_version": pipeline_version,
                    "error": error_text,
                }
            },
        )
        if storage_object_id > 0:
            self._documents.updateDocumentStorageObjectState(
                storage_object_id,
                "failed",
                scope_payload,
                actor_member_id=actor_member_id,
                actor_roles=["owner"],
            )


_INGESTION_WORKER_LOCK = threading.Lock()
_INGESTION_WORKER: DocumentIngestionWorker | None = None


def ensure_document_ingestion_worker_started() -> DocumentIngestionWorker:
    global _INGESTION_WORKER
    with _INGESTION_WORKER_LOCK:
        if _INGESTION_WORKER is None:
            _INGESTION_WORKER = DocumentIngestionWorker()
        _INGESTION_WORKER.start()
        return _INGESTION_WORKER


def stop_document_ingestion_worker(timeout: float = 5.0) -> None:
    global _INGESTION_WORKER
    with _INGESTION_WORKER_LOCK:
        worker = _INGESTION_WORKER
    if worker is None:
        return
    worker.stop(timeout=timeout)
