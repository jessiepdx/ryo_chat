import unittest
import tempfile
from pathlib import Path
from typing import Any

from hypermindlabs.document_ingestion_worker import DocumentIngestionWorker


def _job_record() -> dict[str, Any]:
    return {
        "ingestion_job_id": 101,
        "document_source_id": 501,
        "document_version_id": 601,
        "storage_object_id": 701,
        "schema_version": 1,
        "owner_member_id": 5,
        "chat_host_id": 55,
        "chat_type": "member",
        "community_id": None,
        "topic_id": None,
        "platform": "web",
        "pipeline_version": "v1",
        "attempt_count": 0,
        "max_attempts": 3,
        "job_status": "leased",
    }


class _FakeJobStore:
    def __init__(
        self,
        *,
        fail_result: dict[str, Any] | None = None,
        complete_result: dict[str, Any] | None = None,
    ):
        self._leased = False
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.finish_attempt_calls: list[dict[str, Any]] = []
        self._fail_result = dict(fail_result) if isinstance(fail_result, dict) else {"job_status": "retry_wait"}
        self._complete_result = (
            dict(complete_result)
            if isinstance(complete_result, dict)
            else {"ingestion_job_id": 101, "job_status": "completed", "attempt_count": 1}
        )

    def lease_next_job(self, *, worker_id: str) -> dict[str, Any] | None:
        self.calls.append(("lease_next_job", {"worker_id": worker_id}))
        if self._leased:
            return None
        self._leased = True
        return dict(_job_record())

    def mark_job_running(self, ingestion_job_id: int, *, worker_id: str) -> dict[str, Any]:
        self.calls.append(("mark_job_running", {"ingestion_job_id": ingestion_job_id, "worker_id": worker_id}))
        row = _job_record()
        row["attempt_count"] = 1
        row["job_status"] = "running"
        return row

    def start_attempt(self, job_record: dict[str, Any], *, worker_id: str) -> dict[str, Any]:
        self.calls.append(
            (
                "start_attempt",
                {
                    "ingestion_job_id": int(job_record.get("ingestion_job_id", 0)),
                    "worker_id": worker_id,
                },
            )
        )
        return {"ingestion_attempt_id": 901}

    def heartbeat_job(self, ingestion_job_id: int, *, worker_id: str) -> dict[str, Any]:
        self.calls.append(("heartbeat_job", {"ingestion_job_id": ingestion_job_id, "worker_id": worker_id}))
        return {"ingestion_job_id": ingestion_job_id, "worker_id": worker_id}

    def complete_job(
        self,
        ingestion_job_id: int,
        *,
        worker_id: str,
        metadata_patch: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        self.calls.append(
            (
                "complete_job",
                {
                    "ingestion_job_id": ingestion_job_id,
                    "worker_id": worker_id,
                    "metadata_patch": dict(metadata_patch or {}),
                },
            )
        )
        return dict(self._complete_result) if self._complete_result is not None else None

    def fail_job(
        self,
        ingestion_job_id: int,
        *,
        worker_id: str,
        error_message: str,
        error_context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        self.calls.append(
            (
                "fail_job",
                {
                    "ingestion_job_id": ingestion_job_id,
                    "worker_id": worker_id,
                    "error_message": error_message,
                    "error_context": dict(error_context or {}),
                },
            )
        )
        return dict(self._fail_result)

    def finish_attempt(
        self,
        ingestion_attempt_id: int,
        *,
        attempt_status: str,
        error_message: str = "",
        error_context: dict[str, Any] | None = None,
        stage_events: list[dict[str, Any]] | None = None,
        duration_ms: int | None = None,
    ) -> dict[str, Any]:
        payload = {
            "ingestion_attempt_id": ingestion_attempt_id,
            "attempt_status": attempt_status,
            "error_message": error_message,
            "error_context": dict(error_context or {}),
            "stage_events": list(stage_events or []),
            "duration_ms": duration_ms,
        }
        self.finish_attempt_calls.append(payload)
        self.calls.append(("finish_attempt", payload))
        return payload

    def worker_poll_seconds(self) -> float:
        return 0.05

    def worker_heartbeat_seconds(self) -> float:
        return 60.0


class _FakeDocumentManager:
    def __init__(self):
        self.version_updates: list[dict[str, Any]] = []
        self.source_updates: list[dict[str, Any]] = []
        self.storage_updates: list[dict[str, Any]] = []
        self.tree_replacements: list[dict[str, Any]] = []
        self.chunk_replacements: list[dict[str, Any]] = []
        self.storage_rows: list[dict[str, Any]] = []

    def updateDocumentVersionParserStatus(
        self,
        document_version_id: int,
        parser_status: str,
        scope_payload: dict[str, Any],
        **kwargs: Any,
    ) -> dict[str, Any]:
        payload = {
            "document_version_id": int(document_version_id),
            "parser_status": str(parser_status),
            "scope": dict(scope_payload.get("scope", {})),
            "extra": dict(kwargs),
        }
        self.version_updates.append(payload)
        return payload

    def updateDocumentSourceState(
        self,
        document_source_id: int,
        source_state: str,
        scope_payload: dict[str, Any],
        **kwargs: Any,
    ) -> dict[str, Any]:
        payload = {
            "document_source_id": int(document_source_id),
            "source_state": str(source_state),
            "scope": dict(scope_payload.get("scope", {})),
            "extra": dict(kwargs),
        }
        self.source_updates.append(payload)
        return payload

    def updateDocumentStorageObjectState(
        self,
        storage_object_id: int,
        object_state: str,
        scope_payload: dict[str, Any],
        **kwargs: Any,
    ) -> dict[str, Any]:
        payload = {
            "storage_object_id": int(storage_object_id),
            "object_state": str(object_state),
            "scope": dict(scope_payload.get("scope", {})),
            "extra": dict(kwargs),
        }
        self.storage_updates.append(payload)
        return payload

    def getDocumentStorageObjects(
        self,
        _: dict[str, Any],
        *,
        document_source_id: int | None = None,
        **__: Any,
    ) -> list[dict[str, Any]]:
        rows = list(self.storage_rows)
        if document_source_id is not None:
            rows = [row for row in rows if int(row.get("document_source_id", 0)) == int(document_source_id)]
        return [dict(row) for row in rows]

    def replaceDocumentVersionTree(
        self,
        document_version_id: int,
        *,
        scope_payload: dict[str, Any],
        nodes: list[dict[str, Any]] | None,
        edges: list[dict[str, Any]] | None = None,
        **__: Any,
    ) -> dict[str, Any]:
        payload = {
            "document_version_id": int(document_version_id),
            "scope": dict(scope_payload.get("scope", {})),
            "nodes": list(nodes or []),
            "edges": list(edges or []),
        }
        payload["node_count"] = len(payload["nodes"])
        payload["edge_count"] = len(payload["edges"])
        self.tree_replacements.append(payload)
        return payload

    def replaceDocumentVersionChunks(
        self,
        document_version_id: int,
        *,
        scope_payload: dict[str, Any],
        chunks: list[dict[str, Any]] | None,
        **__: Any,
    ) -> dict[str, Any]:
        payload = {
            "document_version_id": int(document_version_id),
            "scope": dict(scope_payload.get("scope", {})),
            "chunks": list(chunks or []),
        }
        payload["chunk_count"] = len(payload["chunks"])
        self.chunk_replacements.append(payload)
        return payload


class _FakeParserRouter:
    def __init__(self):
        self.calls: list[dict[str, Any]] = []

    def parse_document(self, profile_payload: dict[str, Any]) -> dict[str, Any]:
        self.calls.append(dict(profile_payload))
        return {
            "schema_version": 1,
            "canonical_schema": "document.parse.canonical.v1",
            "status": "parsed",
            "content_text": "Overview\n\nparsed text",
            "sections": [
                {
                    "section_id": "s1",
                    "title": "Overview",
                    "level": 1,
                    "text": "Overview",
                    "start_char": 0,
                    "end_char": 8,
                    "page_start": 1,
                    "page_end": 1,
                    "metadata": {"element_type": "heading"},
                },
                {
                    "section_id": "s2",
                    "title": "",
                    "level": 2,
                    "text": "parsed text",
                    "start_char": 10,
                    "end_char": 21,
                    "page_start": 1,
                    "page_end": 1,
                    "metadata": {"element_type": "paragraph"},
                },
            ],
            "metadata": {"router": "fake"},
            "warnings": [],
            "errors": [],
            "provenance": {
                "selected_adapter": "fake-adapter",
                "selected_adapter_version": "1.0.0",
                "adapter_chain": ["fake-adapter"],
                "confidence": 0.95,
                "cost": 0.2,
                "duration_ms": 11,
                "route_debug": {"selected_chain": ["fake-adapter"]},
                "profile_summary": {"file_name": profile_payload.get("file_name")},
            },
        }


class DocumentIngestionWorkerTests(unittest.TestCase):
    def test_run_once_completes_job_and_marks_related_records(self):
        store = _FakeJobStore()
        documents = _FakeDocumentManager()

        worker = DocumentIngestionWorker(
            job_store=store,
            document_manager=documents,
            processor=lambda *_: {
                "parser_name": "unit-parser",
                "parser_version": "1.2.3",
                "job_metadata_patch": {"unit_test": True},
            },
            worker_id="worker-test",
        )

        self.assertTrue(worker.run_once())
        self.assertEqual(len(store.finish_attempt_calls), 1)
        self.assertEqual(store.finish_attempt_calls[0]["attempt_status"], "succeeded")
        self.assertEqual(documents.version_updates[-1]["parser_status"], "parsed")
        self.assertEqual(documents.source_updates[-1]["source_state"], "parsed")
        self.assertEqual(documents.storage_updates[-1]["object_state"], "parsed")
        event_types = [item.get("event_type") for item in store.finish_attempt_calls[0]["stage_events"]]
        self.assertIn("ingestion.job.completed", event_types)

    def test_run_once_transient_failure_moves_job_to_retry_without_marking_failed_records(self):
        store = _FakeJobStore(fail_result={"job_status": "retry_wait"})
        documents = _FakeDocumentManager()

        def _boom(*_: Any, **__: Any) -> dict[str, Any]:
            raise RuntimeError("temporary parser outage")

        worker = DocumentIngestionWorker(
            job_store=store,
            document_manager=documents,
            processor=_boom,
            worker_id="worker-test",
        )

        self.assertTrue(worker.run_once())
        self.assertEqual(len(store.finish_attempt_calls), 1)
        self.assertEqual(store.finish_attempt_calls[0]["attempt_status"], "failed")
        self.assertEqual(documents.version_updates, [])
        self.assertEqual(documents.source_updates, [])
        self.assertEqual(documents.storage_updates, [])
        event_types = [item.get("event_type") for item in store.finish_attempt_calls[0]["stage_events"]]
        self.assertIn("ingestion.job.retry", event_types)

    def test_run_once_dead_letters_and_marks_related_records_failed(self):
        store = _FakeJobStore(
            fail_result={
                "job_status": "dead_letter",
                "dead_letter_reason": "max_attempts_exhausted:parser_error",
            }
        )
        documents = _FakeDocumentManager()

        worker = DocumentIngestionWorker(
            job_store=store,
            document_manager=documents,
            processor=lambda *_: (_ for _ in ()).throw(RuntimeError("parser failed")),
            worker_id="worker-test",
        )

        self.assertTrue(worker.run_once())
        self.assertEqual(len(store.finish_attempt_calls), 1)
        self.assertEqual(store.finish_attempt_calls[0]["attempt_status"], "failed")
        self.assertEqual(documents.version_updates[-1]["parser_status"], "failed")
        self.assertEqual(documents.source_updates[-1]["source_state"], "failed")
        self.assertEqual(documents.storage_updates[-1]["object_state"], "failed")
        event_types = [item.get("event_type") for item in store.finish_attempt_calls[0]["stage_events"]]
        self.assertIn("ingestion.job.dead_letter", event_types)

    def test_default_worker_processor_invokes_parser_router(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "sample.txt"
            path.write_text("hello parser", encoding="utf-8")

            store = _FakeJobStore()
            documents = _FakeDocumentManager()
            documents.storage_rows = [
                {
                    "storage_object_id": 701,
                    "document_source_id": 501,
                    "file_name": "sample.txt",
                    "file_mime": "text/plain",
                    "file_size_bytes": path.stat().st_size,
                    "storage_path": str(path),
                    "file_sha256": "",
                    "storage_backend": "local_fs",
                    "storage_key": "k1",
                    "record_metadata": {},
                }
            ]
            parser_router = _FakeParserRouter()

            worker = DocumentIngestionWorker(
                job_store=store,
                document_manager=documents,
                parser_router=parser_router,
                worker_id="worker-test",
            )

            self.assertTrue(worker.run_once())
            self.assertEqual(len(parser_router.calls), 1)
            self.assertEqual(documents.version_updates[-1]["parser_status"], "parsed")
            self.assertEqual(documents.version_updates[-1]["extra"]["parser_name"], "fake-adapter")
            self.assertEqual(len(documents.tree_replacements), 1)
            self.assertGreaterEqual(documents.tree_replacements[0]["node_count"], 2)
            self.assertGreaterEqual(documents.tree_replacements[0]["edge_count"], 1)
            self.assertEqual(len(documents.chunk_replacements), 1)
            self.assertGreaterEqual(documents.chunk_replacements[0]["chunk_count"], 1)


if __name__ == "__main__":
    unittest.main()
