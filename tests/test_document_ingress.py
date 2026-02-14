from __future__ import annotations

import io
import tempfile
import unittest
from pathlib import Path
from typing import Any

from hypermindlabs.document_ingress import DocumentIngressError, DocumentIngressService
from hypermindlabs.document_storage import LocalDocumentStorage


def _base_scope() -> dict[str, Any]:
    return {
        "scope": {
            "owner_member_id": 5,
            "chat_host_id": 55,
            "chat_type": "member",
            "community_id": None,
            "topic_id": None,
            "platform": "web",
        }
    }


class _TrackingStream(io.BytesIO):
    def __init__(self, data: bytes):
        super().__init__(data)
        self.read_calls = 0
        self.max_requested_size = 0

    def read(self, size: int = -1) -> bytes:
        self.read_calls += 1
        if size is not None and size > 0:
            self.max_requested_size = max(self.max_requested_size, int(size))
        return super().read(size)


class _FailingStream:
    def __init__(self, chunks_before_failure: int = 1):
        self._calls = 0
        self._chunks_before_failure = max(0, int(chunks_before_failure))

    def read(self, _: int = -1) -> bytes:
        self._calls += 1
        if self._calls > self._chunks_before_failure:
            raise OSError("simulated stream failure")
        return b"x" * 24


class _DummyConfig:
    def __init__(self, *, retention_days: int = 365):
        self._retention_days = int(retention_days)

    def runtimeInt(self, path: str, default: int) -> int:  # noqa: N802
        if path == "documents.retention_default_days":
            return self._retention_days
        return int(default)

    def runtimeValue(self, _: str, default: Any) -> Any:  # noqa: N802
        return default


class _FakeDocumentManager:
    def __init__(self):
        self._source_id = 100
        self._version_id = 900
        self._storage_id = 500
        self.sources: dict[int, dict[str, Any]] = {}
        self.versions: list[dict[str, Any]] = []
        self.storage_objects: dict[int, dict[str, Any]] = {}
        self.create_source_calls = 0
        self.create_version_calls = 0
        self.create_storage_calls = 0

    @staticmethod
    def _scope(payload: dict[str, Any]) -> dict[str, Any]:
        scope = payload.get("scope")
        return dict(scope) if isinstance(scope, dict) else {}

    def _source_match(self, row: dict[str, Any], scope: dict[str, Any]) -> bool:
        return (
            row.get("owner_member_id") == scope.get("owner_member_id")
            and row.get("chat_host_id") == scope.get("chat_host_id")
            and str(row.get("chat_type")) == str(scope.get("chat_type"))
            and str(row.get("platform")) == str(scope.get("platform"))
            and row.get("community_id") == scope.get("community_id")
            and row.get("topic_id") == scope.get("topic_id")
        )

    def findLatestDocumentSourceByDigest(
        self,
        scope_payload: dict[str, Any],
        source_sha256: str,
        *,
        source_name: str | None = None,
        **_: Any,
    ) -> dict[str, Any] | None:
        scope = self._scope(scope_payload)
        digest = str(source_sha256 or "").strip().lower()
        candidates = []
        for row in self.sources.values():
            if not self._source_match(row, scope):
                continue
            if str(row.get("source_sha256") or "").strip().lower() != digest:
                continue
            if source_name and str(row.get("source_name") or "") != str(source_name):
                continue
            candidates.append(row)
        if not candidates:
            return None
        return dict(sorted(candidates, key=lambda item: int(item.get("document_source_id", 0)), reverse=True)[0])

    def nextDocumentVersionNumber(self, document_source_id: int, _: dict[str, Any], **__: Any) -> int:
        versions = [row for row in self.versions if int(row.get("document_source_id", 0)) == int(document_source_id)]
        if not versions:
            return 1
        return max(int(row.get("version_number", 0)) for row in versions) + 1

    def createDocumentSource(self, payload: dict[str, Any], **_: Any) -> dict[str, Any]:
        self.create_source_calls += 1
        self._source_id += 1
        scope = self._scope(payload)
        row = {
            "document_source_id": self._source_id,
            "owner_member_id": scope.get("owner_member_id"),
            "chat_host_id": scope.get("chat_host_id"),
            "chat_type": scope.get("chat_type"),
            "community_id": scope.get("community_id"),
            "topic_id": scope.get("topic_id"),
            "platform": scope.get("platform"),
            "source_name": payload.get("source_name"),
            "source_sha256": payload.get("source_sha256"),
            "source_state": payload.get("source_state", "received"),
            "source_metadata": dict(payload.get("source_metadata") or {}),
        }
        self.sources[self._source_id] = row
        return dict(row)

    def createDocumentStorageObject(self, payload: dict[str, Any], **_: Any) -> dict[str, Any]:
        self.create_storage_calls += 1
        self._storage_id += 1
        scope = self._scope(payload)
        row = {
            "storage_object_id": self._storage_id,
            "document_source_id": int(payload.get("document_source_id", 0)),
            "owner_member_id": scope.get("owner_member_id"),
            "chat_host_id": scope.get("chat_host_id"),
            "chat_type": scope.get("chat_type"),
            "community_id": scope.get("community_id"),
            "topic_id": scope.get("topic_id"),
            "platform": scope.get("platform"),
            "object_state": payload.get("object_state", "received"),
            "storage_key": payload.get("storage_key"),
            "file_sha256": payload.get("file_sha256"),
            "record_metadata": dict(payload.get("record_metadata") or {}),
        }
        self.storage_objects[self._storage_id] = row
        return dict(row)

    def createDocumentVersion(self, payload: dict[str, Any], **_: Any) -> dict[str, Any]:
        self.create_version_calls += 1
        self._version_id += 1
        row = {
            "document_version_id": self._version_id,
            "document_source_id": int(payload.get("document_source_id", 0)),
            "version_number": int(payload.get("version_number", 1)),
            "parser_status": payload.get("parser_status", "queued"),
        }
        self.versions.append(row)
        return dict(row)

    def updateDocumentSourceState(
        self,
        document_source_id: int,
        source_state: str,
        _: dict[str, Any],
        *,
        metadata_patch: dict[str, Any] | None = None,
        **__: Any,
    ) -> dict[str, Any] | None:
        row = self.sources.get(int(document_source_id))
        if row is None:
            return None
        row["source_state"] = source_state
        if isinstance(metadata_patch, dict):
            merged = dict(row.get("source_metadata") or {})
            merged.update(metadata_patch)
            row["source_metadata"] = merged
        return dict(row)

    def updateDocumentStorageObjectState(
        self,
        storage_object_id: int,
        object_state: str,
        _: dict[str, Any],
        **__: Any,
    ) -> dict[str, Any] | None:
        row = self.storage_objects.get(int(storage_object_id))
        if row is None:
            return None
        row["object_state"] = object_state
        return dict(row)

    def getDocumentStorageObjects(self, _: dict[str, Any], *, document_source_id: int | None = None, **__: Any) -> list[dict[str, Any]]:
        rows = list(self.storage_objects.values())
        if document_source_id is not None:
            rows = [row for row in rows if int(row.get("document_source_id", 0)) == int(document_source_id)]
        return [dict(row) for row in rows]


class _FakeIngestionJobStore:
    def __init__(self):
        self.calls: list[dict[str, Any]] = []
        self._job_id = 700

    def enqueue_job(
        self,
        *,
        scope_payload: dict[str, Any],
        document_source_id: int,
        document_version_id: int,
        storage_object_id: int | None = None,
        **_: Any,
    ) -> dict[str, Any]:
        self._job_id += 1
        payload = {
            "ingestion_job_id": self._job_id,
            "scope": dict(scope_payload.get("scope", {})),
            "document_source_id": int(document_source_id),
            "document_version_id": int(document_version_id),
            "storage_object_id": int(storage_object_id or 0) or None,
            "job_status": "queued",
        }
        self.calls.append(payload)
        return dict(payload)


class DocumentIngressTests(unittest.TestCase):
    def test_ingest_streams_upload_and_queues_version(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            manager = _FakeDocumentManager()
            storage = LocalDocumentStorage(
                Path(tmp_dir),
                max_file_bytes=10 * 1024 * 1024,
                chunk_size_bytes=64 * 1024,
            )
            service = DocumentIngressService(
                config_manager=_DummyConfig(retention_days=90),
                document_manager=manager,
                storage_backend=storage,
            )
            stream = _TrackingStream(b"a" * 200000)
            result = service.ingest_upload(
                stream=stream,
                filename="large.pdf",
                scope_payload=_base_scope(),
                actor_member_id=5,
                actor_roles=["user"],
                source_metadata={"channel": "tests"},
                mime_hint="application/pdf",
            )

            self.assertEqual(result["status"], "ok")
            self.assertEqual(result["dedupe_status"], "new")
            self.assertEqual(result["source"]["source_state"], "queued")
            self.assertEqual(result["storage_object"]["object_state"], "queued")
            self.assertEqual(result["version"]["version_number"], 1)
            self.assertGreater(stream.read_calls, 2)
            self.assertEqual(stream.max_requested_size, 64 * 1024)

    def test_duplicate_upload_reuses_source_and_increments_version(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            manager = _FakeDocumentManager()
            storage = LocalDocumentStorage(Path(tmp_dir), chunk_size_bytes=32 * 1024, max_file_bytes=1024 * 1024)
            service = DocumentIngressService(
                config_manager=_DummyConfig(),
                document_manager=manager,
                storage_backend=storage,
            )

            first = service.ingest_upload(
                stream=io.BytesIO(b"duplicate payload"),
                filename="dup.pdf",
                scope_payload=_base_scope(),
                actor_member_id=5,
                actor_roles=["user"],
            )
            second = service.ingest_upload(
                stream=io.BytesIO(b"duplicate payload"),
                filename="dup.pdf",
                scope_payload=_base_scope(),
                actor_member_id=5,
                actor_roles=["user"],
            )

            self.assertEqual(first["dedupe_status"], "new")
            self.assertEqual(second["dedupe_status"], "duplicate_digest")
            self.assertEqual(first["source"]["document_source_id"], second["source"]["document_source_id"])
            self.assertEqual(second["version"]["version_number"], 2)
            self.assertEqual(manager.create_source_calls, 1)
            self.assertEqual(manager.create_version_calls, 2)

    def test_ingest_rejects_oversized_upload(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            manager = _FakeDocumentManager()
            storage = LocalDocumentStorage(Path(tmp_dir), max_file_bytes=1024, chunk_size_bytes=64)
            service = DocumentIngressService(
                config_manager=_DummyConfig(),
                document_manager=manager,
                storage_backend=storage,
            )

            with self.assertRaises(DocumentIngressError):
                service.ingest_upload(
                    stream=io.BytesIO(b"x" * 4096),
                    filename="too-large.txt",
                    scope_payload=_base_scope(),
                    actor_member_id=5,
                    actor_roles=["user"],
                )

            self.assertEqual(manager.create_source_calls, 0)
            self.assertEqual(manager.create_version_calls, 0)

    def test_partial_write_failure_cleans_temp_files(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            manager = _FakeDocumentManager()
            storage = LocalDocumentStorage(Path(tmp_dir), max_file_bytes=1024 * 1024, chunk_size_bytes=32)
            service = DocumentIngressService(
                config_manager=_DummyConfig(),
                document_manager=manager,
                storage_backend=storage,
            )

            with self.assertRaises(DocumentIngressError):
                service.ingest_upload(
                    stream=_FailingStream(chunks_before_failure=1),
                    filename="broken.pdf",
                    scope_payload=_base_scope(),
                    actor_member_id=5,
                    actor_roles=["user"],
                )

            tmp_path = Path(tmp_dir) / ".tmp"
            self.assertEqual(list(tmp_path.glob("*.part")), [])
            self.assertEqual(manager.create_source_calls, 0)
            self.assertEqual(manager.create_storage_calls, 0)

    def test_ingest_enqueues_persistent_ingestion_job(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            manager = _FakeDocumentManager()
            jobs = _FakeIngestionJobStore()
            storage = LocalDocumentStorage(Path(tmp_dir), chunk_size_bytes=16 * 1024, max_file_bytes=5 * 1024 * 1024)
            service = DocumentIngressService(
                config_manager=_DummyConfig(),
                document_manager=manager,
                storage_backend=storage,
                ingestion_job_store=jobs,
            )

            result = service.ingest_upload(
                stream=io.BytesIO(b"queue me"),
                filename="queue.pdf",
                scope_payload=_base_scope(),
                actor_member_id=5,
                actor_roles=["user"],
            )

            self.assertEqual(len(jobs.calls), 1)
            self.assertEqual(result["job"]["job_status"], "queued")
            self.assertGreater(int(result["job"]["ingestion_job_id"]), 0)
            self.assertEqual(result["job"]["document_source_id"], result["source"]["document_source_id"])


if __name__ == "__main__":
    unittest.main()
