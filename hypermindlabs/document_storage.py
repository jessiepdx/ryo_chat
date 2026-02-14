from __future__ import annotations

import hashlib
import mimetypes
import os
import re
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


_SAFE_FILENAME_PATTERN = re.compile(r"[^A-Za-z0-9._-]+")


class DocumentStorageError(RuntimeError):
    """Raised when storage backend operations fail."""


class DocumentStorageLimitError(DocumentStorageError):
    """Raised when uploaded content exceeds configured limits."""


class DocumentStorageWriteError(DocumentStorageError):
    """Raised when a storage write fails while streaming payload data."""


@dataclass(frozen=True)
class DocumentStorageWriteResult:
    storage_backend: str
    storage_key: str
    absolute_path: str
    file_sha256: str
    file_size_bytes: int
    file_mime: str
    deduped_existing: bool
    created_at: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "storage_backend": self.storage_backend,
            "storage_key": self.storage_key,
            "absolute_path": self.absolute_path,
            "file_sha256": self.file_sha256,
            "file_size_bytes": int(self.file_size_bytes),
            "file_mime": self.file_mime,
            "deduped_existing": bool(self.deduped_existing),
            "created_at": self.created_at,
        }


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _safe_filename(filename: str) -> str:
    cleaned = _SAFE_FILENAME_PATTERN.sub("_", str(filename or "").strip())
    cleaned = cleaned.strip("._")
    return cleaned or "upload.bin"


def _sniff_mime(header: bytes, filename: str, mime_hint: str | None = None) -> str:
    hint = str(mime_hint or "").strip().lower()
    if hint and hint != "application/octet-stream":
        return hint

    if header.startswith(b"%PDF-"):
        return "application/pdf"
    if header.startswith(b"PK\x03\x04"):
        return "application/zip"
    if header.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if header.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"

    guessed, _ = mimetypes.guess_type(filename)
    if guessed:
        return str(guessed).strip().lower()
    return "application/octet-stream"


class LocalDocumentStorage:
    """Filesystem-backed streaming storage for raw uploaded document bytes."""

    def __init__(
        self,
        base_path: str | Path,
        *,
        max_file_bytes: int = 100 * 1024 * 1024,
        chunk_size_bytes: int = 1024 * 1024,
        allow_empty_files: bool = False,
    ):
        self._base_path = Path(base_path)
        self._objects_path = self._base_path / "objects"
        self._tmp_path = self._base_path / ".tmp"
        self._max_file_bytes = max(1024, int(max_file_bytes))
        self._chunk_size_bytes = max(16 * 1024, int(chunk_size_bytes))
        self._allow_empty_files = bool(allow_empty_files)
        self._objects_path.mkdir(parents=True, exist_ok=True)
        self._tmp_path.mkdir(parents=True, exist_ok=True)

    @property
    def base_path(self) -> Path:
        return self._base_path

    def absolute_path_for_key(self, storage_key: str) -> Path:
        return self._base_path / Path(str(storage_key).strip())

    def storage_uri_for_key(self, storage_key: str) -> str:
        normalized = str(storage_key).strip().replace("\\", "/")
        return f"local://{normalized}"

    def store_stream(
        self,
        stream: Any,
        *,
        filename: str,
        mime_hint: str | None = None,
    ) -> DocumentStorageWriteResult:
        if not hasattr(stream, "read"):
            raise DocumentStorageError("Uploaded stream object must expose a read() method.")

        safe_name = _safe_filename(filename)
        tmp_name = f"{uuid.uuid4().hex}.{safe_name}.part"
        tmp_path = self._tmp_path / tmp_name
        digest = hashlib.sha256()
        file_size = 0
        header = bytearray()

        try:
            with tmp_path.open("wb") as handle:
                while True:
                    chunk = stream.read(self._chunk_size_bytes)
                    if chunk is None:
                        chunk = b""
                    if isinstance(chunk, str):
                        chunk = chunk.encode("utf-8")
                    if not isinstance(chunk, (bytes, bytearray)):
                        raise DocumentStorageWriteError("Upload stream returned a non-bytes chunk.")
                    if not chunk:
                        break

                    file_size += len(chunk)
                    if file_size > self._max_file_bytes:
                        raise DocumentStorageLimitError(
                            f"Upload exceeded configured max file size ({self._max_file_bytes} bytes)."
                        )

                    if len(header) < 4096:
                        header.extend(chunk[: 4096 - len(header)])

                    digest.update(chunk)
                    handle.write(chunk)

            if file_size <= 0 and not self._allow_empty_files:
                raise DocumentStorageError("Empty uploads are not allowed.")

            file_sha256 = digest.hexdigest().lower()
            storage_key = f"objects/{file_sha256[:2]}/{file_sha256}"
            destination = self.absolute_path_for_key(storage_key)
            destination.parent.mkdir(parents=True, exist_ok=True)

            deduped_existing = destination.exists()
            if deduped_existing:
                tmp_path.unlink(missing_ok=True)
            else:
                os.replace(tmp_path, destination)

            return DocumentStorageWriteResult(
                storage_backend="local_fs",
                storage_key=storage_key,
                absolute_path=str(destination),
                file_sha256=file_sha256,
                file_size_bytes=file_size,
                file_mime=_sniff_mime(bytes(header), safe_name, mime_hint=mime_hint),
                deduped_existing=deduped_existing,
                created_at=_now_iso(),
            )
        except DocumentStorageError:
            tmp_path.unlink(missing_ok=True)
            raise
        except Exception as error:  # noqa: BLE001
            tmp_path.unlink(missing_ok=True)
            raise DocumentStorageWriteError(f"Failed to persist upload stream: {error}") from error
