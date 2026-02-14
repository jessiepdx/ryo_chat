from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


def normalize_extension(filename: str) -> str:
    suffix = Path(str(filename or "").strip()).suffix.lower()
    if suffix and not suffix.startswith("."):
        return "." + suffix
    return suffix


def clamp_confidence(value: Any, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        parsed = float(default)
    if parsed < 0.0:
        parsed = 0.0
    if parsed > 1.0:
        parsed = 1.0
    return parsed


def clamp_cost(value: Any, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        parsed = float(default)
    if parsed < 0.0:
        parsed = 0.0
    return parsed


@dataclass(frozen=True)
class DocumentParseProfile:
    document_source_id: int
    document_version_id: int
    storage_object_id: int | None
    file_name: str
    file_mime: str
    file_extension: str
    file_size_bytes: int
    file_path: str
    file_sha256: str = ""
    storage_backend: str = ""
    storage_key: str = ""
    probes: dict[str, Any] = field(default_factory=dict)
    complexity_score: float = 0.0
    complexity_label: str = "simple"
    record_metadata: dict[str, Any] = field(default_factory=dict)
    scope: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "document_source_id": int(self.document_source_id),
            "document_version_id": int(self.document_version_id),
            "storage_object_id": self.storage_object_id,
            "file_name": str(self.file_name),
            "file_mime": str(self.file_mime),
            "file_extension": str(self.file_extension),
            "file_size_bytes": int(self.file_size_bytes),
            "file_path": str(self.file_path),
            "file_sha256": str(self.file_sha256),
            "storage_backend": str(self.storage_backend),
            "storage_key": str(self.storage_key),
            "probes": dict(self.probes),
            "complexity_score": float(self.complexity_score),
            "complexity_label": str(self.complexity_label),
            "record_metadata": dict(self.record_metadata),
            "scope": dict(self.scope),
        }


def build_parse_profile(payload: dict[str, Any]) -> DocumentParseProfile:
    source = dict(payload) if isinstance(payload, dict) else {}
    file_name = str(source.get("file_name") or "").strip()
    extension = str(source.get("file_extension") or "").strip().lower() or normalize_extension(file_name)
    try:
        source_id = int(source.get("document_source_id", 0))
    except (TypeError, ValueError):
        source_id = 0
    try:
        version_id = int(source.get("document_version_id", 0))
    except (TypeError, ValueError):
        version_id = 0
    try:
        object_id = int(source.get("storage_object_id", 0))
    except (TypeError, ValueError):
        object_id = 0
    try:
        file_size = int(source.get("file_size_bytes", 0))
    except (TypeError, ValueError):
        file_size = 0
    try:
        complexity = float(source.get("complexity_score", 0.0))
    except (TypeError, ValueError):
        complexity = 0.0
    if complexity < 0.0:
        complexity = 0.0
    if complexity > 1.0:
        complexity = 1.0
    return DocumentParseProfile(
        document_source_id=max(0, source_id),
        document_version_id=max(0, version_id),
        storage_object_id=object_id if object_id > 0 else None,
        file_name=file_name,
        file_mime=str(source.get("file_mime") or "").strip().lower(),
        file_extension=extension,
        file_size_bytes=max(0, file_size),
        file_path=str(source.get("file_path") or "").strip(),
        file_sha256=str(source.get("file_sha256") or "").strip().lower(),
        storage_backend=str(source.get("storage_backend") or "").strip().lower(),
        storage_key=str(source.get("storage_key") or "").strip(),
        probes=dict(source.get("probes") or {}) if isinstance(source.get("probes"), dict) else {},
        complexity_score=complexity,
        complexity_label=str(source.get("complexity_label") or "").strip().lower() or "simple",
        record_metadata=dict(source.get("record_metadata") or {}) if isinstance(source.get("record_metadata"), dict) else {},
        scope=dict(source.get("scope") or {}) if isinstance(source.get("scope"), dict) else {},
    )


class DocumentParserAdapter(ABC):
    adapter_name: str = "base-parser"
    adapter_version: str = "v1"

    @abstractmethod
    def can_parse(self, profile: DocumentParseProfile) -> bool:
        """Return whether this adapter can parse the provided profile."""

    @abstractmethod
    def confidence(self, profile: DocumentParseProfile) -> float:
        """Return confidence [0,1] for parsing this profile."""

    @abstractmethod
    def cost(self, profile: DocumentParseProfile) -> float:
        """Return relative cost estimate for this adapter/profile."""

    @abstractmethod
    def parse(self, profile: DocumentParseProfile) -> dict[str, Any]:
        """Return canonicalizable parser output for this profile."""
