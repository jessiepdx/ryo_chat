from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any


DOCUMENT_SCHEMA_VERSION = 1
DOCUMENT_SCOPE_KEYS: tuple[str, ...] = (
    "owner_member_id",
    "chat_host_id",
    "chat_type",
    "community_id",
    "topic_id",
    "platform",
)
DOCUMENT_SOURCE_STATES: tuple[str, ...] = (
    "received",
    "queued",
    "parsed",
    "failed",
    "archived",
    "deleted",
)
DOCUMENT_NODE_TYPES: tuple[str, ...] = (
    "document",
    "section",
    "subsection",
    "paragraph",
    "list",
    "table",
    "code",
    "figure",
    "footnote",
)
DOCUMENT_NODE_EDGE_TYPES: tuple[str, ...] = (
    "parent_child",
    "next_sibling",
    "reference",
)


def _deep_copy_dict(value: dict[str, Any]) -> dict[str, Any]:
    return copy.deepcopy(value)


def _deep_copy_list(value: list[Any]) -> list[Any]:
    return copy.deepcopy(value)


@dataclass(frozen=True)
class DocumentScope:
    owner_member_id: int
    chat_host_id: int
    chat_type: str
    community_id: int | None
    topic_id: int | None
    platform: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "owner_member_id": int(self.owner_member_id),
            "chat_host_id": int(self.chat_host_id),
            "chat_type": str(self.chat_type),
            "community_id": self.community_id,
            "topic_id": self.topic_id,
            "platform": str(self.platform),
        }


@dataclass(frozen=True)
class ParseArtifactContract:
    schema_version: int
    scope: DocumentScope
    parser_name: str
    parser_version: str
    parse_mode: str
    status: str
    artifact: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "schema_version": int(self.schema_version),
            "scope": self.scope.to_dict(),
            "parser_name": str(self.parser_name),
            "parser_version": str(self.parser_version),
            "parse_mode": str(self.parse_mode),
            "status": str(self.status),
            "artifact": _deep_copy_dict(self.artifact),
            "warnings": list(self.warnings),
            "errors": list(self.errors),
        }
        return payload


@dataclass(frozen=True)
class DocumentSourceContract:
    schema_version: int
    scope: DocumentScope
    source_name: str
    source_external_id: str = ""
    source_mime: str = ""
    source_sha256: str = ""
    source_size_bytes: int | None = None
    source_uri: str = ""
    source_metadata: dict[str, Any] = field(default_factory=dict)
    source_state: str = "received"

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "schema_version": int(self.schema_version),
            "scope": self.scope.to_dict(),
            "source_name": str(self.source_name),
            "source_external_id": str(self.source_external_id),
            "source_mime": str(self.source_mime),
            "source_sha256": str(self.source_sha256),
            "source_size_bytes": self.source_size_bytes,
            "source_uri": str(self.source_uri),
            "source_metadata": _deep_copy_dict(self.source_metadata),
            "source_state": str(self.source_state),
        }
        return payload


@dataclass(frozen=True)
class DocumentVersionContract:
    schema_version: int
    scope: DocumentScope
    document_source_id: int
    version_number: int
    parser_name: str = ""
    parser_version: str = ""
    parser_status: str = "queued"
    parse_artifact: dict[str, Any] = field(default_factory=dict)
    record_metadata: dict[str, Any] = field(default_factory=dict)
    source_sha256: str = ""

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "schema_version": int(self.schema_version),
            "scope": self.scope.to_dict(),
            "document_source_id": int(self.document_source_id),
            "version_number": int(self.version_number),
            "parser_name": str(self.parser_name),
            "parser_version": str(self.parser_version),
            "parser_status": str(self.parser_status),
            "parse_artifact": _deep_copy_dict(self.parse_artifact),
            "record_metadata": _deep_copy_dict(self.record_metadata),
            "source_sha256": str(self.source_sha256),
        }
        return payload


@dataclass(frozen=True)
class DocumentNodeContract:
    schema_version: int
    scope: DocumentScope
    document_version_id: int
    node_key: str
    node_type: str
    parent_node_id: int | None = None
    node_title: str = ""
    ordinal: int = 0
    token_count: int = 0
    page_start: int | None = None
    page_end: int | None = None
    char_start: int | None = None
    char_end: int | None = None
    path: str = ""
    node_metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "schema_version": int(self.schema_version),
            "scope": self.scope.to_dict(),
            "document_version_id": int(self.document_version_id),
            "node_key": str(self.node_key),
            "node_type": str(self.node_type),
            "parent_node_id": self.parent_node_id,
            "node_title": str(self.node_title),
            "ordinal": int(self.ordinal),
            "token_count": int(self.token_count),
            "page_start": self.page_start,
            "page_end": self.page_end,
            "char_start": self.char_start,
            "char_end": self.char_end,
            "path": str(self.path),
            "node_metadata": _deep_copy_dict(self.node_metadata),
        }
        return payload


@dataclass(frozen=True)
class DocumentNodeEdgeContract:
    schema_version: int
    scope: DocumentScope
    document_version_id: int
    source_node_id: int
    target_node_id: int
    edge_type: str
    ordinal: int = 0
    edge_metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "schema_version": int(self.schema_version),
            "scope": self.scope.to_dict(),
            "document_version_id": int(self.document_version_id),
            "source_node_id": int(self.source_node_id),
            "target_node_id": int(self.target_node_id),
            "edge_type": str(self.edge_type),
            "ordinal": int(self.ordinal),
            "edge_metadata": _deep_copy_dict(self.edge_metadata),
        }
        return payload


@dataclass(frozen=True)
class DocumentChunkContract:
    schema_version: int
    scope: DocumentScope
    document_version_id: int
    chunk_key: str
    chunk_index: int
    chunk_text: str
    document_node_id: int | None = None
    token_count: int = 0
    start_char: int | None = None
    end_char: int | None = None
    start_page: int | None = None
    end_page: int | None = None
    chunk_digest: str = ""
    chunk_metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "schema_version": int(self.schema_version),
            "scope": self.scope.to_dict(),
            "document_version_id": int(self.document_version_id),
            "chunk_key": str(self.chunk_key),
            "chunk_index": int(self.chunk_index),
            "chunk_text": str(self.chunk_text),
            "document_node_id": self.document_node_id,
            "token_count": int(self.token_count),
            "start_char": self.start_char,
            "end_char": self.end_char,
            "start_page": self.start_page,
            "end_page": self.end_page,
            "chunk_digest": str(self.chunk_digest),
            "chunk_metadata": _deep_copy_dict(self.chunk_metadata),
        }
        return payload


@dataclass(frozen=True)
class DocumentEmbeddingContract:
    schema_version: int
    scope: DocumentScope
    document_chunk_id: int
    embedding_model: str
    embedding_role: str = "content"
    embedding_vector: list[float] = field(default_factory=list)
    embedding_metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "schema_version": int(self.schema_version),
            "scope": self.scope.to_dict(),
            "document_chunk_id": int(self.document_chunk_id),
            "embedding_model": str(self.embedding_model),
            "embedding_role": str(self.embedding_role),
            "embedding_vector": _deep_copy_list(self.embedding_vector),
            "embedding_metadata": _deep_copy_dict(self.embedding_metadata),
        }
        return payload


@dataclass(frozen=True)
class CitationSpanContract:
    schema_version: int
    scope: DocumentScope
    document_chunk_id: int | None = None
    document_node_id: int | None = None
    start_char: int | None = None
    end_char: int | None = None
    start_page: int | None = None
    end_page: int | None = None
    quote_text: str = ""
    citation_label: str = ""
    score: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "schema_version": int(self.schema_version),
            "scope": self.scope.to_dict(),
            "document_chunk_id": self.document_chunk_id,
            "document_node_id": self.document_node_id,
            "start_char": self.start_char,
            "end_char": self.end_char,
            "start_page": self.start_page,
            "end_page": self.end_page,
            "quote_text": str(self.quote_text),
            "citation_label": str(self.citation_label),
            "score": self.score,
            "metadata": _deep_copy_dict(self.metadata),
        }
        return payload


@dataclass(frozen=True)
class DocumentRetrievalEventContract:
    schema_version: int
    scope: DocumentScope
    request_id: str
    query_text: str
    document_source_id: int | None = None
    document_version_id: int | None = None
    result_count: int = 0
    max_distance: float | None = None
    query_metadata: dict[str, Any] = field(default_factory=dict)
    retrieval_metadata: dict[str, Any] = field(default_factory=dict)
    citations: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "schema_version": int(self.schema_version),
            "scope": self.scope.to_dict(),
            "request_id": str(self.request_id),
            "query_text": str(self.query_text),
            "document_source_id": self.document_source_id,
            "document_version_id": self.document_version_id,
            "result_count": int(self.result_count),
            "max_distance": self.max_distance,
            "query_metadata": _deep_copy_dict(self.query_metadata),
            "retrieval_metadata": _deep_copy_dict(self.retrieval_metadata),
            "citations": _deep_copy_list(self.citations),
        }
        return payload
