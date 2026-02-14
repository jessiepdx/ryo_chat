from __future__ import annotations

import copy
import re
import uuid
from typing import Any

from hypermindlabs.document_models import (
    DOCUMENT_SCHEMA_VERSION,
    DOCUMENT_NODE_EDGE_TYPES,
    DOCUMENT_NODE_TYPES,
    DOCUMENT_SCOPE_KEYS,
    DOCUMENT_SOURCE_STATES,
    CitationSpanContract,
    DocumentChunkContract,
    DocumentEmbeddingContract,
    DocumentNodeEdgeContract,
    DocumentNodeContract,
    DocumentRetrievalEventContract,
    DocumentScope,
    DocumentSourceContract,
    DocumentVersionContract,
    ParseArtifactContract,
)
from hypermindlabs.document_scope import (
    DocumentScopeValidationError,
    resolve_document_scope,
)


SUPPORTED_SCHEMA_VERSIONS: tuple[int, ...] = (1,)
_SHA256_PATTERN = re.compile(r"^[A-Fa-f0-9]{64}$")


class DocumentContractValidationError(ValueError):
    """Raised when a document contract payload is invalid."""


def _coerce_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return copy.deepcopy(value)
    return {}


def _coerce_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return copy.deepcopy(value)
    return []


def _as_text(value: Any, default: str = "") -> str:
    text = str(value if value is not None else "").strip()
    return text if text else default


def _positive_int(value: Any, field_name: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        raise DocumentContractValidationError(f"{field_name} must be an integer.") from None
    if parsed <= 0:
        raise DocumentContractValidationError(f"{field_name} must be greater than zero.")
    return parsed


def _non_negative_int(value: Any, field_name: str, default: int = 0) -> int:
    if value is None:
        return int(default)
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        raise DocumentContractValidationError(f"{field_name} must be an integer.") from None
    if parsed < 0:
        raise DocumentContractValidationError(f"{field_name} cannot be negative.")
    return parsed


def _optional_int(value: Any, field_name: str) -> int | None:
    if value is None or str(value).strip() == "":
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        raise DocumentContractValidationError(f"{field_name} must be an integer or null.") from None
    return parsed


def _optional_non_negative_float(value: Any, field_name: str) -> float | None:
    if value is None or str(value).strip() == "":
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        raise DocumentContractValidationError(f"{field_name} must be a number or null.") from None
    if parsed < 0.0:
        raise DocumentContractValidationError(f"{field_name} cannot be negative.")
    return parsed


def _schema_version_value(value: Any) -> int:
    if value is None:
        return int(DOCUMENT_SCHEMA_VERSION)
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        raise DocumentContractValidationError("schema_version must be an integer.") from None
    return parsed


def schema_version_is_compatible(schema_version: int) -> bool:
    return int(schema_version) in set(SUPPORTED_SCHEMA_VERSIONS)


def validate_schema_version(schema_version: Any) -> int:
    parsed = _schema_version_value(schema_version)
    if not schema_version_is_compatible(parsed):
        supported = ", ".join(str(item) for item in SUPPORTED_SCHEMA_VERSIONS)
        raise DocumentContractValidationError(
            f"Unsupported schema_version={parsed}. Supported versions: {supported}."
        )
    return parsed


def schema_version_compatibility_policy() -> dict[str, Any]:
    return {
        "current_schema_version": int(DOCUMENT_SCHEMA_VERSION),
        "supported_schema_versions": list(SUPPORTED_SCHEMA_VERSIONS),
        "policy": (
            "Backward-compatible additive changes are allowed within a supported "
            "schema_version. Breaking changes require a new schema_version."
        ),
    }


def _scope_source(payload: dict[str, Any]) -> dict[str, Any]:
    source = {}
    nested = payload.get("scope")
    if isinstance(nested, dict):
        source.update(_coerce_dict(nested))
    for key in DOCUMENT_SCOPE_KEYS:
        if key in payload:
            source[key] = payload.get(key)
    return source


def _inherit_scope_and_schema(root_payload: dict[str, Any], child_payload: dict[str, Any]) -> dict[str, Any]:
    root_scope = _scope_source(root_payload)
    child_scope = _scope_source(child_payload)
    merged_scope: dict[str, Any] = {}
    for key in DOCUMENT_SCOPE_KEYS:
        if key in child_scope:
            merged_scope[key] = child_scope.get(key)
        elif key in root_scope:
            merged_scope[key] = root_scope.get(key)

    merged = _coerce_dict(child_payload)
    merged["scope"] = merged_scope
    if "schema_version" not in merged and "schema_version" in root_payload:
        merged["schema_version"] = root_payload.get("schema_version")
    return merged


def validate_document_scope(
    payload: dict[str, Any],
    *,
    authenticated_member_id: int | None = None,
) -> DocumentScope:
    try:
        resolved = resolve_document_scope(
            payload,
            authenticated_member_id=authenticated_member_id,
        )
    except DocumentScopeValidationError as error:
        raise DocumentContractValidationError(str(error)) from None

    return DocumentScope(
        owner_member_id=resolved.owner_member_id,
        chat_host_id=resolved.chat_host_id,
        chat_type=resolved.chat_type,
        community_id=resolved.community_id,
        topic_id=resolved.topic_id,
        platform=resolved.platform,
    )


def validate_parse_artifact_contract(
    payload: dict[str, Any],
    *,
    scope: DocumentScope | None = None,
    schema_version: int | None = None,
    authenticated_member_id: int | None = None,
) -> ParseArtifactContract:
    source = _coerce_dict(payload)
    resolved_schema_version = validate_schema_version(
        schema_version if schema_version is not None else source.get("schema_version")
    )
    resolved_scope = (
        scope
        if isinstance(scope, DocumentScope)
        else validate_document_scope(source, authenticated_member_id=authenticated_member_id)
    )

    parser_name = _as_text(source.get("parser_name"), "unknown-parser")
    parser_version = _as_text(source.get("parser_version"))
    parse_mode = _as_text(source.get("parse_mode"), "full")
    status = _as_text(source.get("status"), "queued").lower()
    if status not in DOCUMENT_SOURCE_STATES:
        raise DocumentContractValidationError(
            f"Invalid parse status '{status}'."
        )

    artifact_payload = source.get("artifact")
    if not isinstance(artifact_payload, dict):
        artifact_payload = source.get("parse_artifact")
    artifact = _coerce_dict(artifact_payload)

    warnings = [str(item).strip() for item in _coerce_list(source.get("warnings")) if str(item).strip()]
    errors = [str(item).strip() for item in _coerce_list(source.get("errors")) if str(item).strip()]

    return ParseArtifactContract(
        schema_version=resolved_schema_version,
        scope=resolved_scope,
        parser_name=parser_name,
        parser_version=parser_version,
        parse_mode=parse_mode,
        status=status,
        artifact=artifact,
        warnings=warnings,
        errors=errors,
    )


def validate_document_source_contract(
    payload: dict[str, Any],
    *,
    authenticated_member_id: int | None = None,
) -> DocumentSourceContract:
    source = _coerce_dict(payload)
    schema_version = validate_schema_version(source.get("schema_version"))
    scope = validate_document_scope(source, authenticated_member_id=authenticated_member_id)

    source_name = _as_text(
        source.get("source_name")
        or source.get("file_name")
        or source.get("name")
    )
    if not source_name:
        raise DocumentContractValidationError("source_name is required.")

    sha256 = _as_text(source.get("source_sha256")).lower()
    if sha256 and not _SHA256_PATTERN.fullmatch(sha256):
        raise DocumentContractValidationError("source_sha256 must be a 64-character hex digest.")

    source_state = _as_text(source.get("source_state") or source.get("state"), "received").lower()
    if source_state not in DOCUMENT_SOURCE_STATES:
        raise DocumentContractValidationError(
            f"source_state must be one of: {', '.join(DOCUMENT_SOURCE_STATES)}."
        )

    size_value = source.get("source_size_bytes")
    source_size_bytes = None if size_value is None else _non_negative_int(
        size_value,
        "source_size_bytes",
    )

    return DocumentSourceContract(
        schema_version=schema_version,
        scope=scope,
        source_name=source_name,
        source_external_id=_as_text(source.get("source_external_id") or source.get("source_id")),
        source_mime=_as_text(source.get("source_mime") or source.get("mime")),
        source_sha256=sha256,
        source_size_bytes=source_size_bytes,
        source_uri=_as_text(source.get("source_uri")),
        source_metadata=_coerce_dict(source.get("source_metadata") or source.get("metadata")),
        source_state=source_state,
    )


def normalize_document_version_seed(
    payload: dict[str, Any],
    *,
    scope: DocumentScope,
    schema_version: int,
) -> dict[str, Any]:
    source = _coerce_dict(payload)
    version_number = _positive_int(source.get("version_number", 1), "version_number")
    artifact_contract = validate_parse_artifact_contract(
        source,
        scope=scope,
        schema_version=schema_version,
    )
    parser_status = _as_text(source.get("parser_status"), artifact_contract.status).lower()
    if parser_status not in DOCUMENT_SOURCE_STATES:
        raise DocumentContractValidationError(
            f"parser_status must be one of: {', '.join(DOCUMENT_SOURCE_STATES)}."
        )

    return {
        "schema_version": schema_version,
        "scope": scope.to_dict(),
        "version_number": version_number,
        "parser_name": _as_text(source.get("parser_name"), artifact_contract.parser_name),
        "parser_version": _as_text(source.get("parser_version"), artifact_contract.parser_version),
        "parser_status": parser_status,
        "parse_artifact": artifact_contract.to_dict(),
        "record_metadata": _coerce_dict(source.get("record_metadata")),
        "source_sha256": _as_text(source.get("source_sha256")).lower(),
    }


def validate_document_version_contract(
    payload: dict[str, Any],
    *,
    authenticated_member_id: int | None = None,
) -> DocumentVersionContract:
    source = _coerce_dict(payload)
    schema_version = validate_schema_version(source.get("schema_version"))
    scope = validate_document_scope(source, authenticated_member_id=authenticated_member_id)
    document_source_id = _positive_int(source.get("document_source_id"), "document_source_id")
    version_number = _positive_int(source.get("version_number"), "version_number")
    parser_status = _as_text(source.get("parser_status"), "queued").lower()
    if parser_status not in DOCUMENT_SOURCE_STATES:
        raise DocumentContractValidationError(
            f"parser_status must be one of: {', '.join(DOCUMENT_SOURCE_STATES)}."
        )
    parse_artifact = _coerce_dict(source.get("parse_artifact"))
    return DocumentVersionContract(
        schema_version=schema_version,
        scope=scope,
        document_source_id=document_source_id,
        version_number=version_number,
        parser_name=_as_text(source.get("parser_name")),
        parser_version=_as_text(source.get("parser_version")),
        parser_status=parser_status,
        parse_artifact=parse_artifact,
        record_metadata=_coerce_dict(source.get("record_metadata")),
        source_sha256=_as_text(source.get("source_sha256")).lower(),
    )


def validate_document_node_contract(
    payload: dict[str, Any],
    *,
    authenticated_member_id: int | None = None,
) -> DocumentNodeContract:
    source = _coerce_dict(payload)
    schema_version = validate_schema_version(source.get("schema_version"))
    scope = validate_document_scope(source, authenticated_member_id=authenticated_member_id)
    document_version_id = _positive_int(source.get("document_version_id"), "document_version_id")
    node_key = _as_text(source.get("node_key"))
    if not node_key:
        raise DocumentContractValidationError("node_key is required.")
    node_type = _as_text(source.get("node_type"))
    if not node_type:
        raise DocumentContractValidationError("node_type is required.")
    node_type = node_type.lower()
    if node_type not in DOCUMENT_NODE_TYPES:
        raise DocumentContractValidationError(
            f"node_type must be one of: {', '.join(DOCUMENT_NODE_TYPES)}."
        )
    parent_node_id = _optional_int(source.get("parent_node_id"), "parent_node_id")
    if parent_node_id is not None and parent_node_id <= 0:
        raise DocumentContractValidationError("parent_node_id must be greater than zero when provided.")

    return DocumentNodeContract(
        schema_version=schema_version,
        scope=scope,
        document_version_id=document_version_id,
        node_key=node_key,
        node_type=node_type,
        parent_node_id=parent_node_id,
        node_title=_as_text(source.get("node_title")),
        ordinal=_non_negative_int(source.get("ordinal"), "ordinal", default=0),
        token_count=_non_negative_int(source.get("token_count"), "token_count", default=0),
        page_start=_optional_int(source.get("page_start"), "page_start"),
        page_end=_optional_int(source.get("page_end"), "page_end"),
        char_start=_optional_int(source.get("char_start"), "char_start"),
        char_end=_optional_int(source.get("char_end"), "char_end"),
        path=_as_text(source.get("path")),
        node_metadata=_coerce_dict(source.get("node_metadata")),
    )


def validate_document_node_edge_contract(
    payload: dict[str, Any],
    *,
    authenticated_member_id: int | None = None,
) -> DocumentNodeEdgeContract:
    source = _coerce_dict(payload)
    schema_version = validate_schema_version(source.get("schema_version"))
    scope = validate_document_scope(source, authenticated_member_id=authenticated_member_id)
    document_version_id = _positive_int(source.get("document_version_id"), "document_version_id")
    source_node_id = _positive_int(source.get("source_node_id"), "source_node_id")
    target_node_id = _positive_int(source.get("target_node_id"), "target_node_id")
    edge_type = _as_text(source.get("edge_type")).lower()
    if not edge_type:
        raise DocumentContractValidationError("edge_type is required.")
    if edge_type not in DOCUMENT_NODE_EDGE_TYPES:
        raise DocumentContractValidationError(
            f"edge_type must be one of: {', '.join(DOCUMENT_NODE_EDGE_TYPES)}."
        )
    return DocumentNodeEdgeContract(
        schema_version=schema_version,
        scope=scope,
        document_version_id=document_version_id,
        source_node_id=source_node_id,
        target_node_id=target_node_id,
        edge_type=edge_type,
        ordinal=_non_negative_int(source.get("ordinal"), "ordinal", default=0),
        edge_metadata=_coerce_dict(source.get("edge_metadata")),
    )


def validate_document_chunk_contract(
    payload: dict[str, Any],
    *,
    authenticated_member_id: int | None = None,
) -> DocumentChunkContract:
    source = _coerce_dict(payload)
    schema_version = validate_schema_version(source.get("schema_version"))
    scope = validate_document_scope(source, authenticated_member_id=authenticated_member_id)
    document_version_id = _positive_int(source.get("document_version_id"), "document_version_id")
    chunk_key = _as_text(source.get("chunk_key"))
    if not chunk_key:
        raise DocumentContractValidationError("chunk_key is required.")
    chunk_text = _as_text(source.get("chunk_text"))
    if not chunk_text:
        raise DocumentContractValidationError("chunk_text is required.")

    chunk_index = _non_negative_int(source.get("chunk_index"), "chunk_index", default=0)
    node_id = _optional_int(source.get("document_node_id"), "document_node_id")
    if node_id is not None and node_id <= 0:
        raise DocumentContractValidationError("document_node_id must be greater than zero when provided.")

    return DocumentChunkContract(
        schema_version=schema_version,
        scope=scope,
        document_version_id=document_version_id,
        chunk_key=chunk_key,
        chunk_index=chunk_index,
        chunk_text=chunk_text,
        document_node_id=node_id,
        token_count=_non_negative_int(source.get("token_count"), "token_count", default=0),
        start_char=_optional_int(source.get("start_char"), "start_char"),
        end_char=_optional_int(source.get("end_char"), "end_char"),
        start_page=_optional_int(source.get("start_page"), "start_page"),
        end_page=_optional_int(source.get("end_page"), "end_page"),
        chunk_digest=_as_text(source.get("chunk_digest")).lower(),
        chunk_metadata=_coerce_dict(source.get("chunk_metadata")),
    )


def validate_document_embedding_contract(
    payload: dict[str, Any],
    *,
    authenticated_member_id: int | None = None,
) -> DocumentEmbeddingContract:
    source = _coerce_dict(payload)
    schema_version = validate_schema_version(source.get("schema_version"))
    scope = validate_document_scope(source, authenticated_member_id=authenticated_member_id)
    document_chunk_id = _positive_int(source.get("document_chunk_id"), "document_chunk_id")
    embedding_model = _as_text(source.get("embedding_model"))
    if not embedding_model:
        raise DocumentContractValidationError("embedding_model is required.")

    vector = source.get("embedding_vector")
    vector_list: list[float] = []
    if vector is not None:
        if not isinstance(vector, list):
            raise DocumentContractValidationError("embedding_vector must be an array when provided.")
        for index, raw in enumerate(vector):
            try:
                vector_list.append(float(raw))
            except (TypeError, ValueError):
                raise DocumentContractValidationError(
                    f"embedding_vector[{index}] must be numeric."
                ) from None

    return DocumentEmbeddingContract(
        schema_version=schema_version,
        scope=scope,
        document_chunk_id=document_chunk_id,
        embedding_model=embedding_model,
        embedding_role=_as_text(source.get("embedding_role"), "content"),
        embedding_vector=vector_list,
        embedding_metadata=_coerce_dict(source.get("embedding_metadata")),
    )


def validate_citation_span_contract(
    payload: dict[str, Any],
    *,
    scope: DocumentScope | None = None,
    schema_version: int | None = None,
    authenticated_member_id: int | None = None,
) -> CitationSpanContract:
    source = _coerce_dict(payload)
    resolved_schema_version = validate_schema_version(
        schema_version if schema_version is not None else source.get("schema_version")
    )
    resolved_scope = (
        scope
        if isinstance(scope, DocumentScope)
        else validate_document_scope(source, authenticated_member_id=authenticated_member_id)
    )
    start_char = _optional_int(source.get("start_char"), "start_char")
    end_char = _optional_int(source.get("end_char"), "end_char")
    if (
        start_char is not None
        and end_char is not None
        and end_char < start_char
    ):
        raise DocumentContractValidationError("end_char cannot be less than start_char.")

    start_page = _optional_int(source.get("start_page"), "start_page")
    end_page = _optional_int(source.get("end_page"), "end_page")
    if start_page is not None and start_page <= 0:
        raise DocumentContractValidationError("start_page must be greater than zero when provided.")
    if end_page is not None and end_page <= 0:
        raise DocumentContractValidationError("end_page must be greater than zero when provided.")
    if (
        start_page is not None
        and end_page is not None
        and end_page < start_page
    ):
        raise DocumentContractValidationError("end_page cannot be less than start_page.")

    return CitationSpanContract(
        schema_version=resolved_schema_version,
        scope=resolved_scope,
        document_chunk_id=_optional_int(source.get("document_chunk_id"), "document_chunk_id"),
        document_node_id=_optional_int(source.get("document_node_id"), "document_node_id"),
        start_char=start_char,
        end_char=end_char,
        start_page=start_page,
        end_page=end_page,
        quote_text=_as_text(source.get("quote_text")),
        citation_label=_as_text(source.get("citation_label")),
        score=_optional_non_negative_float(source.get("score"), "score"),
        metadata=_coerce_dict(source.get("metadata")),
    )


def validate_document_retrieval_event_contract(
    payload: dict[str, Any],
    *,
    authenticated_member_id: int | None = None,
) -> DocumentRetrievalEventContract:
    source = _coerce_dict(payload)
    schema_version = validate_schema_version(source.get("schema_version"))
    scope = validate_document_scope(source, authenticated_member_id=authenticated_member_id)
    request_id = _as_text(source.get("request_id")) or uuid.uuid4().hex
    query_text = _as_text(source.get("query_text") or source.get("query"))
    if not query_text:
        raise DocumentContractValidationError("query_text is required.")

    citations_payload = _coerce_list(source.get("citations"))
    citations: list[dict[str, Any]] = []
    for raw in citations_payload:
        citation = validate_citation_span_contract(
            _coerce_dict(raw),
            scope=scope,
            schema_version=schema_version,
        )
        citations.append(citation.to_dict())

    return DocumentRetrievalEventContract(
        schema_version=schema_version,
        scope=scope,
        request_id=request_id,
        query_text=query_text,
        document_source_id=_optional_int(source.get("document_source_id"), "document_source_id"),
        document_version_id=_optional_int(source.get("document_version_id"), "document_version_id"),
        result_count=_non_negative_int(source.get("result_count"), "result_count", default=0),
        max_distance=_optional_non_negative_float(source.get("max_distance"), "max_distance"),
        query_metadata=_coerce_dict(source.get("query_metadata")),
        retrieval_metadata=_coerce_dict(source.get("retrieval_metadata")),
        citations=citations,
    )


def normalize_document_ingest_request(
    payload: dict[str, Any],
    *,
    authenticated_member_id: int | None = None,
) -> dict[str, Any]:
    root = _coerce_dict(payload)
    if not root:
        raise DocumentContractValidationError("Ingest payload must be a JSON object.")

    source_payload = root.get("source")
    if not isinstance(source_payload, dict):
        source_payload = root
    source_payload = _inherit_scope_and_schema(root, _coerce_dict(source_payload))
    source_contract = validate_document_source_contract(
        source_payload,
        authenticated_member_id=authenticated_member_id,
    )

    version_seed: dict[str, Any] | None = None
    raw_version = root.get("version")
    if isinstance(raw_version, dict):
        merged_version = _inherit_scope_and_schema(root, raw_version)
        version_seed = normalize_document_version_seed(
            merged_version,
            scope=source_contract.scope,
            schema_version=source_contract.schema_version,
        )

    return {
        "schema_version": source_contract.schema_version,
        "scope": source_contract.scope.to_dict(),
        "source": source_contract.to_dict(),
        "version": version_seed,
        "compatibility": schema_version_compatibility_policy(),
    }


def normalize_document_retrieval_request(
    payload: dict[str, Any],
    *,
    authenticated_member_id: int | None = None,
) -> dict[str, Any]:
    root = _coerce_dict(payload)
    if not root:
        raise DocumentContractValidationError("Retrieval payload must be a JSON object.")
    event_payload = root.get("event")
    if not isinstance(event_payload, dict):
        event_payload = root
    merged_event = _inherit_scope_and_schema(root, _coerce_dict(event_payload))
    event_contract = validate_document_retrieval_event_contract(
        merged_event,
        authenticated_member_id=authenticated_member_id,
    )
    return {
        "schema_version": event_contract.schema_version,
        "scope": event_contract.scope.to_dict(),
        "event": event_contract.to_dict(),
        "compatibility": schema_version_compatibility_policy(),
    }
