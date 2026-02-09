from __future__ import annotations

import copy
import json
import re
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    import yaml as _yaml  # type: ignore
except Exception:  # noqa: BLE001
    _yaml = None


STORE_SCHEMA = "ryo.agent_definitions.store.v1"
DEFINITION_SCHEMA = "ryo.agent_definition.v1"
DEFAULT_STORE_PATH = Path(__file__).resolve().parent.parent / "db" / "agent_definitions.json"
_SLUG_PATTERN = re.compile(r"[^a-z0-9]+")


class AgentDefinitionValidationError(ValueError):
    """Raised when an agent definition payload is invalid."""


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _as_text(value: Any, fallback: str = "") -> str:
    text = str(value if value is not None else "").strip()
    return text if text else fallback


def _as_bool(value: Any, fallback: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    return fallback


def _as_number(value: Any, fallback: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(fallback)


def _as_int(value: Any, fallback: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(fallback)


def _as_string_list(value: Any) -> list[str]:
    if isinstance(value, list):
        source = value
    elif isinstance(value, str):
        source = [part for part in value.split(",")]
    else:
        return []
    output: list[str] = []
    for item in source:
        cleaned = _as_text(item)
        if cleaned and cleaned not in output:
            output.append(cleaned)
    return output


def _coerce_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return copy.deepcopy(value)
    return {}


def _build_default_definition(name: str = "New Agent") -> dict[str, Any]:
    clean_name = _as_text(name, "New Agent")
    return {
        "schema": DEFINITION_SCHEMA,
        "identity": {
            "name": clean_name,
            "description": "Describe this agent's responsibilities.",
            "tags": ["playground"],
            "visibility": "private",
        },
        "system_prompt": {
            "strategy": "policy",
            "policy_name": "chat_conversation",
            "text": "",
        },
        "model_policy": {
            "default_model": "",
            "allowed_models": [],
            "capability_models": {},
            "temperature": 0.2,
            "top_p": 0.9,
            "seed": 42,
        },
        "tool_access_policy": {
            "enabled_tools": [],
            "denied_tools": [],
            "per_tool": {},
            "custom_tools": [],
        },
        "memory_strategy": {
            "short_term_strategy": "trim_last_n",
            "long_term_strategy": "episodic",
            "token_budget": 2048,
            "ttl_seconds": 86400,
        },
        "guardrail_hooks": {
            "pre": [],
            "mid": [],
            "post": [],
            "allow_internal_diagnostics": False,
        },
        "orchestration": {
            "pattern": "single",
            "delegation_enabled": False,
            "planner": "",
            "executor": "",
            "verifier": "",
        },
    }


def normalize_agent_definition(payload: dict[str, Any] | None) -> dict[str, Any]:
    source = _coerce_dict(payload)
    if not source:
        source = _build_default_definition()

    identity = _coerce_dict(source.get("identity"))
    system_prompt = _coerce_dict(source.get("system_prompt"))
    model_policy = _coerce_dict(source.get("model_policy"))
    tool_policy = _coerce_dict(source.get("tool_access_policy"))
    memory_strategy = _coerce_dict(source.get("memory_strategy"))
    guardrails = _coerce_dict(source.get("guardrail_hooks"))
    orchestration = _coerce_dict(source.get("orchestration"))

    name = _as_text(identity.get("name"), "Untitled Agent")
    visibility = _as_text(identity.get("visibility"), "private").lower()
    if visibility not in {"private", "workspace", "public"}:
        visibility = "private"

    strategy = _as_text(system_prompt.get("strategy"), "policy").lower()
    if strategy not in {"policy", "inline", "hybrid"}:
        strategy = "policy"

    pattern = _as_text(orchestration.get("pattern"), "single").lower()
    if pattern not in {"single", "delegated", "hierarchical"}:
        pattern = "single"

    capability_models = _coerce_dict(model_policy.get("capability_models"))
    clean_capability_models: dict[str, str] = {}
    for capability, model_name in capability_models.items():
        clean_capability = _as_text(capability).lower()
        clean_model = _as_text(model_name)
        if clean_capability and clean_model:
            clean_capability_models[clean_capability] = clean_model

    per_tool = _coerce_dict(tool_policy.get("per_tool"))
    clean_per_tool: dict[str, dict[str, Any]] = {}
    for tool_name, overrides in per_tool.items():
        clean_tool_name = _as_text(tool_name)
        override_map = _coerce_dict(overrides)
        if not clean_tool_name:
            continue
        clean_per_tool[clean_tool_name] = {
            "timeout_seconds": _as_number(override_map.get("timeout_seconds"), 8.0),
            "max_retries": max(0, _as_int(override_map.get("max_retries"), 0)),
            "required_api_key": _as_text(override_map.get("required_api_key")),
        }

    custom_tools = []
    raw_custom_tools = tool_policy.get("custom_tools")
    if isinstance(raw_custom_tools, list):
        for raw_tool in raw_custom_tools:
            tool_map = _coerce_dict(raw_tool)
            tool_name = _as_text(tool_map.get("name"))
            if not tool_name:
                continue
            custom_tools.append(
                {
                    "name": tool_name,
                    "description": _as_text(tool_map.get("description"), f"Custom tool '{tool_name}'"),
                    "input_schema": _coerce_dict(tool_map.get("input_schema")) or {
                        "type": "object",
                        "properties": {},
                    },
                    "handler_mode": _as_text(tool_map.get("handler_mode"), "echo").lower(),
                    "static_result": copy.deepcopy(tool_map.get("static_result")),
                    "required_api_key": _as_text(tool_map.get("required_api_key")) or None,
                    "timeout_seconds": _as_number(tool_map.get("timeout_seconds"), 8.0),
                    "max_retries": max(0, _as_int(tool_map.get("max_retries"), 0)),
                }
            )

    normalized = {
        "schema": DEFINITION_SCHEMA,
        "identity": {
            "name": name,
            "description": _as_text(identity.get("description")),
            "tags": _as_string_list(identity.get("tags")),
            "visibility": visibility,
        },
        "system_prompt": {
            "strategy": strategy,
            "policy_name": _as_text(system_prompt.get("policy_name"), "chat_conversation"),
            "text": _as_text(system_prompt.get("text")),
        },
        "model_policy": {
            "default_model": _as_text(model_policy.get("default_model")),
            "allowed_models": _as_string_list(model_policy.get("allowed_models")),
            "capability_models": clean_capability_models,
            "temperature": _as_number(model_policy.get("temperature"), 0.2),
            "top_p": _as_number(model_policy.get("top_p"), 0.9),
            "seed": _as_int(model_policy.get("seed"), 42),
            "ollama_host": _as_text(model_policy.get("ollama_host")),
        },
        "tool_access_policy": {
            "enabled_tools": _as_string_list(tool_policy.get("enabled_tools")),
            "denied_tools": _as_string_list(tool_policy.get("denied_tools")),
            "per_tool": clean_per_tool,
            "custom_tools": custom_tools,
        },
        "memory_strategy": {
            "short_term_strategy": _as_text(memory_strategy.get("short_term_strategy"), "trim_last_n"),
            "long_term_strategy": _as_text(memory_strategy.get("long_term_strategy"), "episodic"),
            "token_budget": max(128, _as_int(memory_strategy.get("token_budget"), 2048)),
            "ttl_seconds": max(0, _as_int(memory_strategy.get("ttl_seconds"), 86400)),
        },
        "guardrail_hooks": {
            "pre": _as_string_list(guardrails.get("pre")),
            "mid": _as_string_list(guardrails.get("mid")),
            "post": _as_string_list(guardrails.get("post")),
            "allow_internal_diagnostics": _as_bool(guardrails.get("allow_internal_diagnostics"), False),
        },
        "orchestration": {
            "pattern": pattern,
            "delegation_enabled": _as_bool(orchestration.get("delegation_enabled"), False),
            "planner": _as_text(orchestration.get("planner")),
            "executor": _as_text(orchestration.get("executor")),
            "verifier": _as_text(orchestration.get("verifier")),
        },
    }
    return normalized


def runtime_options_from_agent_definition(payload: dict[str, Any] | None) -> dict[str, Any]:
    definition = normalize_agent_definition(payload)
    model_policy = _coerce_dict(definition.get("model_policy"))
    tool_policy = _coerce_dict(definition.get("tool_access_policy"))
    guardrails = _coerce_dict(definition.get("guardrail_hooks"))
    memory_strategy = _coerce_dict(definition.get("memory_strategy"))
    orchestration = _coerce_dict(definition.get("orchestration"))

    options: dict[str, Any] = {}
    model_requested = _as_text(model_policy.get("default_model"))
    if model_requested:
        options["model_requested"] = model_requested

    ollama_host = _as_text(model_policy.get("ollama_host"))
    if ollama_host:
        options["ollama_host"] = ollama_host

    capability_models = _coerce_dict(model_policy.get("capability_models"))
    if capability_models:
        options["capability_models"] = capability_models

    allowed_models = _as_string_list(model_policy.get("allowed_models"))
    if allowed_models:
        options["allowed_models"] = allowed_models

    options["temperature"] = _as_number(model_policy.get("temperature"), 0.2)
    options["top_p"] = _as_number(model_policy.get("top_p"), 0.9)
    options["seed"] = _as_int(model_policy.get("seed"), 42)

    enabled_tools = _as_string_list(tool_policy.get("enabled_tools"))
    denied_tools = _as_string_list(tool_policy.get("denied_tools"))
    per_tool = _coerce_dict(tool_policy.get("per_tool"))
    custom_tools = tool_policy.get("custom_tools")
    if enabled_tools:
        options["enabled_tools"] = enabled_tools
    if denied_tools:
        options["denied_tools"] = denied_tools
    if per_tool:
        options["tool_policy"] = per_tool
    if isinstance(custom_tools, list):
        options["custom_tools"] = copy.deepcopy(custom_tools)

    options["allow_internal_diagnostics"] = _as_bool(guardrails.get("allow_internal_diagnostics"), False)
    options["memory_strategy"] = memory_strategy
    options["orchestration"] = orchestration
    options["agent_definition_name"] = _as_text(_coerce_dict(definition.get("identity")).get("name"), "agent")
    return options


def _slugify(text: str) -> str:
    lowered = _as_text(text).lower()
    lowered = _SLUG_PATTERN.sub("-", lowered).strip("-")
    return lowered or "agent"


class AgentDefinitionStore:
    """File-backed version store for agent definition artifacts."""

    def __init__(self, storage_path: str | Path = DEFAULT_STORE_PATH):
        self._storage_path = Path(storage_path)
        self._lock = threading.RLock()
        self._ensure_store()

    def _ensure_store(self) -> None:
        self._storage_path.parent.mkdir(parents=True, exist_ok=True)
        if self._storage_path.exists():
            return
        payload = {
            "schema": STORE_SCHEMA,
            "updated_at": _now_iso(),
            "definitions": {},
        }
        self._storage_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def _load(self) -> dict[str, Any]:
        self._ensure_store()
        try:
            raw = json.loads(self._storage_path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            raw = {}
        if not isinstance(raw, dict):
            raw = {}
        definitions = raw.get("definitions")
        if not isinstance(definitions, dict):
            definitions = {}
        return {
            "schema": STORE_SCHEMA,
            "updated_at": _as_text(raw.get("updated_at"), _now_iso()),
            "definitions": definitions,
        }

    def _save(self, payload: dict[str, Any]) -> None:
        payload["updated_at"] = _now_iso()
        self._storage_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def _summary(self, record: dict[str, Any]) -> dict[str, Any]:
        versions = record.get("versions") if isinstance(record, dict) else []
        if not isinstance(versions, list):
            versions = []
        latest = versions[-1] if versions else {}
        identity = _coerce_dict(_coerce_dict(latest).get("definition")).get("identity")
        return {
            "definition_id": _as_text(record.get("definition_id")),
            "owner_member_id": _as_int(record.get("owner_member_id"), 0),
            "created_at": _as_text(record.get("created_at")),
            "updated_at": _as_text(record.get("updated_at")),
            "active_version": _as_int(record.get("active_version"), 1),
            "version_count": len(versions),
            "name": _as_text(_coerce_dict(identity).get("name"), "Untitled Agent"),
            "description": _as_text(_coerce_dict(identity).get("description")),
            "tags": _as_string_list(_coerce_dict(identity).get("tags")),
            "last_change_summary": _as_text(_coerce_dict(latest).get("change_summary")),
        }

    def list_definitions(self, owner_member_id: int | None = None) -> list[dict[str, Any]]:
        with self._lock:
            payload = self._load()
            definitions = payload.get("definitions", {})
            output: list[dict[str, Any]] = []
            for record in definitions.values():
                if not isinstance(record, dict):
                    continue
                if owner_member_id is not None and _as_int(record.get("owner_member_id"), 0) != int(owner_member_id):
                    continue
                output.append(self._summary(record))
            output.sort(key=lambda item: item.get("updated_at", ""), reverse=True)
            return output

    def get_definition(
        self,
        definition_id: str,
        *,
        version: int | None = None,
        include_versions: bool = True,
    ) -> dict[str, Any] | None:
        with self._lock:
            payload = self._load()
            definitions = payload.get("definitions", {})
            record = definitions.get(definition_id)
            if not isinstance(record, dict):
                return None
            versions = record.get("versions") if isinstance(record.get("versions"), list) else []
            if not versions:
                return None
            selected_version = int(record.get("active_version", 1)) if version is None else int(version)
            selected = None
            for item in versions:
                if _as_int(_coerce_dict(item).get("version"), -1) == selected_version:
                    selected = _coerce_dict(item)
                    break
            if selected is None:
                selected = _coerce_dict(versions[-1])
            response = self._summary(record)
            response["active_version"] = _as_int(record.get("active_version"), 1)
            response["selected_version"] = _as_int(selected.get("version"), response["active_version"])
            response["definition"] = normalize_agent_definition(_coerce_dict(selected.get("definition")))
            if include_versions:
                response["versions"] = [
                    {
                        "version": _as_int(_coerce_dict(item).get("version"), 0),
                        "created_at": _as_text(_coerce_dict(item).get("created_at")),
                        "author_member_id": _as_int(_coerce_dict(item).get("author_member_id"), 0),
                        "change_summary": _as_text(_coerce_dict(item).get("change_summary")),
                    }
                    for item in versions
                    if isinstance(item, dict)
                ]
            return response

    def _next_id(self, name: str, definitions: dict[str, Any]) -> str:
        base = _slugify(name)
        candidate = f"ag-{base}"
        if candidate not in definitions:
            return candidate
        while True:
            candidate = f"ag-{base}-{uuid.uuid4().hex[:6]}"
            if candidate not in definitions:
                return candidate

    def create_definition(
        self,
        payload: dict[str, Any] | None,
        *,
        author_member_id: int,
        change_summary: str = "Initial definition",
    ) -> dict[str, Any]:
        source = _coerce_dict(payload)
        source_definition = _coerce_dict(source.get("definition")) if "definition" in source else source
        definition = normalize_agent_definition(source_definition)
        name = _as_text(_coerce_dict(definition.get("identity")).get("name"), "Untitled Agent")
        now = _now_iso()

        with self._lock:
            data = self._load()
            definitions = data.get("definitions", {})
            if not isinstance(definitions, dict):
                definitions = {}
                data["definitions"] = definitions

            definition_id = self._next_id(name, definitions)
            record = {
                "definition_id": definition_id,
                "owner_member_id": int(author_member_id),
                "created_at": now,
                "updated_at": now,
                "active_version": 1,
                "versions": [
                    {
                        "version": 1,
                        "created_at": now,
                        "author_member_id": int(author_member_id),
                        "change_summary": _as_text(change_summary, "Initial definition"),
                        "definition": definition,
                    }
                ],
            }
            definitions[definition_id] = record
            self._save(data)

        detail = self.get_definition(definition_id)
        if detail is None:
            raise AgentDefinitionValidationError("Failed to create definition.")
        return detail

    def create_version(
        self,
        definition_id: str,
        payload: dict[str, Any] | None,
        *,
        author_member_id: int,
        change_summary: str = "Updated definition",
    ) -> dict[str, Any]:
        definition = normalize_agent_definition(_coerce_dict(payload))
        now = _now_iso()
        with self._lock:
            data = self._load()
            definitions = data.get("definitions", {})
            record = definitions.get(definition_id)
            if not isinstance(record, dict):
                raise AgentDefinitionValidationError(f"Unknown definition id: {definition_id}")
            versions = record.get("versions")
            if not isinstance(versions, list):
                versions = []
                record["versions"] = versions
            next_version = 1
            if versions:
                next_version = _as_int(_coerce_dict(versions[-1]).get("version"), 0) + 1
            versions.append(
                {
                    "version": next_version,
                    "created_at": now,
                    "author_member_id": int(author_member_id),
                    "change_summary": _as_text(change_summary, "Updated definition"),
                    "definition": definition,
                }
            )
            record["active_version"] = next_version
            record["updated_at"] = now
            definitions[definition_id] = record
            data["definitions"] = definitions
            self._save(data)
        detail = self.get_definition(definition_id)
        if detail is None:
            raise AgentDefinitionValidationError("Failed to update definition.")
        return detail

    def rollback(
        self,
        definition_id: str,
        *,
        target_version: int,
        author_member_id: int,
        change_summary: str | None = None,
    ) -> dict[str, Any]:
        current = self.get_definition(definition_id)
        if current is None:
            raise AgentDefinitionValidationError(f"Unknown definition id: {definition_id}")
        versions = current.get("versions")
        if not isinstance(versions, list):
            raise AgentDefinitionValidationError("Definition has no version history.")

        selected_version_payload = self.get_definition(definition_id, version=int(target_version))
        if selected_version_payload is None:
            raise AgentDefinitionValidationError(f"Unknown version: {target_version}")
        selected_definition = selected_version_payload.get("definition")
        summary = _as_text(change_summary) or f"Rollback to v{int(target_version)}"
        return self.create_version(
            definition_id,
            _coerce_dict(selected_definition),
            author_member_id=int(author_member_id),
            change_summary=summary,
        )

    def export_definition(
        self,
        definition_id: str,
        *,
        version: int | None = None,
        fmt: str = "json",
    ) -> str:
        detail = self.get_definition(definition_id, version=version, include_versions=False)
        if detail is None:
            raise AgentDefinitionValidationError(f"Unknown definition id: {definition_id}")
        payload = {
            "schema": DEFINITION_SCHEMA,
            "definition_id": detail.get("definition_id"),
            "selected_version": detail.get("selected_version"),
            "exported_at": _now_iso(),
            "definition": detail.get("definition"),
        }
        format_name = _as_text(fmt, "json").lower()
        if format_name == "yaml":
            if _yaml is None:
                raise AgentDefinitionValidationError("YAML export requires PyYAML to be installed.")
            return _yaml.safe_dump(payload, sort_keys=False)  # type: ignore[union-attr]
        if format_name != "json":
            raise AgentDefinitionValidationError("format must be 'json' or 'yaml'.")
        return json.dumps(payload, indent=2)

    def import_definition(
        self,
        *,
        raw_payload: str,
        fmt: str = "json",
        author_member_id: int,
        change_summary: str = "Imported definition",
    ) -> dict[str, Any]:
        serialized = _as_text(raw_payload)
        if not serialized:
            raise AgentDefinitionValidationError("raw_payload cannot be empty.")
        format_name = _as_text(fmt, "json").lower()
        if format_name == "json":
            try:
                loaded = json.loads(serialized)
            except json.JSONDecodeError as error:
                raise AgentDefinitionValidationError(f"Invalid JSON payload: {error}") from error
        elif format_name == "yaml":
            if _yaml is None:
                raise AgentDefinitionValidationError("YAML import requires PyYAML to be installed.")
            try:
                loaded = _yaml.safe_load(serialized)  # type: ignore[union-attr]
            except Exception as error:  # noqa: BLE001
                raise AgentDefinitionValidationError(f"Invalid YAML payload: {error}") from error
        else:
            raise AgentDefinitionValidationError("format must be 'json' or 'yaml'.")

        payload = _coerce_dict(loaded)
        definition = _coerce_dict(payload.get("definition")) if "definition" in payload else payload
        return self.create_definition(
            definition,
            author_member_id=int(author_member_id),
            change_summary=_as_text(change_summary, "Imported definition"),
        )


__all__ = [
    "AgentDefinitionStore",
    "AgentDefinitionValidationError",
    "DEFINITION_SCHEMA",
    "normalize_agent_definition",
    "runtime_options_from_agent_definition",
]
