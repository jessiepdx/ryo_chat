##########################################################################
#                                                                        #
#  This file (tool_registry.py) defines canonical metadata for tools     #
#  used by the tool-calling agent and runtime.                           #
#                                                                        #
##########################################################################

from __future__ import annotations

import copy
import json
from pathlib import Path
import re
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable

from hypermindlabs.runtime_settings import DEFAULT_RUNTIME_SETTINGS
from hypermindlabs.tool_sandbox import (
    merge_sandbox_policies,
    normalize_tool_sandbox_policy,
    resolve_tool_sandbox_policy,
)
from hypermindlabs.tool_runtime import ToolDefinition, ToolRuntime


_TOOL_RUNTIME_DEFAULTS = DEFAULT_RUNTIME_SETTINGS.get("tool_runtime", {})
DEFAULT_TIMEOUT_SECONDS = float(_TOOL_RUNTIME_DEFAULTS.get("default_timeout_seconds", 8.0))
DEFAULT_MAX_RETRIES = int(_TOOL_RUNTIME_DEFAULTS.get("default_max_retries", 1))
BRAVE_TIMEOUT_SECONDS = float(_TOOL_RUNTIME_DEFAULTS.get("brave_timeout_seconds", 10.0))
CHAT_HISTORY_TIMEOUT_SECONDS = float(_TOOL_RUNTIME_DEFAULTS.get("chat_history_timeout_seconds", 6.0))
KNOWLEDGE_TIMEOUT_SECONDS = float(_TOOL_RUNTIME_DEFAULTS.get("knowledge_timeout_seconds", 6.0))
SKIP_TOOLS_TIMEOUT_SECONDS = float(_TOOL_RUNTIME_DEFAULTS.get("skip_tools_timeout_seconds", 2.0))
DEFAULT_TOOL_REGISTRY_PATH = Path(__file__).resolve().parent.parent / "db" / "tool_registry.json"
_TOOL_NAME_PATTERN = re.compile(r"^[A-Za-z][A-Za-z0-9_]{2,64}$")


def _coerce_query(value: Any) -> str:
    if value is None:
        raise ValueError("queryString is required.")
    cleaned = str(value).strip()
    if not cleaned:
        raise ValueError("queryString cannot be empty.")
    return cleaned


def _coerce_positive_int(value: Any) -> int:
    if isinstance(value, bool):
        raise ValueError("boolean values are not valid integers.")
    if isinstance(value, int):
        count = value
    elif isinstance(value, float):
        if not value.is_integer():
            raise ValueError("value must be a whole number.")
        count = int(value)
    else:
        cleaned = str(value).strip()
        if cleaned == "":
            raise ValueError("value cannot be empty.")
        count = int(cleaned)

    if count <= 0:
        raise ValueError("value must be greater than zero.")
    return count


def _coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    raise ValueError("value must be a boolean.")


def _custom_tool_name(value: Any) -> str:
    name = _as_text(value)
    if not _TOOL_NAME_PATTERN.fullmatch(name):
        raise ToolRegistryValidationError(
            "Tool name must match ^[A-Za-z][A-Za-z0-9_]{2,64}$."
        )
    return name


@dataclass(frozen=True)
class ToolArgSpec:
    name: str
    json_type: str
    description: str
    required: bool = False
    default: Any = None
    coercer: Callable[[Any], Any] | None = None


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    function: Callable[..., Any]
    args: tuple[ToolArgSpec, ...]
    required_api_key: str | None = None
    default_timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS
    default_max_retries: int = DEFAULT_MAX_RETRIES
    side_effect_class: str = "read_only"
    sandbox_policy: dict[str, Any] = field(default_factory=dict)
    dry_run_mock_result: Any = None

    @property
    def required_args(self) -> tuple[str, ...]:
        return tuple(arg.name for arg in self.args if arg.required)

    @property
    def optional_args(self) -> dict[str, Any]:
        output: dict[str, Any] = {}
        for arg in self.args:
            if not arg.required:
                output[arg.name] = arg.default
        return output

    @property
    def arg_coercers(self) -> dict[str, Callable[[Any], Any]]:
        output: dict[str, Callable[[Any], Any]] = {}
        for arg in self.args:
            if callable(arg.coercer):
                output[arg.name] = arg.coercer
        return output

    def to_model_tool(self) -> dict[str, Any]:
        properties = {}
        required: list[str] = []
        for arg in self.args:
            properties[arg.name] = {
                "type": arg.json_type,
                "description": arg.description,
            }
            if arg.required:
                required.append(arg.name)

        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
            },
        }


class ToolRegistryValidationError(ValueError):
    """Raised when a custom tool registry payload is invalid."""


def _as_text(value: Any, fallback: str = "") -> str:
    cleaned = str(value if value is not None else "").strip()
    return cleaned if cleaned else fallback


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


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


def _coerce_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return copy.deepcopy(value)
    return {}


def _coerce_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return copy.deepcopy(value)
    return []


def _int_value(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _float_value(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _json_type(schema: dict[str, Any]) -> str:
    raw_type = schema.get("type")
    if isinstance(raw_type, str):
        return raw_type
    if isinstance(raw_type, list):
        for candidate in raw_type:
            if isinstance(candidate, str) and candidate != "null":
                return candidate
    return "string"


def _default_coercer_for_type(json_type: str) -> Callable[[Any], Any] | None:
    normalized = str(json_type).lower()
    if normalized == "integer":
        return _coerce_positive_int
    if normalized == "number":
        return lambda value: float(value)
    if normalized == "boolean":
        return _coerce_bool
    return None


def normalize_custom_tool_payload(payload: dict[str, Any] | None) -> dict[str, Any]:
    source = _coerce_dict(payload)
    if not source:
        raise ToolRegistryValidationError("Tool payload cannot be empty.")

    name = _custom_tool_name(source.get("name"))
    description = _as_text(source.get("description"), f"Custom tool '{name}'.")
    schema = _coerce_dict(source.get("input_schema"))
    if not schema:
        schema = {"type": "object", "properties": {}}
    if _json_type(schema) != "object":
        raise ToolRegistryValidationError("input_schema.type must be 'object'.")
    properties = _coerce_dict(schema.get("properties"))
    required = _coerce_list(schema.get("required"))
    required_names = [str(item).strip() for item in required if str(item).strip()]

    handler_mode = _as_text(source.get("handler_mode"), "echo").lower()
    if handler_mode not in {"echo", "static"}:
        handler_mode = "echo"

    raw_sandbox_policy = _coerce_dict(source.get("sandbox_policy"))
    if not raw_sandbox_policy and isinstance(source.get("sandbox"), dict):
        raw_sandbox_policy = _coerce_dict(source.get("sandbox"))
    if "side_effect_class" in source and "side_effect_class" not in raw_sandbox_policy:
        raw_sandbox_policy["side_effect_class"] = source.get("side_effect_class")
    if "approval_required" in source and "require_approval" not in raw_sandbox_policy:
        raw_sandbox_policy["require_approval"] = source.get("approval_required")
    if "dry_run" in source and "dry_run" not in raw_sandbox_policy:
        raw_sandbox_policy["dry_run"] = source.get("dry_run")
    if "approval_timeout_seconds" in source and "approval_timeout_seconds" not in raw_sandbox_policy:
        raw_sandbox_policy["approval_timeout_seconds"] = source.get("approval_timeout_seconds")
    sandbox_policy = normalize_tool_sandbox_policy(raw_sandbox_policy, default_tool_name=name)

    side_effect_class = _as_text(
        source.get("side_effect_class"),
        _as_text(sandbox_policy.get("side_effect_class"), "read_only"),
    ).lower()
    if side_effect_class not in {"read_only", "mutating", "sensitive"}:
        side_effect_class = "read_only"
    approval_required = _as_bool(source.get("approval_required"), bool(sandbox_policy.get("require_approval", False)))
    dry_run = _as_bool(source.get("dry_run"), bool(sandbox_policy.get("dry_run", False)))

    sandbox_policy["side_effect_class"] = side_effect_class
    sandbox_policy["require_approval"] = approval_required
    sandbox_policy["dry_run"] = dry_run

    normalized = {
        "name": name,
        "description": description,
        "source": "custom",
        "enabled": _as_bool(source.get("enabled"), True),
        "auth_requirements": _as_text(source.get("auth_requirements")),
        "side_effect_class": side_effect_class,
        "approval_required": approval_required,
        "dry_run": dry_run,
        "rate_limit_per_minute": max(0, _int_value(source.get("rate_limit_per_minute"), 0)),
        "handler_mode": handler_mode,
        "static_result": copy.deepcopy(source.get("static_result")),
        "dry_run_result": copy.deepcopy(source.get("dry_run_result")),
        "required_api_key": _as_text(source.get("required_api_key")) or None,
        "timeout_seconds": _float_value(source.get("timeout_seconds"), DEFAULT_TIMEOUT_SECONDS),
        "max_retries": max(0, _int_value(source.get("max_retries"), DEFAULT_MAX_RETRIES)),
        "sandbox_policy": sandbox_policy,
        "input_schema": {
            "type": "object",
            "properties": properties,
            "required": required_names,
        },
    }
    return normalized


def _custom_tool_args(input_schema: dict[str, Any]) -> tuple[ToolArgSpec, ...]:
    properties = _coerce_dict(input_schema.get("properties"))
    required = set([str(item).strip() for item in _coerce_list(input_schema.get("required")) if str(item).strip()])
    output: list[ToolArgSpec] = []
    for arg_name, arg_schema in properties.items():
        arg_schema_map = _coerce_dict(arg_schema)
        json_type = _json_type(arg_schema_map)
        description = _as_text(arg_schema_map.get("description"), f"Argument '{arg_name}'.")
        default = arg_schema_map.get("default")
        coercer = _default_coercer_for_type(json_type)
        output.append(
            ToolArgSpec(
                name=str(arg_name),
                json_type=json_type,
                description=description,
                required=str(arg_name) in required,
                default=default,
                coercer=coercer,
            )
        )
    return tuple(output)


def _custom_tool_function(entry: dict[str, Any]) -> Callable[..., Any]:
    mode = _as_text(entry.get("handler_mode"), "echo").lower()
    name = _as_text(entry.get("name"), "customTool")
    static_result = copy.deepcopy(entry.get("static_result"))

    def _handler(**kwargs: Any) -> Any:
        if mode == "static":
            if static_result is None:
                return {"tool": name, "status": "success"}
            return copy.deepcopy(static_result)
        return {
            "tool": name,
            "status": "success",
            "mode": "echo",
            "arguments": copy.deepcopy(kwargs),
        }

    return _handler


def custom_tool_specs(custom_tool_entries: list[dict[str, Any]] | None) -> dict[str, ToolSpec]:
    entries = custom_tool_entries if isinstance(custom_tool_entries, list) else []
    specs: dict[str, ToolSpec] = {}
    for raw_entry in entries:
        try:
            entry = normalize_custom_tool_payload(_coerce_dict(raw_entry))
        except ToolRegistryValidationError:
            continue
        if not _as_bool(entry.get("enabled"), True):
            continue
        input_schema = _coerce_dict(entry.get("input_schema"))
        args = _custom_tool_args(input_schema)
        specs[entry["name"]] = ToolSpec(
            name=entry["name"],
            description=_as_text(entry.get("description"), f"Custom tool '{entry['name']}'."),
            function=_custom_tool_function(entry),
            args=args,
            required_api_key=entry.get("required_api_key"),
            default_timeout_seconds=_float_value(entry.get("timeout_seconds"), DEFAULT_TIMEOUT_SECONDS),
            default_max_retries=max(0, _int_value(entry.get("max_retries"), DEFAULT_MAX_RETRIES)),
            side_effect_class=_as_text(entry.get("side_effect_class"), "read_only"),
            sandbox_policy=normalize_tool_sandbox_policy(
                _coerce_dict(entry.get("sandbox_policy")),
                default_tool_name=entry["name"],
            ),
            dry_run_mock_result=copy.deepcopy(entry.get("dry_run_result")),
        )
    return specs


def build_tool_specs(
    brave_search_fn: Callable[..., Any],
    chat_history_search_fn: Callable[..., Any],
    knowledge_search_fn: Callable[..., Any],
    skip_tools_fn: Callable[..., Any],
    knowledge_domains: list[str] | None = None,
    custom_tool_entries: list[dict[str, Any]] | None = None,
) -> dict[str, ToolSpec]:
    domains = knowledge_domains or []
    domain_text = ", ".join([str(domain).strip() for domain in domains if str(domain).strip()])
    if not domain_text:
        domain_text = "General"

    query_arg = ToolArgSpec(
        name="queryString",
        json_type="string",
        description="Search query used for retrieval.",
        required=True,
        coercer=_coerce_query,
    )
    count_arg = ToolArgSpec(
        name="count",
        json_type="integer",
        description="Maximum number of results to return.",
        required=False,
        default=5,
        coercer=_coerce_positive_int,
    )
    narrow_count_arg = ToolArgSpec(
        name="count",
        json_type="integer",
        description="Maximum number of results to return.",
        required=False,
        default=2,
        coercer=_coerce_positive_int,
    )

    brave_sandbox = resolve_tool_sandbox_policy(
        {
            "tool_name": "braveSearch",
            "side_effect_class": "read_only",
            "network": {
                "enabled": True,
                "allowlist_domains": ["api.search.brave.com"],
            },
            "filesystem": {"mode": "none"},
        }
    )
    local_read_sandbox = resolve_tool_sandbox_policy(
        {
            "side_effect_class": "read_only",
            "network": {"enabled": False, "allowlist_domains": []},
            "filesystem": {"mode": "none"},
        }
    )

    specs = {
        "braveSearch": ToolSpec(
            name="braveSearch",
            description="Search the public web for relevant information.",
            function=brave_search_fn,
            args=(query_arg, count_arg),
            required_api_key="brave_search",
            default_timeout_seconds=BRAVE_TIMEOUT_SECONDS,
            default_max_retries=1,
            side_effect_class="read_only",
            sandbox_policy=brave_sandbox,
        ),
        "chatHistorySearch": ToolSpec(
            name="chatHistorySearch",
            description="Search prior chat history for semantically related messages.",
            function=chat_history_search_fn,
            args=(query_arg, narrow_count_arg),
            default_timeout_seconds=CHAT_HISTORY_TIMEOUT_SECONDS,
            default_max_retries=0,
            side_effect_class="read_only",
            sandbox_policy=merge_sandbox_policies(local_read_sandbox, {"tool_name": "chatHistorySearch"}),
        ),
        "knowledgeSearch": ToolSpec(
            name="knowledgeSearch",
            description=(
                "Search project knowledge documents for semantically related results. "
                f"Known domains: {domain_text}."
            ),
            function=knowledge_search_fn,
            args=(query_arg, narrow_count_arg),
            default_timeout_seconds=KNOWLEDGE_TIMEOUT_SECONDS,
            default_max_retries=0,
            side_effect_class="read_only",
            sandbox_policy=merge_sandbox_policies(local_read_sandbox, {"tool_name": "knowledgeSearch"}),
        ),
        "skipTools": ToolSpec(
            name="skipTools",
            description="Skip tool usage and continue to the next agent.",
            function=skip_tools_fn,
            args=(),
            default_timeout_seconds=SKIP_TOOLS_TIMEOUT_SECONDS,
            default_max_retries=0,
            side_effect_class="read_only",
            sandbox_policy=merge_sandbox_policies(local_read_sandbox, {"tool_name": "skipTools"}),
        ),
    }
    specs.update(custom_tool_specs(custom_tool_entries))
    return specs


def ordered_tool_names(specs: dict[str, ToolSpec]) -> list[str]:
    builtins = ("braveSearch", "chatHistorySearch", "knowledgeSearch", "skipTools")
    output = [name for name in builtins if name in specs]
    custom_names = sorted([name for name in specs.keys() if name not in builtins])
    output.extend(custom_names)
    return output


def model_tool_definitions(specs: dict[str, ToolSpec]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for name in ordered_tool_names(specs):
        spec = specs.get(name)
        if spec is not None:
            output.append(spec.to_model_tool())
    return output


def register_runtime_tools(
    runtime: ToolRuntime,
    specs: dict[str, ToolSpec],
    tool_policy: dict[str, Any] | None = None,
    default_timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    default_max_retries: int = DEFAULT_MAX_RETRIES,
    reject_unknown_args: bool = False,
) -> None:
    tool_policy = tool_policy if isinstance(tool_policy, dict) else {}
    for name in ordered_tool_names(specs):
        spec = specs.get(name)
        if spec is None:
            continue

        policy_overrides = tool_policy.get(name, {})
        if not isinstance(policy_overrides, dict):
            policy_overrides = {}

        timeout_seconds = _float_value(
            policy_overrides.get("timeout_seconds"),
            spec.default_timeout_seconds if spec.default_timeout_seconds is not None else default_timeout_seconds,
        )
        max_retries = _int_value(
            policy_overrides.get("max_retries"),
            spec.default_max_retries if spec.default_max_retries is not None else default_max_retries,
        )
        required_api_key = policy_overrides.get("required_api_key", spec.required_api_key)
        side_effect_class = _as_text(policy_overrides.get("side_effect_class"), spec.side_effect_class).lower()
        if side_effect_class not in {"read_only", "mutating", "sensitive"}:
            side_effect_class = "read_only"

        merged_sandbox = merge_sandbox_policies(
            spec.sandbox_policy,
            _coerce_dict(policy_overrides.get("sandbox_policy")),
        )
        if "require_approval" in policy_overrides:
            merged_sandbox["require_approval"] = _as_bool(policy_overrides.get("require_approval"), False)
        if "dry_run" in policy_overrides:
            merged_sandbox["dry_run"] = _as_bool(policy_overrides.get("dry_run"), False)
        if "approval_timeout_seconds" in policy_overrides:
            merged_sandbox["approval_timeout_seconds"] = _float_value(
                policy_overrides.get("approval_timeout_seconds"),
                DEFAULT_TIMEOUT_SECONDS,
            )
        merged_sandbox["side_effect_class"] = side_effect_class
        merged_sandbox["tool_name"] = spec.name
        sandbox_policy = normalize_tool_sandbox_policy(merged_sandbox, default_tool_name=spec.name)
        dry_run_mock_result = copy.deepcopy(policy_overrides.get("dry_run_result", spec.dry_run_mock_result))

        runtime.register_tool(
            ToolDefinition(
                name=spec.name,
                function=spec.function,
                required_args=spec.required_args,
                optional_args=spec.optional_args,
                arg_coercers=spec.arg_coercers,
                reject_unknown_args=reject_unknown_args,
                required_api_key=required_api_key,
                timeout_seconds=timeout_seconds,
                max_retries=max_retries,
                side_effect_class=side_effect_class,
                sandbox_policy=sandbox_policy,
                mock_result=dry_run_mock_result,
                metadata={"description": spec.description},
            )
        )


def tool_catalog_entries(
    specs: dict[str, ToolSpec],
    *,
    custom_entries: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    custom_index = {}
    for entry in custom_entries if isinstance(custom_entries, list) else []:
        try:
            normalized = normalize_custom_tool_payload(_coerce_dict(entry))
        except ToolRegistryValidationError:
            continue
        custom_index[normalized["name"]] = normalized

    for name in ordered_tool_names(specs):
        spec = specs.get(name)
        if spec is None:
            continue
        model_tool = spec.to_model_tool()
        function_payload = _coerce_dict(model_tool.get("function"))
        custom_meta = custom_index.get(name, {})
        effective_sandbox = resolve_tool_sandbox_policy(
            spec.sandbox_policy,
            _coerce_dict(custom_meta.get("sandbox_policy")),
        )
        effective_sandbox["side_effect_class"] = _as_text(
            custom_meta.get("side_effect_class"),
            _as_text(effective_sandbox.get("side_effect_class"), spec.side_effect_class),
        ).lower()
        entries.append(
            {
                "name": spec.name,
                "description": spec.description,
                "source": "custom" if name in custom_index else "builtin",
                "enabled": True,
                "required_api_key": spec.required_api_key,
                "timeout_seconds": float(spec.default_timeout_seconds),
                "max_retries": int(spec.default_max_retries),
                "auth_requirements": _as_text(custom_meta.get("auth_requirements")),
                "side_effect_class": _as_text(effective_sandbox.get("side_effect_class"), "read_only"),
                "approval_required": _as_bool(
                    custom_meta.get("approval_required"),
                    bool(effective_sandbox.get("require_approval", False)),
                ),
                "dry_run": _as_bool(custom_meta.get("dry_run"), bool(effective_sandbox.get("dry_run", False))),
                "rate_limit_per_minute": max(0, _int_value(custom_meta.get("rate_limit_per_minute"), 0)),
                "handler_mode": _as_text(custom_meta.get("handler_mode"), "native"),
                "static_result": copy.deepcopy(custom_meta.get("static_result")),
                "dry_run_result": copy.deepcopy(custom_meta.get("dry_run_result", spec.dry_run_mock_result)),
                "sandbox_policy": effective_sandbox,
                "input_schema": _coerce_dict(function_payload.get("parameters")) or {
                    "type": "object",
                    "properties": {},
                },
            }
        )

    # Include disabled custom tools in catalog output.
    for name, entry in custom_index.items():
        if _as_bool(entry.get("enabled"), True):
            continue
        entries.append(
            {
                "name": name,
                "description": _as_text(entry.get("description")),
                "source": "custom",
                "enabled": False,
                "required_api_key": entry.get("required_api_key"),
                "timeout_seconds": _float_value(entry.get("timeout_seconds"), DEFAULT_TIMEOUT_SECONDS),
                "max_retries": max(0, _int_value(entry.get("max_retries"), DEFAULT_MAX_RETRIES)),
                "auth_requirements": _as_text(entry.get("auth_requirements")),
                "side_effect_class": _as_text(entry.get("side_effect_class"), "read_only"),
                "approval_required": _as_bool(entry.get("approval_required"), False),
                "dry_run": _as_bool(entry.get("dry_run"), False),
                "rate_limit_per_minute": max(0, _int_value(entry.get("rate_limit_per_minute"), 0)),
                "handler_mode": _as_text(entry.get("handler_mode"), "echo"),
                "static_result": copy.deepcopy(entry.get("static_result")),
                "dry_run_result": copy.deepcopy(entry.get("dry_run_result")),
                "sandbox_policy": normalize_tool_sandbox_policy(
                    _coerce_dict(entry.get("sandbox_policy")),
                    default_tool_name=name,
                ),
                "input_schema": _coerce_dict(entry.get("input_schema")) or {"type": "object", "properties": {}},
            }
        )
    return entries


class ToolRegistryStore:
    """File-backed custom tool registry for the agent playground."""

    def __init__(self, storage_path: str | Path = DEFAULT_TOOL_REGISTRY_PATH):
        self._storage_path = Path(storage_path)
        self._lock = threading.RLock()
        self._ensure_store()

    def _ensure_store(self) -> None:
        self._storage_path.parent.mkdir(parents=True, exist_ok=True)
        if self._storage_path.exists():
            return
        payload = {
            "schema": "ryo.tool_registry.v1",
            "updated_at": _now_iso(),
            "custom_tools": {},
        }
        self._storage_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def _load(self) -> dict[str, Any]:
        self._ensure_store()
        try:
            raw = json.loads(self._storage_path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            raw = {}
        payload = _coerce_dict(raw)
        custom_tools = payload.get("custom_tools")
        if not isinstance(custom_tools, dict):
            custom_tools = {}
        return {
            "schema": "ryo.tool_registry.v1",
            "updated_at": _as_text(payload.get("updated_at")),
            "custom_tools": custom_tools,
        }

    def _save(self, payload: dict[str, Any]) -> None:
        payload["updated_at"] = _now_iso()
        self._storage_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def list_custom_tools(self, *, include_disabled: bool = True) -> list[dict[str, Any]]:
        with self._lock:
            payload = self._load()
            output: list[dict[str, Any]] = []
            for item in payload.get("custom_tools", {}).values():
                if not isinstance(item, dict):
                    continue
                try:
                    normalized = normalize_custom_tool_payload(item)
                except ToolRegistryValidationError:
                    continue
                if not include_disabled and not _as_bool(normalized.get("enabled"), True):
                    continue
                output.append(normalized)
            output.sort(key=lambda item: _as_text(item.get("name")).lower())
            return output

    def get_custom_tool(self, tool_name: str) -> dict[str, Any] | None:
        clean_name = _custom_tool_name(tool_name)
        with self._lock:
            payload = self._load()
            item = _coerce_dict(payload.get("custom_tools", {}).get(clean_name))
            if not item:
                return None
            return normalize_custom_tool_payload(item)

    def upsert_custom_tool(
        self,
        payload: dict[str, Any] | None,
        *,
        actor_member_id: int,
    ) -> dict[str, Any]:
        normalized = normalize_custom_tool_payload(payload)
        normalized["updated_by_member_id"] = int(actor_member_id)
        with self._lock:
            data = self._load()
            custom_tools = data.get("custom_tools")
            if not isinstance(custom_tools, dict):
                custom_tools = {}
                data["custom_tools"] = custom_tools
            existing = _coerce_dict(custom_tools.get(normalized["name"]))
            if not existing:
                normalized["created_by_member_id"] = int(actor_member_id)
                normalized["created_at"] = _now_iso()
            else:
                normalized["created_by_member_id"] = _int_value(existing.get("created_by_member_id"), int(actor_member_id))
                normalized["created_at"] = _as_text(existing.get("created_at"), _now_iso())
            normalized["updated_at"] = _now_iso()
            custom_tools[normalized["name"]] = normalized
            data["custom_tools"] = custom_tools
            self._save(data)
        return normalize_custom_tool_payload(normalized)

    def remove_custom_tool(self, tool_name: str) -> bool:
        clean_name = _custom_tool_name(tool_name)
        with self._lock:
            data = self._load()
            custom_tools = data.get("custom_tools")
            if not isinstance(custom_tools, dict):
                return False
            if clean_name not in custom_tools:
                return False
            custom_tools.pop(clean_name, None)
            data["custom_tools"] = custom_tools
            self._save(data)
            return True

    def list_catalog(
        self,
        *,
        brave_search_fn: Callable[..., Any],
        chat_history_search_fn: Callable[..., Any],
        knowledge_search_fn: Callable[..., Any],
        skip_tools_fn: Callable[..., Any],
        knowledge_domains: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        custom_tools = self.list_custom_tools(include_disabled=True)
        specs = build_tool_specs(
            brave_search_fn=brave_search_fn,
            chat_history_search_fn=chat_history_search_fn,
            knowledge_search_fn=knowledge_search_fn,
            skip_tools_fn=skip_tools_fn,
            knowledge_domains=knowledge_domains,
            custom_tool_entries=custom_tools,
        )
        return tool_catalog_entries(specs, custom_entries=custom_tools)
