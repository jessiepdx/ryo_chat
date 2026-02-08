from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from hypermindlabs.run_events import event_schema
from hypermindlabs.run_mode_handlers import run_modes_manifest
from hypermindlabs.runtime_settings import DEFAULT_RUNTIME_SETTINGS
from hypermindlabs.tool_registry import build_tool_specs
from hypermindlabs.utils import ConfigManager


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")



def _noop_tool(**_: Any) -> dict[str, Any]:
    return {"status": "noop"}



def _tool_capability() -> dict[str, Any]:
    specs = build_tool_specs(
        brave_search_fn=_noop_tool,
        chat_history_search_fn=_noop_tool,
        knowledge_search_fn=_noop_tool,
        skip_tools_fn=_noop_tool,
        knowledge_domains=[],
    )
    items: list[dict[str, Any]] = []
    for spec in specs.values():
        model_tool = spec.to_model_tool()
        function = model_tool.get("function", {})
        items.append(
            {
                "name": spec.name,
                "description": spec.description,
                "required_api_key": spec.required_api_key,
                "timeout_seconds": float(spec.default_timeout_seconds),
                "max_retries": int(spec.default_max_retries),
                "input_schema": function.get("parameters", {"type": "object", "properties": {}}),
            }
        )

    return {
        "id": "tools.list",
        "title": "Tool Registry",
        "permissions": {"read": ["user", "admin", "owner"], "write": ["admin", "owner"]},
        "schema": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["name", "description", "input_schema"],
                "properties": {
                    "name": {"type": "string"},
                    "description": {"type": "string"},
                    "required_api_key": {"type": ["string", "null"]},
                    "timeout_seconds": {"type": "number"},
                    "max_retries": {"type": "integer"},
                    "input_schema": {"type": "object"},
                },
            },
        },
        "items": items,
    }



def _models_capability(config: ConfigManager) -> dict[str, Any]:
    inference = config.inference if isinstance(config.inference, dict) else {}
    models: dict[str, str] = {}
    for capability in ("chat", "tool", "generate", "embedding", "multimodal"):
        section = inference.get(capability)
        if isinstance(section, dict):
            model_name = str(section.get("model") or "").strip()
            if model_name:
                models[capability] = model_name

    settings = DEFAULT_RUNTIME_SETTINGS.get("inference", {})
    defaults = {
        "temperature": 0.2,
        "top_p": 0.9,
        "seed": 42,
        "context_window": int(settings.get("model_context_window", 4096)),
    }

    return {
        "id": "models.list",
        "title": "Model Operations",
        "permissions": {"read": ["user", "admin", "owner"], "write": ["admin", "owner"]},
        "schema": {
            "type": "object",
            "properties": {
                "model_requested": {"type": "string"},
                "temperature": {"type": "number", "minimum": 0, "maximum": 2, "default": defaults["temperature"]},
                "top_p": {"type": "number", "minimum": 0, "maximum": 1, "default": defaults["top_p"]},
                "seed": {"type": "integer", "default": defaults["seed"]},
                "ollama_host": {"type": "string", "default": config.runtimeValue("inference.default_ollama_host", "http://127.0.0.1:11434")},
            },
        },
        "items": {
            "configured_models": models,
            "defaults": defaults,
        },
    }



def _agents_capability() -> dict[str, Any]:
    policy_dir = Path(__file__).resolve().parent.parent / "policies" / "agent"
    agent_entries: list[dict[str, Any]] = []
    if policy_dir.exists():
        for policy_file in sorted(policy_dir.glob("*_policy.json")):
            policy_name = policy_file.stem.replace("_policy", "")
            allowed_models: list[str] = []
            try:
                payload = json.loads(policy_file.read_text(encoding="utf-8"))
                maybe_models = payload.get("allowed_models")
                if isinstance(maybe_models, list):
                    allowed_models = [str(item).strip() for item in maybe_models if str(item).strip()]
            except Exception:  # noqa: BLE001
                allowed_models = []
            agent_entries.append(
                {
                    "name": policy_name,
                    "policy_file": str(policy_file.name),
                    "allowed_models": allowed_models,
                }
            )

    return {
        "id": "agents.list",
        "title": "Agent Definitions",
        "permissions": {"read": ["user", "admin", "owner"], "write": ["admin", "owner"]},
        "schema": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "policy_file": {"type": "string"},
                "allowed_models": {"type": "array", "items": {"type": "string"}},
            },
        },
        "items": agent_entries,
    }



def _memory_capability() -> dict[str, Any]:
    items = [
        {
            "id": "memory.short_term",
            "label": "Short Term Context",
            "strategies": ["none", "trim_last_n", "summary_compress"],
        },
        {
            "id": "memory.long_term",
            "label": "Long Term Memory",
            "strategies": ["episodic", "semantic", "procedural"],
        },
    ]
    return {
        "id": "memory.strategies",
        "title": "Memory Strategies",
        "permissions": {"read": ["user", "admin", "owner"], "write": ["admin", "owner"]},
        "schema": {
            "type": "object",
            "properties": {
                "short_term_strategy": {"type": "string"},
                "long_term_strategy": {"type": "string"},
                "token_budget": {"type": "integer", "minimum": 128, "default": 2048},
            },
        },
        "items": items,
    }



def _evals_capability() -> dict[str, Any]:
    items = [
        {"id": "eval.exact_match", "label": "Exact Match", "input_schema": {"type": "object"}},
        {"id": "eval.fuzzy_match", "label": "Fuzzy Match", "input_schema": {"type": "object"}},
        {"id": "eval.latency_budget", "label": "Latency Budget", "input_schema": {"type": "object"}},
    ]
    return {
        "id": "evals.list",
        "title": "Evaluators",
        "permissions": {"read": ["user", "admin", "owner"], "write": ["admin", "owner"]},
        "schema": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["id", "label"],
                "properties": {
                    "id": {"type": "string"},
                    "label": {"type": "string"},
                    "input_schema": {"type": "object"},
                },
            },
        },
        "items": items,
    }



def _artifacts_capability() -> dict[str, Any]:
    return {
        "id": "artifacts.types",
        "title": "Artifact Renderers",
        "permissions": {"read": ["user", "admin", "owner"], "write": ["admin", "owner"]},
        "schema": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["id", "label"],
                "properties": {
                    "id": {"type": "string"},
                    "label": {"type": "string"},
                },
            },
        },
        "items": [
            {"id": "markdown", "label": "Markdown"},
            {"id": "json", "label": "JSON"},
            {"id": "table", "label": "Table"},
            {"id": "diff", "label": "Diff"},
            {"id": "text", "label": "Text"},
        ],
    }



def _run_lifecycle_capability() -> dict[str, Any]:
    return {
        "id": "runs.lifecycle",
        "title": "Run Lifecycle",
        "permissions": {"read": ["user", "admin", "owner"], "write": ["user", "admin", "owner"]},
        "schema": {
            "type": "object",
            "properties": {
                "mode": {"type": "string", "enum": [item["id"] for item in run_modes_manifest()]},
                "request": {"type": "object"},
            },
        },
        "items": {
            "event_schema": event_schema(),
            "run_modes": run_modes_manifest(),
        },
    }



def build_capability_manifest(member_roles: list[str] | None = None) -> dict[str, Any]:
    roles = [str(role).strip() for role in (member_roles or []) if str(role).strip()]
    config = ConfigManager()

    capabilities = [
        _run_lifecycle_capability(),
        _models_capability(config),
        _tool_capability(),
        _agents_capability(),
        _memory_capability(),
        _evals_capability(),
        _artifacts_capability(),
    ]

    for capability in capabilities:
        read_roles = capability.get("permissions", {}).get("read", [])
        capability["allowed"] = True if not read_roles else any(role in read_roles for role in roles) or (not roles)

    return {
        "manifest_version": "1.0.0",
        "generated_at": _now_iso(),
        "roles": roles,
        "capabilities": capabilities,
    }



def find_capability(manifest: dict[str, Any], capability_id: str) -> dict[str, Any] | None:
    capabilities = manifest.get("capabilities") if isinstance(manifest, dict) else None
    if not isinstance(capabilities, list):
        return None
    for capability in capabilities:
        if isinstance(capability, dict) and capability.get("id") == capability_id:
            return capability
    return None
