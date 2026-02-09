from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from hypermindlabs.agent_definitions import AgentDefinitionStore
from hypermindlabs.run_events import event_schema
from hypermindlabs.run_mode_handlers import run_modes_manifest
from hypermindlabs.runtime_settings import DEFAULT_RUNTIME_SETTINGS
from hypermindlabs.approval_manager import ApprovalManager
from hypermindlabs.tool_registry import ToolRegistryStore
from hypermindlabs.tool_sandbox import ToolSandboxPolicyStore
from hypermindlabs.tool_test_harness import ToolTestHarnessStore
from hypermindlabs.utils import ConfigManager


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")



def _noop_tool(**_: Any) -> dict[str, Any]:
    return {"status": "noop"}



def _tool_capability() -> dict[str, Any]:
    registry = ToolRegistryStore()
    items = registry.list_catalog(
        brave_search_fn=_noop_tool,
        chat_history_search_fn=_noop_tool,
        knowledge_search_fn=_noop_tool,
        skip_tools_fn=_noop_tool,
        known_users_list_fn=_noop_tool,
        message_known_user_fn=_noop_tool,
        process_workspace_upsert_fn=_noop_tool,
        process_workspace_list_fn=_noop_tool,
        process_workspace_step_update_fn=_noop_tool,
        outbox_list_fn=_noop_tool,
        knowledge_domains=[],
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
                    "source": {"type": "string"},
                    "enabled": {"type": "boolean"},
                    "required_api_key": {"type": ["string", "null"]},
                    "timeout_seconds": {"type": "number"},
                    "max_retries": {"type": "integer"},
                    "auth_requirements": {"type": "string"},
                    "side_effect_class": {"type": "string"},
                    "approval_required": {"type": "boolean"},
                    "dry_run": {"type": "boolean"},
                    "rate_limit_per_minute": {"type": "integer"},
                    "handler_mode": {"type": "string"},
                    "static_result": {"type": ["object", "array", "string", "number", "boolean", "null"]},
                    "dry_run_result": {"type": ["object", "array", "string", "number", "boolean", "null"]},
                    "sandbox_policy": {"type": "object"},
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
    store = AgentDefinitionStore()
    policy_dir = Path(__file__).resolve().parent.parent / "policies" / "agent"
    policy_entries: list[dict[str, Any]] = []
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
            policy_entries.append(
                {
                    "name": policy_name,
                    "policy_file": str(policy_file.name),
                    "allowed_models": allowed_models,
                    "source": "policy",
                }
            )
    saved_definitions = store.list_definitions(owner_member_id=None)
    definition_entries = [
        {
            "definition_id": item.get("definition_id"),
            "name": item.get("name"),
            "description": item.get("description"),
            "tags": item.get("tags"),
            "active_version": item.get("active_version"),
            "version_count": item.get("version_count"),
            "source": "definition",
        }
        for item in saved_definitions
    ]

    return {
        "id": "agents.list",
        "title": "Agent Definitions",
        "permissions": {"read": ["user", "admin", "owner"], "write": ["admin", "owner"]},
        "schema": {
            "type": "object",
            "properties": {
                "definition_id": {"type": "string"},
                "name": {"type": "string"},
                "description": {"type": "string"},
                "tags": {"type": "array", "items": {"type": "string"}},
                "active_version": {"type": "integer"},
                "version_count": {"type": "integer"},
                "policy_file": {"type": "string"},
                "allowed_models": {"type": "array", "items": {"type": "string"}},
                "source": {"type": "string"},
            },
        },
        "items": definition_entries + policy_entries,
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


def _tool_sandbox_capability() -> dict[str, Any]:
    store = ToolSandboxPolicyStore()
    return {
        "id": "tools.sandbox",
        "title": "Tool Sandbox Policies",
        "permissions": {"read": ["user", "admin", "owner"], "write": ["admin", "owner"]},
        "schema": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["tool_name", "sandbox_policy"],
                "properties": {
                    "tool_name": {"type": "string"},
                    "source": {"type": "string"},
                    "side_effect_class": {"type": "string"},
                    "approval_required": {"type": "boolean"},
                    "dry_run": {"type": "boolean"},
                    "sandbox_policy": {"type": "object"},
                },
            },
        },
        "items": store.list_policies(),
    }


def _tool_approvals_capability() -> dict[str, Any]:
    queue = ApprovalManager()
    return {
        "id": "tools.approvals",
        "title": "Tool Approval Queue",
        "permissions": {"read": ["user", "admin", "owner"], "write": ["user", "admin", "owner"]},
        "schema": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["request_id", "run_id", "tool_name", "status"],
                "properties": {
                    "request_id": {"type": "string"},
                    "run_id": {"type": "string"},
                    "tool_name": {"type": "string"},
                    "status": {"type": "string"},
                    "reason": {"type": "string"},
                    "requested_at": {"type": "string"},
                    "expires_at": {"type": "string"},
                    "decided_at": {"type": ["string", "null"]},
                },
            },
        },
        "items": queue.list_requests(limit=50),
    }


def _tool_harness_capability() -> dict[str, Any]:
    harness = ToolTestHarnessStore()
    return {
        "id": "tools.harness",
        "title": "Tool Test Harness",
        "permissions": {"read": ["user", "admin", "owner"], "write": ["admin", "owner"]},
        "schema": {
            "type": "object",
            "required": ["tool_name", "fixture_name", "input_args"],
            "properties": {
                "case_id": {"type": "string"},
                "tool_name": {"type": "string"},
                "fixture_name": {"type": "string"},
                "description": {"type": "string"},
                "execution_mode": {"type": "string", "enum": ["real", "mock"]},
                "enabled": {"type": "boolean"},
                "tags": {"type": "array", "items": {"type": "string"}},
                "input_args": {"type": "object"},
                "contract_snapshot": {"type": "object"},
                "golden_output": {"type": ["object", "array", "string", "number", "boolean", "null"]},
            },
        },
        "items": harness.list_cases(),
    }



def build_capability_manifest(member_roles: list[str] | None = None) -> dict[str, Any]:
    roles = [str(role).strip() for role in (member_roles or []) if str(role).strip()]
    config = ConfigManager()

    capabilities = [
        _run_lifecycle_capability(),
        _models_capability(config),
        _tool_capability(),
        _tool_sandbox_capability(),
        _tool_approvals_capability(),
        _tool_harness_capability(),
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
