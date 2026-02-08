from __future__ import annotations

from typing import Any


RUN_MODE_SCHEMAS: dict[str, dict[str, Any]] = {
    "chat": {
        "id": "chat",
        "label": "Chat",
        "description": "Single interactive agent run.",
        "request_schema": {
            "type": "object",
            "required": ["message"],
            "properties": {
                "message": {"type": "string", "minLength": 1},
                "options": {"type": "object"},
            },
        },
    },
    "workflow": {
        "id": "workflow",
        "label": "Workflow",
        "description": "Execute a multi-step sequence with explicit stage events.",
        "request_schema": {
            "type": "object",
            "required": ["message"],
            "properties": {
                "message": {"type": "string", "minLength": 1},
                "workflow_steps": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "string"},
                            "prompt": {"type": "string"},
                        },
                    },
                },
                "options": {"type": "object"},
            },
        },
    },
    "batch": {
        "id": "batch",
        "label": "Batch",
        "description": "Run multiple inputs through the same agent configuration.",
        "request_schema": {
            "type": "object",
            "required": ["batch_inputs"],
            "properties": {
                "batch_inputs": {
                    "type": "array",
                    "minItems": 1,
                    "items": {"type": "string", "minLength": 1},
                },
                "options": {"type": "object"},
            },
        },
    },
    "compare": {
        "id": "compare",
        "label": "Compare",
        "description": "Run a prompt across multiple model selections side-by-side.",
        "request_schema": {
            "type": "object",
            "required": ["message", "compare_models"],
            "properties": {
                "message": {"type": "string", "minLength": 1},
                "compare_models": {
                    "type": "array",
                    "minItems": 2,
                    "items": {"type": "string", "minLength": 1},
                },
                "options": {"type": "object"},
            },
        },
    },
    "replay": {
        "id": "replay",
        "label": "Replay",
        "description": "Replay an existing run from start or from a selected step.",
        "request_schema": {
            "type": "object",
            "required": ["source_run_id"],
            "properties": {
                "source_run_id": {"type": "string", "minLength": 1},
                "replay_from_seq": {"type": "integer", "minimum": 1},
                "state_overrides": {"type": "object"},
            },
        },
    },
}



def normalize_run_mode(value: Any) -> str:
    mode = str(value or "chat").strip().lower()
    if mode in RUN_MODE_SCHEMAS:
        return mode
    return "chat"



def run_modes_manifest() -> list[dict[str, Any]]:
    return [RUN_MODE_SCHEMAS[key] for key in ("chat", "workflow", "batch", "compare", "replay")]
