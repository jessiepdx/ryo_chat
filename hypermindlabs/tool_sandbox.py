from __future__ import annotations

import copy
import json
from pathlib import Path, PurePosixPath
import re
import threading
from datetime import datetime, timedelta, timezone
from typing import Any
from urllib.parse import urlparse


DEFAULT_TOOL_SANDBOX_POLICY_PATH = Path(__file__).resolve().parent.parent / "db" / "tool_sandbox_policies.json"
_TOOL_NAME_PATTERN = re.compile(r"^[A-Za-z][A-Za-z0-9_]{2,64}$")
_SIDE_EFFECT_CLASSES = {"read_only", "mutating", "sensitive"}
_FILESYSTEM_MODES = {"none", "read_only", "read_write"}
_URLISH_KEY_HINTS = {"url", "uri", "endpoint", "host", "base_url"}
_PATH_KEY_HINTS = {"path", "file", "filepath", "filename", "directory", "dir"}
_WRITE_INTENT_HINTS = {
    "write",
    "append",
    "delete",
    "save",
    "create",
    "truncate",
    "remove",
    "update",
    "output",
}


class ToolSandboxPolicyValidationError(ValueError):
    """Raised when sandbox policy payload is invalid."""


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _as_text(value: Any, fallback: str = "") -> str:
    text = str(value if value is not None else "").strip()
    return text if text else fallback


def _as_bool(value: Any, fallback: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        cleaned = value.strip().lower()
        if cleaned in {"1", "true", "yes", "on"}:
            return True
        if cleaned in {"0", "false", "no", "off"}:
            return False
    return fallback


def _as_float(value: Any, fallback: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(fallback)


def _as_int(value: Any, fallback: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(fallback)


def _coerce_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return copy.deepcopy(value)
    return {}


def _coerce_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return copy.deepcopy(value)
    return []


def _unique_str_list(values: list[Any], *, normalize_lower: bool = False) -> list[str]:
    output: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = _as_text(value)
        if not text:
            continue
        normalized = text.lower() if normalize_lower else text
        if normalized in seen:
            continue
        seen.add(normalized)
        output.append(normalized if normalize_lower else text)
    return output


def default_sandbox_policy() -> dict[str, Any]:
    return {
        "enabled": True,
        "side_effect_class": "read_only",
        "require_approval": False,
        "dry_run": False,
        "approval_timeout_seconds": 45.0,
        "execution_timeout_ceiling": 30.0,
        "max_memory_mb": 512,
        "network": {
            "enabled": True,
            "allowlist_domains": [],
        },
        "filesystem": {
            "mode": "none",
            "allowed_paths": [],
        },
    }


def merge_sandbox_policies(base_policy: dict[str, Any] | None, override_policy: dict[str, Any] | None) -> dict[str, Any]:
    base = _coerce_dict(base_policy)
    override = _coerce_dict(override_policy)
    merged = copy.deepcopy(base)

    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = merge_sandbox_policies(_coerce_dict(merged.get(key)), value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def normalize_tool_sandbox_policy(
    payload: dict[str, Any] | None,
    *,
    default_tool_name: str | None = None,
) -> dict[str, Any]:
    source = _coerce_dict(payload)
    nested_policy = _coerce_dict(source.get("sandbox_policy"))
    flattened_source = merge_sandbox_policies(source, nested_policy)
    if "approval_required" in source and "require_approval" not in flattened_source:
        flattened_source["require_approval"] = source.get("approval_required")
    if "dry_run" in source and "dry_run" not in flattened_source:
        flattened_source["dry_run"] = source.get("dry_run")
    defaults = default_sandbox_policy()
    merged = merge_sandbox_policies(defaults, flattened_source)

    tool_name = _as_text(flattened_source.get("tool_name"), _as_text(default_tool_name))
    if tool_name:
        if not _TOOL_NAME_PATTERN.fullmatch(tool_name):
            raise ToolSandboxPolicyValidationError("tool_name must match ^[A-Za-z][A-Za-z0-9_]{2,64}$.")

    side_effect_class = _as_text(merged.get("side_effect_class"), "read_only").lower()
    if side_effect_class not in _SIDE_EFFECT_CLASSES:
        side_effect_class = "read_only"

    network = merge_sandbox_policies(defaults.get("network"), merged.get("network"))
    filesystem = merge_sandbox_policies(defaults.get("filesystem"), merged.get("filesystem"))

    filesystem_mode = _as_text(filesystem.get("mode"), "none").lower()
    if filesystem_mode not in _FILESYSTEM_MODES:
        filesystem_mode = "none"

    allowlist_domains = _unique_str_list(_coerce_list(network.get("allowlist_domains")), normalize_lower=True)
    allowed_paths = _unique_str_list(_coerce_list(filesystem.get("allowed_paths")), normalize_lower=False)
    require_approval_default = side_effect_class in {"mutating", "sensitive"}
    explicit_require = (
        flattened_source.get("require_approval")
        if "require_approval" in flattened_source
        else None
    )
    if explicit_require is None and "approval_required" in source:
        explicit_require = source.get("approval_required")

    normalized = {
        "tool_name": tool_name or None,
        "enabled": _as_bool(merged.get("enabled"), True),
        "side_effect_class": side_effect_class,
        "require_approval": _as_bool(explicit_require, require_approval_default),
        "dry_run": _as_bool(merged.get("dry_run"), False),
        "approval_timeout_seconds": max(1.0, _as_float(merged.get("approval_timeout_seconds"), 45.0)),
        "execution_timeout_ceiling": max(0.1, _as_float(merged.get("execution_timeout_ceiling"), 30.0)),
        "max_memory_mb": max(16, _as_int(merged.get("max_memory_mb"), 512)),
        "network": {
            "enabled": _as_bool(network.get("enabled"), True),
            "allowlist_domains": allowlist_domains,
        },
        "filesystem": {
            "mode": filesystem_mode,
            "allowed_paths": allowed_paths,
        },
    }
    return normalized


def resolve_tool_sandbox_policy(*policy_parts: Any) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for part in policy_parts:
        if isinstance(part, dict):
            merged = merge_sandbox_policies(merged, part)
    return normalize_tool_sandbox_policy(merged)


def _domain_allowed(hostname: str, allowlist_domains: list[str]) -> bool:
    host = _as_text(hostname).lower()
    if not host:
        return False
    for allowed in allowlist_domains:
        candidate = _as_text(allowed).lower()
        if not candidate:
            continue
        if host == candidate or host.endswith(f".{candidate}"):
            return True
    return False


def _extract_hosts(value: Any) -> set[str]:
    hosts: set[str] = set()
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return hosts
        if "://" in text:
            parsed = urlparse(text)
            if parsed.hostname:
                hosts.add(parsed.hostname.lower())
        elif re.fullmatch(r"[A-Za-z0-9.-]+\.[A-Za-z]{2,}", text):
            hosts.add(text.lower())
        return hosts
    if isinstance(value, dict):
        for key, nested in value.items():
            key_name = _as_text(key).lower()
            if any(hint in key_name for hint in _URLISH_KEY_HINTS):
                hosts.update(_extract_hosts(nested))
                continue
            hosts.update(_extract_hosts(nested))
        return hosts
    if isinstance(value, list):
        for item in value:
            hosts.update(_extract_hosts(item))
    return hosts


def _extract_paths(value: Any) -> set[str]:
    paths: set[str] = set()
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return paths
        looks_like_path = text.startswith(("/", "./", "../", "~")) or "/" in text or "\\" in text
        if looks_like_path:
            paths.add(text)
        return paths
    if isinstance(value, dict):
        for key, nested in value.items():
            key_name = _as_text(key).lower()
            if any(hint in key_name for hint in _PATH_KEY_HINTS):
                paths.update(_extract_paths(nested))
            elif isinstance(nested, (dict, list)):
                paths.update(_extract_paths(nested))
        return paths
    if isinstance(value, list):
        for item in value:
            paths.update(_extract_paths(item))
    return paths


def _path_is_allowed(path_value: str, allowed_roots: list[str]) -> bool:
    if not allowed_roots:
        return True
    candidate = str(PurePosixPath(path_value.replace("\\", "/")))
    for allowed_root in allowed_roots:
        root = str(PurePosixPath(str(allowed_root).replace("\\", "/")))
        if candidate == root or candidate.startswith(f"{root}/"):
            return True
    return False


def _has_write_intent(args: dict[str, Any]) -> bool:
    for key, value in args.items():
        key_name = _as_text(key).lower()
        if any(hint in key_name for hint in _WRITE_INTENT_HINTS):
            if isinstance(value, bool):
                if value:
                    return True
            elif value is not None:
                text = _as_text(value).lower()
                if text in {"w", "wb", "append", "true", "1", "yes", "on"}:
                    return True
                if text:
                    return True
        if key_name == "mode":
            mode = _as_text(value).lower()
            if mode in {"w", "wb", "a", "ab", "append", "write"}:
                return True
        if isinstance(value, dict) and _has_write_intent(_coerce_dict(value)):
            return True
    return False


class ToolSandboxEnforcer:
    """Evaluates tool calls against policy-defined sandbox controls."""

    def __init__(self, default_policy: dict[str, Any] | None = None):
        self._default_policy = normalize_tool_sandbox_policy(default_policy or {})

    def evaluate(
        self,
        *,
        tool_name: str,
        prepared_args: dict[str, Any] | None,
        tool_policy: dict[str, Any] | None = None,
        tool_metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        metadata = _coerce_dict(tool_metadata)
        metadata_policy = _coerce_dict(metadata.get("sandbox_policy"))
        if "side_effect_class" in metadata and "side_effect_class" not in metadata_policy:
            metadata_policy["side_effect_class"] = metadata.get("side_effect_class")

        effective_policy = resolve_tool_sandbox_policy(
            self._default_policy,
            metadata_policy,
            tool_policy,
        )
        args = _coerce_dict(prepared_args)

        if not _as_bool(effective_policy.get("enabled"), True):
            return {
                "tool_name": tool_name,
                "status": "allow",
                "code": "sandbox_disabled",
                "message": "Sandbox policy disabled for this tool.",
                "effective_policy": effective_policy,
                "details": {},
            }

        network_policy = _coerce_dict(effective_policy.get("network"))
        filesystem_policy = _coerce_dict(effective_policy.get("filesystem"))
        side_effect_class = _as_text(effective_policy.get("side_effect_class"), "read_only")

        observed_hosts = sorted(_extract_hosts(args))
        network_enabled = _as_bool(network_policy.get("enabled"), True)
        network_allowlist = _unique_str_list(_coerce_list(network_policy.get("allowlist_domains")), normalize_lower=True)

        if not network_enabled and observed_hosts:
            return {
                "tool_name": tool_name,
                "status": "deny",
                "code": "sandbox_block_network",
                "message": "Network access is disabled by sandbox policy.",
                "effective_policy": effective_policy,
                "details": {"hosts": observed_hosts},
            }

        disallowed_hosts = [
            host for host in observed_hosts if network_allowlist and not _domain_allowed(host, network_allowlist)
        ]
        if disallowed_hosts:
            return {
                "tool_name": tool_name,
                "status": "deny",
                "code": "sandbox_block_network_allowlist",
                "message": "Network host is outside sandbox allowlist.",
                "effective_policy": effective_policy,
                "details": {"hosts": disallowed_hosts, "allowlist": network_allowlist},
            }

        observed_paths = sorted(_extract_paths(args))
        filesystem_mode = _as_text(filesystem_policy.get("mode"), "none")
        allowed_paths = _unique_str_list(_coerce_list(filesystem_policy.get("allowed_paths")))

        if filesystem_mode == "none" and observed_paths:
            return {
                "tool_name": tool_name,
                "status": "deny",
                "code": "sandbox_block_filesystem",
                "message": "Filesystem access is disabled by sandbox policy.",
                "effective_policy": effective_policy,
                "details": {"paths": observed_paths},
            }

        out_of_bounds_paths = [
            path_value for path_value in observed_paths if not _path_is_allowed(path_value, allowed_paths)
        ]
        if out_of_bounds_paths:
            return {
                "tool_name": tool_name,
                "status": "deny",
                "code": "sandbox_block_filesystem_path",
                "message": "Filesystem path is outside sandbox allowlist.",
                "effective_policy": effective_policy,
                "details": {"paths": out_of_bounds_paths, "allowed_paths": allowed_paths},
            }

        if filesystem_mode == "read_only" and _has_write_intent(args):
            return {
                "tool_name": tool_name,
                "status": "deny",
                "code": "sandbox_block_filesystem_write",
                "message": "Filesystem write intent blocked by read-only sandbox policy.",
                "effective_policy": effective_policy,
                "details": {},
            }

        return {
            "tool_name": tool_name,
            "status": "allow",
            "code": "sandbox_allowed",
            "message": "Sandbox policy allows this tool invocation.",
            "effective_policy": effective_policy,
            "details": {
                "side_effect_class": side_effect_class,
                "hosts": observed_hosts,
                "paths": observed_paths,
            },
        }


class ToolSandboxPolicyStore:
    """File-backed per-tool sandbox policy store."""

    def __init__(self, storage_path: str | Path = DEFAULT_TOOL_SANDBOX_POLICY_PATH):
        self._storage_path = Path(storage_path)
        self._lock = threading.RLock()
        self._ensure_store()

    def _ensure_store(self) -> None:
        self._storage_path.parent.mkdir(parents=True, exist_ok=True)
        if self._storage_path.exists():
            return
        payload = {
            "schema": "ryo.tool_sandbox_policy.v1",
            "updated_at": _now_iso(),
            "policies": {},
        }
        self._storage_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def _load(self) -> dict[str, Any]:
        self._ensure_store()
        try:
            raw = json.loads(self._storage_path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            raw = {}
        payload = _coerce_dict(raw)
        policies = payload.get("policies")
        if not isinstance(policies, dict):
            policies = {}
        return {
            "schema": "ryo.tool_sandbox_policy.v1",
            "updated_at": _as_text(payload.get("updated_at")),
            "policies": policies,
        }

    def _save(self, payload: dict[str, Any]) -> None:
        payload["updated_at"] = _now_iso()
        self._storage_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def list_policies(self) -> list[dict[str, Any]]:
        with self._lock:
            data = self._load()
            output: list[dict[str, Any]] = []
            for tool_name, raw_policy in data.get("policies", {}).items():
                try:
                    normalized = normalize_tool_sandbox_policy(raw_policy, default_tool_name=tool_name)
                except ToolSandboxPolicyValidationError:
                    continue
                output.append(normalized)
            output.sort(key=lambda item: _as_text(item.get("tool_name")).lower())
            return output

    def policy_map(self) -> dict[str, dict[str, Any]]:
        output: dict[str, dict[str, Any]] = {}
        for policy in self.list_policies():
            tool_name = _as_text(policy.get("tool_name"))
            if tool_name:
                output[tool_name] = policy
        return output

    def get_policy(self, tool_name: str) -> dict[str, Any] | None:
        clean_name = _as_text(tool_name)
        if not clean_name:
            return None
        with self._lock:
            data = self._load()
            raw_policy = _coerce_dict(data.get("policies", {}).get(clean_name))
            if not raw_policy:
                return None
            return normalize_tool_sandbox_policy(raw_policy, default_tool_name=clean_name)

    def upsert_policy(
        self,
        payload: dict[str, Any] | None,
        *,
        actor_member_id: int,
    ) -> dict[str, Any]:
        normalized = normalize_tool_sandbox_policy(payload)
        tool_name = _as_text(normalized.get("tool_name"))
        if not tool_name:
            raise ToolSandboxPolicyValidationError("tool_name is required.")

        with self._lock:
            data = self._load()
            policies = data.get("policies")
            if not isinstance(policies, dict):
                policies = {}
            existing = _coerce_dict(policies.get(tool_name))
            if existing:
                normalized["created_at"] = _as_text(existing.get("created_at"), _now_iso())
                normalized["created_by_member_id"] = _as_int(existing.get("created_by_member_id"), int(actor_member_id))
            else:
                normalized["created_at"] = _now_iso()
                normalized["created_by_member_id"] = int(actor_member_id)
            normalized["updated_at"] = _now_iso()
            normalized["updated_by_member_id"] = int(actor_member_id)
            policies[tool_name] = normalized
            data["policies"] = policies
            self._save(data)
        return normalize_tool_sandbox_policy(normalized, default_tool_name=tool_name)

    def remove_policy(self, tool_name: str) -> bool:
        clean_name = _as_text(tool_name)
        if not clean_name:
            raise ToolSandboxPolicyValidationError("tool_name is required.")
        with self._lock:
            data = self._load()
            policies = data.get("policies")
            if not isinstance(policies, dict):
                return False
            if clean_name not in policies:
                return False
            policies.pop(clean_name, None)
            data["policies"] = policies
            self._save(data)
            return True


def approval_expiry_from_timeout(timeout_seconds: float | int) -> str:
    seconds = max(1.0, float(timeout_seconds))
    return (datetime.now(timezone.utc) + timedelta(seconds=seconds)).isoformat(timespec="seconds")
