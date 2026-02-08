##########################################################################
#                                                                        #
#  This file (policy_manager.py) manages agent policy validation,        #
#  prompt validation, and safe policy updates.                           #
#                                                                        #
##########################################################################

from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from hypermindlabs.runtime_settings import DEFAULT_RUNTIME_SETTINGS

try:
    from ollama import Client
except Exception:  # noqa: BLE001
    Client = None


DEFAULT_OLLAMA_HOST = str(
    DEFAULT_RUNTIME_SETTINGS.get("inference", {}).get("default_ollama_host", "http://127.0.0.1:11434")
)
FALLBACK_SYSTEM_PROMPT = "You are a helpful AI assistant."
INFERENCE_HOST_ORDER = ("tool", "chat", "generate", "embedding", "multimodal")


@dataclass
class PolicyValidationReport:
    policy_name: str
    policy_path: str
    prompt_path: str
    endpoint_host: str
    available_models: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    normalized_policy: dict[str, Any] | None = None

    @property
    def is_valid(self) -> bool:
        return len(self.errors) == 0


@dataclass
class PolicySaveResult:
    saved: bool
    backup_path: str | None
    rollback_performed: bool
    report: PolicyValidationReport


class PolicyValidationError(Exception):
    """Raised when policy validation fails in strict mode."""


class PolicyManager:
    """Loads, validates, and updates policy and system prompt artifacts."""

    _policy_to_inference_key = {
        "tool_calling": "tool",
        "message_analysis": "tool",
        "chat_conversation": "chat",
        "dev_test": "chat",
    }

    def __init__(
        self,
        policies_dir: str | Path = "policies/agent",
        inference_config: dict | None = None,
        endpoint_override: str | None = None,
        default_host: str = DEFAULT_OLLAMA_HOST,
    ):
        self._policies_dir = Path(policies_dir)
        self._system_prompt_dir = self._policies_dir / "system_prompt"
        self._inference_config = inference_config if isinstance(inference_config, dict) else {}
        self._endpoint_override = endpoint_override
        self._default_host = self._normalize_host(default_host) or DEFAULT_OLLAMA_HOST

    @staticmethod
    def _normalize_host(host: str | None) -> str | None:
        if host is None:
            return None
        return host.strip().rstrip("/")

    @staticmethod
    def _is_valid_host(host: str | None) -> bool:
        if not host:
            return False
        parsed = urlparse(host)
        return parsed.scheme in {"http", "https"} and bool(parsed.netloc)

    def _policy_path(self, policy_name: str) -> Path:
        return self._policies_dir / f"{policy_name}_policy.json"

    def _prompt_path(self, policy_name: str) -> Path:
        return self._system_prompt_dir / f"{policy_name}_sp.txt"

    def list_policy_names(self) -> list[str]:
        if not self._policies_dir.exists():
            return []

        names: list[str] = []
        for path in sorted(self._policies_dir.glob("*_policy.json")):
            stem = path.stem
            if stem.endswith("_policy"):
                names.append(stem.removesuffix("_policy"))
        return names

    def resolve_host(self, policy_name: str | None = None) -> str:
        override = self._normalize_host(self._endpoint_override)
        if self._is_valid_host(override):
            return override

        if policy_name:
            hint_key = self._policy_to_inference_key.get(policy_name)
            if hint_key:
                hinted = self._extract_inference_host(hint_key)
                if hinted:
                    return hinted

        for key in INFERENCE_HOST_ORDER:
            host = self._extract_inference_host(key)
            if host:
                return host

        return self._default_host

    def _extract_inference_host(self, inference_key: str) -> str | None:
        section = self._inference_config.get(inference_key, {})
        if isinstance(section, dict):
            host = self._normalize_host(section.get("url"))
            if self._is_valid_host(host):
                return host
        return None

    def discover_models(self, host: str) -> tuple[list[str], str | None]:
        if Client is None:
            return [], "python package 'ollama' is not installed; model inventory check skipped"

        try:
            response = Client(host=host).list()
        except Exception as error:  # noqa: BLE001
            return [], str(error)

        names: list[str] = []
        for entry in getattr(response, "models", []):
            if hasattr(entry, "model"):
                model_name = getattr(entry, "model")
            elif isinstance(entry, dict):
                model_name = entry.get("model")
            else:
                model_name = None

            if isinstance(model_name, str):
                model_name = model_name.strip()
                if model_name and model_name not in names:
                    names.append(model_name)

        return names, None

    def _load_policy_json(self, policy_path: Path) -> dict[str, Any]:
        if not policy_path.exists():
            raise PolicyValidationError(f"Policy file does not exist: {policy_path}")

        try:
            with policy_path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except json.JSONDecodeError as error:
            raise PolicyValidationError(f"Invalid JSON in {policy_path}: {error}") from error
        except OSError as error:
            raise PolicyValidationError(f"Unable to read policy file {policy_path}: {error}") from error

        if not isinstance(payload, dict):
            raise PolicyValidationError(f"Policy file must contain a JSON object: {policy_path}")

        return payload

    def _default_model_for_policy(self, policy_name: str) -> str:
        inference_key = self._policy_to_inference_key.get(policy_name, "chat")
        section = self._inference_config.get(inference_key, {})
        if isinstance(section, dict):
            model_name = section.get("model")
            if isinstance(model_name, str) and model_name.strip():
                return model_name.strip()

        runtimeKey = {
            "embedding": "inference.default_embedding_model",
            "generate": "inference.default_generate_model",
            "tool": "inference.default_tool_model",
            "chat": "inference.default_chat_model",
            "multimodal": "inference.default_multimodal_model",
        }.get(inference_key, "inference.default_chat_model")
        fallback = DEFAULT_RUNTIME_SETTINGS
        for part in runtimeKey.split("."):
            if not isinstance(fallback, dict):
                break
            fallback = fallback.get(part)
        if isinstance(fallback, str) and fallback.strip():
            return fallback.strip()
        return str(
            DEFAULT_RUNTIME_SETTINGS.get("inference", {}).get("default_chat_model", "llama3.2:latest")
        )

    def default_policy(self, policy_name: str) -> dict[str, Any]:
        fallback = {
            "allow_custom_system_prompt": False,
            "allowed_models": [self._default_model_for_policy(policy_name)],
        }
        if policy_name == "tool_calling":
            runtime_defaults = DEFAULT_RUNTIME_SETTINGS.get("tool_runtime", {})
            fallback["tool_runtime"] = {
                "default_timeout_seconds": float(runtime_defaults.get("default_timeout_seconds", 8.0)),
                "default_max_retries": int(runtime_defaults.get("default_max_retries", 1)),
                "reject_unknown_args": False,
                "unknown_tool_behavior": "structured_error",
                "enable_human_approval": bool(runtime_defaults.get("enable_human_approval", True)),
                "default_approval_timeout_seconds": float(runtime_defaults.get("default_approval_timeout_seconds", 45.0)),
                "approval_poll_interval_seconds": float(runtime_defaults.get("approval_poll_interval_seconds", 0.25)),
                "default_dry_run": bool(runtime_defaults.get("default_dry_run", False)),
                "sandbox": copy.deepcopy(runtime_defaults.get("sandbox", {})),
                "tools": {},
            }
        return fallback

    def _validate_schema(
        self,
        policy_name: str,
        raw_policy: dict[str, Any],
    ) -> tuple[dict[str, Any], list[str], list[str]]:
        normalized = copy.deepcopy(raw_policy)
        errors: list[str] = []
        warnings: list[str] = []

        allow_custom = normalized.get("allow_custom_system_prompt")
        if not isinstance(allow_custom, bool):
            errors.append("Policy key 'allow_custom_system_prompt' must be a boolean.")
            normalized["allow_custom_system_prompt"] = False

        allowed_models = normalized.get("allowed_models")
        if not isinstance(allowed_models, list):
            errors.append("Policy key 'allowed_models' must be a list of model names.")
            cleaned_models: list[str] = []
        else:
            cleaned_models = []
            for model in allowed_models:
                if not isinstance(model, str):
                    warnings.append("Ignored non-string entry in 'allowed_models'.")
                    continue
                cleaned = model.strip()
                if not cleaned:
                    warnings.append("Ignored empty model entry in 'allowed_models'.")
                    continue
                if cleaned not in cleaned_models:
                    cleaned_models.append(cleaned)
                else:
                    warnings.append(f"Ignored duplicate model entry '{cleaned}' in 'allowed_models'.")

        if not cleaned_models:
            errors.append("Policy key 'allowed_models' must contain at least one model.")
            cleaned_models = [self._default_model_for_policy(policy_name)]

        normalized["allowed_models"] = cleaned_models

        if "tool_runtime" in normalized and not isinstance(normalized.get("tool_runtime"), dict):
            errors.append("Policy key 'tool_runtime' must be a JSON object when present.")
        elif isinstance(normalized.get("tool_runtime"), dict):
            runtime_payload = copy.deepcopy(normalized.get("tool_runtime"))
            if "tools" in runtime_payload and not isinstance(runtime_payload.get("tools"), dict):
                errors.append("Policy key 'tool_runtime.tools' must be a JSON object when present.")
                runtime_payload["tools"] = {}
            if "sandbox" in runtime_payload and not isinstance(runtime_payload.get("sandbox"), dict):
                errors.append("Policy key 'tool_runtime.sandbox' must be a JSON object when present.")
                runtime_payload["sandbox"] = {}
            for numeric_key in (
                "default_timeout_seconds",
                "default_approval_timeout_seconds",
                "approval_poll_interval_seconds",
            ):
                if numeric_key in runtime_payload:
                    try:
                        runtime_payload[numeric_key] = float(runtime_payload[numeric_key])
                    except (TypeError, ValueError):
                        warnings.append(f"Ignored invalid numeric value for 'tool_runtime.{numeric_key}'.")
                        runtime_payload.pop(numeric_key, None)
            for int_key in ("default_max_retries",):
                if int_key in runtime_payload:
                    try:
                        runtime_payload[int_key] = int(runtime_payload[int_key])
                    except (TypeError, ValueError):
                        warnings.append(f"Ignored invalid integer value for 'tool_runtime.{int_key}'.")
                        runtime_payload.pop(int_key, None)
            normalized["tool_runtime"] = runtime_payload

        return normalized, errors, warnings

    def _validate_prompt_path(self, prompt_path: Path) -> tuple[list[str], list[str]]:
        errors: list[str] = []
        warnings: list[str] = []

        if not prompt_path.exists():
            errors.append(f"Missing system prompt file: {prompt_path}")
            return errors, warnings

        try:
            content = prompt_path.read_text(encoding="utf-8")
        except OSError as error:
            errors.append(f"Failed reading system prompt file {prompt_path}: {error}")
            return errors, warnings

        if not content.strip():
            warnings.append(f"System prompt file is empty: {prompt_path}")

        return errors, warnings

    def _report_for_policy_data(
        self,
        policy_name: str,
        policy_data: dict[str, Any],
        strict_model_check: bool,
    ) -> PolicyValidationReport:
        policy_path = self._policy_path(policy_name)
        prompt_path = self._prompt_path(policy_name)
        host = self.resolve_host(policy_name)
        report = PolicyValidationReport(
            policy_name=policy_name,
            policy_path=str(policy_path),
            prompt_path=str(prompt_path),
            endpoint_host=host,
        )

        normalized, schema_errors, schema_warnings = self._validate_schema(policy_name, policy_data)
        report.normalized_policy = normalized
        report.errors.extend(schema_errors)
        report.warnings.extend(schema_warnings)

        prompt_errors, prompt_warnings = self._validate_prompt_path(prompt_path)
        report.errors.extend(prompt_errors)
        report.warnings.extend(prompt_warnings)

        discovered_models, probe_error = self.discover_models(host)
        report.available_models = discovered_models
        if probe_error:
            report.warnings.append(
                "Model inventory probe failed for host "
                f"{host}: {probe_error}"
            )
        elif discovered_models:
            missing_models = [
                model_name
                for model_name in normalized.get("allowed_models", [])
                if model_name not in discovered_models
            ]
            if missing_models:
                issue = (
                    "Policy contains models not found on endpoint "
                    f"{host}: {', '.join(missing_models)}"
                )
                if strict_model_check:
                    report.errors.append(issue)
                else:
                    report.warnings.append(issue)
        else:
            report.warnings.append(
                f"No models reported by endpoint {host}; policy model compatibility not fully verified."
            )

        return report

    def validate_policy(
        self,
        policy_name: str,
        strict_model_check: bool = False,
    ) -> PolicyValidationReport:
        policy_path = self._policy_path(policy_name)
        prompt_path = self._prompt_path(policy_name)
        host = self.resolve_host(policy_name)
        report = PolicyValidationReport(
            policy_name=policy_name,
            policy_path=str(policy_path),
            prompt_path=str(prompt_path),
            endpoint_host=host,
        )

        try:
            raw_policy = self._load_policy_json(policy_path)
        except PolicyValidationError as error:
            report.errors.append(str(error))
            prompt_errors, prompt_warnings = self._validate_prompt_path(prompt_path)
            report.errors.extend(prompt_errors)
            report.warnings.extend(prompt_warnings)
            return report

        return self._report_for_policy_data(
            policy_name=policy_name,
            policy_data=raw_policy,
            strict_model_check=strict_model_check,
        )

    def validate_all_policies(self, strict_model_check: bool = False) -> dict[str, PolicyValidationReport]:
        reports: dict[str, PolicyValidationReport] = {}
        for policy_name in self.list_policy_names():
            reports[policy_name] = self.validate_policy(
                policy_name=policy_name,
                strict_model_check=strict_model_check,
            )
        return reports

    def load_policy(
        self,
        policy_name: str,
        strict: bool = False,
        strict_model_check: bool = False,
    ) -> dict[str, Any]:
        report = self.validate_policy(
            policy_name=policy_name,
            strict_model_check=strict_model_check,
        )
        if report.errors:
            if strict:
                raise PolicyValidationError("; ".join(report.errors))
            return self.default_policy(policy_name)
        if isinstance(report.normalized_policy, dict):
            return report.normalized_policy
        return self.default_policy(policy_name)

    def load_system_prompt(self, policy_name: str, strict: bool = False) -> str:
        prompt_path = self._prompt_path(policy_name)
        try:
            content = prompt_path.read_text(encoding="utf-8")
        except OSError as error:
            if strict:
                raise PolicyValidationError(f"Failed reading system prompt file {prompt_path}: {error}") from error
            return FALLBACK_SYSTEM_PROMPT

        if not content.strip():
            if strict:
                raise PolicyValidationError(f"System prompt file is empty: {prompt_path}")
            return FALLBACK_SYSTEM_PROMPT

        return content

    @staticmethod
    def _write_json(path: Path, data: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        with tmp_path.open("w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2)
            handle.write("\n")
        tmp_path.replace(path)

    @staticmethod
    def _backup_file(path: Path) -> Path | None:
        if not path.exists():
            return None
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        backup_path = path.with_suffix(path.suffix + f".bak-{stamp}")
        backup_path.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
        return backup_path

    def save_policy(
        self,
        policy_name: str,
        updates: dict[str, Any],
        strict_model_check: bool = False,
    ) -> PolicySaveResult:
        policy_path = self._policy_path(policy_name)
        try:
            current_policy = self._load_policy_json(policy_path)
        except PolicyValidationError as error:
            report = PolicyValidationReport(
                policy_name=policy_name,
                policy_path=str(policy_path),
                prompt_path=str(self._prompt_path(policy_name)),
                endpoint_host=self.resolve_host(policy_name),
                errors=[str(error)],
            )
            return PolicySaveResult(
                saved=False,
                backup_path=None,
                rollback_performed=False,
                report=report,
            )

        candidate = copy.deepcopy(current_policy)
        for key, value in (updates or {}).items():
            candidate[key] = value

        pre_report = self._report_for_policy_data(
            policy_name=policy_name,
            policy_data=candidate,
            strict_model_check=strict_model_check,
        )
        if pre_report.errors:
            return PolicySaveResult(
                saved=False,
                backup_path=None,
                rollback_performed=False,
                report=pre_report,
            )

        backup_path = self._backup_file(policy_path)
        try:
            self._write_json(policy_path, pre_report.normalized_policy or candidate)
        except OSError as error:
            pre_report.errors.append(f"Failed writing policy file {policy_path}: {error}")
            return PolicySaveResult(
                saved=False,
                backup_path=str(backup_path) if backup_path else None,
                rollback_performed=False,
                report=pre_report,
            )

        post_report = self.validate_policy(
            policy_name=policy_name,
            strict_model_check=strict_model_check,
        )
        if post_report.errors:
            rollback_performed = False
            if backup_path and backup_path.exists():
                try:
                    policy_path.write_text(backup_path.read_text(encoding="utf-8"), encoding="utf-8")
                    rollback_performed = True
                except OSError as rollback_error:
                    post_report.errors.append(f"Rollback failed for {policy_path}: {rollback_error}")
            return PolicySaveResult(
                saved=False,
                backup_path=str(backup_path) if backup_path else None,
                rollback_performed=rollback_performed,
                report=post_report,
            )

        return PolicySaveResult(
            saved=True,
            backup_path=str(backup_path) if backup_path else None,
            rollback_performed=False,
            report=post_report,
        )
