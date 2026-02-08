##########################################################################
#                                                                        #
#  This file (tool_runtime.py) provides validated, fault-tolerant        #
#  execution for model-invoked tools.                                    #
#                                                                        #
##########################################################################

from __future__ import annotations

import copy
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from dataclasses import dataclass, field
import json
from typing import Any, Callable

from hypermindlabs.approval_manager import ApprovalManager, ApprovalValidationError
from hypermindlabs.runtime_settings import DEFAULT_RUNTIME_SETTINGS
from hypermindlabs.tool_sandbox import ToolSandboxEnforcer, resolve_tool_sandbox_policy


DEFAULT_TOOL_TIMEOUT_SECONDS = float(
    DEFAULT_RUNTIME_SETTINGS.get("tool_runtime", {}).get("default_timeout_seconds", 8.0)
)
DEFAULT_TOOL_MAX_RETRIES = int(
    DEFAULT_RUNTIME_SETTINGS.get("tool_runtime", {}).get("default_max_retries", 1)
)


def _as_int(value: Any, fallback: int | None = None) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return fallback


@dataclass
class ToolDefinition:
    name: str
    function: Callable[..., Any]
    required_args: tuple[str, ...] = ()
    optional_args: dict[str, Any] = field(default_factory=dict)
    arg_coercers: dict[str, Callable[[Any], Any]] = field(default_factory=dict)
    reject_unknown_args: bool = False
    required_api_key: str | None = None
    timeout_seconds: float = DEFAULT_TOOL_TIMEOUT_SECONDS
    max_retries: int = DEFAULT_TOOL_MAX_RETRIES
    side_effect_class: str = "read_only"
    sandbox_policy: dict[str, Any] = field(default_factory=dict)
    mock_result: Any = None
    metadata: dict[str, Any] = field(default_factory=dict)


class ToolRuntime:
    """Validates and executes tool calls with bounded retry/timeout behavior."""

    def __init__(
        self,
        api_keys: dict[str, str] | None = None,
        *,
        sandbox_enforcer: ToolSandboxEnforcer | None = None,
        approval_manager: ApprovalManager | None = None,
        runtime_context: dict[str, Any] | None = None,
        decision_callback: Callable[[dict[str, Any]], None] | None = None,
        enable_human_approval: bool = True,
        default_approval_timeout_seconds: float = 45.0,
        approval_poll_interval_seconds: float = 0.25,
        default_dry_run: bool = False,
    ):
        self._registry: dict[str, ToolDefinition] = {}
        self._api_keys = api_keys or {}
        self._sandbox = sandbox_enforcer or ToolSandboxEnforcer()
        self._approvals = approval_manager or ApprovalManager()
        self._context = runtime_context if isinstance(runtime_context, dict) else {}
        self._decision_callback = decision_callback if callable(decision_callback) else None
        self._enable_human_approval = bool(enable_human_approval)
        self._default_approval_timeout_seconds = max(1.0, float(default_approval_timeout_seconds))
        self._approval_poll_interval_seconds = max(0.05, float(approval_poll_interval_seconds))
        self._default_dry_run = bool(default_dry_run)

    def register_tool(self, tool: ToolDefinition) -> None:
        self._registry[tool.name] = tool

    def set_runtime_context(self, context: dict[str, Any] | None) -> None:
        self._context = context if isinstance(context, dict) else {}

    def _emit_decision(self, event: dict[str, Any]) -> None:
        if not callable(self._decision_callback):
            return
        try:
            self._decision_callback(copy.deepcopy(event))
        except Exception:  # noqa: BLE001
            return

    def _audit_event(
        self,
        *,
        event_type: str,
        stage: str,
        status: str,
        detail: str,
        meta: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload = {
            "event_type": str(event_type or "run.stage"),
            "stage": str(stage or "tools.runtime"),
            "status": str(status or "info"),
            "detail": str(detail or ""),
            "meta": meta if isinstance(meta, dict) else {},
        }
        self._emit_decision(payload)
        return payload

    @staticmethod
    def _error_result(
        tool_name: str,
        code: str,
        message: str,
        attempts: int = 0,
        details: dict[str, Any] | None = None,
        audit: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        return {
            "tool_name": tool_name,
            "status": "error",
            "tool_results": None,
            "attempts": attempts,
            "audit": audit if isinstance(audit, list) else [],
            "error": {
                "code": code,
                "message": message,
                "attempts": attempts,
                "details": details or {},
            },
        }

    @staticmethod
    def _success_result(
        tool_name: str,
        output: Any,
        attempts: int = 1,
        audit: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        return {
            "tool_name": tool_name,
            "status": "success",
            "tool_results": output,
            "error": None,
            "attempts": attempts,
            "audit": audit if isinstance(audit, list) else [],
        }

    @staticmethod
    def _normalize_args(raw_args: Any) -> Any:
        if isinstance(raw_args, str):
            stripped = raw_args.strip()
            if not stripped:
                return {}
            try:
                parsed = json.loads(stripped)
            except json.JSONDecodeError:
                return {}
            return ToolRuntime._normalize_args(parsed)

        if isinstance(raw_args, dict):
            return raw_args

        return {}

    @staticmethod
    def _candidate_arg_maps(raw_args: Any) -> list[dict[str, Any]]:
        normalized = ToolRuntime._normalize_args(raw_args)
        if not isinstance(normalized, dict):
            return []

        candidates: list[dict[str, Any]] = [normalized]
        properties = normalized.get("properties")
        if isinstance(properties, dict):
            candidates.append(properties)

        nested_arguments = normalized.get("arguments")
        if nested_arguments is not None:
            nested = ToolRuntime._normalize_args(nested_arguments)
            if isinstance(nested, dict):
                candidates.append(nested)

        unique: list[dict[str, Any]] = []
        seen_ids: set[int] = set()
        for candidate in candidates:
            if id(candidate) not in seen_ids:
                unique.append(candidate)
                seen_ids.add(id(candidate))
        return unique

    def _validate_and_prepare_args(
        self,
        tool: ToolDefinition,
        raw_args: Any,
    ) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
        candidate_maps = self._candidate_arg_maps(raw_args)
        args = {}
        accepted_keys = set(tool.required_args) | set(tool.optional_args.keys())
        best_overlap = -1
        for candidate in candidate_maps:
            overlap = len([key for key in candidate.keys() if key in accepted_keys])
            if overlap > best_overlap:
                best_overlap = overlap
                args = candidate

        accepted_keys = set(tool.required_args) | set(tool.optional_args.keys())
        unknown_args = [key for key in args.keys() if key not in accepted_keys]
        if tool.reject_unknown_args and unknown_args:
            return None, {
                "code": "invalid_arguments",
                "message": f"Unexpected arguments provided: {', '.join(unknown_args)}",
                "details": {"unknown_args": unknown_args},
            }

        prepared = {key: value for key, value in args.items() if key in accepted_keys}
        coercion_errors: dict[str, str] = {}
        for arg_name, coercer in tool.arg_coercers.items():
            if arg_name in prepared and prepared[arg_name] is not None:
                try:
                    prepared[arg_name] = coercer(prepared[arg_name])
                except Exception as error:  # noqa: BLE001
                    coercion_errors[arg_name] = str(error)

        if coercion_errors:
            return None, {
                "code": "invalid_arguments",
                "message": "Argument coercion failed.",
                "details": {"coercion_errors": coercion_errors},
            }

        missing_required = [
            arg_name
            for arg_name in tool.required_args
            if prepared.get(arg_name) is None
            or (isinstance(prepared.get(arg_name), str) and prepared.get(arg_name).strip() == "")
        ]
        if missing_required:
            return None, {
                "code": "invalid_arguments",
                "message": f"Missing required arguments: {', '.join(missing_required)}",
                "details": {"missing_required": missing_required},
            }

        for opt_key, default_value in tool.optional_args.items():
            if prepared.get(opt_key) is None:
                prepared[opt_key] = default_value
            if opt_key in tool.arg_coercers and prepared.get(opt_key) is not None:
                try:
                    prepared[opt_key] = tool.arg_coercers[opt_key](prepared[opt_key])
                except Exception as error:  # noqa: BLE001
                    return None, {
                        "code": "invalid_arguments",
                        "message": f"Invalid value for optional argument '{opt_key}'.",
                        "details": {"coercion_errors": {opt_key: str(error)}},
                    }

        return prepared, None

    @staticmethod
    def _extract_tool_invocation(tool_call: Any) -> tuple[str | None, Any, dict[str, Any] | None]:
        if tool_call is None:
            return None, {}, {
                "code": "invalid_tool_call",
                "message": "Tool call is empty.",
                "details": {},
            }

        tool_name = None
        tool_args: Any = {}

        if hasattr(tool_call, "function"):
            function_data = getattr(tool_call, "function")
            tool_name = getattr(function_data, "name", None)
            tool_args = getattr(function_data, "arguments", {})
        elif isinstance(tool_call, dict):
            function_data = tool_call.get("function")
            if isinstance(function_data, dict):
                tool_name = function_data.get("name")
                tool_args = function_data.get("arguments", {})
            else:
                tool_name = tool_call.get("name")
                tool_args = tool_call.get("arguments", {})
        else:
            return None, {}, {
                "code": "invalid_tool_call",
                "message": "Tool call format is unsupported.",
                "details": {"received_type": type(tool_call).__name__},
            }

        if not isinstance(tool_name, str) or tool_name.strip() == "":
            return None, tool_args, {
                "code": "invalid_tool_call",
                "message": "Tool call is missing a valid function name.",
                "details": {},
            }

        return tool_name.strip(), tool_args, None

    def execute_tool_call(self, tool_call: Any) -> dict[str, Any]:
        tool_name, tool_args, parse_error = self._extract_tool_invocation(tool_call)
        if parse_error:
            return self._error_result(
                tool_name=tool_name or "unknown",
                code=parse_error["code"],
                message=parse_error["message"],
                details=parse_error["details"],
            )
        return self.execute(tool_name=tool_name, raw_args=tool_args)

    def execute(self, tool_name: str, raw_args: Any) -> dict[str, Any]:
        audit: list[dict[str, Any]] = []
        tool = self._registry.get(tool_name)
        if tool is None:
            return self._error_result(
                tool_name=tool_name,
                code="tool_not_registered",
                message=f"Tool '{tool_name}' is not registered.",
                audit=audit,
            )

        if tool.required_api_key:
            key_value = self._api_keys.get(tool.required_api_key)
            if key_value is None or (isinstance(key_value, str) and key_value.strip() == ""):
                return self._error_result(
                    tool_name=tool_name,
                    code="missing_api_key",
                    message=f"Required API key '{tool.required_api_key}' is missing.",
                    audit=audit,
                )

        prepared_args, arg_error = self._validate_and_prepare_args(tool, raw_args)
        if arg_error:
            return self._error_result(
                tool_name=tool_name,
                code=arg_error["code"],
                message=arg_error["message"],
                details=arg_error["details"],
                audit=audit,
            )

        sandbox_decision = self._sandbox.evaluate(
            tool_name=tool_name,
            prepared_args=prepared_args,
            tool_policy=tool.sandbox_policy,
            tool_metadata={
                "side_effect_class": tool.side_effect_class,
                "sandbox_policy": tool.sandbox_policy,
                **(tool.metadata if isinstance(tool.metadata, dict) else {}),
            },
        )
        sandbox_allowed = str(sandbox_decision.get("status")) == "allow"
        audit.append(
            self._audit_event(
                event_type="run.sandbox",
                stage="tools.sandbox.decision",
                status="info" if sandbox_allowed else "error",
                detail=str(sandbox_decision.get("message") or "Sandbox decision recorded."),
                meta={
                    "tool_name": tool_name,
                    "code": sandbox_decision.get("code"),
                    "decision": sandbox_decision.get("status"),
                    "policy": sandbox_decision.get("effective_policy", {}),
                    "details": sandbox_decision.get("details", {}),
                },
            )
        )
        if not sandbox_allowed:
            return self._error_result(
                tool_name=tool_name,
                code="sandbox_blocked",
                message=str(sandbox_decision.get("message") or "Sandbox policy blocked this tool."),
                details={
                    "decision_code": sandbox_decision.get("code"),
                    "policy": sandbox_decision.get("effective_policy", {}),
                    "details": sandbox_decision.get("details", {}),
                },
                audit=audit,
            )

        effective_policy = resolve_tool_sandbox_policy(
            tool.sandbox_policy,
            sandbox_decision.get("effective_policy"),
        )
        requires_approval = bool(effective_policy.get("require_approval"))
        dry_run_enabled = self._default_dry_run or bool(effective_policy.get("dry_run"))
        approval_timeout_seconds = max(
            1.0,
            float(
                effective_policy.get("approval_timeout_seconds")
                or self._default_approval_timeout_seconds
            ),
        )

        if dry_run_enabled:
            mock_output = copy.deepcopy(tool.mock_result)
            if mock_output is None:
                mock_output = {
                    "tool": tool_name,
                    "status": "dry_run",
                    "arguments": copy.deepcopy(prepared_args),
                }
            audit.append(
                self._audit_event(
                    event_type="run.sandbox",
                    stage="tools.sandbox.dry_run",
                    status="info",
                    detail=f"Dry-run enabled for tool '{tool_name}'.",
                    meta={"tool_name": tool_name},
                )
            )
            return self._success_result(
                tool_name=tool_name,
                output=mock_output,
                attempts=0,
                audit=audit,
            )

        if requires_approval:
            if not self._enable_human_approval:
                return self._error_result(
                    tool_name=tool_name,
                    code="approval_required",
                    message="Tool requires human approval but approval workflow is disabled.",
                    details={"tool_name": tool_name},
                    audit=audit,
                )

            request_record = self._approvals.request_approval(
                run_id=str(self._context.get("run_id") or "unknown"),
                tool_name=tool_name,
                tool_args=prepared_args,
                requested_by_member_id=_as_int(self._context.get("member_id"), None),
                run_owner_member_id=_as_int(self._context.get("member_id"), None),
                reason=f"Tool '{tool_name}' requires approval before execution.",
                timeout_seconds=approval_timeout_seconds,
                meta={
                    "side_effect_class": tool.side_effect_class,
                    "run_id": self._context.get("run_id"),
                },
            )
            audit.append(
                self._audit_event(
                    event_type="run.approval",
                    stage="tools.approval.pending",
                    status="waiting",
                    detail=f"Approval required for tool '{tool_name}'.",
                    meta={
                        "tool_name": tool_name,
                        "request_id": request_record.get("request_id"),
                        "expires_at": request_record.get("expires_at"),
                    },
                )
            )
            try:
                decision = self._approvals.wait_for_decision(
                    str(request_record.get("request_id")),
                    timeout_seconds=approval_timeout_seconds,
                    poll_interval_seconds=self._approval_poll_interval_seconds,
                )
            except ApprovalValidationError as error:
                return self._error_result(
                    tool_name=tool_name,
                    code="approval_failed",
                    message=str(error),
                    details={"tool_name": tool_name},
                    audit=audit,
                )

            decision_status = str(decision.get("status") or "pending").lower()
            decision_reason = str(decision.get("reason") or "").strip()
            audit.append(
                self._audit_event(
                    event_type="run.approval",
                    stage="tools.approval.decision",
                    status="info" if decision_status == "approved" else "error",
                    detail=f"Approval decision for '{tool_name}': {decision_status}.",
                    meta={
                        "tool_name": tool_name,
                        "request_id": decision.get("request_id"),
                        "decision": decision_status,
                        "reason": decision_reason,
                    },
                )
            )
            if decision_status != "approved":
                code = "approval_timeout" if decision_status == "expired" else "approval_denied"
                return self._error_result(
                    tool_name=tool_name,
                    code=code,
                    message=decision_reason or f"Tool '{tool_name}' was not approved.",
                    details={"decision": decision_status, "request_id": decision.get("request_id")},
                    audit=audit,
                )

        attempts = 0
        last_error: Exception | None = None
        # Max retries means additional attempts after the first one.
        max_attempts = 1 + max(0, tool.max_retries)

        while attempts < max_attempts:
            attempts += 1
            executor = ThreadPoolExecutor(max_workers=1)
            future = None
            try:
                future = executor.submit(tool.function, **prepared_args)
                output = future.result(timeout=tool.timeout_seconds)
                return self._success_result(
                    tool_name=tool_name,
                    output=output,
                    attempts=attempts,
                    audit=audit,
                )
            except FuturesTimeoutError as error:
                if future is not None:
                    future.cancel()
                last_error = error
            except Exception as error:  # noqa: BLE001
                last_error = error
            finally:
                executor.shutdown(wait=False, cancel_futures=True)

        error_code = "tool_timeout" if isinstance(last_error, FuturesTimeoutError) else "tool_execution_failed"
        return self._error_result(
            tool_name=tool_name,
            code=error_code,
            message=f"Tool '{tool_name}' failed after {attempts} attempt(s).",
            attempts=attempts,
            details={"last_error": str(last_error)},
            audit=audit,
        )
