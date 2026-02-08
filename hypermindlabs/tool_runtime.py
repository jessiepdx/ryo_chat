##########################################################################
#                                                                        #
#  This file (tool_runtime.py) provides validated, fault-tolerant        #
#  execution for model-invoked tools.                                    #
#                                                                        #
##########################################################################

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class ToolDefinition:
    name: str
    function: Callable[..., Any]
    required_args: tuple[str, ...] = ()
    optional_args: dict[str, Any] = field(default_factory=dict)
    required_api_key: str | None = None
    timeout_seconds: float = 8.0
    max_retries: int = 1


class ToolRuntime:
    """Validates and executes tool calls with bounded retry/timeout behavior."""

    def __init__(self, api_keys: dict[str, str] | None = None):
        self._registry: dict[str, ToolDefinition] = {}
        self._api_keys = api_keys or {}

    def register_tool(self, tool: ToolDefinition) -> None:
        self._registry[tool.name] = tool

    @staticmethod
    def _error_result(
        tool_name: str,
        code: str,
        message: str,
        attempts: int = 0,
        details: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return {
            "tool_name": tool_name,
            "status": "error",
            "tool_results": None,
            "error": {
                "code": code,
                "message": message,
                "attempts": attempts,
                "details": details or {},
            },
        }

    @staticmethod
    def _success_result(tool_name: str, output: Any, attempts: int = 1) -> dict[str, Any]:
        return {
            "tool_name": tool_name,
            "status": "success",
            "tool_results": output,
            "error": None,
            "attempts": attempts,
        }

    @staticmethod
    def _normalize_args(raw_args: Any) -> dict[str, Any]:
        if isinstance(raw_args, dict):
            properties = raw_args.get("properties")
            if isinstance(properties, dict):
                return properties
            return raw_args
        return {}

    def _validate_and_prepare_args(
        self,
        tool: ToolDefinition,
        raw_args: Any,
    ) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
        args = self._normalize_args(raw_args)
        accepted_keys = set(tool.required_args) | set(tool.optional_args.keys())
        prepared = {key: value for key, value in args.items() if key in accepted_keys}

        missing_required = [
            arg_name for arg_name in tool.required_args if prepared.get(arg_name) is None
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

        return prepared, None

    def execute(self, tool_name: str, raw_args: Any) -> dict[str, Any]:
        tool = self._registry.get(tool_name)
        if tool is None:
            return self._error_result(
                tool_name=tool_name,
                code="tool_not_registered",
                message=f"Tool '{tool_name}' is not registered.",
            )

        if tool.required_api_key:
            key_value = self._api_keys.get(tool.required_api_key)
            if key_value is None or (isinstance(key_value, str) and key_value.strip() == ""):
                return self._error_result(
                    tool_name=tool_name,
                    code="missing_api_key",
                    message=f"Required API key '{tool.required_api_key}' is missing.",
                )

        prepared_args, arg_error = self._validate_and_prepare_args(tool, raw_args)
        if arg_error:
            return self._error_result(
                tool_name=tool_name,
                code=arg_error["code"],
                message=arg_error["message"],
                details=arg_error["details"],
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
                return self._success_result(tool_name=tool_name, output=output, attempts=attempts)
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
        )
