##########################################################################
#                                                                        #
#  This file (tool_registry.py) defines canonical metadata for tools     #
#  used by the tool-calling agent and runtime.                           #
#                                                                        #
##########################################################################

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from hypermindlabs.tool_runtime import ToolDefinition, ToolRuntime


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
    default_timeout_seconds: float = 8.0
    default_max_retries: int = 1

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


def build_tool_specs(
    brave_search_fn: Callable[..., Any],
    chat_history_search_fn: Callable[..., Any],
    knowledge_search_fn: Callable[..., Any],
    skip_tools_fn: Callable[..., Any],
    knowledge_domains: list[str] | None = None,
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

    specs = {
        "braveSearch": ToolSpec(
            name="braveSearch",
            description="Search the public web for relevant information.",
            function=brave_search_fn,
            args=(query_arg, count_arg),
            required_api_key="brave_search",
            default_timeout_seconds=10.0,
            default_max_retries=1,
        ),
        "chatHistorySearch": ToolSpec(
            name="chatHistorySearch",
            description="Search prior chat history for semantically related messages.",
            function=chat_history_search_fn,
            args=(query_arg, narrow_count_arg),
            default_timeout_seconds=6.0,
            default_max_retries=0,
        ),
        "knowledgeSearch": ToolSpec(
            name="knowledgeSearch",
            description=(
                "Search project knowledge documents for semantically related results. "
                f"Known domains: {domain_text}."
            ),
            function=knowledge_search_fn,
            args=(query_arg, narrow_count_arg),
            default_timeout_seconds=6.0,
            default_max_retries=0,
        ),
        "skipTools": ToolSpec(
            name="skipTools",
            description="Skip tool usage and continue to the next agent.",
            function=skip_tools_fn,
            args=(),
            default_timeout_seconds=2.0,
            default_max_retries=0,
        ),
    }
    return specs


def model_tool_definitions(specs: dict[str, ToolSpec]) -> list[dict[str, Any]]:
    ordered_names = ("braveSearch", "chatHistorySearch", "knowledgeSearch", "skipTools")
    output: list[dict[str, Any]] = []
    for name in ordered_names:
        spec = specs.get(name)
        if spec is not None:
            output.append(spec.to_model_tool())
    return output


def _float_value(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _int_value(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def register_runtime_tools(
    runtime: ToolRuntime,
    specs: dict[str, ToolSpec],
    tool_policy: dict[str, Any] | None = None,
    default_timeout_seconds: float = 8.0,
    default_max_retries: int = 1,
    reject_unknown_args: bool = False,
) -> None:
    tool_policy = tool_policy if isinstance(tool_policy, dict) else {}
    ordered_names = ("braveSearch", "chatHistorySearch", "knowledgeSearch", "skipTools")
    for name in ordered_names:
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
            )
        )
