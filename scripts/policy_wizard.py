#!/usr/bin/env python3
"""
Policy walkthrough and setup helper.

WO-006 implementation:
- Guided policy editing for allow_custom_system_prompt and allowed_models.
- Endpoint-aware model discovery using setup/runtime host precedence.
- Validation report output before and after save.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from hypermindlabs.policy_manager import PolicyManager, PolicyValidationReport


def load_config(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def print_report(report: PolicyValidationReport) -> None:
    print(f"Policy: {report.policy_name}")
    print(f"Policy file: {report.policy_path}")
    print(f"Prompt file: {report.prompt_path}")
    print(f"Endpoint host: {report.endpoint_host}")
    if report.available_models:
        print(f"Available models ({len(report.available_models)}): {', '.join(report.available_models)}")
    else:
        print("Available models: none reported")

    if report.warnings:
        print("Warnings:")
        for warning in report.warnings:
            print(f"  - {warning}")
    if report.errors:
        print("Errors:")
        for error in report.errors:
            print(f"  - {error}")


def prompt_text(label: str, default: str = "") -> str:
    default_hint = f" [{default}]" if default else ""
    raw = input(f"{label}{default_hint}: ").strip()
    if raw:
        return raw
    return default


def prompt_bool(label: str, default: bool) -> bool:
    hint = "Y/n" if default else "y/N"
    raw = input(f"{label} [{hint}]: ").strip().lower()
    if not raw:
        return default
    return raw in {"y", "yes", "1", "true"}


def choose_policy_name(manager: PolicyManager, explicit: str | None) -> str:
    available = manager.list_policy_names()
    if not available:
        raise RuntimeError("No policy files discovered in policies directory.")

    if explicit:
        if explicit in available:
            return explicit
        raise RuntimeError(
            f"Requested policy '{explicit}' was not found. Available: {', '.join(available)}"
        )

    if len(available) == 1:
        return available[0]

    print("Available policies:")
    for index, name in enumerate(available, start=1):
        print(f"  {index}. {name}")

    while True:
        selected = prompt_text("Select policy by number", default="1")
        if selected.isdigit():
            idx = int(selected) - 1
            if 0 <= idx < len(available):
                return available[idx]
        print("Please enter a valid number from the list.")


def parse_models(raw_value: str, available_models: list[str]) -> list[str]:
    tokens = [token.strip() for token in raw_value.split(",") if token.strip()]
    parsed: list[str] = []
    for token in tokens:
        if token.isdigit() and available_models:
            idx = int(token) - 1
            if 0 <= idx < len(available_models):
                model_name = available_models[idx]
                if model_name not in parsed:
                    parsed.append(model_name)
                continue
        if token not in parsed:
            parsed.append(token)
    return parsed


def select_allowed_models(current_models: list[str], available_models: list[str]) -> list[str]:
    if available_models:
        print("Discovered models:")
        for index, name in enumerate(available_models, start=1):
            marker = "*" if name in current_models else " "
            print(f"  {index}. {marker} {name}")
        default_input = ",".join(current_models) if current_models else "1"
        raw = prompt_text(
            "Allowed models (comma separated numbers or names)",
            default=default_input,
        )
        selected = parse_models(raw, available_models)
    else:
        default_input = ",".join(current_models)
        raw = prompt_text(
            "Allowed models (comma separated names)",
            default=default_input,
        )
        selected = parse_models(raw, available_models)

    if selected:
        return selected
    return current_models


def parse_bool_argument(value: str | None) -> bool | None:
    if value is None:
        return None
    lowered = value.strip().lower()
    if lowered in {"true", "1", "yes", "y"}:
        return True
    if lowered in {"false", "0", "no", "n"}:
        return False
    raise ValueError(
        "Expected boolean value for --allow-custom-system-prompt "
        "(true|false|yes|no|1|0)."
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Guided policy validator/editor for RYO agent policies."
    )
    parser.add_argument("--config", default="config.json", help="Path to runtime config file.")
    parser.add_argument("--policies-dir", default="policies/agent", help="Path to policy directory.")
    parser.add_argument("--policy", default=None, help="Policy name, for example: tool_calling.")
    parser.add_argument("--ollama-host", default=None, help="Explicit Ollama endpoint override.")
    parser.add_argument(
        "--strict-models",
        action="store_true",
        help="Treat model/endpoint mismatches as validation errors.",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Only validate policy/prompt files; do not edit.",
    )
    parser.add_argument(
        "--non-interactive",
        action="store_true",
        help="Apply updates from flags without prompts.",
    )
    parser.add_argument(
        "--allow-custom-system-prompt",
        default=None,
        help="Non-interactive override for allow_custom_system_prompt.",
    )
    parser.add_argument(
        "--allowed-models",
        default=None,
        help="Non-interactive comma-separated model names or indexes.",
    )
    parser.add_argument(
        "--preview-prompt-lines",
        type=int,
        default=16,
        help="Number of system prompt lines to preview in interactive mode.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()

    config_data = load_config(Path(args.config))
    inference = config_data.get("inference", {})
    manager = PolicyManager(
        policies_dir=args.policies_dir,
        inference_config=inference,
        endpoint_override=args.ollama_host,
    )

    policy_name = choose_policy_name(manager, args.policy)
    report = manager.validate_policy(
        policy_name=policy_name,
        strict_model_check=args.strict_models,
    )
    print_report(report)

    if args.validate_only:
        return 1 if report.errors else 0

    current_policy = manager.load_policy(
        policy_name=policy_name,
        strict=False,
        strict_model_check=args.strict_models,
    )
    current_models = current_policy.get("allowed_models", [])
    if not isinstance(current_models, list):
        current_models = []

    try:
        override_allow_custom = parse_bool_argument(args.allow_custom_system_prompt)
    except ValueError as error:
        print(error)
        return 1

    if args.non_interactive:
        updates: dict = {}
        if override_allow_custom is not None:
            updates["allow_custom_system_prompt"] = override_allow_custom
        if args.allowed_models is not None:
            updates["allowed_models"] = parse_models(args.allowed_models, report.available_models)

        if not updates:
            print("No non-interactive updates provided. Nothing changed.")
            return 0

        save_result = manager.save_policy(
            policy_name=policy_name,
            updates=updates,
            strict_model_check=args.strict_models,
        )
        print_report(save_result.report)
        if save_result.backup_path:
            print(f"Backup file: {save_result.backup_path}")
        if save_result.rollback_performed:
            print("Rollback executed due to failed post-save validation.")
        return 0 if save_result.saved else 1

    prompt_text_content = manager.load_system_prompt(policy_name=policy_name, strict=False)
    prompt_lines = prompt_text_content.splitlines()
    preview_count = max(1, args.preview_prompt_lines)
    print("\nSystem prompt preview:")
    for line in prompt_lines[:preview_count]:
        print(f"  {line}")
    if len(prompt_lines) > preview_count:
        print("  ...")
    print("")

    current_allow_custom = bool(current_policy.get("allow_custom_system_prompt", False))
    if override_allow_custom is None:
        allow_custom = prompt_bool(
            "Allow custom system prompt overrides?",
            default=current_allow_custom,
        )
    else:
        allow_custom = override_allow_custom

    if args.allowed_models is None:
        selected_models = select_allowed_models(current_models, report.available_models)
    else:
        selected_models = parse_models(args.allowed_models, report.available_models)

    print("\nProposed policy updates:")
    print(f"  allow_custom_system_prompt: {allow_custom}")
    print(f"  allowed_models: {', '.join(selected_models)}")
    confirm = prompt_bool("Save these policy changes?", default=True)
    if not confirm:
        print("No changes saved.")
        return 0

    save_result = manager.save_policy(
        policy_name=policy_name,
        updates={
            "allow_custom_system_prompt": allow_custom,
            "allowed_models": selected_models,
        },
        strict_model_check=args.strict_models,
    )
    print("")
    print_report(save_result.report)
    if save_result.backup_path:
        print(f"Backup file: {save_result.backup_path}")
    if save_result.rollback_performed:
        print("Rollback executed due to failed post-save validation.")

    if save_result.saved:
        print("Policy update saved successfully.")
        return 0

    print("Policy update failed.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
