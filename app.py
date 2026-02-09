#!/usr/bin/env python3
"""
Single-entry DevOps launcher for RYO Chat.

Responsibilities:
1. Ensure a local virtual environment exists and re-exec inside it.
2. Install requirements only when needed (requirements hash changed).
3. Ensure baseline runtime artifacts exist (`logs/`, `config.json`, `.env`).
4. Run first-time (or recovery) setup wizard flow when needed.
5. Validate Ollama host reachability, ensure/pull required models, and allow
   selecting a default text model for chat/generate/tool capabilities.
6. Provide a central watchdog menu to toggle UI routes on/off.
"""

from __future__ import annotations

import argparse
from collections import deque
import copy
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import select
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import urlopen

try:
    import curses
except Exception:  # noqa: BLE001
    curses = None

try:
    import pwd
except Exception:  # noqa: BLE001
    pwd = None

from hypermindlabs.runtime_settings import (
    DEFAULT_RUNTIME_SETTINGS,
    build_runtime_settings,
    get_runtime_setting,
    load_dotenv_file,
)
from hypermindlabs.policy_manager import PolicyManager


PROJECT_ROOT = Path(__file__).resolve().parent
VENV_DIR = PROJECT_ROOT / ".venv"
STATE_PATH = PROJECT_ROOT / ".app_bootstrap_state.json"
REQUIREMENTS_FILE = PROJECT_ROOT / "requirements.txt"
REQUIREMENTS_STAMP = VENV_DIR / ".requirements.sha256"
CONFIG_TEMPLATE = PROJECT_ROOT / "config.empty.json"
CONFIG_FILE = PROJECT_ROOT / "config.json"
ENV_TEMPLATE = PROJECT_ROOT / ".env.example"
ENV_FILE = PROJECT_ROOT / ".env"
LOGS_DIR = PROJECT_ROOT / "logs"
WATCHDOG_LOG_DIR = LOGS_DIR / "watchdog"
POLICIES_DIR = PROJECT_ROOT / "policies" / "agent"
LIVE_LOG_BUFFER_LINE_LIMIT = 400
ANSI_ESCAPE_RE = re.compile(r"\x1B\[[0-?]*[ -/]*[@-~]")
DB_CONNINFO_PASSWORD_RE = re.compile(r"(password=)([^\s]+)", re.IGNORECASE)

DEFAULT_OLLAMA_HOST = "http://127.0.0.1:11434"
INFERENCE_KEYS = ("embedding", "generate", "chat", "tool", "multimodal")
DEFAULT_MODELS = {
    "embedding": str(DEFAULT_RUNTIME_SETTINGS.get("inference", {}).get("default_embedding_model", "nomic-embed-text:latest")),
    "generate": str(DEFAULT_RUNTIME_SETTINGS.get("inference", {}).get("default_generate_model", "llama3.2:latest")),
    "chat": str(DEFAULT_RUNTIME_SETTINGS.get("inference", {}).get("default_chat_model", "llama3.2:latest")),
    "tool": str(DEFAULT_RUNTIME_SETTINGS.get("inference", {}).get("default_tool_model", "llama3.2:latest")),
    "multimodal": str(DEFAULT_RUNTIME_SETTINGS.get("inference", {}).get("default_multimodal_model", "llama3.2-vision:latest")),
}
DEFAULT_OLLAMA_HOST = str(DEFAULT_RUNTIME_SETTINGS.get("inference", {}).get("default_ollama_host", DEFAULT_OLLAMA_HOST))
SETUP_PLACEHOLDER_VALUES = {
    "telegram_bot_username",
    "telegram_bot_token",
}
TELEGRAM_GET_ME_TIMEOUT_SECONDS = 3.0
TELEGRAM_BOT_USERNAME_CACHE_TTL_SECONDS = 300.0
_TELEGRAM_BOT_USERNAME_CACHE: dict[str, tuple[float, str | None]] = {}
COMMUNITY_SCORE_REQUIREMENTS: tuple[tuple[str, str, str], ...] = (
    ("private_chat", "minimum_community_score_private_chat", "Private chat"),
    ("private_image", "minimum_community_score_private_image", "Private image"),
    ("group_image", "minimum_community_score_group_image", "Group image"),
    ("other_group", "minimum_community_score_other_group", "Other group"),
    ("link_sharing", "minimum_community_score_link", "Link sharing"),
    ("message_forwarding", "minimum_community_score_forward", "Message forwarding"),
)

POLICY_TO_CAPABILITY_ORDER: dict[str, tuple[str, ...]] = {
    "message_analysis": ("tool", "chat", "generate"),
    "tool_calling": ("tool", "chat", "generate"),
    "chat_conversation": ("chat", "generate", "tool"),
    "dev_test": ("chat", "generate", "tool"),
}

CAPABILITY_TO_RUNTIME_MODEL_PATH: dict[str, str] = {
    "embedding": "inference.default_embedding_model",
    "generate": "inference.default_generate_model",
    "chat": "inference.default_chat_model",
    "tool": "inference.default_tool_model",
    "multimodal": "inference.default_multimodal_model",
}
LOCAL_DATABASE_HOSTS: set[str] = {"127.0.0.1", "localhost", "0.0.0.0", "::1", "::"}
POLICY_MODELS_PATH_PREFIX = "policy_models."
POLICY_MODELS_PATH_SUFFIX = ".allowed_models"


def is_windows() -> bool:
    return os.name == "nt"


def venv_python_path(venv_dir: Path) -> Path:
    if is_windows():
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


def _normalized_path(value: str | Path | None) -> Path | None:
    if value is None:
        return None
    try:
        return Path(value).expanduser().resolve()
    except (OSError, RuntimeError, ValueError):
        return None


def is_active_target_venv(
    target_venv_dir: Path,
    *,
    current_prefix: str | Path | None = None,
    virtual_env_var: str | None = None,
) -> bool:
    target = _normalized_path(target_venv_dir)
    if target is None:
        return False

    if current_prefix is None:
        current_prefix = sys.prefix
    if virtual_env_var is None:
        virtual_env_var = os.getenv("VIRTUAL_ENV")

    active_from_env = _normalized_path(virtual_env_var)
    if active_from_env is not None and active_from_env == target:
        return True

    active_from_prefix = _normalized_path(current_prefix)
    return active_from_prefix is not None and active_from_prefix == target


def run_command(
    command: list[str],
    *,
    cwd: Path | None = None,
    check: bool = False,
    capture_output: bool = False,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=str(cwd) if cwd else None,
        check=check,
        text=True,
        capture_output=capture_output,
    )


def ensure_venv_and_reexec() -> None:
    venv_python = venv_python_path(VENV_DIR)
    if not venv_python.exists():
        print(f"[bootstrap] Creating virtual environment at {VENV_DIR} ...")
        run_command([sys.executable, "-m", "venv", str(VENV_DIR)], cwd=PROJECT_ROOT, check=True)

    if not is_active_target_venv(VENV_DIR):
        target = venv_python
        print(f"[bootstrap] Re-launching under virtual environment: {target}")
        os.execv(str(target), [str(target), str(Path(__file__).resolve()), *sys.argv[1:]])


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def ensure_requirements_installed() -> None:
    if not REQUIREMENTS_FILE.exists():
        raise FileNotFoundError(f"Missing requirements file: {REQUIREMENTS_FILE}")

    expected_hash = file_sha256(REQUIREMENTS_FILE)
    current_hash = None
    if REQUIREMENTS_STAMP.exists():
        current_hash = REQUIREMENTS_STAMP.read_text(encoding="utf-8").strip()

    if current_hash == expected_hash:
        print("[bootstrap] Requirements already satisfied.")
        return

    print("[bootstrap] Installing/updating requirements ...")
    pip_python = venv_python_path(VENV_DIR)
    if not pip_python.exists():
        pip_python = Path(sys.executable)
    run_command([str(pip_python), "-m", "pip", "install", "-r", str(REQUIREMENTS_FILE)], cwd=PROJECT_ROOT, check=True)
    REQUIREMENTS_STAMP.write_text(expected_hash + "\n", encoding="utf-8")
    print("[bootstrap] Requirements installation complete.")


def load_json(path: Path, fallback: Any = None) -> Any:
    if not path.exists():
        return fallback
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError):
        return fallback


def write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
    tmp.replace(path)


def load_state() -> dict[str, Any]:
    payload = load_json(STATE_PATH, fallback={})
    if isinstance(payload, dict):
        return payload
    return {}


def save_state(state: dict[str, Any]) -> None:
    write_json_atomic(STATE_PATH, state)


def copy_if_missing(source: Path, destination: Path) -> bool:
    if destination.exists():
        return False
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return True


def ensure_project_artifacts() -> dict[str, bool]:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    WATCHDOG_LOG_DIR.mkdir(parents=True, exist_ok=True)

    created_config = copy_if_missing(CONFIG_TEMPLATE, CONFIG_FILE)
    created_env = copy_if_missing(ENV_TEMPLATE, ENV_FILE)

    if created_config:
        print(f"[bootstrap] Created config from template: {CONFIG_FILE}")
    if created_env:
        print(f"[bootstrap] Created env from template: {ENV_FILE}")

    return {
        "created_config": created_config,
        "created_env": created_env,
    }


def looks_like_placeholder_config(config_data: dict[str, Any]) -> bool:
    bot_name = str(config_data.get("bot_name", "")).strip()
    bot_token = str(config_data.get("bot_token", "")).strip()
    return bot_name in SETUP_PLACEHOLDER_VALUES or bot_token in SETUP_PLACEHOLDER_VALUES


def should_run_setup(state: dict[str, Any], artifacts: dict[str, bool], config_data: dict[str, Any]) -> bool:
    if artifacts.get("created_config"):
        return True
    if state.get("setup_completed") is not True:
        return True
    if not isinstance(config_data, dict):
        return True
    if looks_like_placeholder_config(config_data):
        return True
    return False


def run_setup_wizard(non_interactive: bool) -> bool:
    command = [sys.executable, "-m", "scripts.setup_wizard"]
    if non_interactive:
        command.append("--non-interactive")
    print(f"[bootstrap] Running setup wizard: {' '.join(command)}")
    result = run_command(command, cwd=PROJECT_ROOT, check=False)
    if result.returncode != 0:
        print(
            "[bootstrap] Setup wizard did not complete successfully. "
            "You can rerun later with: python3 -m scripts.setup_wizard"
        )
        return False
    return True


def _redact_conninfo(conninfo: str | None) -> str | None:
    if conninfo is None:
        return None
    text = str(conninfo).strip()
    if not text:
        return None
    return DB_CONNINFO_PASSWORD_RE.sub(r"\1***", text)


def _is_local_database_host(value: Any) -> bool:
    host = str(value or "").strip().lower()
    return host in LOCAL_DATABASE_HOSTS


def _database_section_complete(section: Any) -> bool:
    if not isinstance(section, dict):
        return False
    required = ("db_name", "user", "password", "host")
    for key in required:
        value = str(section.get(key) or "").strip()
        if not value:
            return False
    return True


def _fallback_enabled(config_data: dict[str, Any]) -> bool:
    fallback = config_data.get("database_fallback")
    return isinstance(fallback, dict) and bool(fallback.get("enabled", False))


def _bootstrap_target_from_config(config_data: dict[str, Any]) -> str:
    fallback = config_data.get("database_fallback")
    if isinstance(fallback, dict) and bool(fallback.get("enabled", False)) and _database_section_complete(fallback):
        return "both"
    return "primary"


def _is_local_bootstrap_candidate(config_data: dict[str, Any]) -> bool:
    database = config_data.get("database")
    if not _database_section_complete(database):
        return False
    if not _is_local_database_host(database.get("host")):
        return False

    fallback = config_data.get("database_fallback")
    if isinstance(fallback, dict) and bool(fallback.get("enabled", False)):
        if not _database_section_complete(fallback):
            return False
        if not _is_local_database_host(fallback.get("host")):
            return False
    return True


def read_database_route_status(
    *,
    config_data: dict[str, Any],
    runtime_settings: dict[str, Any],
) -> dict[str, Any]:
    status: dict[str, Any] = {
        "status": "unknown",
        "active_target": "unknown",
        "active_conninfo": None,
        "primary_conninfo": None,
        "fallback_conninfo": None,
        "primary_available": False,
        "fallback_available": False,
        "fallback_enabled": _fallback_enabled(config_data),
        "errors": [],
    }

    try:
        from hypermindlabs.database_router import DatabaseRouter
    except Exception as error:  # noqa: BLE001
        status["status"] = "error"
        status["errors"] = [f"database router import failed: {error}"]
        return status

    try:
        connect_timeout = max(1, int(get_runtime_setting(runtime_settings, "database.connect_timeout_seconds", 2)))
        router = DatabaseRouter(
            primary_database=config_data.get("database"),
            fallback_database=config_data.get("database_fallback"),
            fallback_enabled=_fallback_enabled(config_data),
            connect_timeout=connect_timeout,
        )
        route = router.resolve().to_dict()
        status.update(route)
        status["active_conninfo"] = _redact_conninfo(status.get("active_conninfo"))
        status["primary_conninfo"] = _redact_conninfo(status.get("primary_conninfo"))
        status["fallback_conninfo"] = _redact_conninfo(status.get("fallback_conninfo"))
        errors = status.get("errors")
        if not isinstance(errors, list):
            status["errors"] = [str(errors)]
        else:
            status["errors"] = [str(item) for item in errors]
    except Exception as error:  # noqa: BLE001
        status["status"] = "error"
        status["errors"] = [f"database route inspection failed: {error}"]

    return status


def _print_command_output(prefix: str, text: str, *, max_lines: int = 24) -> None:
    cleaned = str(text or "").strip()
    if not cleaned:
        return
    lines = cleaned.splitlines()
    for line in lines[:max(1, int(max_lines))]:
        print(f"{prefix}{line}")
    remaining = len(lines) - min(len(lines), max(1, int(max_lines)))
    if remaining > 0:
        print(f"{prefix}... ({remaining} additional line(s) omitted)")


def ensure_database_route_ready(
    *,
    config_data: dict[str, Any],
    runtime_settings: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    db_status = read_database_route_status(config_data=config_data, runtime_settings=runtime_settings)
    status_value = str(db_status.get("status", "unknown")).strip().lower()
    if status_value not in {"failed_all", "error"}:
        return config_data, runtime_settings, db_status

    auto_bootstrap = bool(get_runtime_setting(runtime_settings, "database.auto_bootstrap_on_app_start", True))
    if not auto_bootstrap:
        print("[db] Auto-bootstrap disabled (`runtime.database.auto_bootstrap_on_app_start=false`).")
        return config_data, runtime_settings, db_status

    if not _is_local_bootstrap_candidate(config_data):
        print("[db] Route unhealthy and local auto-bootstrap criteria not met; leaving DB config unchanged.")
        return config_data, runtime_settings, db_status

    use_docker = bool(get_runtime_setting(runtime_settings, "database.auto_bootstrap_use_docker", True))
    target = _bootstrap_target_from_config(config_data)
    command = [
        sys.executable,
        "-m",
        "scripts.bootstrap_postgres",
        "--config",
        str(CONFIG_FILE),
        "--target",
        target,
    ]
    if use_docker:
        command.append("--docker")

    print(
        "[db] Route unhealthy "
        f"(status={db_status.get('status')}, target={db_status.get('active_target')}). "
        f"Attempting auto-bootstrap (target={target}, docker={'yes' if use_docker else 'no'})."
    )
    result = run_command(command, cwd=PROJECT_ROOT, check=False, capture_output=True)
    _print_command_output("[db/bootstrap] ", result.stdout or "")
    _print_command_output("[db/bootstrap] ", result.stderr or "")

    config_data = load_config()
    runtime_settings = build_runtime_settings(config_data=config_data)
    refreshed_status = read_database_route_status(config_data=config_data, runtime_settings=runtime_settings)
    refreshed_value = str(refreshed_status.get("status", "unknown")).strip().lower()
    if result.returncode == 0 and refreshed_value in {"primary", "fallback"}:
        print(
            "[db] Auto-bootstrap recovered database route "
            f"(status={refreshed_status.get('status')}, target={refreshed_status.get('active_target')})."
        )
    elif refreshed_value in {"primary", "fallback"}:
        print(
            "[db] Database route became healthy after bootstrap attempt "
            f"(status={refreshed_status.get('status')}, target={refreshed_status.get('active_target')})."
        )
    else:
        error_count = len(refreshed_status.get("errors", []))
        print(
            "[db] WARNING: database route still unhealthy after bootstrap attempt "
            f"(status={refreshed_status.get('status')}, errors={error_count})."
        )
    return config_data, runtime_settings, refreshed_status


def ensure_database_migrations(runtime_settings: dict[str, Any]) -> None:
    auto_migrate = bool(get_runtime_setting(runtime_settings, "database.auto_migrate_on_app_start", True))
    if not auto_migrate:
        print("[db] Auto migration disabled (`runtime.database.auto_migrate_on_app_start=false`).")
        return

    try:
        from hypermindlabs.utils import ensure_startup_database_migrations
    except Exception as error:  # noqa: BLE001
        print(f"[db] WARNING: unable to import migration runner: {error}")
        return

    report = ensure_startup_database_migrations()
    route_status = str(report.get("route_status", "unknown"))
    active_target = str(report.get("active_target", "unknown"))
    connection_error = report.get("connection_error")
    core_applied = report.get("core_applied", [])
    core_failed = report.get("core_failed", [])
    vector_applied = report.get("vector_applied", [])
    vector_failed = report.get("vector_failed", [])

    if connection_error:
        print(
            "[db] WARNING: migration connection failed "
            f"(route={route_status}/{active_target}): {connection_error}"
        )
        return

    if core_failed:
        print(
            "[db] WARNING: core migration failures detected "
            f"(route={route_status}/{active_target}, failed={len(core_failed)})."
        )
        for item in core_failed:
            migration_name = str(item.get("migration", "unknown"))
            error_text = str(item.get("error", "")).splitlines()[0]
            print(f"  - {migration_name}: {error_text}")
    else:
        print(
            "[db] Core migrations ensured "
            f"(route={route_status}/{active_target}, applied={len(core_applied)})."
        )

    if vector_failed:
        print(
            "[db] Vector migrations partial/unavailable "
            f"(applied={len(vector_applied)}, failed={len(vector_failed)})."
        )
    elif vector_applied:
        print(
            "[db] Vector migrations ensured "
            f"(route={route_status}/{active_target}, applied={len(vector_applied)})."
        )


def valid_http_url(value: str | None) -> bool:
    if not value:
        return False
    parsed = urlparse(str(value).strip())
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def resolve_ollama_host(
    config_data: dict[str, Any],
    runtime_settings: dict[str, Any] | None = None,
) -> str:
    inference = config_data.get("inference", {})
    if isinstance(inference, dict):
        for key in ("tool", "chat", "generate", "embedding", "multimodal"):
            section = inference.get(key, {})
            if isinstance(section, dict):
                url = section.get("url")
                if valid_http_url(url):
                    return str(url).strip().rstrip("/")

    env_host = os.getenv("OLLAMA_HOST")
    if valid_http_url(env_host):
        return str(env_host).strip().rstrip("/")

    if runtime_settings is None:
        runtime_settings = build_runtime_settings(config_data=config_data)
    default_host = get_runtime_setting(
        runtime_settings,
        "inference.default_ollama_host",
        DEFAULT_OLLAMA_HOST,
    )
    return str(default_host).strip().rstrip("/")


def fetch_ollama_models(host: str, timeout: float = 3.0) -> tuple[list[str], str | None]:
    tags_url = host.rstrip("/") + "/api/tags"
    try:
        with urlopen(tags_url, timeout=timeout) as response:  # noqa: S310
            payload = json.loads(response.read().decode("utf-8"))
    except HTTPError as error:
        return [], f"HTTP error {error.code} from {tags_url}"
    except URLError as error:
        return [], f"Connection error to {tags_url}: {error.reason}"
    except TimeoutError:
        return [], f"Timeout while connecting to {tags_url}"
    except Exception as error:  # noqa: BLE001
        return [], f"Unexpected Ollama error: {error}"

    models: list[str] = []
    for entry in payload.get("models", []) if isinstance(payload, dict) else []:
        name = None
        if isinstance(entry, dict):
            name = entry.get("name") or entry.get("model")
        if isinstance(name, str):
            cleaned = name.strip()
            if cleaned and cleaned not in models:
                models.append(cleaned)
    return models, None


def current_model(config_data: dict[str, Any], key: str) -> str | None:
    inference = config_data.get("inference", {})
    if not isinstance(inference, dict):
        return None
    section = inference.get(key, {})
    if isinstance(section, dict):
        model = section.get("model")
        if isinstance(model, str) and model.strip():
            return model.strip()
    return None


def collect_required_models(config_data: dict[str, Any]) -> list[str]:
    output: list[str] = []

    def add(value: str | None) -> None:
        if isinstance(value, str):
            cleaned = value.strip()
            if cleaned and cleaned not in output:
                output.append(cleaned)

    # Use only models explicitly configured in config.json.
    # Do not force-pull hardcoded defaults (for example multimodal) if the
    # operator selected different models during setup.
    for key in INFERENCE_KEYS:
        add(current_model(config_data, key))

    return output


def _dedupe_models(values: list[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        cleaned = str(value or "").strip()
        if not cleaned or cleaned in seen:
            continue
        seen.add(cleaned)
        output.append(cleaned)
    return output


def _preferred_model_for_capability(
    config_data: dict[str, Any],
    *,
    runtime_settings: dict[str, Any],
    capability: str,
) -> str | None:
    configured = current_model(config_data, capability)
    if configured:
        return configured
    runtime_path = CAPABILITY_TO_RUNTIME_MODEL_PATH.get(capability)
    default_value = DEFAULT_MODELS.get(capability, DEFAULT_MODELS["chat"])
    if runtime_path:
        value = get_runtime_setting(runtime_settings, runtime_path, default_value)
    else:
        value = default_value
    cleaned = str(value or "").strip()
    return cleaned if cleaned else None


def _generated_allowed_models_for_policy(
    policy_name: str,
    *,
    config_data: dict[str, Any],
    runtime_settings: dict[str, Any],
    available_models: list[str] | None = None,
) -> list[str]:
    capability_order = POLICY_TO_CAPABILITY_ORDER.get(policy_name, ("chat", "tool", "generate"))
    preferred_models: list[str] = []
    for capability in capability_order:
        preferred = _preferred_model_for_capability(
            config_data,
            runtime_settings=runtime_settings,
            capability=capability,
        )
        if preferred:
            preferred_models.append(preferred)
    preferred_models = _dedupe_models(preferred_models)

    if preferred_models:
        return preferred_models

    discovered = _dedupe_models(list(available_models or []))
    if discovered:
        return discovered[:1]

    fallback = _preferred_model_for_capability(
        config_data,
        runtime_settings=runtime_settings,
        capability="chat",
    )
    return [fallback] if fallback else [DEFAULT_MODELS["chat"]]


def sync_agent_policies_from_runtime_models(
    config_data: dict[str, Any],
    *,
    runtime_settings: dict[str, Any],
    available_models: list[str] | None = None,
) -> tuple[int, int, list[str]]:
    inference_config = config_data.get("inference")
    manager = PolicyManager(
        inference_config=inference_config if isinstance(inference_config, dict) else {},
        endpoint_override=resolve_ollama_host(config_data, runtime_settings=runtime_settings),
    )
    policy_names = manager.list_policy_names()
    if not policy_names:
        return 0, 0, []

    changed = 0
    unchanged = 0
    failures: list[str] = []
    for policy_name in policy_names:
        policy_payload = load_json(_policy_file_path(policy_name), fallback={})
        if not isinstance(policy_payload, dict):
            policy_payload = {}
        existing_allowed = _policy_models_from_payload(policy_payload)
        generated_allowed = _generated_allowed_models_for_policy(
            policy_name,
            config_data=config_data,
            runtime_settings=runtime_settings,
            available_models=available_models,
        )
        if existing_allowed == generated_allowed:
            unchanged += 1
            continue

        save_result = manager.save_policy(
            policy_name=policy_name,
            updates={"allowed_models": generated_allowed},
            strict_model_check=False,
        )
        if save_result.saved:
            changed += 1
            continue

        reasons = save_result.report.errors or save_result.report.warnings
        reason_text = "; ".join(reasons[:3]) if reasons else "unknown error"
        failures.append(f"{policy_name}: {reason_text}")

    return changed, unchanged, failures


def policy_model_sync_fingerprint(
    config_data: dict[str, Any],
    *,
    runtime_settings: dict[str, Any],
    available_models: list[str] | None = None,
) -> str:
    policy_names = sorted(
        path.stem.removesuffix("_policy")
        for path in POLICIES_DIR.glob("*_policy.json")
        if path.stem.endswith("_policy")
    )
    model_map: dict[str, list[str]] = {}
    for policy_name in policy_names:
        model_map[policy_name] = _generated_allowed_models_for_policy(
            policy_name,
            config_data=config_data,
            runtime_settings=runtime_settings,
            available_models=available_models,
        )
    payload = {
        "host": resolve_ollama_host(config_data, runtime_settings=runtime_settings),
        "models": model_map,
    }
    return json.dumps(payload, sort_keys=True, ensure_ascii=True)


def prompt_yes_no(prompt: str, default: bool = True, non_interactive: bool = False) -> bool:
    if non_interactive:
        return default
    hint = "Y/n" if default else "y/N"
    raw = input(f"{prompt} [{hint}]: ").strip().lower()
    if raw == "":
        return default
    return raw in {"y", "yes", "1", "true"}


def can_use_curses_ui(non_interactive: bool) -> bool:
    return (not non_interactive) and (curses is not None) and sys.stdin.isatty() and sys.stdout.isatty()


def _trim_text(value: str, width: int) -> str:
    if width <= 0:
        return ""
    if len(value) <= width:
        return value
    if width <= 3:
        return value[:width]
    return value[: width - 3] + "..."


def _safe_addstr(stdscr: Any, row: int, col: int, text: str, attr: int = 0) -> None:
    height, width = stdscr.getmaxyx()
    if row < 0 or row >= height or col >= width:
        return
    clipped = _trim_text(str(text), max(0, width - col - 1))
    try:
        if attr:
            stdscr.addstr(row, col, clipped, attr)
        else:
            stdscr.addstr(row, col, clipped)
    except Exception:
        return


def _curses_prompt_yes_no(stdscr: Any, title: str, body_lines: list[str], default: bool = True) -> bool:
    selected = 0 if default else 1
    options = ["Yes", "No"]

    while True:
        stdscr.erase()
        _safe_addstr(stdscr, 0, 0, title, curses.A_BOLD if curses else 0)
        row = 2
        for line in body_lines:
            _safe_addstr(stdscr, row, 0, line)
            row += 1

        _safe_addstr(stdscr, row + 1, 0, "Use left/right arrows, Enter to confirm.")
        for idx, label in enumerate(options):
            col = idx * 10
            attr = curses.A_REVERSE if (curses and idx == selected) else 0
            _safe_addstr(stdscr, row + 3, col, label, attr)
        stdscr.refresh()

        key = stdscr.getch()
        if key in (curses.KEY_LEFT if curses else -1, ord("h")):
            selected = max(0, selected - 1)
            continue
        if key in (curses.KEY_RIGHT if curses else -1, ord("l")):
            selected = min(len(options) - 1, selected + 1)
            continue
        if key in (10, 13):
            return selected == 0


def _curses_message(stdscr: Any, title: str, lines: list[str]) -> None:
    stdscr.erase()
    _safe_addstr(stdscr, 0, 0, title, curses.A_BOLD if curses else 0)
    row = 2
    for line in lines:
        _safe_addstr(stdscr, row, 0, line)
        row += 1
    _safe_addstr(stdscr, row + 1, 0, "Press any key to continue.")
    stdscr.refresh()
    stdscr.getch()


def _curses_prompt_text(
    stdscr: Any,
    title: str,
    prompt: str,
    default: str = "",
    *,
    allow_empty: bool = False,
) -> str:
    while True:
        stdscr.erase()
        _safe_addstr(stdscr, 0, 0, title, curses.A_BOLD if curses else 0)
        _safe_addstr(stdscr, 2, 0, prompt)
        if default:
            _safe_addstr(stdscr, 3, 0, f"Default: {default}")
        elif allow_empty:
            _safe_addstr(stdscr, 3, 0, "Default: (empty)")
        _safe_addstr(stdscr, 5, 0, "> ")
        stdscr.refresh()
        if curses:
            curses.echo()
        raw = stdscr.getstr(5, 2, 256)
        if curses:
            curses.noecho()
        value = raw.decode(errors="ignore").strip()
        if value == "":
            if default:
                return default
            if allow_empty:
                return ""
            continue
        if value or allow_empty:
            return value


def _curses_select_model(stdscr: Any, models: list[str], current: str | None) -> str | None:
    if not models:
        return _curses_prompt_text(
            stdscr,
            "Model Selection",
            "No models discovered. Enter model name manually:",
            default=current or "",
        )

    options = list(models)
    try:
        index = options.index(current) if current else 0
    except ValueError:
        index = 0

    while True:
        stdscr.erase()
        _safe_addstr(stdscr, 0, 0, "Select Default Text Model", curses.A_BOLD if curses else 0)
        _safe_addstr(stdscr, 1, 0, "Capabilities affected: chat + generate + tool")
        _safe_addstr(stdscr, 2, 0, "Use up/down arrows, Enter to confirm.")
        start = max(0, index - 10)
        end = min(len(options), start + 18)
        row = 4
        for visible_idx in range(start, end):
            model_name = options[visible_idx]
            marker = "* " if model_name == current else "  "
            attr = curses.A_REVERSE if (curses and visible_idx == index) else 0
            _safe_addstr(stdscr, row, 0, marker + model_name, attr)
            row += 1
        stdscr.refresh()

        key = stdscr.getch()
        if key in (curses.KEY_UP if curses else -1, ord("k")):
            index = max(0, index - 1)
            continue
        if key in (curses.KEY_DOWN if curses else -1, ord("j")):
            index = min(len(options) - 1, index + 1)
            continue
        if key in (10, 13):
            return options[index]
        if key in (27, ord("q")):
            return current


def pull_model_with_output(model_name: str) -> tuple[bool, str]:
    if shutil.which("ollama") is None:
        return False, "'ollama' command not found on PATH"
    result = run_command(
        ["ollama", "pull", model_name],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
    )
    output_lines: list[str] = []
    for chunk in (result.stdout, result.stderr):
        if chunk:
            output_lines.extend(str(chunk).splitlines())
    tail = "\n".join(output_lines[-6:]) if output_lines else "(no output)"
    return result.returncode == 0, tail


def bootstrap_ollama_and_models_curses(
    config_data: dict[str, Any],
    *,
    runtime_settings: dict[str, Any],
) -> dict[str, Any]:
    holder = {"config": config_data}

    def _runner(stdscr: Any) -> None:
        if curses:
            curses.curs_set(0)
            stdscr.keypad(True)

        working = holder["config"]
        host = resolve_ollama_host(working, runtime_settings=runtime_settings)
        probe_timeout = float(get_runtime_setting(runtime_settings, "inference.probe_timeout_seconds", 3.0))
        models, error = fetch_ollama_models(host, timeout=probe_timeout)
        if error:
            _curses_message(
                stdscr,
                "Ollama Warning",
                [
                    f"Could not reach Ollama host: {host}",
                    error,
                    "Start/verify Ollama and rerun app.py.",
                ],
            )
            holder["config"] = working
            return

        required = collect_required_models(working)
        missing = [name for name in required if name not in models]
        if missing:
            should_pull = _curses_prompt_yes_no(
                stdscr,
                "Missing Models Detected",
                [
                    f"Host: {host}",
                    "Missing models:",
                    *[f"- {name}" for name in missing[:8]],
                    "Pull missing models now?",
                ],
                default=True,
            )
            if should_pull:
                for model_name in missing:
                    _curses_message(
                        stdscr,
                        "Pulling Model",
                        [f"Pulling {model_name}", "Please wait..."],
                    )
                    ok, details = pull_model_with_output(model_name)
                    status = "Pulled successfully." if ok else "Model pull failed."
                    _curses_message(
                        stdscr,
                        "Model Pull Result",
                        [f"Model: {model_name}", status, details],
                    )
                refreshed, refresh_error = fetch_ollama_models(host, timeout=probe_timeout)
                if refresh_error is None:
                    models = refreshed

        current_text_model = current_model(working, "chat")
        if models:
            should_select = should_prompt_model_selection(runtime_settings, current_text_model)
            if current_text_model and should_select:
                should_select = _curses_prompt_yes_no(
                    stdscr,
                    "Default Text Model",
                    [
                        f"Ollama host: {host}",
                        f"Discovered models: {len(models)}",
                        f"Current chat model: {current_text_model or '(none)'}",
                        "Select/update default text model for chat/generate/tool?",
                    ],
                    default=False,
                )
            if should_select:
                selected = _curses_select_model(stdscr, models=models, current=current_text_model)
                if selected:
                    working = apply_default_text_model(
                        working,
                        host=host,
                        model_name=selected,
                        runtime_settings=runtime_settings,
                    )
                    _curses_message(
                        stdscr,
                        "Model Updated",
                        [f"Default text model set to: {selected}"],
                    )
        else:
            _curses_message(
                stdscr,
                "No Models Found",
                [
                    f"Ollama host: {host}",
                    "No models were discovered. Setup will continue.",
                ],
            )

        holder["config"] = working

    curses.wrapper(_runner)
    return holder["config"]


def pull_model(model_name: str) -> bool:
    if shutil.which("ollama") is None:
        print("[ollama] 'ollama' command not found on PATH; cannot auto-pull models.")
        return False
    print(f"[ollama] Pulling model: {model_name}")
    result = run_command(["ollama", "pull", model_name], cwd=PROJECT_ROOT, check=False)
    return result.returncode == 0


def should_prompt_model_selection(runtime_settings: dict[str, Any], current_text_model: str | None) -> bool:
    if not current_text_model:
        return True
    return bool(
        get_runtime_setting(
            runtime_settings,
            "inference.prompt_model_selection_on_startup",
            False,
        )
    )


def _coerce_nonnegative_int(value: Any, default: int) -> int:
    try:
        parsed = int(str(value).strip())
    except (TypeError, ValueError):
        parsed = int(default)
    if parsed < 0:
        return 0
    return parsed


def community_score_requirements_configured(config_data: dict[str, Any]) -> bool:
    section = config_data.get("community_score_requirements")
    if not isinstance(section, dict):
        return False
    for config_key, _, _ in COMMUNITY_SCORE_REQUIREMENTS:
        raw_value = section.get(config_key)
        try:
            parsed = int(str(raw_value).strip())
        except (TypeError, ValueError):
            return False
        if parsed < 0:
            return False
    return True


def _community_score_defaults(runtime_settings: dict[str, Any]) -> dict[str, int]:
    defaults: dict[str, int] = {}
    telegram_defaults = DEFAULT_RUNTIME_SETTINGS.get("telegram", {})
    for config_key, runtime_key, _ in COMMUNITY_SCORE_REQUIREMENTS:
        fallback = _coerce_nonnegative_int(telegram_defaults.get(runtime_key, 0), 0)
        value = get_runtime_setting(
            runtime_settings,
            f"telegram.{runtime_key}",
            fallback,
        )
        defaults[config_key] = _coerce_nonnegative_int(value, fallback)
    return defaults


def read_community_score_requirements(
    config_data: dict[str, Any],
    *,
    runtime_settings: dict[str, Any],
) -> dict[str, int]:
    values = _community_score_defaults(runtime_settings)

    runtime_section = config_data.get("runtime")
    if isinstance(runtime_section, dict):
        telegram_section = runtime_section.get("telegram")
        if isinstance(telegram_section, dict):
            for config_key, runtime_key, _ in COMMUNITY_SCORE_REQUIREMENTS:
                if runtime_key in telegram_section:
                    values[config_key] = _coerce_nonnegative_int(
                        telegram_section.get(runtime_key),
                        values[config_key],
                    )

    config_section = config_data.get("community_score_requirements")
    if isinstance(config_section, dict):
        for config_key, _, _ in COMMUNITY_SCORE_REQUIREMENTS:
            if config_key in config_section:
                values[config_key] = _coerce_nonnegative_int(
                    config_section.get(config_key),
                    values[config_key],
                )
    return values


def apply_community_score_requirements(
    config_data: dict[str, Any],
    requirements: dict[str, Any],
    *,
    runtime_settings: dict[str, Any],
) -> dict[str, Any]:
    defaults = _community_score_defaults(runtime_settings)
    normalized: dict[str, int] = {}
    for config_key, _, _ in COMMUNITY_SCORE_REQUIREMENTS:
        normalized[config_key] = _coerce_nonnegative_int(
            requirements.get(config_key),
            defaults[config_key],
        )

    visible_section = config_data.get("community_score_requirements")
    if not isinstance(visible_section, dict):
        visible_section = {}
        config_data["community_score_requirements"] = visible_section
    visible_section.update(normalized)

    runtime_section = config_data.get("runtime")
    if not isinstance(runtime_section, dict):
        runtime_section = {}
        config_data["runtime"] = runtime_section
    telegram_section = runtime_section.get("telegram")
    if not isinstance(telegram_section, dict):
        telegram_section = {}
        runtime_section["telegram"] = telegram_section

    for config_key, runtime_key, _ in COMMUNITY_SCORE_REQUIREMENTS:
        telegram_section[runtime_key] = normalized[config_key]
    return config_data


def _community_requirements_lines(values: dict[str, Any]) -> list[str]:
    lines: list[str] = []
    for config_key, _, label in COMMUNITY_SCORE_REQUIREMENTS:
        lines.append(f"{label}: {int(values.get(config_key, 0))}")
    return lines


def _prompt_nonnegative_int(label: str, default_value: int, *, non_interactive: bool) -> int:
    if non_interactive:
        return int(default_value)
    while True:
        raw = input(f"{label} [{default_value}]: ").strip()
        if raw == "":
            return int(default_value)
        try:
            parsed = int(raw)
        except ValueError:
            print("Enter a whole number greater than or equal to 0.")
            continue
        if parsed < 0:
            print("Enter a whole number greater than or equal to 0.")
            continue
        return parsed


def _curses_prompt_nonnegative_int(
    stdscr: Any,
    title: str,
    prompt: str,
    *,
    default_value: int,
) -> int:
    while True:
        raw = _curses_prompt_text(
            stdscr,
            title,
            prompt,
            default=str(default_value),
        )
        try:
            parsed = int(str(raw).strip())
        except (TypeError, ValueError):
            _curses_message(
                stdscr,
                title,
                ["Enter a whole number greater than or equal to 0."],
            )
            continue
        if parsed < 0:
            _curses_message(
                stdscr,
                title,
                ["Enter a whole number greater than or equal to 0."],
            )
            continue
        return parsed


def bootstrap_community_score_requirements(
    config_data: dict[str, Any],
    *,
    runtime_settings: dict[str, Any],
    non_interactive: bool,
    prompt_on_startup: bool,
) -> dict[str, Any]:
    current_requirements = read_community_score_requirements(
        config_data,
        runtime_settings=runtime_settings,
    )
    config_data = apply_community_score_requirements(
        config_data,
        current_requirements,
        runtime_settings=runtime_settings,
    )

    if non_interactive:
        return config_data
    if not prompt_on_startup:
        return config_data

    print("\n[setup] Community score requirements")
    for line in _community_requirements_lines(current_requirements):
        print(f"  - {line}")

    if not prompt_yes_no(
        "Update community score requirements now?",
        default=True,
        non_interactive=non_interactive,
    ):
        return config_data

    updated: dict[str, int] = {}
    for config_key, _, label in COMMUNITY_SCORE_REQUIREMENTS:
        updated[config_key] = _prompt_nonnegative_int(
            f"{label} minimum score",
            current_requirements[config_key],
            non_interactive=False,
        )

    config_data = apply_community_score_requirements(
        config_data,
        updated,
        runtime_settings=runtime_settings,
    )
    print("[setup] Updated community score requirements.")
    return config_data


def bootstrap_community_score_requirements_curses(
    config_data: dict[str, Any],
    *,
    runtime_settings: dict[str, Any],
    prompt_on_startup: bool,
) -> dict[str, Any]:
    holder = {"config": config_data}

    def _runner(stdscr: Any) -> None:
        if curses:
            curses.curs_set(0)
            stdscr.keypad(True)

        working = holder["config"]
        current_requirements = read_community_score_requirements(
            working,
            runtime_settings=runtime_settings,
        )
        working = apply_community_score_requirements(
            working,
            current_requirements,
            runtime_settings=runtime_settings,
        )
        holder["config"] = working

        if not prompt_on_startup:
            return

        lines = [
            "Set minimum community score gates for Telegram features.",
            "",
            *_community_requirements_lines(current_requirements),
            "",
            "Do you want to update these values now?",
        ]
        should_edit = _curses_prompt_yes_no(
            stdscr,
            "Community Score Requirements",
            lines,
            default=True,
        )
        if not should_edit:
            return

        updated: dict[str, int] = {}
        for config_key, _, label in COMMUNITY_SCORE_REQUIREMENTS:
            updated[config_key] = _curses_prompt_nonnegative_int(
                stdscr,
                "Community Score Requirements",
                f"{label} minimum score",
                default_value=current_requirements[config_key],
            )

        working = apply_community_score_requirements(
            working,
            updated,
            runtime_settings=runtime_settings,
        )
        holder["config"] = working
        _curses_message(
            stdscr,
            "Community Score Requirements",
            ["Saved.", *_community_requirements_lines(updated)],
        )

    curses.wrapper(_runner)
    return holder["config"]


def choose_default_text_model(
    models: list[str],
    current: str | None,
    *,
    non_interactive: bool,
) -> str | None:
    if not models:
        return None
    if non_interactive:
        return current

    print("\nAvailable Ollama models:")
    for index, model in enumerate(models, start=1):
        marker = "*" if model == current else " "
        print(f"  {index}. {marker} {model}")

    print(
        "\nChoose default text model for capabilities: "
        "chat + generate + tool"
    )
    default_label = current if current else models[0]
    raw = input(f"Model number or name [{default_label}]: ").strip()
    if raw == "":
        return default_label
    if raw.isdigit():
        selected_index = int(raw) - 1
        if 0 <= selected_index < len(models):
            return models[selected_index]
        return default_label
    if raw in models:
        return raw
    return default_label


def apply_default_text_model(
    config_data: dict[str, Any],
    *,
    host: str,
    model_name: str,
    runtime_settings: dict[str, Any] | None = None,
) -> dict[str, Any]:
    inference = config_data.setdefault("inference", {})
    if not isinstance(inference, dict):
        inference = {}
        config_data["inference"] = inference

    for key in INFERENCE_KEYS:
        section = inference.get(key)
        if not isinstance(section, dict):
            section = {}
            inference[key] = section
        section["url"] = str(section.get("url") or host).strip().rstrip("/")

    for key in ("chat", "generate", "tool"):
        inference[key]["model"] = model_name

    if runtime_settings is None:
        runtime_settings = build_runtime_settings(config_data=config_data)
    defaultEmbeddingModel = str(
        get_runtime_setting(runtime_settings, "inference.default_embedding_model", DEFAULT_MODELS["embedding"])
    ).strip()
    defaultMultimodalModel = str(
        get_runtime_setting(runtime_settings, "inference.default_multimodal_model", DEFAULT_MODELS["multimodal"])
    ).strip()

    if not current_model(config_data, "embedding"):
        inference["embedding"]["model"] = defaultEmbeddingModel
    if not current_model(config_data, "multimodal"):
        inference["multimodal"]["model"] = defaultMultimodalModel

    return config_data


def backup_file(path: Path) -> Path | None:
    if not path.exists():
        return None
    stamp = time.strftime("%Y%m%d-%H%M%S")
    backup = path.with_suffix(path.suffix + f".bak-{stamp}")
    shutil.copy2(path, backup)
    return backup


def load_config() -> dict[str, Any]:
    payload = load_json(CONFIG_FILE, fallback={})
    if not isinstance(payload, dict):
        return {}
    return payload


@dataclass
class RouteSpec:
    key: str
    script: str
    restart_on_exit: bool = True


@dataclass
class RouteState:
    spec: RouteSpec
    desired: bool = False
    process: subprocess.Popen[str] | None = None
    log_handle: Any = None
    restart_count: int = 0
    last_exit_code: int | None = None
    restart_timestamps: list[float] = field(default_factory=list)
    last_event: str = "idle"
    last_event_epoch: float | None = None
    started_epoch: float | None = None
    external_pid: int | None = None


@dataclass
class LiveLogTailState:
    lines: deque[str] = field(default_factory=lambda: deque(maxlen=LIVE_LOG_BUFFER_LINE_LIMIT))
    cursor: int = 0
    inode: int | None = None
    device: int | None = None
    partial: str = ""
    path: str = ""


@dataclass(frozen=True)
class RouteSettingSpec:
    id: str
    label: str
    path: str
    value_type: str
    description: str
    default: Any = None
    default_runtime_path: str | None = None
    required: bool = False
    min_value: float | None = None
    max_value: float | None = None
    choices: tuple[str, ...] = ()
    sensitive: bool = False
    restart_required: bool = True
    env_override_keys: tuple[str, ...] = ()


@dataclass(frozen=True)
class RouteCategorySpec:
    id: str
    label: str
    settings: tuple[RouteSettingSpec, ...]


@dataclass(frozen=True)
class RouteConfigSpec:
    route_key: str
    label: str
    categories: tuple[RouteCategorySpec, ...]


ROUTE_CONFIG_SPECS: dict[str, RouteConfigSpec] = {
    "telegram": RouteConfigSpec(
        route_key="telegram",
        label="Telegram",
        categories=(
            RouteCategorySpec(
                id="pipeline",
                label="Pipeline",
                settings=(
                    RouteSettingSpec(
                        id="show_stage_progress",
                        label="Show Stage Progress",
                        path="runtime.telegram.show_stage_progress",
                        value_type="bool",
                        description="Show editable transient stage updates before final Telegram reply.",
                        default_runtime_path="telegram.show_stage_progress",
                        env_override_keys=("RYO_TELEGRAM_SHOW_STAGE_PROGRESS",),
                    ),
                    RouteSettingSpec(
                        id="show_stage_json_details",
                        label="Show Stage JSON Details",
                        path="runtime.telegram.show_stage_json_details",
                        value_type="bool",
                        description="Include stage JSON payloads in transient Telegram status message.",
                        default_runtime_path="telegram.show_stage_json_details",
                        env_override_keys=("RYO_TELEGRAM_SHOW_STAGE_JSON_DETAILS",),
                    ),
                    RouteSettingSpec(
                        id="stage_detail_level",
                        label="Stage Detail Level",
                        path="runtime.telegram.stage_detail_level",
                        value_type="string",
                        description="Telemetry verbosity for Telegram stage updates: minimal, normal, debug.",
                        default_runtime_path="telegram.stage_detail_level",
                        choices=("minimal", "normal", "debug"),
                        env_override_keys=("RYO_TELEGRAM_STAGE_DETAIL_LEVEL",),
                    ),
                    RouteSettingSpec(
                        id="get_updates_write_timeout",
                        label="Updates Write Timeout",
                        path="runtime.telegram.get_updates_write_timeout",
                        value_type="int",
                        description="Telegram polling write timeout in seconds.",
                        default_runtime_path="telegram.get_updates_write_timeout",
                        min_value=1,
                        max_value=3600,
                        env_override_keys=("RYO_TELEGRAM_GET_UPDATES_WRITE_TIMEOUT",),
                    ),
                ),
            ),
            RouteCategorySpec(
                id="brevity",
                label="Brevity / Topic Shift",
                settings=(
                    RouteSettingSpec(
                        id="fast_path_small_talk_enabled",
                        label="Fast Path Brevity",
                        path="runtime.orchestrator.fast_path_small_talk_enabled",
                        value_type="bool",
                        description="Allow analysis-reasoned brief turns to skip tool stage and reduce latency.",
                        default_runtime_path="orchestrator.fast_path_small_talk_enabled",
                        env_override_keys=("RYO_ORCHESTRATOR_FAST_PATH_BREVITY_ENABLED", "RYO_ORCHESTRATOR_FAST_PATH_SMALL_TALK_ENABLED"),
                    ),
                    RouteSettingSpec(
                        id="fast_path_small_talk_max_chars",
                        label="Fast Path Brevity Max Chars",
                        path="runtime.orchestrator.fast_path_small_talk_max_chars",
                        value_type="int",
                        description="Max input length eligible for analysis-reasoned brevity fast-path.",
                        default_runtime_path="orchestrator.fast_path_small_talk_max_chars",
                        min_value=1,
                        max_value=1000,
                        env_override_keys=("RYO_ORCHESTRATOR_FAST_PATH_BREVITY_MAX_CHARS", "RYO_ORCHESTRATOR_FAST_PATH_SMALL_TALK_MAX_CHARS"),
                    ),
                    RouteSettingSpec(
                        id="analysis_max_output_tokens",
                        label="Analysis Max Output Tokens",
                        path="runtime.orchestrator.analysis_max_output_tokens",
                        value_type="int",
                        description="Hard cap for message-analysis JSON generation token count.",
                        default_runtime_path="orchestrator.analysis_max_output_tokens",
                        min_value=32,
                        max_value=2048,
                        env_override_keys=("RYO_ORCHESTRATOR_ANALYSIS_MAX_OUTPUT_TOKENS",),
                    ),
                    RouteSettingSpec(
                        id="analysis_temperature",
                        label="Analysis Temperature",
                        path="runtime.orchestrator.analysis_temperature",
                        value_type="float",
                        description="Sampling temperature for message-analysis stage (lower is faster/more deterministic).",
                        default_runtime_path="orchestrator.analysis_temperature",
                        min_value=0.0,
                        max_value=1.0,
                        env_override_keys=("RYO_ORCHESTRATOR_ANALYSIS_TEMPERATURE",),
                    ),
                    RouteSettingSpec(
                        id="analysis_context_summary_max_chars",
                        label="Analysis Context Summary Max Chars",
                        path="runtime.orchestrator.analysis_context_summary_max_chars",
                        value_type="int",
                        description="Post-normalization cap for context_summary field size.",
                        default_runtime_path="orchestrator.analysis_context_summary_max_chars",
                        min_value=60,
                        max_value=2000,
                        env_override_keys=("RYO_ORCHESTRATOR_ANALYSIS_CONTEXT_SUMMARY_MAX_CHARS",),
                    ),
                    RouteSettingSpec(
                        id="topic_shift_trim_temporal_on_small_talk",
                        label="Trim Temporal On Small Talk",
                        path="runtime.topic_shift.trim_temporal_history_on_small_talk",
                        value_type="bool",
                        description="When true, reduce temporal timeline payload during short social turns.",
                        default_runtime_path="topic_shift.trim_temporal_history_on_small_talk",
                        env_override_keys=("RYO_TOPIC_SHIFT_TRIM_TEMPORAL_ON_SMALL_TALK",),
                    ),
                    RouteSettingSpec(
                        id="topic_shift_small_talk_history_limit",
                        label="Small Talk History Limit",
                        path="runtime.topic_shift.small_talk_history_limit",
                        value_type="int",
                        description="History items retained during small-talk turns.",
                        default_runtime_path="topic_shift.small_talk_history_limit",
                        min_value=0,
                        max_value=50,
                        env_override_keys=("RYO_TOPIC_SHIFT_SMALL_TALK_HISTORY_LIMIT",),
                    ),
                    RouteSettingSpec(
                        id="topic_shift_min_token_count",
                        label="Topic Shift Min Token Count",
                        path="runtime.topic_shift.min_token_count",
                        value_type="int",
                        description="Minimum token count before lexical topic-switch logic engages.",
                        default_runtime_path="topic_shift.min_token_count",
                        min_value=1,
                        max_value=100,
                        env_override_keys=("RYO_TOPIC_SHIFT_MIN_TOKEN_COUNT",),
                    ),
                    RouteSettingSpec(
                        id="topic_shift_recent_user_messages",
                        label="Recent User Messages Window",
                        path="runtime.topic_shift.recent_user_messages",
                        value_type="int",
                        description="How many recent user turns are considered when building topic memory circuit.",
                        default_runtime_path="topic_shift.recent_user_messages",
                        min_value=1,
                        max_value=100,
                        env_override_keys=("RYO_TOPIC_SHIFT_RECENT_USER_MESSAGES",),
                    ),
                ),
            ),
            RouteCategorySpec(
                id="access",
                label="Access / Score Gates",
                settings=(
                    RouteSettingSpec(
                        id="min_private_chat",
                        label="Min Score Private Chat",
                        path="community_score_requirements.private_chat",
                        value_type="int",
                        description="Minimum score required for private chat with bot.",
                        default_runtime_path="telegram.minimum_community_score_private_chat",
                        min_value=0,
                        env_override_keys=("RYO_TELEGRAM_MIN_SCORE_PRIVATE_CHAT",),
                    ),
                    RouteSettingSpec(
                        id="min_private_image",
                        label="Min Score Private Image",
                        path="community_score_requirements.private_image",
                        value_type="int",
                        description="Minimum score required for private image actions.",
                        default_runtime_path="telegram.minimum_community_score_private_image",
                        min_value=0,
                        env_override_keys=("RYO_TELEGRAM_MIN_SCORE_PRIVATE_IMAGE",),
                    ),
                    RouteSettingSpec(
                        id="min_group_image",
                        label="Min Score Group Image",
                        path="community_score_requirements.group_image",
                        value_type="int",
                        description="Minimum score required for group image actions.",
                        default_runtime_path="telegram.minimum_community_score_group_image",
                        min_value=0,
                        env_override_keys=("RYO_TELEGRAM_MIN_SCORE_GROUP_IMAGE",),
                    ),
                    RouteSettingSpec(
                        id="min_other_group",
                        label="Min Score Other Group",
                        path="community_score_requirements.other_group",
                        value_type="int",
                        description="Minimum score threshold for non-directed group content.",
                        default_runtime_path="telegram.minimum_community_score_other_group",
                        min_value=0,
                        env_override_keys=("RYO_TELEGRAM_MIN_SCORE_OTHER_GROUP",),
                    ),
                    RouteSettingSpec(
                        id="min_link",
                        label="Min Score Link Sharing",
                        path="community_score_requirements.link_sharing",
                        value_type="int",
                        description="Minimum score required for link sharing behaviors.",
                        default_runtime_path="telegram.minimum_community_score_link",
                        min_value=0,
                        env_override_keys=("RYO_TELEGRAM_MIN_SCORE_LINK",),
                    ),
                    RouteSettingSpec(
                        id="min_forward",
                        label="Min Score Message Forwarding",
                        path="community_score_requirements.message_forwarding",
                        value_type="int",
                        description="Minimum score required for message forwarding behaviors.",
                        default_runtime_path="telegram.minimum_community_score_forward",
                        min_value=0,
                        env_override_keys=("RYO_TELEGRAM_MIN_SCORE_FORWARD",),
                    ),
                ),
            ),
            RouteCategorySpec(
                id="identity",
                label="Identity",
                settings=(
                    RouteSettingSpec(
                        id="bot_name",
                        label="Bot Username",
                        path="bot_name",
                        value_type="string",
                        description="Telegram bot username used for direct link fallback.",
                        env_override_keys=("TELEGRAM_BOT_NAME",),
                    ),
                    RouteSettingSpec(
                        id="bot_id",
                        label="Bot ID",
                        path="bot_id",
                        value_type="int",
                        description="Telegram bot id used by command handlers.",
                        min_value=0,
                        env_override_keys=("TELEGRAM_BOT_ID",),
                    ),
                    RouteSettingSpec(
                        id="bot_token",
                        label="Bot Token",
                        path="bot_token",
                        value_type="secret",
                        description="Telegram bot token (sensitive).",
                        sensitive=True,
                        env_override_keys=("TELEGRAM_BOT_TOKEN",),
                    ),
                ),
            ),
            RouteCategorySpec(
                id="runtime",
                label="Runtime",
                settings=(
                    RouteSettingSpec(
                        id="default_ollama_host",
                        label="Default Ollama Host",
                        path="runtime.inference.default_ollama_host",
                        value_type="url",
                        description="Default Ollama endpoint used when route capability URLs are missing.",
                        default_runtime_path="inference.default_ollama_host",
                        required=True,
                        env_override_keys=("RYO_DEFAULT_OLLAMA_HOST", "OLLAMA_HOST"),
                    ),
                    RouteSettingSpec(
                        id="embedding_timeout_seconds",
                        label="Embedding Timeout (s)",
                        path="runtime.inference.embedding_timeout_seconds",
                        value_type="float",
                        description="Max seconds to wait for each Ollama embedding request before fallback.",
                        default_runtime_path="inference.embedding_timeout_seconds",
                        min_value=0.5,
                        max_value=300.0,
                        env_override_keys=("RYO_OLLAMA_EMBED_TIMEOUT_SECONDS",),
                    ),
                    RouteSettingSpec(
                        id="embedding_max_input_chars",
                        label="Embedding Max Input Chars",
                        path="runtime.inference.embedding_max_input_chars",
                        value_type="int",
                        description="Truncate embedding input above this size to avoid long blocking inference calls.",
                        default_runtime_path="inference.embedding_max_input_chars",
                        min_value=0,
                        max_value=200000,
                        env_override_keys=("RYO_EMBEDDING_MAX_INPUT_CHARS",),
                    ),
                ),
            ),
            RouteCategorySpec(
                id="models",
                label="Models / Profiles",
                settings=(
                    RouteSettingSpec(
                        id="inference_embedding_model",
                        label="Inference Embedding Model",
                        path="inference.embedding.model",
                        value_type="string",
                        description="Embedding model used for vector generation and retrieval.",
                        default_runtime_path="inference.default_embedding_model",
                        required=True,
                        env_override_keys=("OLLAMA_EMBED_MODEL", "RYO_DEFAULT_EMBEDDING_MODEL"),
                    ),
                    RouteSettingSpec(
                        id="inference_chat_model",
                        label="Inference Chat Model",
                        path="inference.chat.model",
                        value_type="string",
                        description="Primary chat model used by conversation responses.",
                        default_runtime_path="inference.default_chat_model",
                        required=True,
                        env_override_keys=("OLLAMA_CHAT_MODEL", "RYO_DEFAULT_CHAT_MODEL"),
                    ),
                    RouteSettingSpec(
                        id="inference_tool_model",
                        label="Inference Tool Model",
                        path="inference.tool.model",
                        value_type="string",
                        description="Model used for tool planning/execution stages.",
                        default_runtime_path="inference.default_tool_model",
                        required=True,
                        env_override_keys=("OLLAMA_TOOL_MODEL", "RYO_DEFAULT_TOOL_MODEL"),
                    ),
                    RouteSettingSpec(
                        id="inference_generate_model",
                        label="Inference Generate Model",
                        path="inference.generate.model",
                        value_type="string",
                        description="Model used for non-chat generate paths.",
                        default_runtime_path="inference.default_generate_model",
                        required=True,
                        env_override_keys=("OLLAMA_GENERATE_MODEL", "RYO_DEFAULT_GENERATE_MODEL"),
                    ),
                    RouteSettingSpec(
                        id="inference_multimodal_model",
                        label="Inference Multimodal Model",
                        path="inference.multimodal.model",
                        value_type="string",
                        description="Model used when multimodal/image capability is requested.",
                        default_runtime_path="inference.default_multimodal_model",
                        required=True,
                        env_override_keys=("OLLAMA_MULTIMODAL_MODEL", "RYO_DEFAULT_MULTIMODAL_MODEL"),
                    ),
                    RouteSettingSpec(
                        id="profile_message_analysis_models",
                        label="Profile: message_analysis",
                        path="policy_models.message_analysis.allowed_models",
                        value_type="model_list",
                        description="Comma-separated model priority chain for message analysis profile.",
                        required=True,
                    ),
                    RouteSettingSpec(
                        id="profile_tool_calling_models",
                        label="Profile: tool_calling",
                        path="policy_models.tool_calling.allowed_models",
                        value_type="model_list",
                        description="Comma-separated model priority chain for tool calling profile.",
                        required=True,
                    ),
                    RouteSettingSpec(
                        id="profile_chat_conversation_models",
                        label="Profile: chat_conversation",
                        path="policy_models.chat_conversation.allowed_models",
                        value_type="model_list",
                        description="Comma-separated model priority chain for chat conversation profile.",
                        required=True,
                    ),
                    RouteSettingSpec(
                        id="profile_dev_test_models",
                        label="Profile: dev_test",
                        path="policy_models.dev_test.allowed_models",
                        value_type="model_list",
                        description="Comma-separated model priority chain for dev_test profile.",
                        required=True,
                    ),
                ),
            ),
            RouteCategorySpec(
                id="temporal",
                label="Temporal Context",
                settings=(
                    RouteSettingSpec(
                        id="temporal_enabled",
                        label="Temporal Context Enabled",
                        path="runtime.temporal.enabled",
                        value_type="bool",
                        description="Inject structured temporal envelope into orchestrator stages.",
                        default_runtime_path="temporal.enabled",
                        env_override_keys=("RYO_TEMPORAL_CONTEXT_ENABLED",),
                    ),
                    RouteSettingSpec(
                        id="temporal_timezone",
                        label="Default Timezone",
                        path="runtime.temporal.default_timezone",
                        value_type="string",
                        description="IANA timezone used to localize naive history timestamps.",
                        default_runtime_path="temporal.default_timezone",
                        required=True,
                        env_override_keys=("RYO_TEMPORAL_DEFAULT_TIMEZONE",),
                    ),
                    RouteSettingSpec(
                        id="temporal_history_limit",
                        label="Temporal History Limit",
                        path="runtime.temporal.history_limit",
                        value_type="int",
                        description="Max recent history items exposed in temporal timeline context.",
                        default_runtime_path="temporal.history_limit",
                        min_value=0,
                        max_value=200,
                        env_override_keys=("RYO_TEMPORAL_HISTORY_LIMIT",),
                    ),
                    RouteSettingSpec(
                        id="temporal_excerpt_max_chars",
                        label="Temporal Excerpt Max Chars",
                        path="runtime.temporal.excerpt_max_chars",
                        value_type="int",
                        description="Max characters for each timeline excerpt in temporal context.",
                        default_runtime_path="temporal.excerpt_max_chars",
                        min_value=0,
                        max_value=2000,
                        env_override_keys=("RYO_TEMPORAL_EXCERPT_MAX_CHARS",),
                    ),
                ),
            ),
        ),
    ),
    "web": RouteConfigSpec(
        route_key="web",
        label="Web",
        categories=(
            RouteCategorySpec(
                id="bind_port",
                label="Bind / Port",
                settings=(
                    RouteSettingSpec(
                        id="web_host",
                        label="Bind Host",
                        path="runtime.web.host",
                        value_type="string",
                        description="Host/interface web_ui binds to.",
                        default_runtime_path="web.host",
                        required=True,
                        env_override_keys=("RYO_WEB_HOST",),
                    ),
                    RouteSettingSpec(
                        id="web_port",
                        label="Bind Port",
                        path="runtime.web.port",
                        value_type="port",
                        description="Preferred local web port (auto-increments if occupied).",
                        default_runtime_path="web.port",
                        env_override_keys=("RYO_WEB_PORT",),
                    ),
                    RouteSettingSpec(
                        id="web_port_scan_limit",
                        label="Port Scan Limit",
                        path="runtime.web.port_scan_limit",
                        value_type="int",
                        description="How many incremental ports to probe when bind port is occupied.",
                        default_runtime_path="web.port_scan_limit",
                        min_value=0,
                        max_value=2000,
                        env_override_keys=("RYO_WEB_PORT_SCAN_LIMIT",),
                    ),
                ),
            ),
            RouteCategorySpec(
                id="runtime",
                label="Runtime",
                settings=(
                    RouteSettingSpec(
                        id="web_debug",
                        label="Debug Mode",
                        path="runtime.web.debug",
                        value_type="bool",
                        description="Enable Flask debug mode for web UI route.",
                        default_runtime_path="web.debug",
                        env_override_keys=("RYO_WEB_DEBUG",),
                    ),
                    RouteSettingSpec(
                        id="web_reloader",
                        label="Use Reloader",
                        path="runtime.web.use_reloader",
                        value_type="bool",
                        description="Enable Flask auto-reloader for web UI route.",
                        default_runtime_path="web.use_reloader",
                        env_override_keys=("RYO_WEB_USE_RELOADER",),
                    ),
                ),
            ),
            RouteCategorySpec(
                id="external_url",
                label="External URL",
                settings=(
                    RouteSettingSpec(
                        id="web_ui_url",
                        label="External Web UI URL",
                        path="web_ui_url",
                        value_type="url",
                        description="Public URL used by integrations (for example Telegram miniapp).",
                        required=True,
                        env_override_keys=("WEB_UI_URL",),
                    ),
                ),
            ),
        ),
    ),
    "cli": RouteConfigSpec(
        route_key="cli",
        label="CLI",
        categories=(
            RouteCategorySpec(
                id="conversation",
                label="Conversation",
                settings=(
                    RouteSettingSpec(
                        id="knowledge_threshold",
                        label="Knowledge Lookup Word Threshold",
                        path="runtime.conversation.knowledge_lookup_word_threshold",
                        value_type="int",
                        description="Minimum prompt word count before knowledge lookup runs.",
                        default_runtime_path="conversation.knowledge_lookup_word_threshold",
                        min_value=0,
                        env_override_keys=("RYO_CONVERSATION_KNOWLEDGE_WORD_THRESHOLD",),
                    ),
                    RouteSettingSpec(
                        id="knowledge_result_limit",
                        label="Knowledge Result Limit",
                        path="runtime.conversation.knowledge_lookup_result_limit",
                        value_type="int",
                        description="Maximum knowledge retrieval records per lookup.",
                        default_runtime_path="conversation.knowledge_lookup_result_limit",
                        min_value=0,
                        env_override_keys=("RYO_CONVERSATION_KNOWLEDGE_RESULT_LIMIT",),
                    ),
                ),
            ),
            RouteCategorySpec(
                id="retrieval",
                label="Retrieval",
                settings=(
                    RouteSettingSpec(
                        id="short_history_limit",
                        label="Short History Limit",
                        path="runtime.retrieval.conversation_short_history_limit",
                        value_type="int",
                        description="Short history window used by orchestrator context preload.",
                        default_runtime_path="retrieval.conversation_short_history_limit",
                        min_value=1,
                        env_override_keys=("RYO_RETRIEVAL_SHORT_HISTORY_LIMIT",),
                    ),
                ),
            ),
            RouteCategorySpec(
                id="tool_runtime",
                label="Tool Runtime",
                settings=(
                    RouteSettingSpec(
                        id="tool_default_timeout",
                        label="Default Tool Timeout (s)",
                        path="runtime.tool_runtime.default_timeout_seconds",
                        value_type="float",
                        description="Default timeout for tool runtime execution calls.",
                        default_runtime_path="tool_runtime.default_timeout_seconds",
                        min_value=0.1,
                        env_override_keys=("RYO_TOOL_RUNTIME_DEFAULT_TIMEOUT_SECONDS",),
                    ),
                    RouteSettingSpec(
                        id="tool_default_retries",
                        label="Default Tool Retries",
                        path="runtime.tool_runtime.default_max_retries",
                        value_type="int",
                        description="Default retry attempts for tool runtime execution calls.",
                        default_runtime_path="tool_runtime.default_max_retries",
                        min_value=0,
                        env_override_keys=("RYO_TOOL_RUNTIME_DEFAULT_MAX_RETRIES",),
                    ),
                ),
            ),
        ),
    ),
    "x": RouteConfigSpec(
        route_key="x",
        label="X / Twitter",
        categories=(
            RouteCategorySpec(
                id="twitter_keys",
                label="Twitter Keys",
                settings=(
                    RouteSettingSpec(
                        id="consumer_key",
                        label="Consumer Key",
                        path="twitter_keys.consumer_key",
                        value_type="secret",
                        description="Twitter API consumer key.",
                        sensitive=True,
                        env_override_keys=("TWITTER_CONSUMER_KEY",),
                    ),
                    RouteSettingSpec(
                        id="consumer_secret",
                        label="Consumer Secret",
                        path="twitter_keys.consumer_secret",
                        value_type="secret",
                        description="Twitter API consumer secret.",
                        sensitive=True,
                        env_override_keys=("TWITTER_CONSUMER_SECRET",),
                    ),
                    RouteSettingSpec(
                        id="access_token",
                        label="Access Token",
                        path="twitter_keys.access_token",
                        value_type="secret",
                        description="Twitter access token.",
                        sensitive=True,
                        env_override_keys=("TWITTER_ACCESS_TOKEN",),
                    ),
                    RouteSettingSpec(
                        id="access_token_secret",
                        label="Access Token Secret",
                        path="twitter_keys.access_token_secret",
                        value_type="secret",
                        description="Twitter access token secret.",
                        sensitive=True,
                        env_override_keys=("TWITTER_ACCESS_TOKEN_SECRET",),
                    ),
                ),
            ),
            RouteCategorySpec(
                id="runtime",
                label="Runtime",
                settings=(
                    RouteSettingSpec(
                        id="watchdog_auto_start",
                        label="Watchdog Auto Start Routes",
                        path="runtime.watchdog.auto_start_routes",
                        value_type="bool",
                        description="Whether launcher auto-starts managed routes on dashboard start.",
                        default_runtime_path="watchdog.auto_start_routes",
                        env_override_keys=("RYO_WATCHDOG_AUTO_START_ROUTES",),
                    ),
                ),
            ),
        ),
    ),
}


_MISSING = object()


def _dot_path_parts(path: str) -> list[str]:
    return [part for part in str(path).split(".") if part]


def _policy_name_from_setting_path(path: str) -> str | None:
    path_text = str(path or "").strip()
    if not path_text.startswith(POLICY_MODELS_PATH_PREFIX):
        return None
    if not path_text.endswith(POLICY_MODELS_PATH_SUFFIX):
        return None
    policy_name = path_text[
        len(POLICY_MODELS_PATH_PREFIX) : len(path_text) - len(POLICY_MODELS_PATH_SUFFIX)
    ]
    if not policy_name:
        return None
    if "." in policy_name:
        return None
    return policy_name


def _get_config_path(config_data: dict[str, Any], path: str, default: Any = _MISSING) -> Any:
    cursor: Any = config_data
    for part in _dot_path_parts(path):
        if not isinstance(cursor, dict) or part not in cursor:
            return default
        cursor = cursor.get(part)
    return cursor


def _set_config_path(config_data: dict[str, Any], path: str, value: Any) -> None:
    parts = _dot_path_parts(path)
    if not parts:
        return
    cursor: dict[str, Any] = config_data
    for part in parts[:-1]:
        next_value = cursor.get(part)
        if not isinstance(next_value, dict):
            next_value = {}
            cursor[part] = next_value
        cursor = next_value
    cursor[parts[-1]] = value


def _setting_runtime_path(setting: RouteSettingSpec) -> str | None:
    if setting.default_runtime_path:
        return setting.default_runtime_path
    if setting.path.startswith("runtime."):
        return setting.path[len("runtime.") :]
    return None


def _empty_value_for_setting(setting: RouteSettingSpec) -> Any:
    if setting.value_type == "model_list":
        return []
    if setting.value_type in {"string", "secret", "url"}:
        return ""
    if setting.value_type == "bool":
        return False
    if setting.value_type in {"int", "port"}:
        if setting.min_value is not None:
            return int(setting.min_value)
        return 0
    if setting.value_type == "float":
        if setting.min_value is not None:
            return float(setting.min_value)
        return 0.0
    return ""


def _default_value_for_setting(setting: RouteSettingSpec) -> Any:
    policy_name = _policy_name_from_setting_path(setting.path)
    if setting.value_type == "model_list" and policy_name:
        return _generated_allowed_models_for_policy(
            policy_name,
            config_data={},
            runtime_settings=DEFAULT_RUNTIME_SETTINGS,
            available_models=None,
        )

    if setting.default is not None:
        return copy.deepcopy(setting.default)

    runtime_path = _setting_runtime_path(setting)
    if runtime_path:
        value = get_runtime_setting(DEFAULT_RUNTIME_SETTINGS, runtime_path, _MISSING)
        if value is not _MISSING:
            return copy.deepcopy(value)

    return _empty_value_for_setting(setting)


def _setting_env_override_active(setting: RouteSettingSpec) -> bool:
    if _policy_name_from_setting_path(setting.path):
        return False
    for env_key in setting.env_override_keys:
        raw = os.getenv(env_key)
        if raw is None:
            continue
        if str(raw).strip() == "":
            continue
        return True
    return False


def _resolve_setting_value(
    setting: RouteSettingSpec,
    *,
    config_data: dict[str, Any],
    pending_changes: dict[str, Any],
    runtime_settings: dict[str, Any],
) -> tuple[Any, str, bool]:
    if setting.path in pending_changes:
        value = pending_changes[setting.path]
        source = "pending"
    else:
        policy_name = _policy_name_from_setting_path(setting.path)
        if policy_name:
            payload = _read_policy_payload_for_editor(policy_name)
            explicit_policy_models = _policy_models_from_payload(payload)
            if explicit_policy_models:
                value = explicit_policy_models
                source = "policy"
            else:
                value = _generated_allowed_models_for_policy(
                    policy_name,
                    config_data=config_data,
                    runtime_settings=runtime_settings,
                    available_models=None,
                )
                source = "generated"
        else:
            configured = _get_config_path(config_data, setting.path, _MISSING)
            if configured is not _MISSING:
                value = configured
                source = "config"
            else:
                runtime_path = _setting_runtime_path(setting)
                if runtime_path:
                    value = get_runtime_setting(runtime_settings, runtime_path, _default_value_for_setting(setting))
                    source = "runtime-default"
                elif setting.default is not None:
                    value = copy.deepcopy(setting.default)
                    source = "spec-default"
                else:
                    value = _default_value_for_setting(setting)
                    source = "unset"

    env_override_active = _setting_env_override_active(setting)
    if env_override_active:
        source = f"{source}+env"
    return value, source, env_override_active


def _mask_secret(value: Any) -> str:
    text = str(value or "")
    if text == "":
        return "(empty)"
    if len(text) <= 4:
        return "*" * len(text)
    return text[:2] + ("*" * (len(text) - 4)) + text[-2:]


def _format_setting_value(value: Any, *, sensitive: bool = False) -> str:
    if sensitive:
        return _mask_secret(value)
    if value is None:
        return "(unset)"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (dict, list)):
        try:
            return json.dumps(value, ensure_ascii=True)
        except (TypeError, ValueError):
            return str(value)
    text = str(value)
    return text if text.strip() != "" else "(empty)"


def _setting_type_label(setting: RouteSettingSpec) -> str:
    if setting.value_type == "model_list":
        return "model-list"
    if setting.value_type == "bool":
        return "bool"
    if setting.value_type == "int":
        return "int"
    if setting.value_type == "float":
        return "float"
    if setting.value_type == "port":
        return "port"
    if setting.value_type == "url":
        return "url"
    if setting.value_type == "secret":
        return "secret"
    if setting.choices:
        return "enum"
    return "string"


def _coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    cleaned = str(value).strip().lower()
    if cleaned in {"1", "true", "yes", "y", "on"}:
        return True
    if cleaned in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError("expected boolean value (true/false)")


def _coerce_setting_value(setting: RouteSettingSpec, raw_value: Any) -> tuple[bool, Any, str]:
    value_type = _setting_type_label(setting)
    value: Any = raw_value

    try:
        if value_type == "bool":
            value = _coerce_bool(raw_value)
        elif value_type == "model-list":
            if isinstance(raw_value, list):
                value = _dedupe_models([str(item).strip() for item in raw_value if str(item).strip()])
            else:
                value = _parse_allowed_models_input(str(raw_value))
        elif value_type == "int":
            if isinstance(raw_value, bool):
                raise ValueError("expected integer")
            value = int(str(raw_value).strip())
        elif value_type == "float":
            if isinstance(raw_value, bool):
                raise ValueError("expected float")
            value = float(str(raw_value).strip())
        elif value_type == "port":
            if isinstance(raw_value, bool):
                raise ValueError("expected port number")
            value = int(str(raw_value).strip())
            if value < 1 or value > 65535:
                raise ValueError("port must be between 1 and 65535")
        elif value_type == "url":
            value = str(raw_value).strip()
            if setting.required and value == "":
                raise ValueError("value is required")
            if value != "" and not valid_http_url(value):
                raise ValueError("expected URL like http://127.0.0.1:11434")
        else:
            value = str(raw_value)
            if setting.value_type != "secret":
                value = value.strip()
    except ValueError as error:
        return False, None, str(error)

    if value_type in {"int", "float"}:
        numeric = float(value)
        if setting.min_value is not None and numeric < float(setting.min_value):
            return False, None, f"value must be >= {setting.min_value}"
        if setting.max_value is not None and numeric > float(setting.max_value):
            return False, None, f"value must be <= {setting.max_value}"

    if setting.choices:
        value_str = str(value)
        if value_str not in setting.choices:
            allowed = ", ".join(setting.choices)
            return False, None, f"value must be one of: {allowed}"

    if setting.required:
        if value is None:
            return False, None, "value is required"
        if isinstance(value, list) and len(value) == 0:
            return False, None, "value is required"
        if isinstance(value, str) and value.strip() == "":
            return False, None, "value is required"

    return True, value, ""


def _iter_route_settings(route_spec: RouteConfigSpec) -> list[RouteSettingSpec]:
    settings: list[RouteSettingSpec] = []
    for category in route_spec.categories:
        settings.extend(category.settings)
    return settings


def _settings_by_path(route_spec: RouteConfigSpec) -> dict[str, RouteSettingSpec]:
    return {setting.path: setting for setting in _iter_route_settings(route_spec)}


class InterfaceWatchdog:
    def __init__(
        self,
        python_exec: str,
        *,
        runtime_settings: dict[str, Any] | None = None,
        restart_window_seconds: int = 60,
        max_restarts_per_window: int = 5,
        terminate_timeout_seconds: float = 8.0,
        kill_timeout_seconds: float = 4.0,
        thread_join_timeout_seconds: float = 2.0,
    ):
        self._python_exec = python_exec
        self._routes: dict[str, RouteState] = {
            "web": RouteState(RouteSpec("web", "web_ui.py", restart_on_exit=True)),
            "telegram": RouteState(RouteSpec("telegram", "telegram_ui.py", restart_on_exit=True)),
            "cli": RouteState(RouteSpec("cli", "cli_ui.py", restart_on_exit=False)),
            "x": RouteState(RouteSpec("x", "x_ui.py", restart_on_exit=False)),
        }
        self._runtime_settings = runtime_settings if isinstance(runtime_settings, dict) else {}
        self._route_env_overrides: dict[str, dict[str, str]] = {}
        self._restart_window_seconds = restart_window_seconds
        self._max_restarts = max_restarts_per_window
        self._terminate_timeout_seconds = terminate_timeout_seconds
        self._kill_timeout_seconds = kill_timeout_seconds
        self._thread_join_timeout_seconds = thread_join_timeout_seconds
        self._lock = threading.Lock()
        self._alive = True
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()

    def route_keys(self) -> list[str]:
        return list(self._routes.keys())

    def update_runtime_settings(self, runtime_settings: dict[str, Any] | None) -> None:
        with self._lock:
            self._runtime_settings = runtime_settings if isinstance(runtime_settings, dict) else {}
            # Force fresh port resolution for future web spawns after settings edits.
            self._route_env_overrides.pop("web", None)

    @staticmethod
    def _stamp_now() -> float:
        return time.time()

    def _set_event(self, state: RouteState, message: str) -> None:
        state.last_event = message
        state.last_event_epoch = self._stamp_now()

    def _log_path(self, key: str) -> Path:
        return WATCHDOG_LOG_DIR / f"{key}.log"

    def _prepare_web_env_overrides(self) -> None:
        bind_host = _web_runtime_bind_host(self._runtime_settings)
        start_port = _web_runtime_start_port(self._runtime_settings)
        scan_limit = _web_runtime_port_scan_limit(self._runtime_settings)

        existing = self._route_env_overrides.get("web", {})
        previous_port = existing.get("RYO_WEB_RESOLVED_PORT")
        if previous_port is not None:
            start_port = _coerce_port(previous_port, start_port)

        selected_port = _find_available_port(bind_host, start_port, scan_limit)
        endpoint = f"http://{_web_public_host(bind_host)}:{selected_port}/"
        self._route_env_overrides["web"] = {
            "RYO_WEB_RESOLVED_HOST": bind_host,
            "RYO_WEB_RESOLVED_PORT": str(selected_port),
            "RYO_WEB_RESOLVED_ENDPOINT": endpoint,
        }

    def _prepare_route_env(self, key: str) -> dict[str, str]:
        if key == "web":
            self._prepare_web_env_overrides()
        env = os.environ.copy()
        env.update(self._route_env_overrides.get(key, {}))
        env.setdefault("PYTHONUNBUFFERED", "1")
        env.setdefault("PYTHONIOENCODING", "utf-8")
        env.setdefault("NO_COLOR", "1")
        env.setdefault("PY_COLORS", "0")
        env.setdefault("CLICOLOR", "0")
        env.setdefault("CLICOLOR_FORCE", "0")
        return env

    def _spawn(self, state: RouteState) -> None:
        log_path = self._log_path(state.spec.key)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        state.log_handle = log_path.open("a", encoding="utf-8", buffering=1)
        command = [self._python_exec, "-u", state.spec.script]
        launch_stamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        state.log_handle.write(
            f"\n[{launch_stamp}] [watchdog] launching route={state.spec.key} command={' '.join(command)}\n"
        )
        state.log_handle.flush()
        process_env = self._prepare_route_env(state.spec.key)
        state.process = subprocess.Popen(
            command,
            cwd=str(PROJECT_ROOT),
            env=process_env,
            stdin=subprocess.DEVNULL,
            stdout=state.log_handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
        state.last_exit_code = None
        state.started_epoch = self._stamp_now()
        self._set_event(state, f"started pid={state.process.pid}")

    def _cleanup_process(self, state: RouteState, *, close_handle: bool = True) -> None:
        state.process = None
        state.started_epoch = None
        state.external_pid = None
        if close_handle and state.log_handle is not None:
            try:
                state.log_handle.close()
            except OSError:
                pass
            state.log_handle = None

    def _can_restart(self, state: RouteState) -> bool:
        now = time.time()
        keep_after = now - self._restart_window_seconds
        state.restart_timestamps = [stamp for stamp in state.restart_timestamps if stamp >= keep_after]
        return len(state.restart_timestamps) < self._max_restarts

    def _record_restart(self, state: RouteState) -> None:
        state.restart_timestamps.append(time.time())
        state.restart_count += 1

    def start(self, key: str) -> bool:
        with self._lock:
            state = self._routes.get(key)
            if state is None:
                return False

            # Manual interface routes should be launched in a transient terminal
            # instead of a detached watchdog process with stdin=DEVNULL.
            if key in {"cli", "x"} and not bool(state.spec.restart_on_exit):
                self._cleanup_process(state)
                opened, message = _launch_script_in_transient_terminal(key, state.spec.script)
                state.desired = bool(opened)
                state.last_exit_code = None if opened else 1
                if opened:
                    detected_pid = _wait_for_external_route_pid(state.spec.script, timeout_seconds=2.0)
                    state.external_pid = detected_pid
                    if detected_pid is not None:
                        self._set_event(state, f"running external pid={detected_pid}")
                    else:
                        self._set_event(state, message)
                else:
                    self._set_event(state, f"launch failed: {message}")
                return opened

            state.desired = True
            if state.process is not None and state.process.poll() is None:
                self._set_event(state, "already running")
                return True
            self._cleanup_process(state)
            try:
                self._spawn(state)
            except Exception as error:  # noqa: BLE001
                state.desired = False
                self._set_event(state, f"failed to start: {error}")
                self._cleanup_process(state)
                return False
            return True

    def stop(self, key: str) -> bool:
        with self._lock:
            state = self._routes.get(key)
            if state is None:
                return False
            state.desired = False
            process = state.process
            if process is None and not bool(state.spec.restart_on_exit):
                pids = _discover_external_route_pids(state.spec.script)
                if not pids:
                    self._cleanup_process(state)
                    self._set_event(state, "already stopped")
                    return True

                terminated: list[int] = []
                failed: list[int] = []
                for pid in pids:
                    if _terminate_external_pid(
                        pid,
                        terminate_timeout_seconds=self._terminate_timeout_seconds,
                        kill_timeout_seconds=self._kill_timeout_seconds,
                    ):
                        terminated.append(pid)
                    else:
                        failed.append(pid)

                if terminated:
                    state.last_exit_code = 0
                    state.external_pid = None
                    self._set_event(
                        state,
                        "stopped external pid(s): " + ", ".join(str(pid) for pid in terminated[:3]),
                    )
                else:
                    state.last_exit_code = 1
                    self._set_event(
                        state,
                        "failed stopping external pid(s): " + ", ".join(str(pid) for pid in failed[:3]),
                    )
                return len(failed) == 0

            if process is None:
                self._cleanup_process(state)
                self._set_event(state, "already stopped")
                return True

            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=self._terminate_timeout_seconds)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=self._kill_timeout_seconds)
            state.last_exit_code = process.returncode
            self._set_event(state, f"stopped exit={process.returncode}")
            self._cleanup_process(state)
            return True

    def toggle(self, key: str) -> bool:
        manual_script: str | None = None
        with self._lock:
            state = self._routes.get(key)
            if state is None:
                return False
            if not bool(state.spec.restart_on_exit):
                manual_script = state.spec.script
        if manual_script is not None:
            running_manual = bool(_discover_external_route_pids(manual_script))
            if running_manual:
                return self.stop(key)
            return self.start(key)
        status = self.status()
        entry = status.get(key)
        if entry is None:
            return False
        if entry["desired"]:
            return self.stop(key)
        return self.start(key)

    def start_all(self, *, include_manual: bool = True) -> None:
        for key in self.route_keys():
            state = self._routes.get(key)
            if state is None:
                continue
            if not include_manual and not bool(state.spec.restart_on_exit):
                continue
            self.start(key)

    def stop_all(self) -> None:
        for key in self.route_keys():
            self.stop(key)

    def status(self) -> dict[str, dict[str, Any]]:
        with self._lock:
            output: dict[str, dict[str, Any]] = {}
            for key, state in self._routes.items():
                running = state.process is not None and state.process.poll() is None
                pid = state.process.pid if running else None
                uptime_seconds = (
                    int(max(0.0, self._stamp_now() - state.started_epoch))
                    if running and state.started_epoch
                    else None
                )
                if (not running) and (not bool(state.spec.restart_on_exit)):
                    external_pids = _discover_external_route_pids(state.spec.script)
                    if external_pids:
                        pid = external_pids[0]
                        running = True
                        state.external_pid = pid
                        uptime_seconds = _pid_uptime_seconds(pid)
                    else:
                        state.external_pid = None

                desired = state.desired
                if not bool(state.spec.restart_on_exit):
                    desired = running
                output[key] = {
                    "desired": desired,
                    "running": running,
                    "pid": pid,
                    "process_user": _pid_username(pid if running else None),
                    "script": state.spec.script,
                    "restart_count": state.restart_count,
                    "last_exit_code": state.last_exit_code,
                    "restart_on_exit": state.spec.restart_on_exit,
                    "uptime_seconds": uptime_seconds,
                    "last_event": state.last_event,
                    "last_event_epoch": state.last_event_epoch,
                    "log_file": str(self._log_path(key)),
                    "endpoint_url": self._route_env_overrides.get(key, {}).get("RYO_WEB_RESOLVED_ENDPOINT"),
                }
            return output

    def _monitor_loop(self) -> None:
        while self._alive:
            with self._lock:
                for state in self._routes.values():
                    process = state.process
                    if process is None:
                        continue
                    exit_code = process.poll()
                    if exit_code is None:
                        continue

                    state.last_exit_code = exit_code
                    self._set_event(state, f"exited code={exit_code}")
                    self._cleanup_process(state)

                    if state.desired and state.spec.restart_on_exit:
                        if self._can_restart(state):
                            self._record_restart(state)
                            self._spawn(state)
                            self._set_event(
                                state,
                                f"restarted after exit={exit_code} (restart #{state.restart_count})",
                            )
                        else:
                            state.desired = False
                            self._set_event(
                                state,
                                "disabled after hitting restart window limit",
                            )
                    elif state.desired and not state.spec.restart_on_exit:
                        # One-shot routes (for example CLI/X in current codebase) are
                        # intentionally not auto-restarted.
                        state.desired = False
                        self._set_event(state, f"exited code={exit_code} (manual route)")
            time.sleep(1.0)

    def shutdown(self) -> None:
        self._alive = False
        self.stop_all()
        self._thread.join(timeout=self._thread_join_timeout_seconds)


def _clear_screen() -> None:
    if is_windows():
        os.system("cls")
    else:
        print("\033[2J\033[H", end="")


def _format_duration(seconds: int | None) -> str:
    if seconds is None:
        return "-"
    seconds = max(0, int(seconds))
    hours, rem = divmod(seconds, 3600)
    minutes, sec = divmod(rem, 60)
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{sec:02d}"
    return f"{minutes:02d}:{sec:02d}"


def _proc_path_for_pid(pid: int, suffix: str) -> Path:
    return Path("/proc") / str(int(pid)) / suffix


def _pid_exists(pid: int | None) -> bool:
    if pid is None:
        return False
    try:
        return _proc_path_for_pid(int(pid), "").exists()
    except (TypeError, ValueError):
        return False


def _pid_cmdline(pid: int) -> list[str]:
    path = _proc_path_for_pid(pid, "cmdline")
    try:
        payload = path.read_bytes()
    except OSError:
        return []
    if not payload:
        return []
    values = []
    for raw in payload.split(b"\x00"):
        if not raw:
            continue
        values.append(raw.decode("utf-8", errors="replace"))
    return values


def _pid_cwd(pid: int) -> Path | None:
    path = _proc_path_for_pid(pid, "cwd")
    try:
        return path.resolve()
    except OSError:
        return None


def _pid_uptime_seconds(pid: int) -> int | None:
    stat_path = _proc_path_for_pid(pid, "stat")
    try:
        stat_line = stat_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None

    try:
        close_idx = stat_line.rfind(")")
        if close_idx < 0:
            return None
        fields = stat_line[close_idx + 2 :].split()
        start_ticks = int(fields[19])
        clk_tck = int(os.sysconf("SC_CLK_TCK"))
        uptime_raw = Path("/proc/uptime").read_text(encoding="utf-8", errors="replace").split()[0]
        uptime_total = float(uptime_raw)
        return max(0, int(uptime_total - (start_ticks / float(clk_tck))))
    except (IndexError, ValueError, OSError):
        return None


def _arg_matches_script(arg: str, script: str) -> bool:
    script_name = Path(script).name
    cleaned = str(arg or "").strip().strip("'\"")
    if cleaned == "":
        return False
    candidate = Path(cleaned)
    if candidate.name == script_name:
        return True
    if candidate.is_absolute():
        try:
            return candidate.resolve() == (PROJECT_ROOT / script_name).resolve()
        except OSError:
            return False
    return False


def _python_script_arg_from_argv(argv: list[str]) -> str | None:
    if not argv:
        return None
    exe_name = Path(str(argv[0])).name.lower()
    if "python" not in exe_name:
        return None

    for arg in argv[1:]:
        token = str(arg or "").strip()
        if token == "":
            continue
        if token in {"-c", "-m"}:
            return None
        if token.startswith("-"):
            continue
        return token
    return None


def _discover_external_route_pids(script: str) -> list[int]:
    proc_root = Path("/proc")
    if not proc_root.exists():
        return []
    try:
        project_root = PROJECT_ROOT.resolve()
    except OSError:
        project_root = PROJECT_ROOT
    script_name = Path(script).name
    script_abs = (PROJECT_ROOT / script_name).resolve()
    this_pid = os.getpid()
    matches: list[int] = []
    for entry in proc_root.iterdir():
        if not entry.name.isdigit():
            continue
        try:
            pid = int(entry.name)
        except ValueError:
            continue
        if pid == this_pid:
            continue
        argv = _pid_cmdline(pid)
        if len(argv) < 2:
            continue
        script_arg = _python_script_arg_from_argv(argv)
        if script_arg is None:
            continue
        if not _arg_matches_script(script_arg, script_name):
            continue

        accepted = False
        script_arg_path = Path(script_arg)
        if script_arg_path.is_absolute():
            try:
                if script_arg_path.resolve() == script_abs:
                    accepted = True
            except OSError:
                pass

        if not accepted:
            cwd = _pid_cwd(pid)
            if cwd is not None:
                try:
                    cwd.relative_to(project_root)
                    accepted = True
                except ValueError:
                    accepted = False

        if not accepted:
            try:
                argv0_path = Path(argv[0]).resolve()
                if argv0_path == (PROJECT_ROOT / ".venv" / "bin" / "python").resolve():
                    accepted = True
            except OSError:
                accepted = False

        if not accepted:
            continue
        matches.append(pid)
    return sorted(matches, reverse=True)


def _wait_for_external_route_pid(script: str, timeout_seconds: float = 2.0) -> int | None:
    deadline = time.time() + max(0.2, float(timeout_seconds))
    while time.time() <= deadline:
        pids = _discover_external_route_pids(script)
        if pids:
            return pids[0]
        time.sleep(0.05)
    return None


def _terminate_external_pid(pid: int, *, terminate_timeout_seconds: float, kill_timeout_seconds: float) -> bool:
    if not _pid_exists(pid):
        return True
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        return True
    except OSError:
        return False

    deadline = time.time() + max(0.1, float(terminate_timeout_seconds))
    while time.time() <= deadline:
        if not _pid_exists(pid):
            return True
        time.sleep(0.05)

    try:
        os.kill(pid, signal.SIGKILL)
    except ProcessLookupError:
        return True
    except OSError:
        return False

    kill_deadline = time.time() + max(0.1, float(kill_timeout_seconds))
    while time.time() <= kill_deadline:
        if not _pid_exists(pid):
            return True
        time.sleep(0.05)
    return not _pid_exists(pid)


def _pid_username(pid: int | None) -> str:
    if pid is None:
        return "-"
    try:
        status = os.stat(f"/proc/{int(pid)}")
    except (OSError, ValueError, TypeError):
        return "-"

    uid = status.st_uid
    if pwd is None:
        return str(uid)
    try:
        return pwd.getpwuid(uid).pw_name
    except KeyError:
        return str(uid)


def _postgres_links(config_data: dict[str, Any]) -> tuple[str, str | None]:
    database = config_data.get("database")
    if not isinstance(database, dict):
        database = {}
    primary_user = str(database.get("user") or "postgres").strip() or "postgres"
    primary_host = str(database.get("host") or "127.0.0.1").strip() or "127.0.0.1"
    primary_port = str(database.get("port") or "5432").strip() or "5432"
    primary_name = str(database.get("db_name") or "postgres").strip() or "postgres"
    primary = f"postgresql://{primary_user}@{primary_host}:{primary_port}/{primary_name}"

    fallback = config_data.get("database_fallback")
    if not isinstance(fallback, dict) or not bool(fallback.get("enabled", False)):
        return primary, None

    fallback_user = str(fallback.get("user") or primary_user).strip() or primary_user
    fallback_host = str(fallback.get("host") or "127.0.0.1").strip() or "127.0.0.1"
    fallback_port = str(fallback.get("port") or "5433").strip() or "5433"
    fallback_name = str(fallback.get("db_name") or primary_name).strip() or primary_name
    fallback_link = f"postgresql://{fallback_user}@{fallback_host}:{fallback_port}/{fallback_name}"
    return primary, fallback_link


def _database_status_label(db_status: dict[str, Any] | None) -> str:
    if not isinstance(db_status, dict):
        return "unknown"

    status_value = str(db_status.get("status", "unknown")).strip().lower()
    active_target = str(db_status.get("active_target", "unknown")).strip().lower()
    primary_available = bool(db_status.get("primary_available", False))
    fallback_available = bool(db_status.get("fallback_available", False))

    if status_value == "primary":
        return f"primary healthy (active={active_target or 'primary'})"
    if status_value == "fallback":
        return f"fallback healthy (active={active_target or 'fallback'})"
    if status_value == "failed_all":
        return (
            f"unhealthy (primary={'up' if primary_available else 'down'}, "
            f"fallback={'up' if fallback_available else 'down'})"
        )
    if status_value == "error":
        return "route-inspection error"
    return status_value or "unknown"


def _database_errors_preview(db_status: dict[str, Any] | None, *, limit: int = 2) -> str:
    if not isinstance(db_status, dict):
        return ""
    errors = db_status.get("errors")
    if not isinstance(errors, list):
        return ""
    chunks = [str(item).strip() for item in errors if str(item).strip()]
    if not chunks:
        return ""
    preview = "; ".join(chunks[: max(1, int(limit))])
    extra = len(chunks) - min(len(chunks), max(1, int(limit)))
    if extra > 0:
        preview = f"{preview}; +{extra} more"
    return preview


def _route_runtime_state(entry: dict[str, Any]) -> str:
    if entry.get("running"):
        return "running"
    if entry.get("desired") and not entry.get("restart_on_exit"):
        return "exited (manual)"
    if entry.get("desired") and entry.get("restart_on_exit"):
        return "awaiting restart"
    if entry.get("last_exit_code") is not None:
        return "stopped (exited)"
    return "stopped"


def _parse_route_token(token: str, keys: list[str]) -> str | None:
    cleaned = str(token or "").strip().lower()
    if not cleaned:
        return None
    if cleaned.isdigit():
        idx = int(cleaned) - 1
        if 0 <= idx < len(keys):
            return keys[idx]
        return None
    if cleaned in keys:
        return cleaned
    return None


def _tail_log_lines(log_path: Path, line_count: int = 40) -> list[str]:
    if not log_path.exists():
        return [f"(log file not found: {log_path})"]
    try:
        lines = log_path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        return [f"(failed reading log file: {error})"]
    if not lines:
        return ["(log is empty)"]
    return [_strip_ansi(line) for line in lines[-max(1, int(line_count)) :]]


def _strip_ansi(value: Any) -> str:
    text = str(value if value is not None else "")
    return ANSI_ESCAPE_RE.sub("", text)


def _append_live_log_line(state: LiveLogTailState, line: str) -> None:
    cleaned = _strip_ansi(line).replace("\t", "    ").strip("\r\n")
    if cleaned == "":
        return
    state.lines.append(cleaned)


def _update_live_log_tail(state: LiveLogTailState, log_path: Path) -> None:
    state.path = str(log_path)
    if not log_path.exists():
        state.cursor = 0
        state.inode = None
        state.device = None
        state.partial = ""
        if not state.lines or not state.lines[-1].startswith("(log file not found:"):
            state.lines.clear()
            state.lines.append(f"(log file not found: {log_path})")
        return

    try:
        stat = log_path.stat()
    except OSError as error:
        if not state.lines or not state.lines[-1].startswith("(failed reading log file:"):
            state.lines.append(f"(failed reading log file: {error})")
        return

    current_inode = int(getattr(stat, "st_ino", 0) or 0)
    current_device = int(getattr(stat, "st_dev", 0) or 0)

    rotated = False
    if state.inode is None or state.device is None:
        state.inode = current_inode
        state.device = current_device
    elif state.inode != current_inode or state.device != current_device:
        rotated = True
    elif int(stat.st_size) < int(state.cursor):
        rotated = True

    if rotated:
        state.cursor = 0
        state.partial = ""
        state.inode = current_inode
        state.device = current_device
        _append_live_log_line(state, "[watchdog] log rotated/truncated; tail restarted")

    try:
        with log_path.open("r", encoding="utf-8", errors="replace") as handle:
            handle.seek(max(0, int(state.cursor)))
            chunk = handle.read()
            state.cursor = int(handle.tell())
    except OSError as error:
        if not state.lines or not state.lines[-1].startswith("(failed reading log file:"):
            state.lines.append(f"(failed reading log file: {error})")
        return

    if not chunk:
        return

    text = f"{state.partial}{chunk}"
    segments = text.splitlines(keepends=True)
    state.partial = ""
    for segment in segments:
        if segment.endswith("\n") or segment.endswith("\r"):
            _append_live_log_line(state, segment)
        else:
            state.partial = segment

    if state.partial and len(state.partial) > 1200:
        _append_live_log_line(state, _trim_text(state.partial, 1200))
        state.partial = ""


def _live_log_lines_for_display(state: LiveLogTailState, *, max_lines: int) -> list[str]:
    lines = list(state.lines)
    if state.partial.strip():
        lines.append(_trim_text(state.partial.strip(), 1200) + " …")
    if not lines:
        lines = ["(no log output yet)"]
    return lines[-max(1, int(max_lines)) :]


def _infer_route_agent_state(entry: dict[str, Any], live_log_lines: list[str]) -> str:
    if not entry.get("running"):
        return _route_runtime_state(entry)

    lowered = [str(line).lower() for line in live_log_lines]
    for line in reversed(lowered):
        if "orchestrator.complete" in line or "completed end-to-end orchestration" in line:
            return "orchestrator complete"
        if "response.complete" in line or "final response generated" in line:
            return "response complete"
        if "response.start" in line or "generating final response" in line:
            return "response generation"
        if "tools.complete" in line or "tool execution stage complete" in line:
            return "tools complete"
        if "tools.start" in line or "evaluating tool calls" in line:
            return "tool execution"
        if "analysis.complete" in line or "analysis stage complete" in line:
            return "analysis complete"
        if "analysis.start" in line or "running message analysis" in line:
            return "analysis running"
        if "context.built" in line or "known context assembled" in line:
            return "context built"
        if "orchestrator.start" in line or "accepted request and preparing context" in line:
            return "request accepted"
    return "running (no stage markers yet)"


def _coerce_port(value: Any, default: int) -> int:
    try:
        port = int(str(value).strip())
    except (TypeError, ValueError):
        port = int(default)
    if port <= 0:
        port = int(default)
    if port > 65535:
        return 65535
    return port


def _normalize_web_bind_host(value: Any) -> str:
    host = str(value if value is not None else "").strip()
    if host == "::":
        return "0.0.0.0"
    return host if host else "127.0.0.1"


def _web_public_host(bind_host: str) -> str:
    cleaned = _normalize_web_bind_host(bind_host)
    if cleaned in {"0.0.0.0", "::"}:
        return "127.0.0.1"
    return cleaned


def _web_runtime_bind_host(runtime_settings: dict[str, Any]) -> str:
    return _normalize_web_bind_host(get_runtime_setting(runtime_settings, "web.host", "127.0.0.1"))


def _web_runtime_start_port(runtime_settings: dict[str, Any]) -> int:
    raw = get_runtime_setting(runtime_settings, "web.port", 4747)
    return _coerce_port(raw, 4747)


def _web_runtime_port_scan_limit(runtime_settings: dict[str, Any]) -> int:
    raw = get_runtime_setting(runtime_settings, "web.port_scan_limit", 100)
    try:
        return max(0, int(raw))
    except (TypeError, ValueError):
        return 100


def _web_local_endpoint_from_runtime(runtime_settings: dict[str, Any]) -> str:
    bind_host = _web_runtime_bind_host(runtime_settings)
    port = _web_runtime_start_port(runtime_settings)
    return f"http://{_web_public_host(bind_host)}:{port}/"


def _is_tcp_port_available(bind_host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            probe.bind((bind_host, int(port)))
        except OSError:
            return False
    return True


def _find_available_port(bind_host: str, start_port: int, scan_limit: int) -> int:
    attempts = max(0, int(scan_limit))
    current = _coerce_port(start_port, 4747)
    for _ in range(attempts + 1):
        if _is_tcp_port_available(bind_host, current):
            return current
        if current >= 65535:
            break
        current += 1
    raise RuntimeError(
        f"No available TCP port for host={bind_host}, start_port={start_port}, scan_limit={scan_limit}"
    )


def _normalized_telegram_bot_username(config_data: dict[str, Any]) -> str | None:
    raw = str(config_data.get("bot_name", "")).strip()
    if not raw:
        return None
    if raw in SETUP_PLACEHOLDER_VALUES:
        return None
    if raw.startswith("@"):
        raw = raw[1:]
    cleaned = raw.strip()
    return cleaned or None


def _normalized_telegram_bot_token(config_data: dict[str, Any]) -> str | None:
    raw = str(config_data.get("bot_token", "")).strip()
    if not raw:
        return None
    if raw in SETUP_PLACEHOLDER_VALUES:
        return None
    return raw


def _telegram_username_from_bot_token(bot_token: str) -> str | None:
    token = str(bot_token or "").strip()
    if not token:
        return None

    cache_key = hashlib.sha256(token.encode("utf-8")).hexdigest()
    now = time.time()
    cached = _TELEGRAM_BOT_USERNAME_CACHE.get(cache_key)
    if cached is not None:
        expires_at, cached_username = cached
        if now < expires_at:
            return cached_username

    url = f"https://api.telegram.org/bot{token}/getMe"
    username: str | None = None
    try:
        with urlopen(url, timeout=TELEGRAM_GET_ME_TIMEOUT_SECONDS) as response:  # noqa: S310
            payload = json.loads(response.read().decode("utf-8"))
    except (HTTPError, URLError, TimeoutError, ValueError, json.JSONDecodeError):
        payload = None
    except Exception:  # noqa: BLE001
        payload = None
    else:
        if isinstance(payload, dict) and bool(payload.get("ok")):
            result = payload.get("result")
            if isinstance(result, dict):
                raw_username = str(result.get("username") or "").strip()
                if raw_username:
                    username = raw_username.lstrip("@")

    _TELEGRAM_BOT_USERNAME_CACHE[cache_key] = (
        now + TELEGRAM_BOT_USERNAME_CACHE_TTL_SECONDS,
        username,
    )
    return username


def _telegram_bot_link(config_data: dict[str, Any]) -> str | None:
    token = _normalized_telegram_bot_token(config_data)
    if token:
        derived_username = _telegram_username_from_bot_token(token)
        if derived_username:
            return f"https://t.me/{derived_username}"

    username = _normalized_telegram_bot_username(config_data)
    if not username:
        return None
    return f"https://t.me/{username}"


def _route_access_summary(
    route_key: str,
    config_data: dict[str, Any],
    runtime_settings: dict[str, Any],
    *,
    entry: dict[str, Any] | None = None,
) -> str:
    if route_key == "web":
        local_url = None
        if isinstance(entry, dict):
            local_url = str(entry.get("endpoint_url") or "").strip() or None
        if not valid_http_url(local_url):
            local_url = _web_local_endpoint_from_runtime(runtime_settings)
        return f"local web url: {local_url}"
    if route_key == "telegram":
        link = _telegram_bot_link(config_data)
        return (
            f"telegram link: {link}"
            if link
            else "telegram link unavailable (set bot_token or bot_name in config.json)"
        )
    if route_key == "cli":
        return "interactive terminal launch available (press r or start)"
    if route_key == "x":
        return "interactive terminal launch available (press r or start)"
    return "no interface action available"


def _open_url_with_system_handler(url: str) -> tuple[bool, str]:
    if not valid_http_url(url):
        return False, f"invalid URL: {url}"
    candidates = (
        ["xdg-open", url],
        ["open", url],
    )
    last_error = ""
    for command in candidates:
        executable = command[0]
        if shutil.which(executable) is None:
            continue
        try:
            subprocess.Popen(
                command,
                cwd=str(PROJECT_ROOT),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
            return True, f"opened in default browser via '{executable}'"
        except OSError as error:
            last_error = str(error)
    if last_error:
        return False, f"failed launching browser handler: {last_error}"
    return False, "no supported URL opener found (tried: xdg-open, open)"


def _launch_script_in_transient_terminal(route_key: str, script: str) -> tuple[bool, str]:
    script_path = PROJECT_ROOT / script
    if not script_path.exists():
        return False, f"missing script file: {script_path}"

    title = f"RYO {route_key} interface"
    quoted_root = shlex.quote(str(PROJECT_ROOT))
    quoted_python = shlex.quote(str(sys.executable))
    quoted_script = shlex.quote(script)
    shell_command = (
        f"cd {quoted_root} && {quoted_python} -u {quoted_script}; "
        "status=$?; "
        "printf '\\n[launcher] process exited with code %s. Press Enter to close...' \"$status\"; "
        "read -r _"
    )

    candidates: list[list[str]] = []
    if shutil.which("x-terminal-emulator") is not None:
        candidates.append(["x-terminal-emulator", "-T", title, "-e", "bash", "-lc", shell_command])
    if shutil.which("gnome-terminal") is not None:
        candidates.append(["gnome-terminal", "--title", title, "--", "bash", "-lc", shell_command])
    if shutil.which("konsole") is not None:
        candidates.append(["konsole", "-p", f"tabtitle={title}", "-e", "bash", "-lc", shell_command])
    if shutil.which("xterm") is not None:
        candidates.append(["xterm", "-T", title, "-e", "bash", "-lc", shell_command])

    if not candidates:
        return False, "no supported terminal emulator found (x-terminal-emulator, gnome-terminal, konsole, xterm)"

    errors: list[str] = []
    for command in candidates:
        try:
            subprocess.Popen(
                command,
                cwd=str(PROJECT_ROOT),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
            return True, f"launched {script} in new terminal"
        except OSError as error:
            errors.append(f"{command[0]}: {error}")
    return False, "failed launching terminal: " + " | ".join(errors)


def _route_open_action(
    route_key: str,
    entry: dict[str, Any],
    config_data: dict[str, Any],
    runtime_settings: dict[str, Any],
) -> tuple[bool, list[str]]:
    if route_key in {"cli", "x"}:
        script = str(entry.get("script") or f"{route_key}_ui.py")
        ok, message = _launch_script_in_transient_terminal(route_key, script)
        return ok, [f"Route: {route_key}", f"Script: {script}", message]

    if route_key == "web":
        web_url = str(entry.get("endpoint_url") or "").strip()
        if not valid_http_url(web_url):
            web_url = _web_local_endpoint_from_runtime(runtime_settings)
        opened, opener_message = _open_url_with_system_handler(web_url)
        return opened, [
            "Route: web",
            f"Local Web UI URL: {web_url}",
            opener_message if opened else f"Open manually: {web_url}",
        ]

    if route_key == "telegram":
        tg_link = _telegram_bot_link(config_data)
        if tg_link is None:
            return False, [
                "Route: telegram",
                "Could not determine Telegram bot link.",
                "Set a valid bot_token (preferred) or bot_name in config.json, then relaunch app.py.",
            ]
        opened, opener_message = _open_url_with_system_handler(tg_link)
        return opened, [
            "Route: telegram",
            f"Bot chat URL: {tg_link}",
            opener_message if opened else f"Open manually: {tg_link}",
        ]

    return False, [f"Route: {route_key}", "No open action implemented."]


def print_launcher_summary(
    config_data: dict[str, Any],
    runtime_settings: dict[str, Any],
    *,
    db_status: dict[str, Any] | None = None,
) -> None:
    ollama_host = resolve_ollama_host(config_data, runtime_settings=runtime_settings)
    primary_pg_link, fallback_pg_link = _postgres_links(config_data)
    chat_model = current_model(config_data, "chat") or "-"
    tool_model = current_model(config_data, "tool") or "-"
    generate_model = current_model(config_data, "generate") or "-"
    embedding_model = current_model(config_data, "embedding") or "-"
    watchdog_window = int(get_runtime_setting(runtime_settings, "watchdog.restart_window_seconds", 60))
    watchdog_restarts = int(get_runtime_setting(runtime_settings, "watchdog.max_restarts_per_window", 5))
    print("=== RYO Launcher ===")
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Config file:  {CONFIG_FILE}")
    print(f"Ollama host:  {ollama_host}")
    print(f"Postgres:     {primary_pg_link}")
    if fallback_pg_link:
        print(f"Postgres fb:  {fallback_pg_link}")
    print(f"DB route:     {_database_status_label(db_status)}")
    db_errors = _database_errors_preview(db_status, limit=2)
    if db_errors:
        print(f"DB errors:    {db_errors}")
    print(
        "Models: "
        f"chat={chat_model} | tool={tool_model} | generate={generate_model} | embedding={embedding_model}"
    )
    print(
        "Watchdog policy: "
        f"max {watchdog_restarts} restart(s) per {watchdog_window}s window (auto routes)"
    )


def print_watchdog_status(watchdog: InterfaceWatchdog) -> None:
    status = watchdog.status()
    print("\n=== Interface Watchdog ===")
    print("Route      Script         Desired  Running  PID      User      Uptime   Restarts  LastExit  Policy  State")
    print("----------------------------------------------------------------------------------------------------------")
    for key in watchdog.route_keys():
        entry = status[key]
        desired = "on" if entry["desired"] else "off"
        running = "yes" if entry["running"] else "no"
        pid = str(entry["pid"] or "-")
        process_user = str(entry.get("process_user") or "-")
        uptime = _format_duration(entry.get("uptime_seconds"))
        restarts = str(entry["restart_count"])
        last_exit = str(entry["last_exit_code"]) if entry["last_exit_code"] is not None else "-"
        policy = "auto" if entry["restart_on_exit"] else "manual"
        runtime_state = _route_runtime_state(entry)
        print(
            f"{key:<10} {entry['script']:<14} {desired:<7} {running:<7} {pid:<8} {process_user:<9} "
            f"{uptime:<8} {restarts:<9} {last_exit:<8} {policy:<7} {runtime_state}"
        )
        event = str(entry.get("last_event") or "").strip()
        if event:
            print(f"  event: {event}")
    print("----------------------------------------------------------------------------------------------------------")
    print("Log files live under: logs/watchdog/")


def _select_route_interactive(watchdog: InterfaceWatchdog, prompt: str = "Route (number/key): ") -> str | None:
    keys = watchdog.route_keys()
    print("\nRoutes:")
    for index, key in enumerate(keys, start=1):
        print(f"  {index}. {key}")
    selected = input(prompt).strip()
    return _parse_route_token(selected, keys)


def _show_route_log_tail(watchdog: InterfaceWatchdog, route_key: str, line_count: int = 40) -> None:
    status = watchdog.status()
    entry = status.get(route_key)
    if entry is None:
        print(f"[launcher] Unknown route: {route_key}")
        return
    log_path = Path(str(entry["log_file"]))
    print(f"\n=== Log Tail: {route_key} ({log_path}) ===")
    for line in _tail_log_lines(log_path, line_count=line_count):
        print(line)


def monitor_dashboard(
    watchdog: InterfaceWatchdog,
    config_data: dict[str, Any],
    runtime_settings: dict[str, Any],
    db_status: dict[str, Any] | None = None,
    *,
    refresh_seconds: float = 1.0,
) -> None:
    if not sys.stdin.isatty() or not sys.stdout.isatty():
        print("[launcher] Live monitor requires an interactive terminal.")
        return

    keys = watchdog.route_keys()
    while True:
        _clear_screen()
        print_launcher_summary(
            config_data=config_data,
            runtime_settings=runtime_settings,
            db_status=db_status,
        )
        print_watchdog_status(watchdog)
        print("\nDashboard commands:")
        print("  q                 Back to launcher menu")
        print("  a                 Start auto-managed routes")
        print("  o                 Stop all routes")
        print("  t <route|number>  Toggle route")
        print("  l <route|number>  Tail route log")
        print("  <Enter>           Refresh")
        print("\ncommand> ", end="", flush=True)

        ready, _, _ = select.select([sys.stdin], [], [], max(0.2, float(refresh_seconds)))
        if not ready:
            continue

        raw = sys.stdin.readline()
        if raw is None:
            continue
        command = raw.strip()
        if command == "":
            continue

        chunks = command.split()
        action = chunks[0].lower()
        arg = chunks[1] if len(chunks) > 1 else ""

        if action == "q":
            _clear_screen()
            return
        if action == "a":
            watchdog.start_all(include_manual=False)
            continue
        if action == "o":
            watchdog.stop_all()
            continue
        if action == "t":
            route = _parse_route_token(arg, keys)
            if route is None:
                continue
            watchdog.toggle(route)
            continue
        if action == "l":
            route = _parse_route_token(arg, keys)
            if route is None:
                continue
            _clear_screen()
            _show_route_log_tail(watchdog, route_key=route, line_count=60)
            input("\nPress Enter to return to monitor...")
            continue


def route_menu(
    watchdog: InterfaceWatchdog,
    config_data: dict[str, Any],
    runtime_settings: dict[str, Any],
    db_status: dict[str, Any] | None = None,
) -> None:
    while True:
        _clear_screen()
        print_launcher_summary(
            config_data=config_data,
            runtime_settings=runtime_settings,
            db_status=db_status,
        )
        print_watchdog_status(watchdog)
        print("\nActions:")
        print("  1. Toggle a route")
        print("  2. Start auto-managed routes")
        print("  3. Stop all routes")
        print("  4. Show route log paths")
        print("  5. Open live monitor dashboard")
        print("  6. Tail a route log")
        print("  0. Exit launcher")
        choice = input("Select action: ").strip()

        if choice == "1":
            route_key = _select_route_interactive(watchdog, prompt="Route number or key to toggle: ")
            if route_key:
                watchdog.toggle(route_key)
            continue

        if choice == "2":
            watchdog.start_all(include_manual=False)
            print("[launcher] Start signal sent to auto-managed routes (web, telegram).")
            print("[launcher] Use route open action for interactive/manual routes (cli, x).")
            continue

        if choice == "3":
            watchdog.stop_all()
            print("[launcher] Stop signal sent to all routes.")
            continue

        if choice == "4":
            status = watchdog.status()
            for key in watchdog.route_keys():
                print(f"{key}: {status[key]['log_file']}")
            input("\nPress Enter to continue...")
            continue

        if choice == "5":
            monitor_dashboard(
                watchdog,
                config_data=config_data,
                runtime_settings=runtime_settings,
                db_status=db_status,
            )
            continue

        if choice == "6":
            route_key = _select_route_interactive(watchdog, prompt="Route number or key to tail: ")
            if route_key:
                _show_route_log_tail(watchdog, route_key=route_key, line_count=60)
                input("\nPress Enter to continue...")
            continue

        if choice == "0":
            return


def _curses_select_option(
    stdscr: Any,
    *,
    title: str,
    body_lines: list[str],
    options: list[str],
    current: str | None = None,
) -> str | None:
    if not options:
        return None

    try:
        index = options.index(current) if current is not None else 0
    except ValueError:
        index = 0

    while True:
        stdscr.erase()
        _safe_addstr(stdscr, 0, 0, title, curses.A_BOLD if curses else 0)
        row = 2
        for line in body_lines:
            _safe_addstr(stdscr, row, 0, line)
            row += 1
        _safe_addstr(stdscr, row + 1, 0, "Use up/down arrows, Enter to confirm, q/Esc to cancel.")
        row += 3
        start = max(0, index - 8)
        end = min(len(options), start + 16)
        for option_idx in range(start, end):
            option = options[option_idx]
            marker = "* " if option == current else "  "
            attr = curses.A_REVERSE if (curses and option_idx == index) else 0
            _safe_addstr(stdscr, row, 0, f"{marker}{option}", attr)
            row += 1
        stdscr.refresh()

        key = stdscr.getch()
        if key in (curses.KEY_UP if curses else -1, ord("k")):
            index = max(0, index - 1)
            continue
        if key in (curses.KEY_DOWN if curses else -1, ord("j")):
            index = min(len(options) - 1, index + 1)
            continue
        if key in (10, 13):
            return options[index]
        if key in (27, ord("q")):
            return None


def _is_ollama_model_field(path: str) -> bool:
    normalized = str(path or "").strip()
    return normalized.startswith("inference.") and normalized.endswith(".model")


def _discover_ollama_models_for_editor(
    *,
    config_data: dict[str, Any],
    runtime_settings: dict[str, Any],
) -> tuple[list[str], str | None]:
    host = resolve_ollama_host(config_data, runtime_settings=runtime_settings)
    probe_timeout = float(get_runtime_setting(runtime_settings, "inference.probe_timeout_seconds", 3.0))
    return fetch_ollama_models(host, timeout=probe_timeout)


def _curses_select_model_chain(
    stdscr: Any,
    *,
    title: str,
    body_lines: list[str],
    options: list[str],
    current_selected: list[str] | None = None,
) -> list[str] | None:
    if not options:
        return None

    selected_order = _dedupe_models(list(current_selected or []))
    display_options = _dedupe_models(selected_order + list(options))
    selected_set = set(selected_order)

    index = 0
    if selected_order:
        try:
            index = display_options.index(selected_order[0])
        except ValueError:
            index = 0

    while True:
        stdscr.erase()
        _safe_addstr(stdscr, 0, 0, title, curses.A_BOLD if curses else 0)
        row = 2
        for line in body_lines:
            _safe_addstr(stdscr, row, 0, line)
            row += 1
        selected_preview = ", ".join(selected_order) if selected_order else "(none)"
        _safe_addstr(stdscr, row + 1, 0, f"Selected order: {selected_preview}")
        _safe_addstr(
            stdscr,
            row + 2,
            0,
            "Use up/down to move, Space to toggle, Enter to confirm, q/Esc to cancel.",
        )
        row += 4
        start = max(0, index - 8)
        end = min(len(display_options), start + 16)
        for option_idx in range(start, end):
            model_name = display_options[option_idx]
            marker = "[x]" if model_name in selected_set else "[ ]"
            attr = curses.A_REVERSE if (curses and option_idx == index) else 0
            _safe_addstr(stdscr, row, 0, f"{marker} {model_name}", attr)
            row += 1
        stdscr.refresh()

        key = stdscr.getch()
        if key in (curses.KEY_UP if curses else -1, ord("k")):
            index = max(0, index - 1)
            continue
        if key in (curses.KEY_DOWN if curses else -1, ord("j")):
            index = min(len(display_options) - 1, index + 1)
            continue
        if key == ord(" "):
            model_name = display_options[index]
            if model_name in selected_set:
                selected_set.remove(model_name)
                selected_order = [item for item in selected_order if item != model_name]
            else:
                selected_set.add(model_name)
                selected_order.append(model_name)
            continue
        if key in (10, 13):
            return list(selected_order)
        if key in (27, ord("q")):
            return None


def _edit_route_setting_curses(
    stdscr: Any,
    *,
    route_spec: RouteConfigSpec,
    category_spec: RouteCategorySpec,
    setting: RouteSettingSpec,
    config_data: dict[str, Any],
    runtime_settings: dict[str, Any],
    pending_changes: dict[str, Any],
) -> tuple[bool, Any, str]:
    current_value, source, env_override = _resolve_setting_value(
        setting,
        config_data=config_data,
        pending_changes=pending_changes,
        runtime_settings=runtime_settings,
    )
    default_value = _default_value_for_setting(setting)
    type_label = _setting_type_label(setting)
    title = f"{route_spec.label} > {category_spec.label} > {setting.label}"
    details = [
        f"Path: {setting.path}",
        f"Type: {type_label}",
        f"Current: {_format_setting_value(current_value, sensitive=setting.sensitive)}",
        f"Default: {_format_setting_value(default_value, sensitive=setting.sensitive)}",
        f"Source: {source}",
        setting.description,
    ]
    if env_override and setting.env_override_keys:
        details.append("Environment override present: " + ", ".join(setting.env_override_keys))

    if type_label == "bool":
        selected = _curses_prompt_yes_no(
            stdscr,
            title,
            details + ["Set new value for this flag?"],
            default=bool(current_value),
        )
        ok, parsed, error = _coerce_setting_value(setting, selected)
        if not ok:
            return False, current_value, error
        return True, parsed, "updated"

    if setting.choices:
        selected = _curses_select_option(
            stdscr,
            title=title,
            body_lines=details,
            options=list(setting.choices),
            current=str(current_value) if current_value is not None else None,
        )
        if selected is None:
            return False, current_value, "cancelled"
        ok, parsed, error = _coerce_setting_value(setting, selected)
        if not ok:
            return False, current_value, error
        return True, parsed, "updated"

    uses_model_picker = _is_ollama_model_field(setting.path) or type_label == "model-list"
    if uses_model_picker:
        available_models, discovery_error = _discover_ollama_models_for_editor(
            config_data=config_data,
            runtime_settings=runtime_settings,
        )
        if available_models:
            preview = ", ".join(available_models[:8])
            suffix = " ..." if len(available_models) > 8 else ""
            details.append(f"Discovered models ({len(available_models)}): {preview}{suffix}")
        elif discovery_error:
            details.append(f"Model discovery warning: {discovery_error}")

        if type_label == "model-list":
            current_models = current_value if isinstance(current_value, list) else []
            if available_models:
                selected_models = _curses_select_model_chain(
                    stdscr,
                    title=title,
                    body_lines=details,
                    options=available_models,
                    current_selected=current_models,
                )
                if selected_models is None:
                    return False, current_value, "cancelled"
                ok, parsed, error = _coerce_setting_value(setting, selected_models)
                if not ok:
                    return False, current_value, error
                return True, parsed, "updated"

            default_text = ", ".join([str(item) for item in current_models if str(item).strip()])
            if default_text == "":
                fallback_default = _default_value_for_setting(setting)
                if isinstance(fallback_default, list):
                    default_text = ", ".join([str(item) for item in fallback_default if str(item).strip()])
            raw_value = _curses_prompt_text(
                stdscr,
                title,
                "No models discovered. Enter comma-separated model chain:",
                default=default_text,
                allow_empty=not setting.required,
            )
            ok, parsed, error = _coerce_setting_value(setting, raw_value)
            if not ok:
                return False, current_value, error
            return True, parsed, "updated"

        if available_models:
            selected_model = _curses_select_option(
                stdscr,
                title=title,
                body_lines=details,
                options=available_models,
                current=str(current_value) if current_value is not None else None,
            )
            if selected_model is None:
                return False, current_value, "cancelled"
            ok, parsed, error = _coerce_setting_value(setting, selected_model)
            if not ok:
                return False, current_value, error
            return True, parsed, "updated"

        raw_value = _curses_prompt_text(
            stdscr,
            title,
            "No models discovered. Enter model name:",
            default=str(current_value or ""),
            allow_empty=not setting.required,
        )
        ok, parsed, error = _coerce_setting_value(setting, raw_value)
        if not ok:
            return False, current_value, error
        return True, parsed, "updated"

    if setting.sensitive:
        should_replace = _curses_prompt_yes_no(
            stdscr,
            title,
            details + ["Replace secret value now?"],
            default=False,
        )
        if not should_replace:
            return False, current_value, "cancelled"
        raw_value = _curses_prompt_text(
            stdscr,
            title,
            "Enter new value (input hidden from previews after save):",
            default="",
            allow_empty=not setting.required,
        )
    else:
        default_candidate = default_value if current_value is None else current_value
        default_text = "" if default_candidate is None else str(default_candidate)
        raw_value = _curses_prompt_text(
            stdscr,
            title,
            "Enter new value:",
            default=default_text,
            allow_empty=not setting.required,
        )

    ok, parsed, error = _coerce_setting_value(setting, raw_value)
    if not ok:
        return False, current_value, error
    return True, parsed, "updated"


def _curses_route_category_menu(
    stdscr: Any,
    *,
    watchdog: InterfaceWatchdog,
    route_spec: RouteConfigSpec,
    route_key: str,
    category_spec: RouteCategorySpec,
    config_data: dict[str, Any],
    runtime_settings: dict[str, Any],
    pending_changes: dict[str, Any],
    save_callback: Any,
) -> str:
    selected_idx = 0
    status_text = "ready"
    settings = list(category_spec.settings)

    while True:
        selected_idx = max(0, min(selected_idx, max(0, len(settings) - 1)))
        stdscr.erase()
        status = watchdog.status().get(route_key, {})
        running = "yes" if status.get("running") else "no"
        desired = "on" if status.get("desired") else "off"

        _safe_addstr(
            stdscr,
            0,
            0,
            f"Route Config > {route_spec.label} > {category_spec.label}",
            curses.A_BOLD if curses else 0,
        )
        _safe_addstr(stdscr, 1, 0, f"Route state: desired={desired} running={running}")
        _safe_addstr(
            stdscr,
            2,
            0,
            "Controls: Up/Down select | Enter edit | d default | u reset category | s save | b/q back",
        )
        _safe_addstr(stdscr, 3, 0, f"Pending changes: {len(pending_changes)}")
        _safe_addstr(stdscr, 4, 0, f"Status: {status_text}")

        _safe_addstr(stdscr, 6, 0, "Setting                     Value                          Source           Type")
        _safe_addstr(stdscr, 7, 0, "---------------------------------------------------------------------------------------")
        row = 8
        for idx, setting in enumerate(settings):
            value, source, _ = _resolve_setting_value(
                setting,
                config_data=config_data,
                pending_changes=pending_changes,
                runtime_settings=runtime_settings,
            )
            value_text = _format_setting_value(value, sensitive=setting.sensitive)
            marker = "*" if setting.path in pending_changes else " "
            line = (
                f"{marker} {setting.label:<25} "
                f"{_trim_text(value_text, 30):<30} "
                f"{source:<15} {_setting_type_label(setting)}"
            )
            attr = curses.A_REVERSE if (curses and idx == selected_idx) else 0
            _safe_addstr(stdscr, row, 0, line, attr)
            row += 1

        details_row = row + 1
        if settings:
            selected = settings[selected_idx]
            default_value = _default_value_for_setting(selected)
            _safe_addstr(stdscr, details_row, 0, f"Selected: {selected.label}", curses.A_BOLD if curses else 0)
            _safe_addstr(stdscr, details_row + 1, 0, f"Path: {selected.path}")
            _safe_addstr(
                stdscr,
                details_row + 2,
                0,
                f"Default: {_format_setting_value(default_value, sensitive=selected.sensitive)}",
            )
            _safe_addstr(stdscr, details_row + 3, 0, selected.description)
            if selected.env_override_keys:
                env_text = ", ".join(selected.env_override_keys)
                _safe_addstr(stdscr, details_row + 4, 0, f"Env keys: {env_text}")

        stdscr.refresh()
        key = stdscr.getch()
        if key in (ord("q"), ord("b"), 27):
            return status_text
        if key in (curses.KEY_UP if curses else -1, ord("k")):
            selected_idx = max(0, selected_idx - 1)
            continue
        if key in (curses.KEY_DOWN if curses else -1, ord("j")):
            selected_idx = min(len(settings) - 1, selected_idx + 1) if settings else 0
            continue
        if key in (10, 13):
            if not settings:
                continue
            selected = settings[selected_idx]
            changed, new_value, edit_status = _edit_route_setting_curses(
                stdscr,
                route_spec=route_spec,
                category_spec=category_spec,
                setting=selected,
                config_data=config_data,
                runtime_settings=runtime_settings,
                pending_changes=pending_changes,
            )
            if changed:
                pending_changes[selected.path] = new_value
                status_text = f"updated pending: {selected.label}"
            else:
                status_text = edit_status
                if edit_status and edit_status not in {"cancelled", "updated"}:
                    _curses_message(
                        stdscr,
                        "Invalid Value",
                        [f"Setting: {selected.label}", f"Reason: {edit_status}"],
                    )
            continue
        if key == ord("d"):
            if not settings:
                continue
            selected = settings[selected_idx]
            pending_changes[selected.path] = _default_value_for_setting(selected)
            status_text = f"reset pending to default: {selected.label}"
            continue
        if key == ord("u"):
            should_reset = _curses_prompt_yes_no(
                stdscr,
                f"Reset Category Defaults: {category_spec.label}",
                [
                    f"Route: {route_spec.label}",
                    f"Category: {category_spec.label}",
                    "Reset every setting in this category to defaults?",
                ],
                default=False,
            )
            if should_reset:
                for setting in settings:
                    pending_changes[setting.path] = _default_value_for_setting(setting)
                status_text = f"reset category to defaults: {category_spec.label}"
            else:
                status_text = "reset cancelled"
            continue
        if key == ord("s"):
            status_text = str(save_callback())
            continue


def _curses_route_config_workspace(
    stdscr: Any,
    *,
    watchdog: InterfaceWatchdog,
    route_key: str,
    config_data: dict[str, Any],
    runtime_settings: dict[str, Any],
) -> str:
    route_spec = ROUTE_CONFIG_SPECS.get(route_key)
    if route_spec is None:
        _curses_message(
            stdscr,
            "Route Config",
            [f"No configuration schema defined for route: {route_key}"],
        )
        return f"no config schema for route {route_key}"

    category_idx = 0
    pending_changes: dict[str, Any] = {}
    status_text = "ready"
    settings_by_path = _settings_by_path(route_spec)

    def _save_pending_changes() -> str:
        nonlocal status_text
        if not pending_changes:
            return "no pending changes"

        next_config = copy.deepcopy(config_data)
        parsed_by_path: dict[str, Any] = {}
        for path, pending_value in pending_changes.items():
            setting = settings_by_path.get(path)
            if setting is None:
                continue
            ok, parsed, error = _coerce_setting_value(setting, pending_value)
            if not ok:
                _curses_message(
                    stdscr,
                    "Save Failed",
                    [f"Invalid value for {setting.label}", f"Reason: {error}"],
                )
                return f"save failed: {setting.label}"
            parsed_by_path[path] = parsed

        if not parsed_by_path:
            return "no valid pending changes"

        changed_setting_paths: list[str] = list(parsed_by_path.keys())
        changed_config_paths: list[str] = []
        policy_updates: dict[str, list[str]] = {}

        for path, parsed in parsed_by_path.items():
            policy_name = _policy_name_from_setting_path(path)
            if policy_name:
                if isinstance(parsed, list):
                    policy_updates[policy_name] = _dedupe_models(
                        [str(item).strip() for item in parsed if str(item).strip()]
                    )
                else:
                    policy_updates[policy_name] = _parse_allowed_models_input(str(parsed))
                continue
            _set_config_path(next_config, path, parsed)
            changed_config_paths.append(path)

        policy_saved_count = 0
        policy_warning_count = 0
        policy_backup_names: list[str] = []
        if policy_updates:
            policy_manager = PolicyManager(
                inference_config=next_config.get("inference"),
                endpoint_override=resolve_ollama_host(
                    next_config,
                    runtime_settings=build_runtime_settings(config_data=next_config),
                ),
            )
            policy_failures: list[str] = []
            for policy_name, allowed_models in policy_updates.items():
                save_result = policy_manager.save_policy(
                    policy_name=policy_name,
                    updates={"allowed_models": list(allowed_models)},
                    strict_model_check=False,
                )
                if not save_result.saved:
                    reason = save_result.report.errors[0] if save_result.report.errors else "unknown error"
                    policy_failures.append(f"{policy_name}: {reason}")
                    continue
                policy_saved_count += 1
                policy_warning_count += len(save_result.report.warnings)
                if save_result.backup_path:
                    policy_backup_names.append(Path(save_result.backup_path).name)

            if policy_failures:
                _curses_message(
                    stdscr,
                    "Policy Save Failed",
                    [
                        "One or more policy profiles could not be saved.",
                        *policy_failures[:6],
                    ],
                )
                return f"save failed: {len(policy_failures)} policy profile(s)"

        backup = None
        if changed_config_paths:
            try:
                backup = backup_file(CONFIG_FILE)
                write_json_atomic(CONFIG_FILE, next_config)
            except OSError as error:
                _curses_message(
                    stdscr,
                    "Save Failed",
                    [f"Could not write config file: {error}"],
                )
                return f"save failed: {error}"

        next_runtime = build_runtime_settings(config_data=next_config)
        config_data.clear()
        config_data.update(next_config)
        runtime_settings.clear()
        runtime_settings.update(next_runtime)
        watchdog.update_runtime_settings(runtime_settings)

        restart_required = any(
            settings_by_path[path].restart_required
            for path in changed_setting_paths
            if path in settings_by_path
        )
        pending_changes.clear()

        save_message = f"saved {len(changed_setting_paths)} setting(s)"
        if backup:
            save_message += f" (backup: {backup.name})"
        if policy_saved_count > 0:
            save_message += f" | policy profiles updated: {policy_saved_count}"
            if policy_backup_names:
                save_message += f" (policy backups: {', '.join(policy_backup_names[:2])}"
                if len(policy_backup_names) > 2:
                    save_message += f", +{len(policy_backup_names) - 2} more"
                save_message += ")"
            if policy_warning_count > 0:
                save_message += f" | warnings: {policy_warning_count}"

        route_status = watchdog.status().get(route_key, {})
        is_running = bool(route_status.get("running"))
        if restart_required and is_running:
            restart_now = _curses_prompt_yes_no(
                stdscr,
                f"Restart Route: {route_key}",
                [
                    "One or more saved settings require restart to apply.",
                    f"Restart route '{route_key}' now?",
                ],
                default=True,
            )
            if restart_now:
                watchdog.stop(route_key)
                watchdog.start(route_key)
                save_message += " | route restarted"
            else:
                save_message += " | restart pending"

        return save_message

    while True:
        status = watchdog.status().get(route_key, {})
        categories = list(route_spec.categories)
        category_idx = max(0, min(category_idx, max(0, len(categories) - 1)))

        stdscr.erase()
        running = "yes" if status.get("running") else "no"
        desired = "on" if status.get("desired") else "off"
        policy = "auto" if status.get("restart_on_exit") else "manual"

        _safe_addstr(stdscr, 0, 0, f"Route Config: {route_spec.label}", curses.A_BOLD if curses else 0)
        _safe_addstr(
            stdscr,
            1,
            0,
            f"Route: {route_key} | Script: {status.get('script', '-') } | Desired: {desired} | Running: {running} | Policy: {policy}",
        )
        _safe_addstr(
            stdscr,
            2,
            0,
            f"Access: {_route_access_summary(route_key, config_data, runtime_settings, entry=status)}",
        )
        _safe_addstr(
            stdscr,
            3,
            0,
            "Controls: Up/Down select | Enter category | s save | d discard | r restart route | t toggle | o open | q back",
        )
        _safe_addstr(stdscr, 4, 0, f"Pending changes: {len(pending_changes)}")
        _safe_addstr(stdscr, 5, 0, f"Status: {status_text}")

        _safe_addstr(stdscr, 7, 0, "Categories")
        _safe_addstr(stdscr, 8, 0, "---------------------------------------------------------------")
        row = 9
        for idx, category in enumerate(categories):
            category_pending = sum(1 for setting in category.settings if setting.path in pending_changes)
            line = f"{category.label:<28} settings={len(category.settings):<3} pending={category_pending:<3}"
            attr = curses.A_REVERSE if (curses and idx == category_idx) else 0
            _safe_addstr(stdscr, row, 0, line, attr)
            row += 1

        preview_row = row + 1
        if categories:
            selected_category = categories[category_idx]
            _safe_addstr(
                stdscr,
                preview_row,
                0,
                f"Selected category: {selected_category.label}",
                curses.A_BOLD if curses else 0,
            )
            preview_row += 1
            for setting in selected_category.settings[:6]:
                value, source, _ = _resolve_setting_value(
                    setting,
                    config_data=config_data,
                    pending_changes=pending_changes,
                    runtime_settings=runtime_settings,
                )
                marker = "*" if setting.path in pending_changes else " "
                line = (
                    f"{marker} {setting.label:<24} "
                    f"{_trim_text(_format_setting_value(value, sensitive=setting.sensitive), 26):<26} "
                    f"{source}"
                )
                _safe_addstr(stdscr, preview_row, 0, line)
                preview_row += 1

        stdscr.refresh()
        key = stdscr.getch()
        if key in (ord("q"), ord("b"), 27):
            if pending_changes:
                should_discard = _curses_prompt_yes_no(
                    stdscr,
                    f"Discard Pending Changes: {route_spec.label}",
                    [
                        f"Route: {route_key}",
                        f"Unsaved settings: {len(pending_changes)}",
                        "Discard pending edits and return to dashboard?",
                    ],
                    default=False,
                )
                if not should_discard:
                    status_text = "discard cancelled"
                    continue
            return status_text
        if key in (curses.KEY_UP if curses else -1, ord("k")):
            category_idx = max(0, category_idx - 1)
            continue
        if key in (curses.KEY_DOWN if curses else -1, ord("j")):
            category_idx = min(len(categories) - 1, category_idx + 1) if categories else 0
            continue
        if key in (10, 13):
            if not categories:
                continue
            selected_category = categories[category_idx]
            status_text = _curses_route_category_menu(
                stdscr,
                watchdog=watchdog,
                route_spec=route_spec,
                route_key=route_key,
                category_spec=selected_category,
                config_data=config_data,
                runtime_settings=runtime_settings,
                pending_changes=pending_changes,
                save_callback=_save_pending_changes,
            )
            continue
        if key == ord("s"):
            status_text = _save_pending_changes()
            continue
        if key == ord("d"):
            if not pending_changes:
                status_text = "no pending changes"
                continue
            should_discard = _curses_prompt_yes_no(
                stdscr,
                f"Discard Pending Changes: {route_spec.label}",
                [
                    f"Route: {route_key}",
                    f"Unsaved settings: {len(pending_changes)}",
                    "Discard all pending edits?",
                ],
                default=False,
            )
            if should_discard:
                pending_changes.clear()
                status_text = "discarded pending changes"
            else:
                status_text = "discard cancelled"
            continue
        if key == ord("t"):
            watchdog.toggle(route_key)
            status_text = f"toggled route {route_key}"
            continue
        if key == ord("r"):
            route_status = watchdog.status().get(route_key, {})
            if route_status.get("running"):
                watchdog.stop(route_key)
                watchdog.start(route_key)
                status_text = f"restarted route {route_key}"
            else:
                status_text = f"route {route_key} not running"
            continue
        if key == ord("o"):
            opened, lines = _route_open_action(route_key, status, config_data, runtime_settings)
            _curses_message(stdscr, f"Open Interface: {route_key}", lines)
            status_text = f"opened route {route_key}" if opened else f"open failed for {route_key}"
            continue


def _curses_view_route_log(stdscr: Any, watchdog: InterfaceWatchdog, route_key: str) -> None:
    while True:
        stdscr.erase()
        status = watchdog.status()
        entry = status.get(route_key, {})
        log_path = Path(str(entry.get("log_file", "")))
        _safe_addstr(stdscr, 0, 0, f"Log Tail: {route_key}", curses.A_BOLD if curses else 0)
        _safe_addstr(stdscr, 1, 0, str(log_path))
        _safe_addstr(stdscr, 2, 0, "Press q or Esc to return.")
        lines = _tail_log_lines(log_path, line_count=max(10, stdscr.getmaxyx()[0] - 5))
        row = 4
        for line in lines:
            _safe_addstr(stdscr, row, 0, line)
            row += 1
        stdscr.refresh()
        key = stdscr.getch()
        if key in (ord("q"), 27):
            return


def _policy_file_path(policy_name: str) -> Path:
    return POLICIES_DIR / f"{policy_name}_policy.json"


def _read_policy_payload_for_editor(policy_name: str) -> dict[str, Any]:
    payload = load_json(_policy_file_path(policy_name), fallback={})
    if isinstance(payload, dict):
        return payload
    return {}


def _policy_models_from_payload(payload: dict[str, Any]) -> list[str]:
    allowed = payload.get("allowed_models")
    if not isinstance(allowed, list):
        return []
    return _dedupe_models([str(value) for value in allowed if isinstance(value, str)])


def _parse_allowed_models_input(raw: str) -> list[str]:
    tokens = re.split(r"[,;\n]+", str(raw or ""))
    return _dedupe_models([token.strip() for token in tokens])


def _curses_edit_policy(
    stdscr: Any,
    *,
    manager: PolicyManager,
    policy_name: str,
    config_data: dict[str, Any],
    runtime_settings: dict[str, Any],
    available_models: list[str],
) -> str:
    payload = _read_policy_payload_for_editor(policy_name)
    initial_allow_custom = bool(payload.get("allow_custom_system_prompt", False))
    initial_allowed_models = _policy_models_from_payload(payload)
    if not initial_allowed_models:
        initial_allowed_models = _generated_allowed_models_for_policy(
            policy_name,
            config_data=config_data,
            runtime_settings=runtime_settings,
            available_models=available_models,
        )

    allow_custom = bool(initial_allow_custom)
    allowed_models = list(initial_allowed_models)
    status_text = "ready"

    while True:
        stdscr.erase()
        generated_models = _generated_allowed_models_for_policy(
            policy_name,
            config_data=config_data,
            runtime_settings=runtime_settings,
            available_models=available_models,
        )
        dirty = (allow_custom != initial_allow_custom) or (allowed_models != initial_allowed_models)
        _safe_addstr(stdscr, 0, 0, f"Policy Editor: {policy_name}", curses.A_BOLD if curses else 0)
        _safe_addstr(stdscr, 1, 0, f"Dirty: {'yes' if dirty else 'no'} | allow_custom_system_prompt: {'true' if allow_custom else 'false'}")
        _safe_addstr(
            stdscr,
            2,
            0,
            "Controls: t toggle allow_custom | m edit models | r reset generated | s save | q back",
        )
        _safe_addstr(stdscr, 3, 0, f"Status: {status_text}")
        _safe_addstr(stdscr, 5, 0, f"Generated from setup-selected runtime models: {', '.join(generated_models) or '(none)'}")
        _safe_addstr(stdscr, 6, 0, f"Discovered Ollama models: {len(available_models)}")
        _safe_addstr(stdscr, 8, 0, "Allowed Models:")
        row = 9
        if allowed_models:
            for model_name in allowed_models[: max(3, stdscr.getmaxyx()[0] - 13)]:
                _safe_addstr(stdscr, row, 0, f"- {model_name}")
                row += 1
        else:
            _safe_addstr(stdscr, row, 0, "(none)")
            row += 1
        stdscr.refresh()

        key = stdscr.getch()
        if key in (ord("q"), 27):
            if dirty:
                should_discard = _curses_prompt_yes_no(
                    stdscr,
                    f"Discard Unsaved Policy Changes: {policy_name}",
                    [
                        f"Policy: {policy_name}",
                        "Unsaved changes were detected.",
                        "Discard changes and return?",
                    ],
                    default=False,
                )
                if not should_discard:
                    status_text = "discard cancelled"
                    continue
            return "closed policy editor"
        if key == ord("t"):
            allow_custom = not allow_custom
            status_text = f"allow_custom_system_prompt={'true' if allow_custom else 'false'}"
            continue
        if key == ord("m"):
            default_text = ", ".join(allowed_models) if allowed_models else ", ".join(generated_models)
            raw = _curses_prompt_text(
                stdscr,
                f"Allowed Models: {policy_name}",
                [
                    "Enter comma-separated model list.",
                    "Example: qwen3-vl:latest, llama3.2:latest",
                    f"Discovered models on host: {len(available_models)}",
                ],
                default=default_text,
                required=True,
            )
            parsed = _parse_allowed_models_input(raw)
            if not parsed:
                _curses_message(
                    stdscr,
                    "Invalid Model List",
                    ["Allowed models cannot be empty."],
                )
                status_text = "invalid model list"
                continue
            allowed_models = parsed
            status_text = f"updated allowed_models ({len(allowed_models)})"
            continue
        if key == ord("r"):
            allowed_models = list(generated_models)
            status_text = "reset to generated runtime-selected models"
            continue
        if key == ord("s"):
            if not allowed_models:
                _curses_message(stdscr, "Save Failed", ["Allowed models cannot be empty."])
                status_text = "save failed: empty allowed models"
                continue
            save_result = manager.save_policy(
                policy_name=policy_name,
                updates={
                    "allow_custom_system_prompt": bool(allow_custom),
                    "allowed_models": list(allowed_models),
                },
                strict_model_check=False,
            )
            if not save_result.saved:
                errors = save_result.report.errors or ["unknown save error"]
                _curses_message(
                    stdscr,
                    f"Save Failed: {policy_name}",
                    [*errors[:6], *(save_result.report.warnings[:2] if save_result.report.warnings else [])],
                )
                status_text = "save failed"
                continue

            warning_suffix = ""
            if save_result.report.warnings:
                warning_suffix = f" (warnings: {len(save_result.report.warnings)})"
            return f"saved policy {policy_name}{warning_suffix}"


def _curses_policy_editor_workspace(
    stdscr: Any,
    *,
    config_data: dict[str, Any],
    runtime_settings: dict[str, Any],
) -> str:
    inference_config = config_data.get("inference")
    manager = PolicyManager(
        inference_config=inference_config if isinstance(inference_config, dict) else {},
        endpoint_override=resolve_ollama_host(config_data, runtime_settings=runtime_settings),
    )

    policy_names = manager.list_policy_names()
    if not policy_names:
        _curses_message(stdscr, "Policy Editor", [f"No policy files found under {POLICIES_DIR}"])
        return "no policy files found"

    host = resolve_ollama_host(config_data, runtime_settings=runtime_settings)
    probe_timeout = float(get_runtime_setting(runtime_settings, "inference.probe_timeout_seconds", 3.0))
    available_models, discovery_error = fetch_ollama_models(host, timeout=probe_timeout)

    selected_idx = 0
    status_text = "ready"
    while True:
        policy_names = manager.list_policy_names()
        if not policy_names:
            _curses_message(stdscr, "Policy Editor", [f"No policy files found under {POLICIES_DIR}"])
            return "no policy files found"
        selected_idx = max(0, min(selected_idx, len(policy_names) - 1))

        rows: list[dict[str, Any]] = []
        for policy_name in policy_names:
            payload = _read_policy_payload_for_editor(policy_name)
            allowed_models = _policy_models_from_payload(payload)
            generated_models = _generated_allowed_models_for_policy(
                policy_name,
                config_data=config_data,
                runtime_settings=runtime_settings,
                available_models=available_models,
            )
            rows.append(
                {
                    "policy_name": policy_name,
                    "allow_custom": bool(payload.get("allow_custom_system_prompt", False)),
                    "allowed_models": allowed_models,
                    "generated_models": generated_models,
                    "drifted": allowed_models != generated_models,
                }
            )

        selected = rows[selected_idx]
        stdscr.erase()
        _safe_addstr(stdscr, 0, 0, "Policy Editor", curses.A_BOLD if curses else 0)
        _safe_addstr(stdscr, 1, 0, f"Host: {host}")
        if discovery_error:
            _safe_addstr(stdscr, 2, 0, f"Model discovery warning: {discovery_error}")
        else:
            _safe_addstr(stdscr, 2, 0, f"Discovered Ollama models: {len(available_models)}")
        _safe_addstr(
            stdscr,
            3,
            0,
            "Controls: Up/Down select | Enter edit | g regenerate from setup models | v validate | f refresh models | q back",
        )
        _safe_addstr(stdscr, 4, 0, f"Status: {status_text}")
        _safe_addstr(stdscr, 6, 0, "Policy              Custom  Models Drift  Preview")
        _safe_addstr(stdscr, 7, 0, "--------------------------------------------------------------------------")

        row = 8
        for idx, item in enumerate(rows):
            preview = ", ".join(item["allowed_models"][:3]) if item["allowed_models"] else "(none)"
            if len(item["allowed_models"]) > 3:
                preview = preview + ", ..."
            drift = "*" if item["drifted"] else "-"
            line = (
                f"{item['policy_name']:<19} "
                f"{('yes' if item['allow_custom'] else 'no'):<7} "
                f"{len(item['allowed_models']):<6} "
                f"{drift:<5} "
                f"{preview}"
            )
            attr = curses.A_REVERSE if (curses and idx == selected_idx) else 0
            _safe_addstr(stdscr, row, 0, line, attr)
            row += 1

        detail_row = row + 1
        _safe_addstr(stdscr, detail_row, 0, f"Selected: {selected['policy_name']}", curses.A_BOLD if curses else 0)
        _safe_addstr(
            stdscr,
            detail_row + 1,
            0,
            f"Generated model chain: {', '.join(selected['generated_models']) or '(none)'}",
        )
        _safe_addstr(
            stdscr,
            detail_row + 2,
            0,
            f"Allowed model chain: {', '.join(selected['allowed_models']) or '(none)'}",
        )
        _safe_addstr(
            stdscr,
            detail_row + 3,
            0,
            "Drift marker '*': policy differs from setup-selected generated chain.",
        )
        stdscr.refresh()

        key = stdscr.getch()
        if key in (ord("q"), 27):
            return status_text
        if key in (curses.KEY_UP if curses else -1, ord("k")):
            selected_idx = max(0, selected_idx - 1)
            continue
        if key in (curses.KEY_DOWN if curses else -1, ord("j")):
            selected_idx = min(len(rows) - 1, selected_idx + 1)
            continue
        if key in (10, 13):
            status_text = _curses_edit_policy(
                stdscr,
                manager=manager,
                policy_name=selected["policy_name"],
                config_data=config_data,
                runtime_settings=runtime_settings,
                available_models=available_models,
            )
            continue
        if key == ord("g"):
            changed, unchanged, failures = sync_agent_policies_from_runtime_models(
                config_data,
                runtime_settings=runtime_settings,
                available_models=available_models,
            )
            if failures:
                _curses_message(
                    stdscr,
                    "Policy Sync Failures",
                    [*failures[:8], f"updated={changed}, unchanged={unchanged}"],
                )
                status_text = f"sync failures: {len(failures)}"
            else:
                status_text = f"sync complete: updated={changed}, unchanged={unchanged}"
            continue
        if key == ord("v"):
            report = manager.validate_policy(
                policy_name=selected["policy_name"],
                strict_model_check=False,
            )
            lines = [
                f"Policy: {selected['policy_name']}",
                f"Endpoint: {report.endpoint_host}",
                f"Available models discovered: {len(report.available_models)}",
            ]
            if report.errors:
                lines.append("Errors:")
                lines.extend([f"- {line}" for line in report.errors[:6]])
            if report.warnings:
                lines.append("Warnings:")
                lines.extend([f"- {line}" for line in report.warnings[:6]])
            if not report.errors and not report.warnings:
                lines.append("Validation clean.")
            _curses_message(stdscr, "Policy Validation", lines)
            status_text = f"validated {selected['policy_name']}"
            continue
        if key == ord("f"):
            host = resolve_ollama_host(config_data, runtime_settings=runtime_settings)
            available_models, discovery_error = fetch_ollama_models(host, timeout=probe_timeout)
            status_text = "refreshed model inventory"
            continue


def watchdog_dashboard_curses(
    watchdog: InterfaceWatchdog,
    *,
    config_data: dict[str, Any],
    runtime_settings: dict[str, Any],
    db_status: dict[str, Any] | None = None,
) -> None:
    def _runner(stdscr: Any) -> None:
        if curses:
            curses.curs_set(0)
            stdscr.keypad(True)
        stdscr.timeout(800)

        auto_start_routes = bool(
            get_runtime_setting(
                runtime_settings,
                "watchdog.auto_start_routes",
                True,
            )
        )
        if auto_start_routes:
            watchdog.start_all(include_manual=False)
            last_status_text = "auto-started auto routes (web/telegram)"
        else:
            last_status_text = "ready"

        selected_idx = 0
        live_log_state: dict[str, LiveLogTailState] = {}
        while True:
            status = watchdog.status()
            keys = watchdog.route_keys()
            if keys:
                selected_idx = max(0, min(selected_idx, len(keys) - 1))
            else:
                selected_idx = 0

            for route_key in keys:
                live_log_state.setdefault(route_key, LiveLogTailState())
            for stale_key in [key for key in list(live_log_state.keys()) if key not in keys]:
                live_log_state.pop(stale_key, None)

            selected_key = keys[selected_idx] if keys else None
            selected_entry = status.get(selected_key, {}) if selected_key else {}
            selected_log_lines: list[str] = ["(no route selected)"]
            selected_route_state = "idle"
            if selected_key:
                selected_log_state = live_log_state.setdefault(selected_key, LiveLogTailState())
                selected_log_path = Path(str(selected_entry.get("log_file", "")))
                _update_live_log_tail(selected_log_state, selected_log_path)
                selected_log_lines = _live_log_lines_for_display(
                    selected_log_state,
                    max_lines=max(10, LIVE_LOG_BUFFER_LINE_LIMIT),
                )
                selected_route_state = _infer_route_agent_state(selected_entry, selected_log_lines)

            stdscr.erase()
            height, width = stdscr.getmaxyx()
            _safe_addstr(stdscr, 0, 0, "RYO Launcher Dashboard", curses.A_BOLD if curses else 0)
            ollama_host = resolve_ollama_host(config_data, runtime_settings=runtime_settings)
            primary_pg_link, fallback_pg_link = _postgres_links(config_data)
            web_status = status.get("web", {})
            web_ui_url = str(web_status.get("endpoint_url") or "").strip() or _web_local_endpoint_from_runtime(runtime_settings)
            telegram_url = _telegram_bot_link(config_data) or "(not configured)"
            db_status_text = _database_status_label(db_status)
            db_error_text = _database_errors_preview(db_status, limit=1)
            _safe_addstr(stdscr, 1, 0, f"Ollama: {ollama_host}")
            _safe_addstr(stdscr, 2, 0, f"Postgres: {primary_pg_link}")
            _safe_addstr(stdscr, 3, 0, f"Postgres fallback: {fallback_pg_link or '(disabled)'}")
            _safe_addstr(stdscr, 4, 0, f"DB route: {db_status_text}")
            if db_error_text:
                _safe_addstr(stdscr, 5, 0, f"DB errors: {db_error_text}")
            else:
                _safe_addstr(stdscr, 5, 0, "DB errors: (none)")
            _safe_addstr(stdscr, 6, 0, f"Web: {web_ui_url}")
            _safe_addstr(stdscr, 7, 0, f"Telegram: {telegram_url}")
            stage_progress_enabled = bool(get_runtime_setting(runtime_settings, "telegram.show_stage_progress", True))
            _safe_addstr(
                stdscr,
                8,
                0,
                f"Telegram stage progress: {'on' if stage_progress_enabled else 'off'}",
            )
            _safe_addstr(
                stdscr,
                9,
                0,
                "Controls: Up/Down select | Enter config | p policy editor | r open interface | Space toggle | s start (manual opens terminal) | x stop | a/o auto-all | l logs | q quit",
            )
            _safe_addstr(stdscr, 10, 0, f"Status: {last_status_text}")
            _safe_addstr(stdscr, 12, 0, "Route      Desired Running PID      User      Uptime   Restarts LastExit Policy  State")
            _safe_addstr(stdscr, 13, 0, "------------------------------------------------------------------------------------------------")

            row = 14
            for idx, key in enumerate(keys):
                entry = status[key]
                desired = "on" if entry["desired"] else "off"
                running = "yes" if entry["running"] else "no"
                pid = str(entry["pid"] or "-")
                process_user = str(entry.get("process_user") or "-")
                uptime = _format_duration(entry.get("uptime_seconds"))
                restarts = str(entry["restart_count"])
                last_exit = str(entry["last_exit_code"]) if entry["last_exit_code"] is not None else "-"
                policy = "auto" if entry["restart_on_exit"] else "manual"
                runtime_state = _route_runtime_state(entry)
                line = (
                    f"{key:<10} {desired:<7} {running:<7} {pid:<8} {process_user:<9} {uptime:<8} "
                    f"{restarts:<8} {last_exit:<8} {policy:<7} {runtime_state}"
                )
                attr = curses.A_REVERSE if (curses and idx == selected_idx) else 0
                _safe_addstr(stdscr, row, 0, line, attr)
                row += 1

            details_row = row + 1
            if selected_key:
                _safe_addstr(stdscr, details_row, 0, f"Selected route: {selected_key}", curses.A_BOLD if curses else 0)
                _safe_addstr(stdscr, details_row + 1, 0, f"Script: {selected_entry.get('script')}")
                _safe_addstr(stdscr, details_row + 2, 0, f"Log: {selected_entry.get('log_file')}")
                _safe_addstr(stdscr, details_row + 3, 0, f"Event: {selected_entry.get('last_event')}")
                _safe_addstr(
                    stdscr,
                    details_row + 4,
                    0,
                    f"Access: {_route_access_summary(selected_key, config_data, runtime_settings, entry=selected_entry)}",
                )
                _safe_addstr(stdscr, details_row + 5, 0, f"Agent state: {selected_route_state}")

                live_title_row = details_row + 7
                _safe_addstr(stdscr, live_title_row, 0, "Live Log (rolling buffer)", curses.A_BOLD if curses else 0)
                _safe_addstr(stdscr, live_title_row + 1, 0, "-" * max(0, width - 1))
                available_live_rows = max(3, height - (live_title_row + 3))
                live_lines = selected_log_lines[-available_live_rows:]
                live_row = live_title_row + 2
                for line in live_lines:
                    _safe_addstr(stdscr, live_row, 0, line)
                    live_row += 1

            stdscr.refresh()
            key = stdscr.getch()
            if key == -1:
                continue
            if key in (ord("q"), 27):
                return
            if key in (curses.KEY_UP if curses else -1, ord("k")):
                selected_idx = max(0, selected_idx - 1)
                continue
            if key in (curses.KEY_DOWN if curses else -1, ord("j")):
                selected_idx = min(len(keys) - 1, selected_idx + 1) if keys else 0
                continue
            if key in (10, 13):
                if keys:
                    route_key = keys[selected_idx]
                    last_status_text = _curses_route_config_workspace(
                        stdscr,
                        watchdog=watchdog,
                        route_key=route_key,
                        config_data=config_data,
                        runtime_settings=runtime_settings,
                    )
                continue
            if key == ord("p"):
                last_status_text = _curses_policy_editor_workspace(
                    stdscr,
                    config_data=config_data,
                    runtime_settings=runtime_settings,
                )
                continue
            if key == ord("r"):
                if keys:
                    route_key = keys[selected_idx]
                    opened, lines = _route_open_action(
                        route_key,
                        status.get(route_key, {}),
                        config_data,
                        runtime_settings,
                    )
                    _curses_message(
                        stdscr,
                        f"Open Interface: {route_key}",
                        lines,
                    )
                    last_status_text = f"opened route {route_key}" if opened else f"open failed for {route_key}"
                continue
            if key == ord(" "):
                if keys:
                    route_key = keys[selected_idx]
                    watchdog.toggle(route_key)
                    last_status_text = f"toggled route {route_key}"
                continue
            if key == ord("s"):
                if keys:
                    route_key = keys[selected_idx]
                    watchdog.start(route_key)
                    last_status_text = f"started route {route_key}"
                continue
            if key == ord("x"):
                if keys:
                    route_key = keys[selected_idx]
                    watchdog.stop(route_key)
                    last_status_text = f"stopped route {route_key}"
                continue
            if key == ord("a"):
                watchdog.start_all(include_manual=False)
                last_status_text = "start-all sent (auto-managed routes only)"
                continue
            if key == ord("o"):
                watchdog.stop_all()
                last_status_text = "stop-all signal sent"
                continue
            if key == ord("l"):
                if keys:
                    _curses_view_route_log(stdscr, watchdog, keys[selected_idx])
                    last_status_text = f"viewed logs for {keys[selected_idx]}"
                continue

    curses.wrapper(_runner)


def bootstrap_ollama_and_models(
    config_data: dict[str, Any],
    *,
    runtime_settings: dict[str, Any],
    non_interactive: bool,
) -> dict[str, Any]:
    host = resolve_ollama_host(config_data, runtime_settings=runtime_settings)
    print(f"[ollama] Using host: {host}")
    probe_timeout = float(
        get_runtime_setting(runtime_settings, "inference.probe_timeout_seconds", 3.0)
    )

    models, error = fetch_ollama_models(host, timeout=probe_timeout)
    if error:
        print(f"[ollama] Warning: {error}")
        print("[ollama] Start/verify Ollama, then re-run `python3 app.py`.")
        return config_data

    print(f"[ollama] Discovered {len(models)} model(s).")
    required = collect_required_models(config_data)
    missing = [name for name in required if name not in models]
    if missing:
        print("[ollama] Missing models detected:")
        for name in missing:
            print(f"  - {name}")
        if prompt_yes_no(
            "Pull missing models now?",
            default=(not non_interactive),
            non_interactive=non_interactive,
        ):
            for model_name in missing:
                ok = pull_model(model_name)
                if ok:
                    print(f"[ollama] Pulled: {model_name}")
                else:
                    print(f"[ollama] Failed pull: {model_name}")

            refreshed, refresh_error = fetch_ollama_models(host, timeout=probe_timeout)
            if refresh_error is None:
                models = refreshed

    if not models:
        return config_data

    current_text_model = current_model(config_data, "chat")
    should_select_model = should_prompt_model_selection(runtime_settings, current_text_model)
    if current_text_model and should_select_model:
        should_select_model = prompt_yes_no(
            "Select/update default text model (chat/generate/tool)?",
            default=False,
            non_interactive=non_interactive,
        )

    if should_select_model:
        selected = choose_default_text_model(
            models=models,
            current=current_text_model,
            non_interactive=non_interactive,
        )
        if selected:
            config_data = apply_default_text_model(
                config_data,
                host=host,
                model_name=selected,
                runtime_settings=runtime_settings,
            )
            print(f"[ollama] Default text model set to: {selected}")

    return config_data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RYO Chat all-in-one bootstrap + watchdog launcher.")
    parser.add_argument(
        "--non-interactive",
        action="store_true",
        help="Run bootstrap with defaults and no prompts where possible.",
    )
    parser.add_argument(
        "--bootstrap-only",
        action="store_true",
        help="Run setup/bootstrap checks only, then exit (do not open watchdog menu).",
    )
    parser.add_argument(
        "--skip-setup-wizard",
        action="store_true",
        help="Do not invoke scripts/setup_wizard.py automatically.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    load_dotenv_file(PROJECT_ROOT / ".env", override=False)

    ensure_venv_and_reexec()
    ensure_requirements_installed()

    artifacts = ensure_project_artifacts()
    state = load_state()
    config_data = load_config()

    non_interactive = bool(args.non_interactive or not sys.stdin.isatty())
    if not non_interactive and not can_use_curses_ui(non_interactive=False):
        print("[launcher] Interactive mode requires curses-capable TTY.")
        print("[launcher] Use --non-interactive or run in a terminal that supports curses.")
        return 2
    use_curses_ui = not non_interactive

    setup_ran = False
    if not args.skip_setup_wizard and should_run_setup(state, artifacts, config_data):
        setup_ran = True
        setup_ok = run_setup_wizard(non_interactive=non_interactive)
        state["setup_completed"] = setup_ok
        save_state(state)
        config_data = load_config()
    else:
        state["setup_completed"] = bool(state.get("setup_completed", True))
        save_state(state)

    config_before = json.dumps(config_data, sort_keys=True)
    runtime_settings = build_runtime_settings(config_data=config_data)
    if use_curses_ui:
        config_data = bootstrap_ollama_and_models_curses(
            config_data,
            runtime_settings=runtime_settings,
        )
    else:
        config_data = bootstrap_ollama_and_models(
            config_data,
            runtime_settings=runtime_settings,
            non_interactive=non_interactive,
        )
    runtime_settings = build_runtime_settings(config_data=config_data)
    prompt_community_requirements = setup_ran or (not community_score_requirements_configured(config_data))
    if use_curses_ui:
        config_data = bootstrap_community_score_requirements_curses(
            config_data,
            runtime_settings=runtime_settings,
            prompt_on_startup=prompt_community_requirements,
        )
    else:
        config_data = bootstrap_community_score_requirements(
            config_data,
            runtime_settings=runtime_settings,
            non_interactive=non_interactive,
            prompt_on_startup=prompt_community_requirements,
        )
    config_after = json.dumps(config_data, sort_keys=True)
    if config_before != config_after:
        backup = backup_file(CONFIG_FILE)
        write_json_atomic(CONFIG_FILE, config_data)
        if backup:
            print(f"[bootstrap] Backed up config to: {backup}")
        print(f"[bootstrap] Updated config: {CONFIG_FILE}")

    runtime_settings = build_runtime_settings(config_data=config_data)
    db_status: dict[str, Any] | None = None
    config_data, runtime_settings, db_status = ensure_database_route_ready(
        config_data=config_data,
        runtime_settings=runtime_settings,
    )
    policy_probe_timeout = float(
        get_runtime_setting(runtime_settings, "inference.probe_timeout_seconds", 3.0)
    )
    policy_host = resolve_ollama_host(config_data, runtime_settings=runtime_settings)
    policy_models, policy_probe_error = fetch_ollama_models(policy_host, timeout=policy_probe_timeout)
    if policy_probe_error:
        print(f"[policy] Model discovery warning on {policy_host}: {policy_probe_error}")
        policy_models = []
    desired_policy_fingerprint = policy_model_sync_fingerprint(
        config_data,
        runtime_settings=runtime_settings,
        available_models=policy_models,
    )
    current_policy_fingerprint = str(state.get("policy_model_sync_fingerprint", ""))
    should_sync_policies = desired_policy_fingerprint != current_policy_fingerprint
    if should_sync_policies:
        changed_policies, unchanged_policies, policy_failures = sync_agent_policies_from_runtime_models(
            config_data,
            runtime_settings=runtime_settings,
            available_models=policy_models,
        )
        if changed_policies > 0:
            print(f"[policy] Updated {changed_policies} policy file(s) from setup-selected runtime models.")
        elif unchanged_policies > 0:
            print("[policy] Policy model chains already aligned with setup-selected runtime models.")
        if policy_failures:
            print(f"[policy] Failed to update {len(policy_failures)} policy file(s):")
            for failure in policy_failures:
                print(f"  - {failure}")
        else:
            state["policy_model_sync_fingerprint"] = desired_policy_fingerprint

    ensure_database_migrations(runtime_settings=runtime_settings)
    db_status = read_database_route_status(config_data=config_data, runtime_settings=runtime_settings)

    state["last_run_epoch"] = int(time.time())
    save_state(state)

    if args.non_interactive and not args.bootstrap_only:
        print("[launcher] --non-interactive mode completed bootstrap only.")
        return 0

    if args.bootstrap_only:
        print("[launcher] Bootstrap complete.")
        return 0

    runtime_settings = build_runtime_settings(config_data=config_data)
    watchdog = InterfaceWatchdog(
        sys.executable,
        runtime_settings=runtime_settings,
        restart_window_seconds=int(get_runtime_setting(runtime_settings, "watchdog.restart_window_seconds", 60)),
        max_restarts_per_window=int(get_runtime_setting(runtime_settings, "watchdog.max_restarts_per_window", 5)),
        terminate_timeout_seconds=float(get_runtime_setting(runtime_settings, "watchdog.terminate_timeout_seconds", 8.0)),
        kill_timeout_seconds=float(get_runtime_setting(runtime_settings, "watchdog.kill_timeout_seconds", 4.0)),
        thread_join_timeout_seconds=float(get_runtime_setting(runtime_settings, "watchdog.thread_join_timeout_seconds", 2.0)),
    )
    try:
        if use_curses_ui:
            watchdog_dashboard_curses(
                watchdog,
                config_data=config_data,
                runtime_settings=runtime_settings,
                db_status=db_status,
            )
        else:
            if bool(get_runtime_setting(runtime_settings, "watchdog.auto_start_routes", True)):
                watchdog.start_all(include_manual=False)
                print("[launcher] Auto-started auto routes (web/telegram).")
            route_menu(
                watchdog,
                config_data=config_data,
                runtime_settings=runtime_settings,
                db_status=db_status,
            )
    except KeyboardInterrupt:
        print("\n[launcher] Shutdown requested.")
    finally:
        watchdog.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
