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
import hashlib
import json
import os
from pathlib import Path
import shlex
import select
import shutil
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


def _curses_prompt_text(stdscr: Any, title: str, prompt: str, default: str = "") -> str:
    while True:
        stdscr.erase()
        _safe_addstr(stdscr, 0, 0, title, curses.A_BOLD if curses else 0)
        _safe_addstr(stdscr, 2, 0, prompt)
        if default:
            _safe_addstr(stdscr, 3, 0, f"Default: {default}")
        _safe_addstr(stdscr, 5, 0, "> ")
        stdscr.refresh()
        if curses:
            curses.echo()
        raw = stdscr.getstr(5, 2, 256)
        if curses:
            curses.noecho()
        value = raw.decode(errors="ignore").strip()
        if value == "":
            value = default
        if value:
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
        return env

    def _spawn(self, state: RouteState) -> None:
        log_path = self._log_path(state.spec.key)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        state.log_handle = log_path.open("a", encoding="utf-8")
        command = [self._python_exec, state.spec.script]
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
        status = self.status()
        entry = status.get(key)
        if entry is None:
            return False
        if entry["desired"]:
            return self.stop(key)
        return self.start(key)

    def start_all(self) -> None:
        for key in self.route_keys():
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
                uptime_seconds = int(max(0.0, self._stamp_now() - state.started_epoch)) if running and state.started_epoch else None
                output[key] = {
                    "desired": state.desired,
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
    return lines[-max(1, int(line_count)) :]


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
        return "interactive terminal launch available (press r)"
    if route_key == "x":
        return "interactive terminal launch available (press r)"
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
        f"cd {quoted_root} && {quoted_python} {quoted_script}; "
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


def print_launcher_summary(config_data: dict[str, Any], runtime_settings: dict[str, Any]) -> None:
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
    *,
    refresh_seconds: float = 1.0,
) -> None:
    if not sys.stdin.isatty() or not sys.stdout.isatty():
        print("[launcher] Live monitor requires an interactive terminal.")
        return

    keys = watchdog.route_keys()
    while True:
        _clear_screen()
        print_launcher_summary(config_data=config_data, runtime_settings=runtime_settings)
        print_watchdog_status(watchdog)
        print("\nDashboard commands:")
        print("  q                 Back to launcher menu")
        print("  a                 Start all routes")
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
            watchdog.start_all()
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
) -> None:
    while True:
        _clear_screen()
        print_launcher_summary(config_data=config_data, runtime_settings=runtime_settings)
        print_watchdog_status(watchdog)
        print("\nActions:")
        print("  1. Toggle a route")
        print("  2. Start all routes")
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
            watchdog.start_all()
            print("[launcher] Start signal sent to all routes.")
            print("[launcher] Note: 'cli' and 'x' are manual routes and may exit quickly by design.")
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
            monitor_dashboard(watchdog, config_data=config_data, runtime_settings=runtime_settings)
            continue

        if choice == "6":
            route_key = _select_route_interactive(watchdog, prompt="Route number or key to tail: ")
            if route_key:
                _show_route_log_tail(watchdog, route_key=route_key, line_count=60)
                input("\nPress Enter to continue...")
            continue

        if choice == "0":
            return


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


def watchdog_dashboard_curses(
    watchdog: InterfaceWatchdog,
    *,
    config_data: dict[str, Any],
    runtime_settings: dict[str, Any],
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
            watchdog.start_all()
            last_status_text = "auto-started all routes"
        else:
            last_status_text = "ready"

        selected_idx = 0
        while True:
            status = watchdog.status()
            keys = watchdog.route_keys()
            if keys:
                selected_idx = max(0, min(selected_idx, len(keys) - 1))
            else:
                selected_idx = 0

            stdscr.erase()
            _safe_addstr(stdscr, 0, 0, "RYO Launcher Dashboard", curses.A_BOLD if curses else 0)
            ollama_host = resolve_ollama_host(config_data, runtime_settings=runtime_settings)
            primary_pg_link, fallback_pg_link = _postgres_links(config_data)
            web_status = status.get("web", {})
            web_ui_url = str(web_status.get("endpoint_url") or "").strip() or _web_local_endpoint_from_runtime(runtime_settings)
            telegram_url = _telegram_bot_link(config_data) or "(not configured)"
            _safe_addstr(stdscr, 1, 0, f"Ollama: {ollama_host}")
            _safe_addstr(stdscr, 2, 0, f"Postgres: {primary_pg_link}")
            _safe_addstr(stdscr, 3, 0, f"Postgres fallback: {fallback_pg_link or '(disabled)'}")
            _safe_addstr(stdscr, 4, 0, f"Web: {web_ui_url}")
            _safe_addstr(stdscr, 5, 0, f"Telegram: {telegram_url}")
            _safe_addstr(
                stdscr,
                6,
                0,
                "Controls: Up/Down select | Enter/r open interface | Space toggle | s start | x stop | a/o all | l logs | q quit",
            )
            _safe_addstr(stdscr, 7, 0, f"Status: {last_status_text}")
            _safe_addstr(stdscr, 9, 0, "Route      Desired Running PID      User      Uptime   Restarts LastExit Policy  State")
            _safe_addstr(stdscr, 10, 0, "------------------------------------------------------------------------------------------------")

            row = 11
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
            if keys:
                selected_key = keys[selected_idx]
                selected_entry = status[selected_key]
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
            if key in (10, 13, ord("r")):
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
                watchdog.start_all()
                last_status_text = "start-all signal sent"
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
    ensure_database_migrations(runtime_settings=runtime_settings)

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
            )
        else:
            if bool(get_runtime_setting(runtime_settings, "watchdog.auto_start_routes", True)):
                watchdog.start_all()
                print("[launcher] Auto-started all managed routes.")
            route_menu(watchdog, config_data=config_data, runtime_settings=runtime_settings)
    except KeyboardInterrupt:
        print("\n[launcher] Shutdown requested.")
    finally:
        watchdog.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
