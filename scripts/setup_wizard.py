#!/usr/bin/env python3
"""
RYO setup wizard for endpoint, model, and key ingress configuration.

WO-005 implementation:
- Curses-guided interactive setup
- Required-field validation for first-run setup
- Ollama endpoint choice + model probe
- Capability-to-model mapping
- Config write with backup
- Optional .env write support
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from hypermindlabs.runtime_settings import (
    DEFAULT_RUNTIME_SETTINGS,
    build_runtime_settings,
    get_runtime_setting,
    load_dotenv_file,
)

try:
    import curses
except Exception:  # noqa: BLE001
    curses = None

try:
    from ollama import Client
except Exception:  # noqa: BLE001
    Client = None


DEFAULT_OLLAMA_HOST = str(
    DEFAULT_RUNTIME_SETTINGS.get("inference", {}).get("default_ollama_host", "http://127.0.0.1:11434")
)
DEFAULT_STATE_PATH = ".setup_wizard_state.json"
INFERENCE_KEYS = ("embedding", "generate", "chat", "tool", "multimodal")
DEFAULT_MODELS = {
    "embedding": str(DEFAULT_RUNTIME_SETTINGS.get("inference", {}).get("default_embedding_model", "")),
    "generate": str(DEFAULT_RUNTIME_SETTINGS.get("inference", {}).get("default_generate_model", "")),
    "chat": str(DEFAULT_RUNTIME_SETTINGS.get("inference", {}).get("default_chat_model", "")),
    "tool": str(DEFAULT_RUNTIME_SETTINGS.get("inference", {}).get("default_tool_model", "")),
    "multimodal": str(DEFAULT_RUNTIME_SETTINGS.get("inference", {}).get("default_multimodal_model", "")),
}


class SetupCancelledError(RuntimeError):
    """Raised when setup is cancelled before writing files."""


def load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid JSON in {path}: {error}") from error
    return data if isinstance(data, dict) else {}


def write_json(path: Path, data: dict) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)
        handle.write("\n")


def load_partial_state(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except (json.JSONDecodeError, OSError):
        return {}
    return data if isinstance(data, dict) else {}


def save_partial_state(path: Path, state: dict) -> None:
    if not state:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(state, handle, indent=2)
        handle.write("\n")
    tmp.replace(path)


def clear_partial_state(path: Path) -> None:
    if path.exists():
        path.unlink()


def resolve_bootstrap_target(config_data: dict, explicit_target: str | None = None) -> str:
    if explicit_target in {"primary", "fallback", "both"}:
        return explicit_target
    fallback = config_data.get("database_fallback")
    if isinstance(fallback, dict) and bool(fallback.get("enabled")):
        return "both"
    return "primary"


def run_postgres_bootstrap(
    config_path: Path,
    target: str,
    use_docker: bool = False,
) -> tuple[bool, str]:
    command = [
        sys.executable,
        "-m",
        "scripts.bootstrap_postgres",
        "--config",
        str(config_path),
        "--target",
        target,
    ]
    if use_docker:
        command.append("--docker")

    result = subprocess.run(command, check=False, text=True, capture_output=True)
    output = "\n".join(
        chunk for chunk in [result.stdout.strip(), result.stderr.strip()] if chunk
    )
    return result.returncode == 0, output


def is_valid_http_url(value: str | None) -> bool:
    if not value:
        return False
    parsed = urlparse(value)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def is_valid_numeric_id(value: str | int | None) -> bool:
    if value is None:
        return False
    if isinstance(value, int):
        return value > 0
    text = str(value).strip()
    return text.isdigit() and int(text) > 0


def _is_local_db_host(value: str | None) -> bool:
    host = str(value or "").strip().lower()
    return host in {"127.0.0.1", "localhost", "0.0.0.0", "::1", "::"}


def _is_local_db_mode(value: str | None) -> bool:
    mode = str(value or "").strip().lower()
    return mode in {"", "local", "docker", "container"}


def is_local_database_setup(state: dict[str, Any]) -> bool:
    if not _is_local_db_host(str(state.get("db_host", "")).strip()):
        return False
    if bool(state.get("fallback_enabled", False)):
        if not _is_local_db_host(str(state.get("fallback_db_host", "")).strip()):
            return False
        if not _is_local_db_mode(str(state.get("fallback_mode", "")).strip()):
            return False
    return True


def sync_db_state_from_config(state: dict[str, Any], config_data: dict[str, Any]) -> dict[str, Any]:
    synced: dict[str, Any] = dict(state)
    database = config_data.get("database")
    if isinstance(database, dict):
        if database.get("db_name") not in (None, ""):
            synced["db_name"] = str(database.get("db_name"))
        if database.get("user") not in (None, ""):
            synced["db_user"] = str(database.get("user"))
        if database.get("password") not in (None, ""):
            synced["db_password"] = str(database.get("password"))
        if database.get("host") not in (None, ""):
            synced["db_host"] = str(database.get("host"))
        if database.get("port") not in (None, ""):
            synced["db_port"] = str(database.get("port"))

    fallback = config_data.get("database_fallback")
    if isinstance(fallback, dict):
        synced["fallback_enabled"] = bool(fallback.get("enabled", synced.get("fallback_enabled", False)))
        if fallback.get("mode") not in (None, ""):
            synced["fallback_mode"] = str(fallback.get("mode"))
        if fallback.get("db_name") not in (None, ""):
            synced["fallback_db_name"] = str(fallback.get("db_name"))
        if fallback.get("user") not in (None, ""):
            synced["fallback_db_user"] = str(fallback.get("user"))
        if fallback.get("password") not in (None, ""):
            synced["fallback_db_password"] = str(fallback.get("password"))
        if fallback.get("host") not in (None, ""):
            synced["fallback_db_host"] = str(fallback.get("host"))
        if fallback.get("port") not in (None, ""):
            synced["fallback_db_port"] = str(fallback.get("port"))
    return synced


def infer_existing_host(config_data: dict) -> str | None:
    inference = config_data.get("inference")
    if not isinstance(inference, dict):
        return None

    for key in INFERENCE_KEYS:
        value = inference.get(key)
        if isinstance(value, dict):
            host = value.get("url")
            if is_valid_http_url(host):
                return host.rstrip("/")
    return None


def runtime_settings_for_config(config_data: dict) -> dict[str, Any]:
    return build_runtime_settings(config_data=config_data)


def runtime_string(config_data: dict, path: str, fallback: str) -> str:
    settings = runtime_settings_for_config(config_data)
    value = get_runtime_setting(settings, path, fallback)
    text = str(value).strip() if value is not None else ""
    return text if text else fallback


def choose_ollama_host(args: argparse.Namespace, existing_host: str | None) -> str:
    if args.ollama_host:
        if not is_valid_http_url(args.ollama_host):
            raise ValueError(f"Invalid --ollama-host value: {args.ollama_host}")
        return args.ollama_host.rstrip("/")

    if args.non_interactive:
        return (existing_host or args.default_host).rstrip("/")

    suggested = (existing_host or args.default_host).rstrip("/")
    while True:
        prompt = f"Enter Ollama endpoint [{suggested}]: "
        entry = input(prompt).strip()
        selected = suggested if entry == "" else entry
        if is_valid_http_url(selected):
            return selected.rstrip("/")
        print(f"Endpoint must be a valid http(s) URL, for example: {DEFAULT_OLLAMA_HOST}")


def probe_ollama_models(host: str) -> tuple[list[str], str | None]:
    if Client is None:
        return [], "python package 'ollama' is not installed; model discovery skipped"
    try:
        models = Client(host=host).list()
        names: list[str] = []
        for model in getattr(models, "models", []):
            model_name = getattr(model, "model", None)
            if isinstance(model_name, str) and model_name not in names:
                names.append(model_name)
        return names, None
    except Exception as error:  # noqa: BLE001
        return [], str(error)


def current_model(config_data: dict, key: str) -> str | None:
    inference = config_data.get("inference", {})
    section = inference.get(key)
    if isinstance(section, dict):
        model = section.get("model")
        if isinstance(model, str) and model.strip():
            return model.strip()
    return None


def resolve_model_value(
    key: str,
    explicit_value: str | None,
    config_data: dict,
    available_models: list[str],
) -> str:
    if explicit_value:
        return explicit_value

    existing = current_model(config_data, key)
    if existing and (not available_models or existing in available_models):
        return existing

    preferred = DEFAULT_MODELS.get(key)
    if available_models:
        if preferred in available_models:
            return preferred
        return available_models[0]

    if existing:
        return existing
    return preferred or ""


def set_inference_urls(config_data: dict, host: str) -> dict:
    inference = config_data.setdefault("inference", {})
    for key in INFERENCE_KEYS:
        section = inference.get(key)
        if not isinstance(section, dict):
            section = {}
            inference[key] = section
        section["url"] = host
    return config_data


def set_inference_models(config_data: dict, model_map: dict[str, str]) -> dict:
    inference = config_data.setdefault("inference", {})
    for key in INFERENCE_KEYS:
        section = inference.get(key)
        if not isinstance(section, dict):
            section = {}
            inference[key] = section
        model_value = model_map.get(key)
        if model_value:
            section["model"] = model_value
    return config_data


def _set_if_value(target: dict, key: str, value: str | None) -> None:
    if value is not None:
        target[key] = value


def set_database_settings(config_data: dict, args: argparse.Namespace) -> dict:
    runtimeSettings = runtime_settings_for_config(config_data)
    defaultPrimaryHost = str(get_runtime_setting(runtimeSettings, "database.default_primary_host", "127.0.0.1"))
    defaultPrimaryPort = str(get_runtime_setting(runtimeSettings, "database.default_primary_port", "5432"))
    defaultFallbackHost = str(get_runtime_setting(runtimeSettings, "database.default_fallback_host", "127.0.0.1"))
    defaultFallbackPort = str(get_runtime_setting(runtimeSettings, "database.default_fallback_port", "5433"))

    database = config_data.setdefault("database", {})
    if not isinstance(database, dict):
        database = {}
        config_data["database"] = database

    _set_if_value(database, "db_name", args.db_name)
    _set_if_value(database, "user", args.db_user)
    _set_if_value(database, "password", args.db_password)
    _set_if_value(database, "host", args.db_host)
    _set_if_value(database, "port", args.db_port)
    if not database.get("host"):
        database["host"] = defaultPrimaryHost
    if not database.get("port"):
        database["port"] = defaultPrimaryPort

    fallback = config_data.get("database_fallback")
    if not isinstance(fallback, dict):
        fallback = {}

    fallback_requested = any(
        value is not None
        for value in (
            args.fallback_db_name,
            args.fallback_db_user,
            args.fallback_db_password,
            args.fallback_db_host,
            args.fallback_db_port,
            args.fallback_mode,
        )
    ) or args.fallback_enabled or args.fallback_disabled or bool(fallback)

    if not fallback_requested:
        return config_data

    if args.fallback_enabled:
        fallback["enabled"] = True
    elif args.fallback_disabled:
        fallback["enabled"] = False
    elif "enabled" not in fallback:
        fallback["enabled"] = False

    _set_if_value(fallback, "mode", args.fallback_mode)
    _set_if_value(fallback, "db_name", args.fallback_db_name)
    _set_if_value(fallback, "user", args.fallback_db_user)
    _set_if_value(fallback, "password", args.fallback_db_password)
    _set_if_value(fallback, "host", args.fallback_db_host)
    _set_if_value(fallback, "port", args.fallback_db_port)

    if fallback.get("enabled") is True:
        if not fallback.get("db_name") and database.get("db_name"):
            fallback["db_name"] = f"{database.get('db_name')}_fallback"
        if not fallback.get("user") and database.get("user"):
            fallback["user"] = database.get("user")
        if not fallback.get("password") and database.get("password"):
            fallback["password"] = database.get("password")
        if not fallback.get("host"):
            fallback["host"] = defaultFallbackHost
        if not fallback.get("port"):
            fallback["port"] = defaultFallbackPort
        if not fallback.get("mode"):
            fallback["mode"] = "local"

    config_data["database_fallback"] = fallback
    return config_data


def ensure_defaults(config_data: dict) -> dict:
    if not isinstance(config_data.get("roles_list"), list):
        config_data["roles_list"] = ["user", "tester", "marketing", "admin", "owner"]
    config_data.setdefault("api_keys", {})
    config_data.setdefault("twitter_keys", {})
    return config_data


def validate_required_config(config_data: dict) -> list[str]:
    missing: list[str] = []

    required_simple = (
        "bot_name",
        "bot_id",
        "bot_token",
        "web_ui_url",
        "owner_info",
        "database",
        "roles_list",
        "defaults",
        "inference",
        "api_keys",
    )
    for key in required_simple:
        if config_data.get(key) in (None, "", []):
            missing.append(key)

    owner = config_data.get("owner_info", {})
    for owner_key in ("first_name", "last_name", "user_id", "username"):
        if owner.get(owner_key) in (None, ""):
            missing.append(f"owner_info.{owner_key}")

    database = config_data.get("database", {})
    for db_key in ("db_name", "user", "password", "host"):
        if database.get(db_key) in (None, ""):
            missing.append(f"database.{db_key}")

    inference = config_data.get("inference", {})
    for key in INFERENCE_KEYS:
        section = inference.get(key, {})
        if section.get("url") in (None, ""):
            missing.append(f"inference.{key}.url")
        if section.get("model") in (None, ""):
            missing.append(f"inference.{key}.model")

    return missing


def validate_required_telegram_config(config_data: dict) -> list[str]:
    missing: list[str] = []

    if not str(config_data.get("bot_name", "")).strip():
        missing.append("bot_name")
    if not is_valid_numeric_id(config_data.get("bot_id")):
        missing.append("bot_id")
    if not str(config_data.get("bot_token", "")).strip():
        missing.append("bot_token")

    web_ui_url = config_data.get("web_ui_url")
    if not is_valid_http_url(str(web_ui_url) if web_ui_url is not None else None):
        missing.append("web_ui_url")

    owner = config_data.get("owner_info")
    if not isinstance(owner, dict):
        owner = {}
    if not str(owner.get("first_name", "")).strip():
        missing.append("owner_info.first_name")
    if not str(owner.get("last_name", "")).strip():
        missing.append("owner_info.last_name")
    if not is_valid_numeric_id(owner.get("user_id")):
        missing.append("owner_info.user_id")
    if not str(owner.get("username", "")).strip():
        missing.append("owner_info.username")

    return missing


def backup_file(path: Path) -> Path | None:
    if not path.exists():
        return None
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    backup_path = path.with_suffix(path.suffix + f".bak-{stamp}")
    backup_path.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
    return backup_path


def parse_env_file(path: Path) -> tuple[list[str], dict[str, str]]:
    if not path.exists():
        return [], {}

    lines = path.read_text(encoding="utf-8").splitlines()
    values: dict[str, str] = {}
    for line in lines:
        striped = line.strip()
        if not striped or striped.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value
    return lines, values


def write_env_with_updates(env_path: Path, template_path: Path, updates: dict[str, str]) -> None:
    base_lines: list[str]
    if env_path.exists():
        base_lines, _ = parse_env_file(env_path)
    elif template_path.exists():
        base_lines = template_path.read_text(encoding="utf-8").splitlines()
    else:
        base_lines = []

    output_lines: list[str] = []
    seen_keys: set[str] = set()
    for line in base_lines:
        striped = line.strip()
        if not striped or striped.startswith("#") or "=" not in line:
            output_lines.append(line)
            continue
        key, _ = line.split("=", 1)
        key = key.strip()
        if key in updates:
            output_lines.append(f"{key}={updates[key]}")
            seen_keys.add(key)
        else:
            output_lines.append(line)

    for key, value in updates.items():
        if key not in seen_keys:
            output_lines.append(f"{key}={value}")

    env_path.write_text("\n".join(output_lines) + "\n", encoding="utf-8")


def _seeded_str(seed_state: dict[str, Any] | None, key: str, fallback: str) -> str:
    if not isinstance(seed_state, dict):
        return fallback
    value = seed_state.get(key)
    if value in (None, ""):
        return fallback
    return str(value)


def _seeded_bool(seed_state: dict[str, Any] | None, key: str, fallback: bool) -> bool:
    if not isinstance(seed_state, dict) or key not in seed_state:
        return fallback
    return bool(seed_state.get(key))


def _seeded_models(seed_state: dict[str, Any] | None) -> dict[str, str]:
    if not isinstance(seed_state, dict):
        return {}
    model_map = seed_state.get("model_map")
    if not isinstance(model_map, dict):
        return {}
    seeded: dict[str, str] = {}
    for key, value in model_map.items():
        if isinstance(key, str) and isinstance(value, str) and value.strip():
            seeded[key] = value.strip()
    return seeded


def _prompt_plain(prompt: str, default: str = "", required: bool = False, validator=None) -> str:
    while True:
        default_hint = f" [{default}]" if default else ""
        value = input(f"{prompt}{default_hint}: ").strip()
        if not value:
            value = default
        if required and not value:
            print("This value is required.")
            continue
        if validator and value and not validator(value):
            print("Invalid value.")
            continue
        return value


def _bool_prompt_plain(prompt: str, default: bool = True) -> bool:
    hint = "Y/n" if default else "y/N"
    raw = input(f"{prompt} [{hint}]: ").strip().lower()
    if not raw:
        return default
    return raw in {"y", "yes", "1", "true"}


def _select_model_plain(capability: str, default_model: str, available_models: list[str]) -> str:
    if available_models:
        print(f"Available models for {capability}:")
        for idx, name in enumerate(available_models, start=1):
            marker = "*" if name == default_model else " "
            print(f"  {idx}. {marker} {name}")
        selected = _prompt_plain(
            f"Select {capability} model by number or name",
            default=default_model,
            required=True,
        )
        if selected.isdigit():
            index = int(selected) - 1
            if 0 <= index < len(available_models):
                return available_models[index]
        return selected
    return _prompt_plain(f"Enter {capability} model", default=default_model, required=True)


class CursesUI:
    def __init__(self, stdscr):
        self.stdscr = stdscr
        curses.curs_set(1)

    def _render_title(self, title: str, subtitle: str | None = None):
        self.stdscr.clear()
        self.stdscr.addstr(0, 0, "RYO Setup Wizard", curses.A_BOLD)
        self.stdscr.addstr(1, 0, title, curses.A_UNDERLINE)
        if subtitle:
            self.stdscr.addstr(3, 0, subtitle)

    def message(self, title: str, text: str) -> None:
        self._render_title(title, text)
        self.stdscr.addstr(5, 0, "Press any key to continue.")
        self.stdscr.refresh()
        self.stdscr.getch()

    def prompt_text(
        self,
        title: str,
        label: str,
        default: str = "",
        required: bool = False,
        validator=None,
    ) -> str:
        while True:
            self._render_title(title, label)
            if default:
                self.stdscr.addstr(5, 0, f"Default: {default}")
            self.stdscr.addstr(7, 0, "> ")
            self.stdscr.refresh()
            curses.echo()
            raw = self.stdscr.getstr(7, 2, 220)
            curses.noecho()
            value = raw.decode(errors="ignore").strip()
            if not value:
                value = default
            if required and not value:
                self.message(title, "This value is required.")
                continue
            if validator and value and not validator(value):
                self.message(title, "Invalid value.")
                continue
            return value

    def prompt_yes_no(self, title: str, label: str, default: bool = True) -> bool:
        options = ["Yes", "No"]
        index = 0 if default else 1
        while True:
            self._render_title(title, label)
            self.stdscr.addstr(5, 0, "Use left/right arrows, Enter to confirm.")
            for i, option in enumerate(options):
                attr = curses.A_REVERSE if i == index else curses.A_NORMAL
                self.stdscr.addstr(7, i * 8, option, attr)
            self.stdscr.refresh()
            key = self.stdscr.getch()
            if key in (curses.KEY_LEFT, ord("h")):
                index = max(0, index - 1)
            elif key in (curses.KEY_RIGHT, ord("l")):
                index = min(len(options) - 1, index + 1)
            elif key in (10, 13):
                return index == 0

    def select_model(self, capability: str, models: list[str], default_model: str) -> str:
        if not models:
            return self.prompt_text(
                title=f"{capability} model",
                label="No models were discovered. Enter model manually:",
                default=default_model,
                required=True,
            )

        options = list(models) + ["Manual entry"]
        try:
            index = options.index(default_model)
        except ValueError:
            index = 0

        while True:
            self._render_title(f"Select model for {capability}", "Use arrows, Enter to confirm.")
            start = max(0, index - 8)
            end = min(len(options), start + 12)
            for row, option in enumerate(options[start:end], start=4):
                real_index = start + (row - 4)
                attr = curses.A_REVERSE if real_index == index else curses.A_NORMAL
                self.stdscr.addstr(row, 0, option, attr)
            self.stdscr.refresh()
            key = self.stdscr.getch()
            if key in (curses.KEY_UP, ord("k")):
                index = max(0, index - 1)
            elif key in (curses.KEY_DOWN, ord("j")):
                index = min(len(options) - 1, index + 1)
            elif key in (10, 13):
                choice = options[index]
                if choice == "Manual entry":
                    return self.prompt_text(
                        title=f"{capability} model",
                        label="Enter model manually:",
                        default=default_model,
                        required=True,
                    )
                return choice


def apply_setup_state(config_data: dict, state: dict) -> dict:
    config_data = apply_telegram_state(config_data, state)
    config_data = ensure_defaults(config_data)
    database = config_data.setdefault("database", {})
    fallback = config_data.get("database_fallback")
    if not isinstance(fallback, dict):
        fallback = {}
        config_data["database_fallback"] = fallback

    database["db_name"] = state["db_name"]
    database["user"] = state["db_user"]
    database["password"] = state["db_password"]
    database["host"] = state["db_host"]
    database["port"] = state["db_port"]

    fallback_enabled = state["fallback_enabled"]
    fallback["enabled"] = fallback_enabled
    if fallback_enabled:
        fallback["mode"] = state["fallback_mode"]
        fallback["db_name"] = state["fallback_db_name"]
        fallback["user"] = state["fallback_db_user"]
        fallback["password"] = state["fallback_db_password"]
        fallback["host"] = state["fallback_db_host"]
        fallback["port"] = state["fallback_db_port"]
    config_data["database_fallback"] = fallback

    api_keys = config_data.setdefault("api_keys", {})
    api_keys["brave_search"] = state.get("brave_search_key", "")

    twitter_keys = config_data.setdefault("twitter_keys", {})
    twitter_keys["consumer_key"] = state.get("twitter_consumer_key", "")
    twitter_keys["consumer_secret"] = state.get("twitter_consumer_secret", "")
    twitter_keys["access_token"] = state.get("twitter_access_token", "")
    twitter_keys["access_token_secret"] = state.get("twitter_access_token_secret", "")

    config_data = set_inference_urls(config_data, state["ollama_host"])
    config_data = set_inference_models(config_data, state["model_map"])
    return config_data


def apply_telegram_state(config_data: dict, state: dict) -> dict:
    config_data["bot_name"] = state["bot_name"]
    config_data["bot_id"] = int(state["bot_id"])
    config_data["bot_token"] = state["bot_token"]
    config_data["web_ui_url"] = state["web_ui_url"]

    owner = config_data.setdefault("owner_info", {})
    owner["first_name"] = state["owner_first_name"]
    owner["last_name"] = state["owner_last_name"]
    owner["user_id"] = int(state["owner_user_id"])
    owner["username"] = state["owner_username"]
    return config_data


def state_to_env_updates(state: dict, telegram_only: bool = False) -> dict[str, str]:
    telegram_updates = {
        "TELEGRAM_BOT_NAME": state["bot_name"],
        "TELEGRAM_BOT_ID": str(state["bot_id"]),
        "TELEGRAM_BOT_TOKEN": state["bot_token"],
        "TELEGRAM_OWNER_FIRST_NAME": state["owner_first_name"],
        "TELEGRAM_OWNER_LAST_NAME": state["owner_last_name"],
        "TELEGRAM_OWNER_USER_ID": str(state["owner_user_id"]),
        "TELEGRAM_OWNER_USERNAME": state["owner_username"],
        "WEB_UI_URL": state["web_ui_url"],
    }
    if telegram_only:
        return telegram_updates

    updates = {
        "OLLAMA_HOST": state["ollama_host"],
        "OLLAMA_EMBED_MODEL": state["model_map"]["embedding"],
        "OLLAMA_GENERATE_MODEL": state["model_map"]["generate"],
        "OLLAMA_CHAT_MODEL": state["model_map"]["chat"],
        "OLLAMA_TOOL_MODEL": state["model_map"]["tool"],
        "OLLAMA_MULTIMODAL_MODEL": state["model_map"]["multimodal"],
        "RYO_DEFAULT_OLLAMA_HOST": state["ollama_host"],
        "RYO_DEFAULT_EMBEDDING_MODEL": state["model_map"]["embedding"],
        "RYO_DEFAULT_GENERATE_MODEL": state["model_map"]["generate"],
        "RYO_DEFAULT_CHAT_MODEL": state["model_map"]["chat"],
        "RYO_DEFAULT_TOOL_MODEL": state["model_map"]["tool"],
        "RYO_DEFAULT_MULTIMODAL_MODEL": state["model_map"]["multimodal"],
        "POSTGRES_DB": state["db_name"],
        "POSTGRES_USER": state["db_user"],
        "POSTGRES_PASSWORD": state["db_password"],
        "POSTGRES_HOST": state["db_host"],
        "POSTGRES_PORT": state["db_port"],
        "POSTGRES_FALLBACK_ENABLED": "true" if state["fallback_enabled"] else "false",
        "POSTGRES_FALLBACK_MODE": state["fallback_mode"] if state["fallback_enabled"] else "local",
        "POSTGRES_FALLBACK_DB": state["fallback_db_name"] if state["fallback_enabled"] else "",
        "POSTGRES_FALLBACK_USER": state["fallback_db_user"] if state["fallback_enabled"] else "",
        "POSTGRES_FALLBACK_PASSWORD": state["fallback_db_password"] if state["fallback_enabled"] else "",
        "POSTGRES_FALLBACK_HOST": state["fallback_db_host"] if state["fallback_enabled"] else "",
        "POSTGRES_FALLBACK_PORT": state["fallback_db_port"] if state["fallback_enabled"] else "",
        "BRAVE_SEARCH_API_KEY": state.get("brave_search_key", ""),
        "TWITTER_CONSUMER_KEY": state.get("twitter_consumer_key", ""),
        "TWITTER_CONSUMER_SECRET": state.get("twitter_consumer_secret", ""),
        "TWITTER_ACCESS_TOKEN": state.get("twitter_access_token", ""),
        "TWITTER_ACCESS_TOKEN_SECRET": state.get("twitter_access_token_secret", ""),
    }
    updates.update(telegram_updates)
    return updates


def build_state_non_interactive(args: argparse.Namespace, config_data: dict) -> dict:
    config_data = ensure_defaults(config_data)
    runtimeSettings = runtime_settings_for_config(config_data)
    existing_host = infer_existing_host(config_data)
    selected_host = choose_ollama_host(args, existing_host)
    models, _ = probe_ollama_models(selected_host)

    model_map = {
        "embedding": resolve_model_value("embedding", args.embedding_model, config_data, models),
        "generate": resolve_model_value("generate", args.generate_model, config_data, models),
        "chat": resolve_model_value("chat", args.chat_model, config_data, models),
        "tool": resolve_model_value("tool", args.tool_model, config_data, models),
        "multimodal": resolve_model_value("multimodal", args.multimodal_model, config_data, models),
    }

    owner = config_data.get("owner_info", {})
    database = config_data.get("database", {})
    fallback = config_data.get("database_fallback", {})
    twitter = config_data.get("twitter_keys", {})
    api_keys = config_data.get("api_keys", {})
    defaultPrimaryHost = str(get_runtime_setting(runtimeSettings, "database.default_primary_host", "127.0.0.1"))
    defaultPrimaryPort = str(get_runtime_setting(runtimeSettings, "database.default_primary_port", "5432"))
    defaultFallbackHost = str(get_runtime_setting(runtimeSettings, "database.default_fallback_host", "127.0.0.1"))
    defaultFallbackPort = str(get_runtime_setting(runtimeSettings, "database.default_fallback_port", "5433"))

    fallback_enabled = args.fallback_enabled or (bool(fallback.get("enabled")) and not args.fallback_disabled)
    state = {
        "ollama_host": selected_host,
        "model_map": model_map,
        "bot_name": args.bot_name or config_data.get("bot_name", ""),
        "bot_id": args.bot_id or config_data.get("bot_id", 0),
        "bot_token": args.bot_token or config_data.get("bot_token", ""),
        "web_ui_url": args.web_ui_url or config_data.get("web_ui_url", ""),
        "owner_first_name": args.owner_first_name or owner.get("first_name", ""),
        "owner_last_name": args.owner_last_name or owner.get("last_name", ""),
        "owner_user_id": args.owner_user_id or owner.get("user_id", 0),
        "owner_username": args.owner_username or owner.get("username", ""),
        "db_name": args.db_name or database.get("db_name", ""),
        "db_user": args.db_user or database.get("user", ""),
        "db_password": args.db_password or database.get("password", ""),
        "db_host": args.db_host or database.get("host", defaultPrimaryHost),
        "db_port": args.db_port or database.get("port", defaultPrimaryPort),
        "fallback_enabled": fallback_enabled,
        "fallback_mode": args.fallback_mode or fallback.get("mode", "local"),
        "fallback_db_name": args.fallback_db_name or fallback.get("db_name", ""),
        "fallback_db_user": args.fallback_db_user or fallback.get("user", ""),
        "fallback_db_password": args.fallback_db_password or fallback.get("password", ""),
        "fallback_db_host": args.fallback_db_host or fallback.get("host", defaultFallbackHost),
        "fallback_db_port": args.fallback_db_port or fallback.get("port", defaultFallbackPort),
        "brave_search_key": args.brave_search_key if args.brave_search_key is not None else api_keys.get("brave_search", ""),
        "twitter_consumer_key": args.twitter_consumer_key if args.twitter_consumer_key is not None else twitter.get("consumer_key", ""),
        "twitter_consumer_secret": args.twitter_consumer_secret if args.twitter_consumer_secret is not None else twitter.get("consumer_secret", ""),
        "twitter_access_token": args.twitter_access_token if args.twitter_access_token is not None else twitter.get("access_token", ""),
        "twitter_access_token_secret": args.twitter_access_token_secret if args.twitter_access_token_secret is not None else twitter.get("access_token_secret", ""),
        "bootstrap_postgres": False,
        "bootstrap_docker": False,
        "write_env": True,
    }
    local_defaults_mode = is_local_database_setup(state)
    state["bootstrap_postgres"] = bool(args.bootstrap_postgres) or local_defaults_mode
    state["bootstrap_docker"] = bool(args.bootstrap_docker) or local_defaults_mode
    return state


def build_state_non_interactive_telegram(args: argparse.Namespace, config_data: dict) -> dict:
    owner = config_data.get("owner_info")
    if not isinstance(owner, dict):
        owner = {}
    return {
        "bot_name": args.bot_name or config_data.get("bot_name", ""),
        "bot_id": args.bot_id or config_data.get("bot_id", 0),
        "bot_token": args.bot_token or config_data.get("bot_token", ""),
        "web_ui_url": args.web_ui_url or config_data.get("web_ui_url", ""),
        "owner_first_name": args.owner_first_name or owner.get("first_name", ""),
        "owner_last_name": args.owner_last_name or owner.get("last_name", ""),
        "owner_user_id": args.owner_user_id or owner.get("user_id", 0),
        "owner_username": args.owner_username or owner.get("username", ""),
    }


def build_state_plain_interactive(
    args: argparse.Namespace,
    config_data: dict,
    seed_state: dict[str, Any] | None = None,
    state_path: Path | None = None,
) -> dict:
    owner = config_data.get("owner_info", {})
    database = config_data.get("database", {})
    fallback = config_data.get("database_fallback", {})
    twitter = config_data.get("twitter_keys", {})
    api_keys = config_data.get("api_keys", {})
    seeded_model_map = _seeded_models(seed_state)
    state: dict[str, Any] = dict(seed_state or {})
    runtimeSettings = runtime_settings_for_config(config_data)
    defaultPrimaryPort = str(get_runtime_setting(runtimeSettings, "database.default_primary_port", "5432"))
    defaultFallbackHost = str(get_runtime_setting(runtimeSettings, "database.default_fallback_host", "127.0.0.1"))
    defaultFallbackPort = str(get_runtime_setting(runtimeSettings, "database.default_fallback_port", "5433"))

    try:
        existing_host = _seeded_str(seed_state, "ollama_host", infer_existing_host(config_data) or "")
        selected_host = choose_ollama_host(args, existing_host or None)
        state["ollama_host"] = selected_host

        models, model_error = probe_ollama_models(selected_host)
        if model_error:
            print(f"Warning: failed to probe models: {model_error}")

        model_map = {}
        for key in INFERENCE_KEYS:
            default_model = seeded_model_map.get(key) or resolve_model_value(key, None, config_data, models)
            model_map[key] = _select_model_plain(key, default_model, models)
        state["model_map"] = model_map

        state["bot_name"] = _prompt_plain(
            "Bot name",
            default=_seeded_str(seed_state, "bot_name", str(config_data.get("bot_name", ""))),
            required=True,
        )
        state["bot_id"] = _prompt_plain(
            "Bot ID",
            default=_seeded_str(seed_state, "bot_id", str(config_data.get("bot_id", ""))),
            required=True,
            validator=lambda v: v.isdigit(),
        )
        state["bot_token"] = _prompt_plain(
            "Bot token",
            default=_seeded_str(seed_state, "bot_token", str(config_data.get("bot_token", ""))),
            required=True,
        )
        state["web_ui_url"] = _prompt_plain(
            "Web UI URL",
            default=_seeded_str(seed_state, "web_ui_url", str(config_data.get("web_ui_url", ""))),
            required=True,
            validator=is_valid_http_url,
        )
        state["owner_first_name"] = _prompt_plain(
            "Owner first name",
            default=_seeded_str(seed_state, "owner_first_name", str(owner.get("first_name", ""))),
            required=True,
        )
        state["owner_last_name"] = _prompt_plain(
            "Owner last name",
            default=_seeded_str(seed_state, "owner_last_name", str(owner.get("last_name", ""))),
            required=True,
        )
        state["owner_user_id"] = _prompt_plain(
            "Owner Telegram user ID",
            default=_seeded_str(seed_state, "owner_user_id", str(owner.get("user_id", ""))),
            required=True,
            validator=lambda v: v.isdigit(),
        )
        state["owner_username"] = _prompt_plain(
            "Owner Telegram username",
            default=_seeded_str(seed_state, "owner_username", str(owner.get("username", ""))),
            required=True,
        )
        state["db_name"] = _prompt_plain(
            "Primary DB name",
            default=_seeded_str(seed_state, "db_name", str(database.get("db_name", ""))),
            required=True,
        )
        state["db_user"] = _prompt_plain(
            "Primary DB user",
            default=_seeded_str(seed_state, "db_user", str(database.get("user", ""))),
            required=True,
        )
        state["db_password"] = _prompt_plain(
            "Primary DB password",
            default=_seeded_str(seed_state, "db_password", str(database.get("password", ""))),
            required=True,
        )
        state["db_host"] = _prompt_plain(
            "Primary DB host",
            default=_seeded_str(seed_state, "db_host", str(database.get("host", ""))),
            required=True,
        )
        state["db_port"] = _prompt_plain(
            "Primary DB port",
            default=_seeded_str(seed_state, "db_port", str(database.get("port", defaultPrimaryPort))),
            required=True,
        )

        state["fallback_enabled"] = _bool_prompt_plain(
            "Enable fallback database routing?",
            default=_seeded_bool(seed_state, "fallback_enabled", bool(fallback.get("enabled", False))),
        )
        state["fallback_mode"] = _seeded_str(seed_state, "fallback_mode", "local")
        state["fallback_db_name"] = _seeded_str(seed_state, "fallback_db_name", "")
        state["fallback_db_user"] = _seeded_str(seed_state, "fallback_db_user", "")
        state["fallback_db_password"] = _seeded_str(seed_state, "fallback_db_password", "")
        state["fallback_db_host"] = _seeded_str(seed_state, "fallback_db_host", "")
        state["fallback_db_port"] = _seeded_str(seed_state, "fallback_db_port", "")

        if state["fallback_enabled"]:
            state["fallback_mode"] = _prompt_plain(
                "Fallback mode",
                default=_seeded_str(seed_state, "fallback_mode", str(fallback.get("mode", "local"))),
                required=True,
            )
            state["fallback_db_name"] = _prompt_plain(
                "Fallback DB name",
                default=_seeded_str(seed_state, "fallback_db_name", str(fallback.get("db_name", f"{state['db_name']}_fallback"))),
                required=True,
            )
            state["fallback_db_user"] = _prompt_plain(
                "Fallback DB user",
                default=_seeded_str(seed_state, "fallback_db_user", str(fallback.get("user", state["db_user"]))),
                required=True,
            )
            state["fallback_db_password"] = _prompt_plain(
                "Fallback DB password",
                default=_seeded_str(seed_state, "fallback_db_password", str(fallback.get("password", state["db_password"]))),
                required=True,
            )
            state["fallback_db_host"] = _prompt_plain(
                "Fallback DB host",
                default=_seeded_str(seed_state, "fallback_db_host", str(fallback.get("host", defaultFallbackHost))),
                required=True,
            )
            state["fallback_db_port"] = _prompt_plain(
                "Fallback DB port",
                default=_seeded_str(seed_state, "fallback_db_port", str(fallback.get("port", defaultFallbackPort))),
                required=True,
            )

        state["brave_search_key"] = _prompt_plain(
            "Brave Search API key (optional)",
            default=_seeded_str(seed_state, "brave_search_key", str(api_keys.get("brave_search", ""))),
            required=False,
        )
        state["twitter_consumer_key"] = _prompt_plain(
            "Twitter consumer key (optional)",
            default=_seeded_str(seed_state, "twitter_consumer_key", str(twitter.get("consumer_key", ""))),
            required=False,
        )
        state["twitter_consumer_secret"] = _prompt_plain(
            "Twitter consumer secret (optional)",
            default=_seeded_str(seed_state, "twitter_consumer_secret", str(twitter.get("consumer_secret", ""))),
            required=False,
        )
        state["twitter_access_token"] = _prompt_plain(
            "Twitter access token (optional)",
            default=_seeded_str(seed_state, "twitter_access_token", str(twitter.get("access_token", ""))),
            required=False,
        )
        state["twitter_access_token_secret"] = _prompt_plain(
            "Twitter access token secret (optional)",
            default=_seeded_str(seed_state, "twitter_access_token_secret", str(twitter.get("access_token_secret", ""))),
            required=False,
        )
        local_defaults_mode = is_local_database_setup(state)
        state["write_env"] = True
        state["bootstrap_postgres"] = bool(args.bootstrap_postgres) or local_defaults_mode
        state["bootstrap_docker"] = bool(args.bootstrap_docker) or local_defaults_mode
        if state["bootstrap_postgres"] and state["bootstrap_docker"]:
            print("Local/default DB mode detected: PostgreSQL bootstrap will run automatically via Docker.")
        elif state["bootstrap_postgres"]:
            print("PostgreSQL bootstrap requested in direct-host mode.")
    except KeyboardInterrupt as error:
        if state_path is not None:
            save_partial_state(state_path, state)
        raise SetupCancelledError("Setup interrupted by user.") from error

    return state


def build_state_plain_interactive_telegram(
    args: argparse.Namespace,
    config_data: dict,
    seed_state: dict[str, Any] | None = None,
    state_path: Path | None = None,
) -> dict:
    owner = config_data.get("owner_info")
    if not isinstance(owner, dict):
        owner = {}
    state: dict[str, Any] = dict(seed_state or {})

    try:
        state["bot_name"] = _prompt_plain(
            "Bot name",
            default=_seeded_str(seed_state, "bot_name", str(config_data.get("bot_name", ""))),
            required=True,
        )
        state["bot_id"] = _prompt_plain(
            "Bot ID",
            default=_seeded_str(seed_state, "bot_id", str(config_data.get("bot_id", ""))),
            required=True,
            validator=lambda v: v.isdigit(),
        )
        state["bot_token"] = _prompt_plain(
            "Bot token",
            default=_seeded_str(seed_state, "bot_token", str(config_data.get("bot_token", ""))),
            required=True,
        )
        state["web_ui_url"] = _prompt_plain(
            "Web UI URL",
            default=_seeded_str(seed_state, "web_ui_url", str(config_data.get("web_ui_url", ""))),
            required=True,
            validator=is_valid_http_url,
        )
        state["owner_first_name"] = _prompt_plain(
            "Owner first name",
            default=_seeded_str(seed_state, "owner_first_name", str(owner.get("first_name", ""))),
            required=True,
        )
        state["owner_last_name"] = _prompt_plain(
            "Owner last name",
            default=_seeded_str(seed_state, "owner_last_name", str(owner.get("last_name", ""))),
            required=True,
        )
        state["owner_user_id"] = _prompt_plain(
            "Owner Telegram user ID",
            default=_seeded_str(seed_state, "owner_user_id", str(owner.get("user_id", ""))),
            required=True,
            validator=lambda v: v.isdigit(),
        )
        state["owner_username"] = _prompt_plain(
            "Owner Telegram username",
            default=_seeded_str(seed_state, "owner_username", str(owner.get("username", ""))),
            required=True,
        )
        state["write_env"] = _bool_prompt_plain(
            "Write/update .env with Telegram values?",
            default=_seeded_bool(seed_state, "write_env", args.write_env),
        )
    except KeyboardInterrupt as error:
        if state_path is not None:
            save_partial_state(state_path, state)
        raise SetupCancelledError("Setup interrupted by user.") from error

    return state


def build_state_curses(
    args: argparse.Namespace,
    config_data: dict,
    state_path: Path,
    seed_state: dict[str, Any] | None = None,
) -> dict:
    if curses is None:
        raise RuntimeError("python curses module is unavailable in this environment")

    owner = config_data.get("owner_info", {})
    database = config_data.get("database", {})
    fallback = config_data.get("database_fallback", {})
    twitter = config_data.get("twitter_keys", {})
    api_keys = config_data.get("api_keys", {})
    seeded_model_map = _seeded_models(seed_state)
    state: dict[str, Any] = dict(seed_state or {})
    runtimeSettings = runtime_settings_for_config(config_data)
    defaultPrimaryPort = str(get_runtime_setting(runtimeSettings, "database.default_primary_port", "5432"))
    defaultFallbackHost = str(get_runtime_setting(runtimeSettings, "database.default_fallback_host", "127.0.0.1"))
    defaultFallbackPort = str(get_runtime_setting(runtimeSettings, "database.default_fallback_port", "5433"))

    def _runner(stdscr):
        ui = CursesUI(stdscr)
        if args.config and Path(args.config).exists():
            proceed = ui.prompt_yes_no(
                title="Existing config detected",
                label=f"Update existing file '{args.config}'?",
                default=True,
            )
            if not proceed:
                raise SetupCancelledError("Setup cancelled by user.")

        existing_host = _seeded_str(seed_state, "ollama_host", infer_existing_host(config_data) or args.default_host)
        state["ollama_host"] = ui.prompt_text(
            title="Ollama endpoint",
            label="Enter custom endpoint or use default/existing:",
            default=existing_host,
            required=True,
            validator=is_valid_http_url,
        ).rstrip("/")

        models, model_error = probe_ollama_models(state["ollama_host"])
        if model_error:
            ui.message("Model probe warning", f"Could not probe models: {model_error}")
        else:
            ui.message("Model probe", f"Discovered {len(models)} model(s).")

        model_map = {}
        for key in INFERENCE_KEYS:
            default_model = seeded_model_map.get(key) or resolve_model_value(key, None, config_data, models)
            model_map[key] = ui.select_model(key, models, default_model)
        state["model_map"] = model_map

        state["bot_name"] = ui.prompt_text(
            "Telegram bot",
            "Bot name:",
            default=_seeded_str(seed_state, "bot_name", str(config_data.get("bot_name", ""))),
            required=True,
        )
        state["bot_id"] = ui.prompt_text(
            "Telegram bot",
            "Bot ID:",
            default=_seeded_str(seed_state, "bot_id", str(config_data.get("bot_id", ""))),
            required=True,
            validator=lambda v: v.isdigit(),
        )
        state["bot_token"] = ui.prompt_text(
            "Telegram bot",
            "Bot token:",
            default=_seeded_str(seed_state, "bot_token", str(config_data.get("bot_token", ""))),
            required=True,
        )
        state["web_ui_url"] = ui.prompt_text(
            "Web UI",
            "Web UI URL:",
            default=_seeded_str(seed_state, "web_ui_url", str(config_data.get("web_ui_url", ""))),
            required=True,
            validator=is_valid_http_url,
        )

        state["owner_first_name"] = ui.prompt_text(
            "Owner info",
            "Owner first name:",
            default=_seeded_str(seed_state, "owner_first_name", str(owner.get("first_name", ""))),
            required=True,
        )
        state["owner_last_name"] = ui.prompt_text(
            "Owner info",
            "Owner last name:",
            default=_seeded_str(seed_state, "owner_last_name", str(owner.get("last_name", ""))),
            required=True,
        )
        state["owner_user_id"] = ui.prompt_text(
            "Owner info",
            "Owner Telegram user ID:",
            default=_seeded_str(seed_state, "owner_user_id", str(owner.get("user_id", ""))),
            required=True,
            validator=lambda v: v.isdigit(),
        )
        state["owner_username"] = ui.prompt_text(
            "Owner info",
            "Owner username:",
            default=_seeded_str(seed_state, "owner_username", str(owner.get("username", ""))),
            required=True,
        )

        state["db_name"] = ui.prompt_text(
            "Primary database",
            "DB name:",
            default=_seeded_str(seed_state, "db_name", str(database.get("db_name", ""))),
            required=True,
        )
        state["db_user"] = ui.prompt_text(
            "Primary database",
            "DB user:",
            default=_seeded_str(seed_state, "db_user", str(database.get("user", ""))),
            required=True,
        )
        state["db_password"] = ui.prompt_text(
            "Primary database",
            "DB password:",
            default=_seeded_str(seed_state, "db_password", str(database.get("password", ""))),
            required=True,
        )
        state["db_host"] = ui.prompt_text(
            "Primary database",
            "DB host:",
            default=_seeded_str(seed_state, "db_host", str(database.get("host", ""))),
            required=True,
        )
        state["db_port"] = ui.prompt_text(
            "Primary database",
            "DB port:",
            default=_seeded_str(seed_state, "db_port", str(database.get("port", defaultPrimaryPort))),
            required=True,
        )

        state["fallback_enabled"] = ui.prompt_yes_no(
            "Fallback database",
            "Enable fallback DB routing?",
            default=_seeded_bool(seed_state, "fallback_enabled", bool(fallback.get("enabled", False))),
        )
        state["fallback_mode"] = _seeded_str(seed_state, "fallback_mode", "local")
        state["fallback_db_name"] = _seeded_str(seed_state, "fallback_db_name", "")
        state["fallback_db_user"] = _seeded_str(seed_state, "fallback_db_user", "")
        state["fallback_db_password"] = _seeded_str(seed_state, "fallback_db_password", "")
        state["fallback_db_host"] = _seeded_str(seed_state, "fallback_db_host", "")
        state["fallback_db_port"] = _seeded_str(seed_state, "fallback_db_port", "")

        if state["fallback_enabled"]:
            state["fallback_mode"] = ui.prompt_text(
                "Fallback database",
                "Fallback mode:",
                default=_seeded_str(seed_state, "fallback_mode", str(fallback.get("mode", "local"))),
                required=True,
            )
            state["fallback_db_name"] = ui.prompt_text(
                "Fallback database",
                "Fallback DB name:",
                default=_seeded_str(seed_state, "fallback_db_name", str(fallback.get("db_name", f"{state['db_name']}_fallback"))),
                required=True,
            )
            state["fallback_db_user"] = ui.prompt_text(
                "Fallback database",
                "Fallback DB user:",
                default=_seeded_str(seed_state, "fallback_db_user", str(fallback.get("user", state["db_user"]))),
                required=True,
            )
            state["fallback_db_password"] = ui.prompt_text(
                "Fallback database",
                "Fallback DB password:",
                default=_seeded_str(seed_state, "fallback_db_password", str(fallback.get("password", state["db_password"]))),
                required=True,
            )
            state["fallback_db_host"] = ui.prompt_text(
                "Fallback database",
                "Fallback DB host:",
                default=_seeded_str(seed_state, "fallback_db_host", str(fallback.get("host", defaultFallbackHost))),
                required=True,
            )
            state["fallback_db_port"] = ui.prompt_text(
                "Fallback database",
                "Fallback DB port:",
                default=_seeded_str(seed_state, "fallback_db_port", str(fallback.get("port", defaultFallbackPort))),
                required=True,
            )

        state["brave_search_key"] = ui.prompt_text(
            "API keys",
            "Brave Search API key (optional):",
            default=_seeded_str(seed_state, "brave_search_key", str(api_keys.get("brave_search", ""))),
            required=False,
        )
        state["twitter_consumer_key"] = ui.prompt_text(
            "API keys",
            "Twitter consumer key (optional):",
            default=_seeded_str(seed_state, "twitter_consumer_key", str(twitter.get("consumer_key", ""))),
            required=False,
        )
        state["twitter_consumer_secret"] = ui.prompt_text(
            "API keys",
            "Twitter consumer secret (optional):",
            default=_seeded_str(seed_state, "twitter_consumer_secret", str(twitter.get("consumer_secret", ""))),
            required=False,
        )
        state["twitter_access_token"] = ui.prompt_text(
            "API keys",
            "Twitter access token (optional):",
            default=_seeded_str(seed_state, "twitter_access_token", str(twitter.get("access_token", ""))),
            required=False,
        )
        state["twitter_access_token_secret"] = ui.prompt_text(
            "API keys",
            "Twitter access token secret (optional):",
            default=_seeded_str(seed_state, "twitter_access_token_secret", str(twitter.get("access_token_secret", ""))),
            required=False,
        )

        local_defaults_mode = is_local_database_setup(state)
        state["write_env"] = True
        state["bootstrap_postgres"] = bool(args.bootstrap_postgres) or local_defaults_mode
        state["bootstrap_docker"] = bool(args.bootstrap_docker) or local_defaults_mode

        summary = (
            f"Endpoint: {state['ollama_host']}\n"
            f"Primary DB: {state['db_host']}:{state['db_port']}/{state['db_name']}\n"
            f"Fallback enabled: {state['fallback_enabled']}\n"
            f"Run DB bootstrap: {state['bootstrap_postgres']} (docker={state['bootstrap_docker']})\n"
            "Write/update .env: True\n"
            "Save changes to config file?"
        )
        confirmed = ui.prompt_yes_no("Confirm", summary, default=True)
        if not confirmed:
            raise SetupCancelledError("Setup cancelled by user.")

    try:
        curses.wrapper(_runner)
    except SetupCancelledError:
        save_partial_state(state_path, state)
        raise
    except KeyboardInterrupt as error:
        save_partial_state(state_path, state)
        raise SetupCancelledError("Setup interrupted by user.") from error
    except Exception:
        save_partial_state(state_path, state)
        raise

    return state


def build_state_curses_telegram(
    args: argparse.Namespace,
    config_data: dict,
    state_path: Path,
    seed_state: dict[str, Any] | None = None,
) -> dict:
    if curses is None:
        raise RuntimeError("python curses module is unavailable in this environment")

    owner = config_data.get("owner_info")
    if not isinstance(owner, dict):
        owner = {}
    state: dict[str, Any] = dict(seed_state or {})

    def _runner(stdscr):
        ui = CursesUI(stdscr)
        if args.config and Path(args.config).exists():
            proceed = ui.prompt_yes_no(
                title="Existing config detected",
                label=f"Update existing file '{args.config}'?",
                default=True,
            )
            if not proceed:
                raise SetupCancelledError("Setup cancelled by user.")

        ui.message(
            "Telegram-only setup",
            "This mode updates Telegram credentials/owner fields only and preserves model/database settings.",
        )

        state["bot_name"] = ui.prompt_text(
            "Telegram bot",
            "Bot name:",
            default=_seeded_str(seed_state, "bot_name", str(config_data.get("bot_name", ""))),
            required=True,
        )
        state["bot_id"] = ui.prompt_text(
            "Telegram bot",
            "Bot ID:",
            default=_seeded_str(seed_state, "bot_id", str(config_data.get("bot_id", ""))),
            required=True,
            validator=lambda v: v.isdigit(),
        )
        state["bot_token"] = ui.prompt_text(
            "Telegram bot",
            "Bot token:",
            default=_seeded_str(seed_state, "bot_token", str(config_data.get("bot_token", ""))),
            required=True,
        )
        state["web_ui_url"] = ui.prompt_text(
            "Web UI",
            "Web UI URL:",
            default=_seeded_str(seed_state, "web_ui_url", str(config_data.get("web_ui_url", ""))),
            required=True,
            validator=is_valid_http_url,
        )

        state["owner_first_name"] = ui.prompt_text(
            "Owner info",
            "Owner first name:",
            default=_seeded_str(seed_state, "owner_first_name", str(owner.get("first_name", ""))),
            required=True,
        )
        state["owner_last_name"] = ui.prompt_text(
            "Owner info",
            "Owner last name:",
            default=_seeded_str(seed_state, "owner_last_name", str(owner.get("last_name", ""))),
            required=True,
        )
        state["owner_user_id"] = ui.prompt_text(
            "Owner info",
            "Owner Telegram user ID:",
            default=_seeded_str(seed_state, "owner_user_id", str(owner.get("user_id", ""))),
            required=True,
            validator=lambda v: v.isdigit(),
        )
        state["owner_username"] = ui.prompt_text(
            "Owner info",
            "Owner username:",
            default=_seeded_str(seed_state, "owner_username", str(owner.get("username", ""))),
            required=True,
        )

        state["write_env"] = ui.prompt_yes_no(
            "Environment file",
            "Write/update .env with Telegram values?",
            default=_seeded_bool(seed_state, "write_env", args.write_env),
        )

        summary = (
            f"Bot: {state['bot_name']}\n"
            f"Bot ID: {state['bot_id']}\n"
            f"Web UI URL: {state['web_ui_url']}\n"
            "Save Telegram-only changes to config file?"
        )
        confirmed = ui.prompt_yes_no("Confirm", summary, default=True)
        if not confirmed:
            raise SetupCancelledError("Setup cancelled by user.")

    try:
        curses.wrapper(_runner)
    except SetupCancelledError:
        save_partial_state(state_path, state)
        raise
    except KeyboardInterrupt as error:
        save_partial_state(state_path, state)
        raise SetupCancelledError("Setup interrupted by user.") from error
    except Exception:
        save_partial_state(state_path, state)
        raise

    return state


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="RYO setup wizard for endpoint, model, and key ingress configuration.")
    parser.add_argument("--config", default="config.json", help="Path to runtime config file.")
    parser.add_argument("--template", default="config.empty.json", help="Path to template used when config file does not exist.")
    parser.add_argument("--state-path", default=DEFAULT_STATE_PATH, help=f"Path for partial setup state (default: {DEFAULT_STATE_PATH}).")
    parser.add_argument("--env-path", default=".env", help="Path to .env file to update when writing env values.")
    parser.add_argument("--env-template", default=".env.example", help="Template used when creating .env.")
    parser.add_argument("--write-env", action="store_true", help="Write/update .env with wizard values.")
    parser.add_argument("--bootstrap-postgres", action="store_true", help="Run PostgreSQL + pgvector bootstrap after writing config.")
    parser.add_argument("--bootstrap-docker", action="store_true", help="Use dockerized PostgreSQL provisioning when running bootstrap.")
    parser.add_argument(
        "--bootstrap-target",
        choices=("primary", "fallback", "both"),
        default=None,
        help="Bootstrap target override. Default: primary unless fallback is enabled, then both.",
    )
    parser.add_argument(
        "--telegram-only",
        action="store_true",
        help="Update Telegram credentials/owner fields only. Preserves inference/model/database sections.",
    )

    parser.add_argument("--ollama-host", default=None, help="Explicit Ollama endpoint URL.")
    parser.add_argument("--default-host", default=DEFAULT_OLLAMA_HOST, help=f"Default local endpoint (default: {DEFAULT_OLLAMA_HOST}).")
    parser.add_argument("--embedding-model", default=None, help="Embedding model override.")
    parser.add_argument("--generate-model", default=None, help="Generate model override.")
    parser.add_argument("--chat-model", default=None, help="Chat model override.")
    parser.add_argument("--tool-model", default=None, help="Tool model override.")
    parser.add_argument("--multimodal-model", default=None, help="Multimodal model override.")

    parser.add_argument("--bot-name", default=None)
    parser.add_argument("--bot-id", default=None)
    parser.add_argument("--bot-token", default=None)
    parser.add_argument("--web-ui-url", default=None)
    parser.add_argument("--owner-first-name", default=None)
    parser.add_argument("--owner-last-name", default=None)
    parser.add_argument("--owner-user-id", default=None)
    parser.add_argument("--owner-username", default=None)

    parser.add_argument("--db-name", default=None, help="Primary PostgreSQL database name.")
    parser.add_argument("--db-user", default=None, help="Primary PostgreSQL username.")
    parser.add_argument("--db-password", default=None, help="Primary PostgreSQL password.")
    parser.add_argument("--db-host", default=None, help="Primary PostgreSQL host.")
    parser.add_argument("--db-port", default=None, help="Primary PostgreSQL port.")
    parser.add_argument("--fallback-enabled", action="store_true", help="Enable fallback PostgreSQL routing.")
    parser.add_argument("--fallback-disabled", action="store_true", help="Disable fallback PostgreSQL routing.")
    parser.add_argument("--fallback-mode", default=None, help="Fallback mode label, for example: local.")
    parser.add_argument("--fallback-db-name", default=None, help="Fallback PostgreSQL database name.")
    parser.add_argument("--fallback-db-user", default=None, help="Fallback PostgreSQL username.")
    parser.add_argument("--fallback-db-password", default=None, help="Fallback PostgreSQL password.")
    parser.add_argument("--fallback-db-host", default=None, help="Fallback PostgreSQL host.")
    parser.add_argument("--fallback-db-port", default=None, help="Fallback PostgreSQL port.")

    parser.add_argument("--brave-search-key", default=None)
    parser.add_argument("--twitter-consumer-key", default=None)
    parser.add_argument("--twitter-consumer-secret", default=None)
    parser.add_argument("--twitter-access-token", default=None)
    parser.add_argument("--twitter-access-token-secret", default=None)

    parser.add_argument("--non-interactive", action="store_true", help="Do not prompt; use argument/config precedence.")
    parser.add_argument("--no-curses", action="store_true", help="Use plain text prompts instead of curses.")
    parser.add_argument("--strict", action="store_true", help="Fail if required config values remain missing after merge.")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    load_dotenv_file(Path(args.env_path), override=False)

    if args.fallback_enabled and args.fallback_disabled:
        raise ValueError("Use either --fallback-enabled or --fallback-disabled, not both.")

    config_path = Path(args.config)
    template_path = Path(args.template)
    state_path = Path(args.state_path)
    env_path = Path(args.env_path)
    env_template = Path(args.env_template)

    if config_path.exists():
        try:
            config_data = load_json(config_path)
        except ValueError as error:
            print(f"Warning: {error}. Falling back to template: {template_path}")
            if not template_path.exists():
                raise FileNotFoundError(f"Missing valid config and template. Expected {config_path} or {template_path}.") from error
            config_data = load_json(template_path)
    else:
        if not template_path.exists():
            raise FileNotFoundError(f"Missing config and template. Expected {config_path} or {template_path}.")
        config_data = load_json(template_path)
    config_data = ensure_defaults(config_data)

    seed_state: dict[str, Any] = {}
    if not args.non_interactive:
        seed_state = load_partial_state(state_path)
        if seed_state:
            print(f"Loaded partial setup state from: {state_path}")

    if args.non_interactive:
        if args.telegram_only:
            state = build_state_non_interactive_telegram(args, config_data)
        else:
            state = build_state_non_interactive(args, config_data)
    else:
        use_curses = (not args.no_curses) and sys.stdin.isatty() and sys.stdout.isatty()
        if use_curses:
            try:
                if args.telegram_only:
                    state = build_state_curses_telegram(
                        args,
                        config_data,
                        state_path=state_path,
                        seed_state=seed_state,
                    )
                else:
                    state = build_state_curses(
                        args,
                        config_data,
                        state_path=state_path,
                        seed_state=seed_state,
                    )
            except SetupCancelledError as error:
                print(f"{error} Partial state saved to: {state_path}")
                return 1
            except Exception as error:  # noqa: BLE001
                print(f"Curses wizard unavailable ({error}). Falling back to plain prompts.")
                seed_state = load_partial_state(state_path)
                try:
                    if args.telegram_only:
                        state = build_state_plain_interactive_telegram(
                            args,
                            config_data,
                            seed_state=seed_state,
                            state_path=state_path,
                        )
                    else:
                        state = build_state_plain_interactive(
                            args,
                            config_data,
                            seed_state=seed_state,
                            state_path=state_path,
                        )
                except SetupCancelledError as inner_error:
                    print(f"{inner_error} Partial state saved to: {state_path}")
                    return 1
        else:
            try:
                if args.telegram_only:
                    state = build_state_plain_interactive_telegram(
                        args,
                        config_data,
                        seed_state=seed_state,
                        state_path=state_path,
                    )
                else:
                    state = build_state_plain_interactive(
                        args,
                        config_data,
                        seed_state=seed_state,
                        state_path=state_path,
                    )
            except SetupCancelledError as error:
                print(f"{error} Partial state saved to: {state_path}")
                return 1

    if args.telegram_only:
        merged = apply_telegram_state(config_data, state)
        missing = validate_required_telegram_config(merged)
    else:
        merged = apply_setup_state(config_data, state)
        missing = validate_required_config(merged)
    strict_mode = args.strict or not args.non_interactive
    if strict_mode and missing:
        raise ValueError(f"Missing required config values after setup: {', '.join(missing)}")

    backup = backup_file(config_path)
    write_json(config_path, merged)

    bootstrap_requested = (
        (not args.telegram_only)
        and (args.bootstrap_postgres or bool(state.get("bootstrap_postgres", False)))
    )
    if (not args.telegram_only) and not bootstrap_requested and is_local_database_setup(state):
        bootstrap_requested = True
        state["bootstrap_postgres"] = True
        state["bootstrap_docker"] = True

    if bootstrap_requested:
        bootstrap_target = resolve_bootstrap_target(
            merged,
            explicit_target=args.bootstrap_target,
        )
        bootstrap_use_docker = args.bootstrap_docker or bool(state.get("bootstrap_docker", False))
        ok, output = run_postgres_bootstrap(
            config_path=config_path,
            target=bootstrap_target,
            use_docker=bootstrap_use_docker,
        )
        print(f"PostgreSQL bootstrap target: {bootstrap_target}")
        if bootstrap_use_docker:
            print("PostgreSQL bootstrap mode: docker")
        else:
            print("PostgreSQL bootstrap mode: direct host")
        if output:
            print(output)
        if not ok:
            print("PostgreSQL bootstrap failed; config was saved but bootstrap did not complete.")
            return 1
        print("PostgreSQL bootstrap completed successfully.")
        state = sync_db_state_from_config(state, load_json(config_path))

    should_write_env = (
        (not args.telegram_only)
        or args.write_env
        or bool(state.get("write_env", False))
    )
    if should_write_env:
        updates = state_to_env_updates(state, telegram_only=args.telegram_only)
        write_env_with_updates(env_path, env_template, updates)
        print(f"Wrote env values: {env_path}")

    if backup:
        print(f"Backed up existing config to: {backup}")
    clear_partial_state(state_path)
    print(f"Wrote config: {config_path}")
    if args.telegram_only:
        print("Setup mode: telegram-only")
        print("Updated Telegram credentials/owner info; preserved inference and database settings.")
    else:
        print("Setup precedence (host): explicit custom -> existing config -> default local")
        print(f"Ollama endpoint in use: {state['ollama_host']}")
    print("Startup hints:")
    print("  python3 scripts/bootstrap_postgres.py --config config.json --target both")
    print("  python3 telegram_ui.py")
    print("  python3 web_ui.py")
    print("  python3 cli_ui.py")
    print("  python3 scripts/policy_wizard.py --validate-only")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
