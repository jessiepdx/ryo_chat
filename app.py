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
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import urlopen


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
    "embedding": "nomic-embed-text:latest",
    "generate": "llama3.2:latest",
    "chat": "llama3.2:latest",
    "tool": "llama3.2:latest",
    "multimodal": "llama3.2-vision:latest",
}
SETUP_PLACEHOLDER_VALUES = {
    "telegram_bot_username",
    "telegram_bot_token",
}


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
    run_command([sys.executable, "-m", "pip", "install", "-r", str(REQUIREMENTS_FILE)], cwd=PROJECT_ROOT, check=True)
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
    command = [sys.executable, "scripts/setup_wizard.py"]
    if non_interactive:
        command.append("--non-interactive")
    print(f"[bootstrap] Running setup wizard: {' '.join(command)}")
    result = run_command(command, cwd=PROJECT_ROOT, check=False)
    if result.returncode != 0:
        print(
            "[bootstrap] Setup wizard did not complete successfully. "
            "You can rerun later with: python3 scripts/setup_wizard.py"
        )
        return False
    return True


def valid_http_url(value: str | None) -> bool:
    if not value:
        return False
    parsed = urlparse(str(value).strip())
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def resolve_ollama_host(config_data: dict[str, Any]) -> str:
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

    return DEFAULT_OLLAMA_HOST


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

    for key in INFERENCE_KEYS:
        add(current_model(config_data, key))
        add(DEFAULT_MODELS.get(key))

    return output


def prompt_yes_no(prompt: str, default: bool = True, non_interactive: bool = False) -> bool:
    if non_interactive:
        return default
    hint = "Y/n" if default else "y/N"
    raw = input(f"{prompt} [{hint}]: ").strip().lower()
    if raw == "":
        return default
    return raw in {"y", "yes", "1", "true"}


def pull_model(model_name: str) -> bool:
    if shutil.which("ollama") is None:
        print("[ollama] 'ollama' command not found on PATH; cannot auto-pull models.")
        return False
    print(f"[ollama] Pulling model: {model_name}")
    result = run_command(["ollama", "pull", model_name], cwd=PROJECT_ROOT, check=False)
    return result.returncode == 0


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

    if not current_model(config_data, "embedding"):
        inference["embedding"]["model"] = DEFAULT_MODELS["embedding"]
    if not current_model(config_data, "multimodal"):
        inference["multimodal"]["model"] = DEFAULT_MODELS["multimodal"]

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


class InterfaceWatchdog:
    def __init__(
        self,
        python_exec: str,
        *,
        restart_window_seconds: int = 60,
        max_restarts_per_window: int = 5,
    ):
        self._python_exec = python_exec
        self._routes: dict[str, RouteState] = {
            "web": RouteState(RouteSpec("web", "web_ui.py", restart_on_exit=True)),
            "telegram": RouteState(RouteSpec("telegram", "telegram_ui.py", restart_on_exit=True)),
            "cli": RouteState(RouteSpec("cli", "cli_ui.py", restart_on_exit=False)),
            "x": RouteState(RouteSpec("x", "x_ui.py", restart_on_exit=False)),
        }
        self._restart_window_seconds = restart_window_seconds
        self._max_restarts = max_restarts_per_window
        self._lock = threading.Lock()
        self._alive = True
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()

    def route_keys(self) -> list[str]:
        return list(self._routes.keys())

    def _log_path(self, key: str) -> Path:
        return WATCHDOG_LOG_DIR / f"{key}.log"

    def _spawn(self, state: RouteState) -> None:
        log_path = self._log_path(state.spec.key)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        state.log_handle = log_path.open("a", encoding="utf-8")
        command = [self._python_exec, state.spec.script]
        state.process = subprocess.Popen(
            command,
            cwd=str(PROJECT_ROOT),
            stdin=subprocess.DEVNULL,
            stdout=state.log_handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
        state.last_exit_code = None

    def _cleanup_process(self, state: RouteState, *, close_handle: bool = True) -> None:
        state.process = None
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
                return True
            self._cleanup_process(state)
            self._spawn(state)
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
                return True

            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=8)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=4)
            state.last_exit_code = process.returncode
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
                output[key] = {
                    "desired": state.desired,
                    "running": running,
                    "pid": pid,
                    "restart_count": state.restart_count,
                    "last_exit_code": state.last_exit_code,
                    "restart_on_exit": state.spec.restart_on_exit,
                    "log_file": str(self._log_path(key)),
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
                    self._cleanup_process(state)

                    if state.desired and state.spec.restart_on_exit:
                        if self._can_restart(state):
                            self._record_restart(state)
                            self._spawn(state)
                        else:
                            state.desired = False
                    elif state.desired and not state.spec.restart_on_exit:
                        # One-shot routes (for example CLI/X in current codebase) are
                        # intentionally not auto-restarted.
                        state.desired = False
            time.sleep(1.0)

    def shutdown(self) -> None:
        self._alive = False
        self.stop_all()
        self._thread.join(timeout=2.0)


def print_watchdog_status(watchdog: InterfaceWatchdog) -> None:
    status = watchdog.status()
    print("\n=== Interface Watchdog ===")
    print("Key       Desired  Running  PID      Restarts  LastExit  RestartPolicy")
    print("-----------------------------------------------------------------------")
    for key in watchdog.route_keys():
        entry = status[key]
        desired = "on" if entry["desired"] else "off"
        running = "yes" if entry["running"] else "no"
        pid = str(entry["pid"] or "-")
        restarts = str(entry["restart_count"])
        last_exit = str(entry["last_exit_code"]) if entry["last_exit_code"] is not None else "-"
        policy = "auto" if entry["restart_on_exit"] else "manual"
        print(
            f"{key:<9} {desired:<7} {running:<7} {pid:<8} {restarts:<9} {last_exit:<8} {policy}"
        )
    print("-----------------------------------------------------------------------")
    print("Log files live under: logs/watchdog/")


def route_menu(watchdog: InterfaceWatchdog) -> None:
    while True:
        print_watchdog_status(watchdog)
        print("\nActions:")
        print("  1. Toggle a route")
        print("  2. Start all routes")
        print("  3. Stop all routes")
        print("  4. Show route log paths")
        print("  0. Exit launcher")
        choice = input("Select action: ").strip()

        if choice == "1":
            keys = watchdog.route_keys()
            print("\nRoutes:")
            for index, key in enumerate(keys, start=1):
                print(f"  {index}. {key}")
            selected = input("Route number to toggle: ").strip()
            if selected.isdigit():
                idx = int(selected) - 1
                if 0 <= idx < len(keys):
                    watchdog.toggle(keys[idx])
            continue

        if choice == "2":
            watchdog.start_all()
            continue

        if choice == "3":
            watchdog.stop_all()
            continue

        if choice == "4":
            status = watchdog.status()
            for key in watchdog.route_keys():
                print(f"{key}: {status[key]['log_file']}")
            continue

        if choice == "0":
            return


def bootstrap_ollama_and_models(
    config_data: dict[str, Any],
    *,
    non_interactive: bool,
) -> dict[str, Any]:
    host = resolve_ollama_host(config_data)
    print(f"[ollama] Using host: {host}")

    models, error = fetch_ollama_models(host)
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

            refreshed, refresh_error = fetch_ollama_models(host)
            if refresh_error is None:
                models = refreshed

    if not models:
        return config_data

    current_text_model = current_model(config_data, "chat")
    if prompt_yes_no(
        "Select/update default text model (chat/generate/tool)?",
        default=(current_text_model is None),
        non_interactive=non_interactive,
    ):
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

    ensure_venv_and_reexec()
    ensure_requirements_installed()

    artifacts = ensure_project_artifacts()
    state = load_state()
    config_data = load_config()

    non_interactive = bool(args.non_interactive or not sys.stdin.isatty())

    if not args.skip_setup_wizard and should_run_setup(state, artifacts, config_data):
        setup_ok = run_setup_wizard(non_interactive=non_interactive)
        state["setup_completed"] = setup_ok
        save_state(state)
        config_data = load_config()
    else:
        state["setup_completed"] = bool(state.get("setup_completed", True))
        save_state(state)

    config_before = json.dumps(config_data, sort_keys=True)
    config_data = bootstrap_ollama_and_models(config_data, non_interactive=non_interactive)
    config_after = json.dumps(config_data, sort_keys=True)
    if config_before != config_after:
        backup = backup_file(CONFIG_FILE)
        write_json_atomic(CONFIG_FILE, config_data)
        if backup:
            print(f"[bootstrap] Backed up config to: {backup}")
        print(f"[bootstrap] Updated config: {CONFIG_FILE}")

    state["last_run_epoch"] = int(time.time())
    save_state(state)

    if args.non_interactive and not args.bootstrap_only:
        print("[launcher] --non-interactive mode completed bootstrap only.")
        return 0

    if args.bootstrap_only:
        print("[launcher] Bootstrap complete.")
        return 0

    watchdog = InterfaceWatchdog(sys.executable)
    if prompt_yes_no(
        "Start all managed routes now?",
        default=False,
        non_interactive=non_interactive,
    ):
        watchdog.start_all()
    try:
        route_menu(watchdog)
    except KeyboardInterrupt:
        print("\n[launcher] Shutdown requested.")
    finally:
        watchdog.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
