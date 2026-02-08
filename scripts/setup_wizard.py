#!/usr/bin/env python3
"""
Minimal setup helper for endpoint and inference host configuration.

This is the first implementation slice for WO-001 endpoint handling:
- Accept custom Ollama endpoint
- Fall back to existing config endpoint
- Fall back to default local endpoint
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse


DEFAULT_OLLAMA_HOST = "http://127.0.0.1:11434"
INFERENCE_KEYS = ("embedding", "generate", "chat", "tool", "multimodal")


def load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, data: dict) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)
        handle.write("\n")


def is_valid_http_url(value: str | None) -> bool:
    if not value:
        return False
    parsed = urlparse(value)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


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
        print("Endpoint must be a valid http(s) URL, for example: http://127.0.0.1:11434")


def set_inference_urls(config_data: dict, host: str) -> dict:
    inference = config_data.setdefault("inference", {})
    for key in INFERENCE_KEYS:
        section = inference.get(key)
        if not isinstance(section, dict):
            section = {}
            inference[key] = section
        section["url"] = host
    return config_data


def backup_file(path: Path) -> Path | None:
    if not path.exists():
        return None
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    backup_path = path.with_suffix(path.suffix + f".bak-{stamp}")
    backup_path.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
    return backup_path


def main() -> int:
    parser = argparse.ArgumentParser(description="RYO setup helper for Ollama endpoint configuration.")
    parser.add_argument("--config", default="config.json", help="Path to runtime config file.")
    parser.add_argument(
        "--template",
        default="config.empty.json",
        help="Path to template used when config file does not exist.",
    )
    parser.add_argument("--ollama-host", default=None, help="Explicit Ollama endpoint URL.")
    parser.add_argument(
        "--default-host",
        default=DEFAULT_OLLAMA_HOST,
        help=f"Default local endpoint when none exists (default: {DEFAULT_OLLAMA_HOST}).",
    )
    parser.add_argument(
        "--non-interactive",
        action="store_true",
        help="Do not prompt; use --ollama-host or fallback precedence.",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    template_path = Path(args.template)

    if config_path.exists():
        config_data = load_json(config_path)
    else:
        if not template_path.exists():
            raise FileNotFoundError(
                f"Missing config and template. Expected {config_path} or {template_path}."
            )
        config_data = load_json(template_path)

    existing_host = infer_existing_host(config_data)
    selected_host = choose_ollama_host(args, existing_host)

    if not is_valid_http_url(selected_host):
        raise ValueError(f"Invalid endpoint selection: {selected_host}")

    config_data = set_inference_urls(config_data, selected_host)

    backup = backup_file(config_path)
    write_json(config_path, config_data)

    if backup:
        print(f"Backed up existing config to: {backup}")
    print(f"Wrote config: {config_path}")
    print(f"Ollama endpoint in use: {selected_host}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
