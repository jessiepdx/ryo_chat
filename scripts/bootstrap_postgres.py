#!/usr/bin/env python3
"""
Idempotent PostgreSQL + pgvector bootstrap utility for RYO Chat.

Supports:
1. Manual target bootstrapping via CLI flags.
2. Config-driven bootstrapping from `config.json` primary/fallback sections.
3. Optional local Docker provisioning with pgvector image.
4. Extension and vector-type verification with actionable pass/fail output.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import psycopg
from psycopg import sql


DEFAULT_VERIFY_SQL = Path(__file__).with_name("verify_pgvector.sql")


@dataclass
class BootstrapTarget:
    label: str
    db_name: str
    user: str
    password: str
    host: str
    port: str
    maintenance_db: str = "postgres"

    @property
    def display(self) -> str:
        return f"{self.label}({self.user}@{self.host}:{self.port}/{self.db_name})"


def _conninfo(target: BootstrapTarget, db_name: str | None = None) -> str:
    database_name = db_name or target.db_name
    return (
        f"dbname={database_name} "
        f"user={target.user} "
        f"password={target.password} "
        f"host={target.host} "
        f"port={target.port}"
    )


def _redacted_conninfo(target: BootstrapTarget, db_name: str | None = None) -> str:
    database_name = db_name or target.db_name
    return (
        f"dbname={database_name} "
        f"user={target.user} "
        "password=*** "
        f"host={target.host} "
        f"port={target.port}"
    )


def _connect_with_retry(conninfo: str, retries: int, retry_delay: float):
    errors: list[str] = []
    attempts = max(1, retries + 1)
    for attempt in range(1, attempts + 1):
        try:
            return psycopg.connect(conninfo=conninfo, autocommit=True)
        except Exception as error:  # noqa: BLE001
            errors.append(str(error))
            if attempt < attempts:
                time.sleep(max(0.0, retry_delay))
    raise RuntimeError(errors[-1] if errors else "unknown connection failure")


def _database_exists(conninfo: str, target_db: str, retries: int, retry_delay: float) -> bool:
    with _connect_with_retry(conninfo, retries, retry_delay) as connection:
        with connection.cursor() as cursor:
            cursor.execute("SELECT 1 FROM pg_database WHERE datname = %s LIMIT 1;", (target_db,))
            return cursor.fetchone() is not None


def _create_database(conninfo: str, target_db: str, retries: int, retry_delay: float) -> None:
    with _connect_with_retry(conninfo, retries, retry_delay) as connection:
        with connection.cursor() as cursor:
            cursor.execute(sql.SQL("CREATE DATABASE {};").format(sql.Identifier(target_db)))


def _ensure_vector_extension(conninfo: str, retries: int, retry_delay: float) -> None:
    with _connect_with_retry(conninfo, retries, retry_delay) as connection:
        with connection.cursor() as cursor:
            cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")


def _run_sql_check(conninfo: str, sql_path: Path, retries: int, retry_delay: float) -> None:
    sql_text = sql_path.read_text(encoding="utf-8")
    with _connect_with_retry(conninfo, retries, retry_delay) as connection:
        with connection.cursor() as cursor:
            cursor.execute(sql_text)


def _verify_vector_support(conninfo: str, retries: int, retry_delay: float) -> None:
    with _connect_with_retry(conninfo, retries, retry_delay) as connection:
        with connection.cursor() as cursor:
            cursor.execute("SELECT EXISTS(SELECT 1 FROM pg_extension WHERE extname = 'vector');")
            extension_enabled = bool(cursor.fetchone()[0])
            if not extension_enabled:
                raise RuntimeError("pgvector extension is not enabled in target database.")

            cursor.execute("SELECT to_regtype('vector') IS NOT NULL;")
            vector_type_exists = bool(cursor.fetchone()[0])
            if not vector_type_exists:
                raise RuntimeError("vector type is unavailable even though extension was expected.")

            cursor.execute("CREATE TEMP TABLE __pgvector_probe (embedding vector(3));")
            cursor.execute("INSERT INTO __pgvector_probe (embedding) VALUES ('[1,2,3]');")
            cursor.execute("SELECT COUNT(*) FROM __pgvector_probe;")
            rows = int(cursor.fetchone()[0])
            if rows < 1:
                raise RuntimeError("vector probe insert/select check failed.")


def _required_fields_present(mapping: dict, required_keys: Iterable[str]) -> bool:
    for key in required_keys:
        value = mapping.get(key)
        if value is None or str(value).strip() == "":
            return False
    return True


def _target_from_mapping(label: str, mapping: dict, maintenance_db: str) -> BootstrapTarget | None:
    required_keys = ("db_name", "user", "password", "host")
    if not _required_fields_present(mapping, required_keys):
        return None
    return BootstrapTarget(
        label=label,
        db_name=str(mapping.get("db_name")).strip(),
        user=str(mapping.get("user")).strip(),
        password=str(mapping.get("password")),
        host=str(mapping.get("host")).strip(),
        port=str(mapping.get("port", "5432")).strip(),
        maintenance_db=maintenance_db,
    )


def load_targets_from_config(config_path: Path, target_mode: str, maintenance_db: str) -> list[BootstrapTarget]:
    with config_path.open("r", encoding="utf-8") as handle:
        config_data = json.load(handle)
    if not isinstance(config_data, dict):
        raise ValueError(f"Expected object JSON in {config_path}.")

    targets: list[BootstrapTarget] = []
    include_primary = target_mode in {"primary", "both"}
    include_fallback = target_mode in {"fallback", "both"}

    if include_primary:
        primary = config_data.get("database", {})
        if not isinstance(primary, dict):
            primary = {}
        target = _target_from_mapping("primary", primary, maintenance_db)
        if target is None:
            raise ValueError("Missing required `database.*` values in config for primary bootstrap target.")
        targets.append(target)

    if include_fallback:
        fallback = config_data.get("database_fallback", {})
        if not isinstance(fallback, dict):
            fallback = {}
        fallback_enabled = bool(fallback.get("enabled", False))
        if target_mode == "both" and not fallback_enabled:
            print("INFO: fallback target is disabled in config (`database_fallback.enabled=false`); skipping fallback.")
        else:
            target = _target_from_mapping("fallback", fallback, maintenance_db)
            if target is None:
                raise ValueError("Missing required `database_fallback.*` values in config for fallback bootstrap target.")
            targets.append(target)

    return targets


def _docker_exec(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, check=False, text=True, capture_output=True)


def _docker_container_exists(name: str) -> bool:
    result = _docker_exec(["docker", "ps", "-a", "--filter", f"name=^{name}$", "--format", "{{.Names}}"])
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "failed to inspect docker containers")
    return any(line.strip() == name for line in result.stdout.splitlines())


def _docker_container_running(name: str) -> bool:
    result = _docker_exec(["docker", "ps", "--filter", f"name=^{name}$", "--format", "{{.Names}}"])
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "failed to inspect running docker containers")
    return any(line.strip() == name for line in result.stdout.splitlines())


def _ensure_docker_container(
    target: BootstrapTarget,
    image: str,
    container_name: str,
    recreate: bool,
    volume: str | None,
) -> None:
    if shutil.which("docker") is None:
        raise RuntimeError("`docker` is not available on PATH.")

    exists = _docker_container_exists(container_name)
    running = _docker_container_running(container_name) if exists else False

    if exists and recreate:
        rm = _docker_exec(["docker", "rm", "-f", container_name])
        if rm.returncode != 0:
            raise RuntimeError(rm.stderr.strip() or f"failed to remove container {container_name}")
        exists = False
        running = False

    if exists and not running:
        start = _docker_exec(["docker", "start", container_name])
        if start.returncode != 0:
            raise RuntimeError(start.stderr.strip() or f"failed to start container {container_name}")
        print(f"Started existing docker container: {container_name}")
        return

    if exists and running:
        print(f"Docker container already running: {container_name}")
        return

    run_args = [
        "docker",
        "run",
        "-d",
        "--name",
        container_name,
        "-e",
        f"POSTGRES_DB={target.db_name}",
        "-e",
        f"POSTGRES_USER={target.user}",
        "-e",
        f"POSTGRES_PASSWORD={target.password}",
        "-p",
        f"{target.port}:5432",
    ]
    if volume:
        run_args.extend(["-v", volume])
    run_args.append(image)

    result = _docker_exec(run_args)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or f"failed to launch docker container {container_name}")
    print(f"Created docker container: {container_name} ({image})")


def bootstrap_target(
    target: BootstrapTarget,
    skip_db_create: bool,
    verify_only: bool,
    sql_check_path: Path | None,
    retries: int,
    retry_delay: float,
) -> bool:
    maintenance_conn = _conninfo(target, db_name=target.maintenance_db)
    target_conn = _conninfo(target, db_name=target.db_name)
    print(f"[{target.label}] target={target.display}")
    print(f"[{target.label}] maintenance-conn={_redacted_conninfo(target, db_name=target.maintenance_db)}")

    try:
        if not verify_only:
            if skip_db_create:
                print(f"[{target.label}] skip-db-create enabled; assuming database exists.")
            else:
                exists = _database_exists(maintenance_conn, target.db_name, retries=retries, retry_delay=retry_delay)
                if exists:
                    print(f"[{target.label}] database already exists: {target.db_name}")
                else:
                    _create_database(maintenance_conn, target.db_name, retries=retries, retry_delay=retry_delay)
                    print(f"[{target.label}] created database: {target.db_name}")

            _ensure_vector_extension(target_conn, retries=retries, retry_delay=retry_delay)
            print(f"[{target.label}] ensured extension: vector")

        if sql_check_path is not None:
            _run_sql_check(target_conn, sql_check_path, retries=retries, retry_delay=retry_delay)
            print(f"[{target.label}] executed SQL check: {sql_check_path}")

        _verify_vector_support(target_conn, retries=retries, retry_delay=retry_delay)
        print(f"[{target.label}] vector extension/type verification passed")
        return True
    except Exception as error:  # noqa: BLE001
        print(f"[{target.label}] ERROR: {error}")
        return False


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Bootstrap PostgreSQL and pgvector for RYO Chat.")
    parser.add_argument("--config", default=None, help="Path to config.json for primary/fallback target loading.")
    parser.add_argument(
        "--target",
        choices=("primary", "fallback", "both"),
        default="primary",
        help="Config target to bootstrap when --config is provided (default: primary).",
    )

    parser.add_argument("--host", default="127.0.0.1", help="Manual target host.")
    parser.add_argument("--port", default="5432", help="Manual target port.")
    parser.add_argument("--user", default=None, help="Manual target user.")
    parser.add_argument("--password", default=None, help="Manual target password.")
    parser.add_argument("--db-name", default=None, help="Manual target database name.")
    parser.add_argument("--maintenance-db", default="postgres", help="Maintenance DB for CREATE DATABASE flow.")

    parser.add_argument("--skip-db-create", action="store_true", help="Skip CREATE DATABASE checks/creation.")
    parser.add_argument("--verify-only", action="store_true", help="Only verify extension/type; do not create DB or extension.")
    parser.add_argument("--connect-retries", type=int, default=15, help="Number of connection retries before failure.")
    parser.add_argument("--retry-delay", type=float, default=1.0, help="Delay in seconds between retries.")

    parser.add_argument("--sql-check", default=str(DEFAULT_VERIFY_SQL), help="Optional SQL file for additional checks.")
    parser.add_argument("--no-sql-check", action="store_true", help="Disable execution of SQL check file.")

    parser.add_argument("--docker", action="store_true", help="Provision/ensure local dockerized PostgreSQL target.")
    parser.add_argument("--docker-image", default="pgvector/pgvector:pg16", help="Docker image for PostgreSQL + pgvector.")
    parser.add_argument("--docker-container", default="ryo-pg", help="Docker container name.")
    parser.add_argument("--docker-recreate", action="store_true", help="Recreate docker container if it already exists.")
    parser.add_argument("--docker-volume", default=None, help="Optional docker volume mount (e.g. ryo_pg_data:/var/lib/postgresql/data).")

    return parser


def _manual_target_from_args(args: argparse.Namespace) -> BootstrapTarget:
    if not args.user or not args.password or not args.db_name:
        raise ValueError("Manual mode requires --user, --password, and --db-name.")
    return BootstrapTarget(
        label="manual",
        db_name=str(args.db_name),
        user=str(args.user),
        password=str(args.password),
        host=str(args.host),
        port=str(args.port),
        maintenance_db=str(args.maintenance_db),
    )


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.config:
        config_path = Path(args.config)
        if not config_path.exists():
            print(f"ERROR: config file not found: {config_path}")
            return 2
        try:
            targets = load_targets_from_config(config_path, target_mode=args.target, maintenance_db=args.maintenance_db)
        except Exception as error:  # noqa: BLE001
            print(f"ERROR: failed to load targets from config: {error}")
            return 2
    else:
        try:
            targets = [_manual_target_from_args(args)]
        except ValueError as error:
            print(f"ERROR: {error}")
            return 2

    sql_check_path: Path | None = None
    if not args.no_sql_check:
        sql_check_path = Path(args.sql_check)
        if not sql_check_path.exists():
            print(f"ERROR: sql-check file not found: {sql_check_path}")
            return 2

    exit_code = 0
    for target in targets:
        if args.docker:
            try:
                _ensure_docker_container(
                    target=target,
                    image=args.docker_image,
                    container_name=args.docker_container if len(targets) == 1 else f"{args.docker_container}-{target.label}",
                    recreate=args.docker_recreate,
                    volume=args.docker_volume,
                )
            except Exception as error:  # noqa: BLE001
                print(f"[{target.label}] ERROR: docker provisioning failed: {error}")
                exit_code = 1
                continue

        ok = bootstrap_target(
            target=target,
            skip_db_create=args.skip_db_create,
            verify_only=args.verify_only,
            sql_check_path=sql_check_path,
            retries=args.connect_retries,
            retry_delay=args.retry_delay,
        )
        if not ok:
            exit_code = 1

    if exit_code == 0:
        labels = ", ".join(target.label for target in targets)
        print(f"Bootstrap completed successfully for target(s): {labels}")
    else:
        print("Bootstrap completed with failures. See errors above.")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
