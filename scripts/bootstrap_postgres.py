#!/usr/bin/env python3
"""
Bootstrap helper for PostgreSQL database creation and pgvector enablement.

This script assumes a reachable PostgreSQL server is already running.
"""

from __future__ import annotations

import argparse
import psycopg


def _conninfo(db_name: str, user: str, password: str, host: str, port: str | int) -> str:
    return f"dbname={db_name} user={user} password={password} host={host} port={port}"


def _database_exists(conninfo: str, target_db: str) -> bool:
    with psycopg.connect(conninfo=conninfo, autocommit=True) as connection:
        with connection.cursor() as cursor:
            cursor.execute("SELECT 1 FROM pg_database WHERE datname = %s LIMIT 1;", (target_db,))
            return cursor.fetchone() is not None


def _create_database(conninfo: str, target_db: str) -> None:
    with psycopg.connect(conninfo=conninfo, autocommit=True) as connection:
        with connection.cursor() as cursor:
            cursor.execute(f'CREATE DATABASE "{target_db}";')


def _enable_vector_extension(conninfo: str) -> None:
    with psycopg.connect(conninfo=conninfo, autocommit=True) as connection:
        with connection.cursor() as cursor:
            cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")


def main() -> int:
    parser = argparse.ArgumentParser(description="Bootstrap PostgreSQL DB and pgvector extension.")
    parser.add_argument("--host", default="127.0.0.1", help="PostgreSQL host.")
    parser.add_argument("--port", default="5432", help="PostgreSQL port.")
    parser.add_argument("--user", required=True, help="PostgreSQL user.")
    parser.add_argument("--password", required=True, help="PostgreSQL password.")
    parser.add_argument("--db-name", required=True, help="Target database name.")
    parser.add_argument(
        "--maintenance-db",
        default="postgres",
        help="Maintenance database used to create target DB if needed.",
    )
    args = parser.parse_args()

    maintenance_conn = _conninfo(
        db_name=args.maintenance_db,
        user=args.user,
        password=args.password,
        host=args.host,
        port=args.port,
    )
    target_conn = _conninfo(
        db_name=args.db_name,
        user=args.user,
        password=args.password,
        host=args.host,
        port=args.port,
    )

    exists = _database_exists(maintenance_conn, args.db_name)
    if not exists:
        _create_database(maintenance_conn, args.db_name)
        print(f"Created database: {args.db_name}")
    else:
        print(f"Database already exists: {args.db_name}")

    _enable_vector_extension(target_conn)
    print(f"Ensured pgvector extension exists in: {args.db_name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
