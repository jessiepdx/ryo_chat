##########################################################################
#                                                                        #
#  This file (database_router.py) handles PostgreSQL primary/fallback    #
#  connection routing for runtime startup.                               #
#                                                                        #
##########################################################################

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import psycopg

from hypermindlabs.runtime_settings import DEFAULT_RUNTIME_SETTINGS


DEFAULT_DB_CONNECT_TIMEOUT = int(
    DEFAULT_RUNTIME_SETTINGS.get("database", {}).get("connect_timeout_seconds", 2)
)


@dataclass
class DatabaseRoute:
    status: str
    active_target: str
    active_conninfo: str | None
    primary_conninfo: str | None
    fallback_conninfo: str | None
    primary_available: bool = False
    fallback_available: bool = False
    fallback_enabled: bool = False
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class DatabaseRouter:
    """Resolve and validate primary/fallback PostgreSQL connection targets."""

    def __init__(
        self,
        primary_database: dict | None,
        fallback_database: dict | None = None,
        fallback_enabled: bool | None = None,
        connect_timeout: int = DEFAULT_DB_CONNECT_TIMEOUT,
    ):
        self._primary_database = primary_database or {}
        self._fallback_database = fallback_database or {}
        self._connect_timeout = connect_timeout

        if fallback_enabled is None:
            fallback_enabled = bool(self._fallback_database.get("enabled", False))
        self._fallback_enabled = bool(fallback_enabled)

    @staticmethod
    def build_conninfo(database: dict | None) -> str | None:
        if not isinstance(database, dict):
            return None

        db_name = database.get("db_name")
        user = database.get("user")
        password = database.get("password")
        host = database.get("host")
        port = database.get("port")

        required = (db_name, user, password, host)
        if any(not item for item in required):
            return None

        conninfo = (
            f"dbname={db_name} user={user} password={password} host={host}"
        )
        if port:
            conninfo = f"{conninfo} port={port}"
        return conninfo

    def _can_connect(self, conninfo: str | None) -> tuple[bool, str | None]:
        if not conninfo:
            return False, "missing conninfo"
        try:
            connection = psycopg.connect(conninfo=conninfo, connect_timeout=self._connect_timeout)
            connection.close()
            return True, None
        except Exception as error:
            return False, str(error)

    def resolve(self) -> DatabaseRoute:
        primary_conninfo = self.build_conninfo(self._primary_database)
        fallback_conninfo = self.build_conninfo(self._fallback_database)

        route = DatabaseRoute(
            status="unknown",
            active_target="none",
            active_conninfo=None,
            primary_conninfo=primary_conninfo,
            fallback_conninfo=fallback_conninfo,
            fallback_enabled=self._fallback_enabled,
        )

        primary_ok, primary_error = self._can_connect(primary_conninfo)
        route.primary_available = primary_ok
        if primary_error:
            route.errors.append(f"primary: {primary_error}")

        if primary_ok:
            route.status = "primary"
            route.active_target = "primary"
            route.active_conninfo = primary_conninfo
            return route

        if self._fallback_enabled and fallback_conninfo:
            fallback_ok, fallback_error = self._can_connect(fallback_conninfo)
            route.fallback_available = fallback_ok
            if fallback_error:
                route.errors.append(f"fallback: {fallback_error}")
            if fallback_ok:
                route.status = "fallback"
                route.active_target = "fallback"
                route.active_conninfo = fallback_conninfo
                return route
        elif self._fallback_enabled and not fallback_conninfo:
            route.errors.append("fallback: enabled but fallback conninfo is missing")

        # No healthy connection found; keep primary conninfo as default active target.
        route.status = "failed_all"
        route.active_target = "primary"
        route.active_conninfo = primary_conninfo
        return route
