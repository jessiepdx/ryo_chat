##########################################################################
#                                                                        #
#  Personality persistence manager                                       #
#                                                                        #
##########################################################################

from __future__ import annotations

import json
import logging
from typing import Any

import psycopg
from psycopg.rows import dict_row

from hypermindlabs.utils import ConfigManager


logger = logging.getLogger(__name__)


def _as_int(value: Any, fallback: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(fallback)


def _as_text(value: Any, fallback: str = "") -> str:
    text = str(value if value is not None else "").strip()
    return text if text else fallback


def _coerce_dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _coerce_list(value: Any) -> list[Any]:
    return list(value) if isinstance(value, list) else []


class PersonalityStoreManager:
    def _connect(self):
        return psycopg.connect(conninfo=ConfigManager()._instance.db_conninfo, row_factory=dict_row)

    def get_profile(self, member_id: int) -> dict[str, Any] | None:
        member_id_int = _as_int(member_id, 0)
        if member_id_int <= 0:
            return None
        connection = None
        try:
            connection = self._connect()
            cursor = connection.cursor()
            cursor.execute(
                """SELECT profile_id, member_id, explicit_directive_json, adaptive_state_json,
                          effective_profile_json, profile_version, locked_fields_json, created_at, updated_at
                   FROM member_personality_profile
                   WHERE member_id = %s
                   LIMIT 1""",
                (member_id_int,),
            )
            row = cursor.fetchone()
            cursor.close()
            return dict(row) if isinstance(row, dict) else None
        except (Exception, psycopg.DatabaseError) as error:
            logger.warning(f"Unable to load personality profile for member {member_id_int}: {error}")
            return None
        finally:
            if connection is not None:
                connection.close()

    def upsert_profile_payload(self, member_id: int, profile: dict[str, Any]) -> dict[str, Any] | None:
        member_id_int = _as_int(member_id, 0)
        if member_id_int <= 0:
            return None
        payload = _coerce_dict(profile)
        explicit_json = _coerce_dict(payload.get("explicit"))
        adaptive_json = _coerce_dict(payload.get("adaptive"))
        narrative_json = _coerce_dict(payload.get("narrative"))
        if narrative_json:
            adaptive_json["narrative"] = narrative_json
        effective_json = _coerce_dict(payload.get("effective"))
        locked_fields = _coerce_list(payload.get("locked_fields"))
        connection = None
        try:
            connection = self._connect()
            cursor = connection.cursor()
            cursor.execute(
                """INSERT INTO member_personality_profile (
                       member_id,
                       explicit_directive_json,
                       adaptive_state_json,
                       effective_profile_json,
                       profile_version,
                       locked_fields_json,
                       created_at,
                       updated_at
                   )
                   VALUES (%s, %s::jsonb, %s::jsonb, %s::jsonb, 1, %s::jsonb, NOW(), NOW())
                   ON CONFLICT(member_id)
                   DO UPDATE SET
                       explicit_directive_json = EXCLUDED.explicit_directive_json,
                       adaptive_state_json = EXCLUDED.adaptive_state_json,
                       effective_profile_json = EXCLUDED.effective_profile_json,
                       locked_fields_json = EXCLUDED.locked_fields_json,
                       profile_version = member_personality_profile.profile_version + 1,
                       updated_at = NOW()
                   RETURNING profile_id, member_id, explicit_directive_json, adaptive_state_json,
                             effective_profile_json, profile_version, locked_fields_json, created_at, updated_at""",
                (
                    member_id_int,
                    json.dumps(explicit_json, ensure_ascii=False, default=str),
                    json.dumps(adaptive_json, ensure_ascii=False, default=str),
                    json.dumps(effective_json, ensure_ascii=False, default=str),
                    json.dumps(locked_fields, ensure_ascii=False, default=str),
                ),
            )
            row = cursor.fetchone()
            connection.commit()
            cursor.close()
            return dict(row) if isinstance(row, dict) else None
        except (Exception, psycopg.DatabaseError) as error:
            if connection is not None:
                connection.rollback()
            logger.warning(f"Unable to upsert personality profile for member {member_id_int}: {error}")
            return None
        finally:
            if connection is not None:
                connection.close()

    def append_event(
        self,
        *,
        member_id: int,
        event_type: str,
        before_json: dict[str, Any] | None,
        after_json: dict[str, Any] | None,
        reason_code: str = "",
        reason_detail: str = "",
    ) -> int | None:
        member_id_int = _as_int(member_id, 0)
        if member_id_int <= 0:
            return None
        connection = None
        try:
            connection = self._connect()
            cursor = connection.cursor()
            cursor.execute(
                """INSERT INTO member_personality_events (
                       member_id,
                       event_type,
                       before_json,
                       after_json,
                       reason_code,
                       reason_detail,
                       created_at
                   )
                   VALUES (%s, %s, %s::jsonb, %s::jsonb, %s, %s, NOW())
                   RETURNING event_id""",
                (
                    member_id_int,
                    _as_text(event_type, "adaptive_update")[:32],
                    json.dumps(_coerce_dict(before_json), ensure_ascii=False, default=str),
                    json.dumps(_coerce_dict(after_json), ensure_ascii=False, default=str),
                    _as_text(reason_code)[:96],
                    _as_text(reason_detail)[:2000],
                ),
            )
            row = cursor.fetchone()
            connection.commit()
            cursor.close()
            return _as_int(row.get("event_id"), 0) if isinstance(row, dict) else None
        except (Exception, psycopg.DatabaseError) as error:
            if connection is not None:
                connection.rollback()
            logger.warning(f"Unable to append personality event for member {member_id_int}: {error}")
            return None
        finally:
            if connection is not None:
                connection.close()

    def get_latest_narrative_chunk(self, member_id: int) -> dict[str, Any] | None:
        member_id_int = _as_int(member_id, 0)
        if member_id_int <= 0:
            return None
        connection = None
        try:
            connection = self._connect()
            cursor = connection.cursor()
            cursor.execute(
                """SELECT chunk_id, member_id, chunk_index, source_turn_start_id, source_turn_end_id,
                          summary_text, summary_json, compression_ratio, created_at, updated_at
                   FROM member_narrative_chunks
                   WHERE member_id = %s
                   ORDER BY chunk_index DESC
                   LIMIT 1""",
                (member_id_int,),
            )
            row = cursor.fetchone()
            cursor.close()
            return dict(row) if isinstance(row, dict) else None
        except (Exception, psycopg.DatabaseError) as error:
            logger.warning(f"Unable to get latest narrative chunk for member {member_id_int}: {error}")
            return None
        finally:
            if connection is not None:
                connection.close()

    def list_narrative_chunks(self, member_id: int, count: int = 6) -> list[dict[str, Any]]:
        member_id_int = _as_int(member_id, 0)
        if member_id_int <= 0:
            return []
        limit = max(1, int(count))
        connection = None
        try:
            connection = self._connect()
            cursor = connection.cursor()
            cursor.execute(
                """SELECT chunk_id, member_id, chunk_index, source_turn_start_id, source_turn_end_id,
                          summary_text, summary_json, compression_ratio, created_at, updated_at
                   FROM member_narrative_chunks
                   WHERE member_id = %s
                   ORDER BY chunk_index DESC
                   LIMIT %s""",
                (member_id_int, limit),
            )
            rows = cursor.fetchall()
            cursor.close()
            return [dict(row) for row in rows] if isinstance(rows, list) else []
        except (Exception, psycopg.DatabaseError) as error:
            logger.warning(f"Unable to list narrative chunks for member {member_id_int}: {error}")
            return []
        finally:
            if connection is not None:
                connection.close()

    def insert_narrative_chunk(
        self,
        *,
        member_id: int,
        chunk_index: int,
        summary_text: str,
        summary_json: dict[str, Any] | None,
        compression_ratio: float = 1.0,
        source_turn_start_id: int | None = None,
        source_turn_end_id: int | None = None,
    ) -> dict[str, Any] | None:
        member_id_int = _as_int(member_id, 0)
        chunk_index_int = _as_int(chunk_index, 0)
        if member_id_int <= 0 or chunk_index_int <= 0:
            return None
        connection = None
        try:
            connection = self._connect()
            cursor = connection.cursor()
            cursor.execute(
                """INSERT INTO member_narrative_chunks (
                       member_id,
                       chunk_index,
                       source_turn_start_id,
                       source_turn_end_id,
                       summary_text,
                       summary_json,
                       compression_ratio,
                       created_at,
                       updated_at
                   )
                   VALUES (%s, %s, %s, %s, %s, %s::jsonb, %s, NOW(), NOW())
                   ON CONFLICT(member_id, chunk_index)
                   DO UPDATE SET
                       source_turn_start_id = EXCLUDED.source_turn_start_id,
                       source_turn_end_id = EXCLUDED.source_turn_end_id,
                       summary_text = EXCLUDED.summary_text,
                       summary_json = EXCLUDED.summary_json,
                       compression_ratio = EXCLUDED.compression_ratio,
                       updated_at = NOW()
                   RETURNING chunk_id, member_id, chunk_index, summary_text, summary_json,
                             compression_ratio, created_at, updated_at""",
                (
                    member_id_int,
                    chunk_index_int,
                    source_turn_start_id,
                    source_turn_end_id,
                    _as_text(summary_text)[:6000],
                    json.dumps(_coerce_dict(summary_json), ensure_ascii=False, default=str),
                    float(compression_ratio),
                ),
            )
            row = cursor.fetchone()
            connection.commit()
            cursor.close()
            return dict(row) if isinstance(row, dict) else None
        except (Exception, psycopg.DatabaseError) as error:
            if connection is not None:
                connection.rollback()
            logger.warning(f"Unable to insert narrative chunk for member {member_id_int}: {error}")
            return None
        finally:
            if connection is not None:
                connection.close()


__all__ = ["PersonalityStoreManager"]

