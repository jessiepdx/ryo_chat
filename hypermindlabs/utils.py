##########################################################################
#                                                                        #
#  This file (utils.py) contains the utility modules for Hypermind Labs  #
#                                                                        #
#  Created by:  Jessie W                                                 #
#  Github: jessiepdx                                                     #
#  Contributors:                                                         #
#      Robit                                                             #
#  Created: February 1st, 2025                                           #
#  Modified: April 3rd, 2025                                             #
#                                                                        #
##########################################################################


###########
# IMPORTS #
###########

import base64
import hashlib
import hmac
import json
import logging
import os
import psycopg
import queue
import re
import requests
import secrets
import sys
import string
import threading
import textstat
import time
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from hypermindlabs.database_router import DatabaseRouter
from hypermindlabs.document_access_policy import (
    DocumentAccessPolicy,
    DocumentScopeAccessError,
)
from hypermindlabs.document_contracts import (
    validate_document_node_contract,
    validate_document_node_edge_contract,
    validate_document_retrieval_event_contract,
    validate_document_source_contract,
    validate_document_version_contract,
)
from hypermindlabs.document_models import DOCUMENT_SOURCE_STATES
from hypermindlabs.document_scope import (
    apply_pg_scope_settings,
    build_scope_where_clause,
    resolve_document_scope,
)
from hypermindlabs.runtime_settings import (
    build_runtime_settings,
    get_runtime_setting,
    load_dotenv_file,
)
from math import ceil
from ollama import Client
from psycopg.rows import dict_row
from urllib.parse import parse_qs, unquote, urlparse



###########
# GLOBALS #
###########

# Create a custom formatter sub class for adding colored outputs
class CustomFormatter(logging.Formatter):
    """Creates a custom formatter for the logging library."""
    grey = "\x1b[38;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    base_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s (%(filename)s:%(lineno)d)"

    COLOR_FORMATS = {
        logging.DEBUG: grey + base_format + reset,
        logging.INFO: grey + base_format + reset,
        logging.WARNING: yellow + base_format + reset,
        logging.ERROR: red + base_format + reset,
        logging.CRITICAL: bold_red + base_format + reset
    }

    def __init__(self, *, use_color: bool | None = None, stream: Any | None = None):
        super().__init__(self.base_format)
        if use_color is None:
            no_color = str(os.getenv("NO_COLOR", "")).strip()
            ryo_color = str(os.getenv("RYO_LOG_COLOR", "")).strip().lower()
            if no_color:
                use_color = False
            elif ryo_color in {"0", "false", "no", "off"}:
                use_color = False
            elif ryo_color in {"1", "true", "yes", "on"}:
                use_color = True
            else:
                target_stream = stream if stream is not None else sys.stderr
                use_color = bool(getattr(target_stream, "isatty", lambda: False)())
        self._use_color = bool(use_color)

    def format(self, record):
        if not self._use_color:
            formatter = logging.Formatter(self.base_format)
            return formatter.format(record)
        log_fmt = self.COLOR_FORMATS.get(record.levelno, self.base_format)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)

# Enable logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s (%(filename)s:%(lineno)d)", 
    level=logging.WARNING
)

logger = logging.getLogger(__name__)

MIGRATIONS_DIR = Path(__file__).resolve().parent.parent / "db" / "migrations"
MIGRATION_TOKEN_PATTERN = re.compile(r"\{\{\s*([A-Za-z_][A-Za-z0-9_]*)\s*\}\}")
STARTUP_CORE_MIGRATIONS: tuple[str, ...] = (
    "001_member_data.sql",
    "002_member_telegram.sql",
    "003_member_telegram_user_id_nullable.sql",
    "004_member_secure.sql",
    "010_chat_history.sql",
    "020_community_data.sql",
    "021_community_telegram.sql",
    "030_community_score.sql",
    "050_proposals.sql",
    "051_proposal_disclosure.sql",
    "070_inference_usage.sql",
    "080_runs.sql",
    "081_run_events.sql",
    "082_run_state_snapshots.sql",
    "083_run_artifacts.sql",
    "084_agent_process_workspace.sql",
    "085_member_outbox.sql",
    "086_member_personality_profile.sql",
    "087_member_narrative_chunks.sql",
    "088_member_personality_events.sql",
    "089_community_isolation.sql",
    "090_document_sources.sql",
    "091_document_versions.sql",
    "092_document_nodes.sql",
    "093_document_chunks.sql",
    "094_document_retrieval_events.sql",
    "095_document_rls_policies.sql",
    "096_document_storage_objects.sql",
    "097_document_ingestion_jobs.sql",
    "098_document_ingestion_attempts.sql",
    "099_document_node_edges.sql",
)
STARTUP_VECTOR_MIGRATIONS: tuple[str, ...] = (
    "011_create_vector_extension.sql",
    "012_chat_history_embeddings.sql",
    "040_knowledge.sql",
    "041_knowledge_retrievals.sql",
    "060_spam.sql",
)
VECTOR_DIMENSION_MIGRATIONS: set[str] = {
    "012_chat_history_embeddings.sql",
    "040_knowledge.sql",
    "060_spam.sql",
}
_EMBEDDING_WRITE_QUEUE: queue.Queue[dict[str, Any]] | None = None
_EMBEDDING_WORKER_THREAD: threading.Thread | None = None
_EMBEDDING_WORKER_LOCK = threading.Lock()
DOCUMENT_INGESTION_JOB_STATES: tuple[str, ...] = (
    "queued",
    "leased",
    "running",
    "retry_wait",
    "completed",
    "cancelled",
    "failed",
    "dead_letter",
)
DOCUMENT_INGESTION_ATTEMPT_STATES: tuple[str, ...] = (
    "leased",
    "running",
    "succeeded",
    "failed",
    "cancelled",
)


def _render_migration_sql(sql_text: str, context: dict[str, Any] | None = None) -> str:
    migration_context = context or {}

    def replace_token(match: re.Match[str]) -> str:
        key = match.group(1)
        if key not in migration_context:
            raise KeyError(f"Missing migration template value: {key}")
        value = migration_context[key]
        if isinstance(value, bool):
            return "true" if value else "false"
        if isinstance(value, (int, float)):
            return str(value)
        raise ValueError(f"Unsupported migration template value for '{key}': {type(value).__name__}")

    return MIGRATION_TOKEN_PATTERN.sub(replace_token, sql_text)


def execute_migration(cursor: Any, migration_filename: str, context: dict[str, Any] | None = None) -> None:
    migration_path = MIGRATIONS_DIR / migration_filename
    migration_sql = migration_path.read_text(encoding="utf-8")
    rendered_sql = _render_migration_sql(migration_sql, context=context)
    cursor.execute(rendered_sql)


def ensure_startup_database_migrations() -> dict[str, Any]:
    report: dict[str, Any] = {
        "route_status": "unknown",
        "active_target": "unknown",
        "core_applied": list(),
        "vector_applied": list(),
        "core_failed": list(),
        "vector_failed": list(),
        "connection_error": None,
    }

    try:
        config_manager = ConfigManager()
        route = config_manager.databaseRoute if isinstance(config_manager.databaseRoute, dict) else {}
        report["route_status"] = str(route.get("status", "unknown"))
        report["active_target"] = str(route.get("active_target", "unknown"))
        conninfo = str(config_manager._instance.db_conninfo or "").strip()
        vector_dimensions = max(1, config_manager.runtimeInt("vectors.embedding_dimensions", 768))
    except Exception as error:  # noqa: BLE001
        report["connection_error"] = str(error)
        return report

    if not conninfo:
        report["connection_error"] = "Empty database connection info."
        return report

    connection = None
    try:
        connection = psycopg.connect(conninfo=conninfo)
        cursor = connection.cursor()

        for migration_filename in STARTUP_CORE_MIGRATIONS:
            try:
                execute_migration(cursor, migration_filename)
                connection.commit()
                report["core_applied"].append(migration_filename)
            except (Exception, psycopg.DatabaseError) as error:  # noqa: PERF203
                connection.rollback()
                report["core_failed"].append(
                    {"migration": migration_filename, "error": str(error)}
                )

        for migration_filename in STARTUP_VECTOR_MIGRATIONS:
            try:
                context = (
                    {"vector_dimensions": vector_dimensions}
                    if migration_filename in VECTOR_DIMENSION_MIGRATIONS
                    else None
                )
                execute_migration(cursor, migration_filename, context=context)
                connection.commit()
                report["vector_applied"].append(migration_filename)
            except (Exception, psycopg.DatabaseError) as error:  # noqa: PERF203
                connection.rollback()
                report["vector_failed"].append(
                    {"migration": migration_filename, "error": str(error)}
                )

        cursor.close()
    except (Exception, psycopg.DatabaseError) as error:
        report["connection_error"] = str(error)
    finally:
        if connection is not None:
            connection.close()

    return report


def _defer_embeddings_on_write_enabled() -> bool:
    try:
        return ConfigManager().runtimeBool("inference.defer_embeddings_on_write", True)
    except Exception:  # noqa: BLE001
        return True


def _embedding_write_queue_max_size() -> int:
    try:
        size = int(ConfigManager().runtimeInt("inference.embedding_write_queue_size", 512))
    except Exception:  # noqa: BLE001
        size = 512
    return max(16, size)


def _persist_chat_embedding(history_id: int, message_text: str, *, source: str = "sync") -> bool:
    if history_id is None:
        return False

    embedding = getEmbeddings(message_text)
    if embedding is None:
        logger.warning("Skipping chat embeddings persistence because embedding model returned no vector.")
        return False

    connection = None
    try:
        connection = psycopg.connect(conninfo=ConfigManager()._instance.db_conninfo, row_factory=dict_row)
        cursor = connection.cursor()
        insertEmbeddings_sql = """INSERT INTO chat_history_embeddings (history_id, embeddings)
        VALUES (%s, %s);"""
        cursor.execute(insertEmbeddings_sql, (history_id, embedding))
        connection.commit()
        cursor.close()
        logger.debug(f"Chat embeddings persisted ({source}) for history id {history_id}.")
        return True
    except (Exception, psycopg.DatabaseError) as error:
        if connection is not None:
            connection.rollback()
        logger.warning(f"Unable to persist chat embeddings ({source}) for history id {history_id}:\n{error}")
        return False
    finally:
        if connection is not None:
            connection.close()


def _embedding_worker_loop() -> None:
    global _EMBEDDING_WRITE_QUEUE
    logger.info("Embedding worker started for deferred chat history persistence.")
    while True:
        workQueue = _EMBEDDING_WRITE_QUEUE
        if workQueue is None:
            break
        try:
            job = workQueue.get(timeout=0.5)
        except queue.Empty:
            continue
        except Exception as error:  # noqa: BLE001
            logger.warning(f"Embedding worker queue error: {error}")
            continue

        if not isinstance(job, dict):
            continue

        history_id = job.get("history_id")
        message_text = str(job.get("message_text") or "")
        if not isinstance(history_id, int) or history_id <= 0:
            continue
        if message_text.strip() == "":
            continue

        try:
            _persist_chat_embedding(history_id, message_text, source="deferred")
        except Exception as error:  # noqa: BLE001
            logger.warning(f"Embedding worker failed for history id {history_id}: {error}")


def _ensure_embedding_worker_started() -> queue.Queue[dict[str, Any]]:
    global _EMBEDDING_WRITE_QUEUE
    global _EMBEDDING_WORKER_THREAD

    with _EMBEDDING_WORKER_LOCK:
        expectedSize = _embedding_write_queue_max_size()
        if _EMBEDDING_WRITE_QUEUE is None:
            _EMBEDDING_WRITE_QUEUE = queue.Queue(maxsize=expectedSize)
        if _EMBEDDING_WORKER_THREAD is None or not _EMBEDDING_WORKER_THREAD.is_alive():
            _EMBEDDING_WORKER_THREAD = threading.Thread(
                target=_embedding_worker_loop,
                name="ryo-embedding-writer",
                daemon=True,
            )
            _EMBEDDING_WORKER_THREAD.start()
    return _EMBEDDING_WRITE_QUEUE


def _enqueue_chat_embedding(history_id: int, message_text: str) -> bool:
    if not _defer_embeddings_on_write_enabled():
        return False
    if not isinstance(history_id, int) or history_id <= 0:
        return False
    if str(message_text or "").strip() == "":
        return False

    try:
        workQueue = _ensure_embedding_worker_started()
        workQueue.put_nowait({"history_id": history_id, "message_text": str(message_text)})
        return True
    except queue.Full:
        logger.warning("Embedding queue is full; skipping deferred embedding enqueue for history id %s.", history_id)
        return False
    except Exception as error:  # noqa: BLE001
        logger.warning(f"Unable to enqueue deferred embedding for history id {history_id}: {error}")
        return False


# NOTE OLD - but other aspects of code still use... Maybe make into a class for dot syntax use
# Create a dictionary containing color codes for console output
ConsoleColors = {
    "default": "\x1b[0m",
    "green": "\x1b[38;5;46m",
    "dark_green": "\x1b[38;5;34m",
    "red": "\x1b[38;5;196m",
    "dark_red": "\x1b[38;5;124m",
    "yellow": "\x1b[38;5;226m",
    "blue": "\x1b[38;5;21m",
    "purple": "\x1b[38;5;201m",
    "pink": "\x1b[38;5;207m"

}

# Set timezone for time
timezone(-timedelta(hours=7), "Pacific")



###########################
# SINGLETON DATA MANAGERS #
###########################

class MemberManager:
    _instance = None
    
    # Private attributes
    __rolesList = ["user","tester","marketing","admin","owner"]

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(MemberManager, cls).__new__(cls)
            # Initialize the singleton instance.

            connection = None
            # Create the members table if it doesn't exist yet
            try:
                connection = psycopg.connect(conninfo=ConfigManager()._instance.db_conninfo, row_factory=dict_row)
                cursor = connection.cursor()
                logger.debug(f"PostgreSQL connection established.")

                # Create the member data table
                execute_migration(cursor, "001_member_data.sql")
                connection.commit()
                
                # Create the member telegram table if it doesn't exist
                execute_migration(cursor, "002_member_telegram.sql")
                connection.commit()

                # Allow web-only accounts that are not yet linked to a Telegram user id.
                execute_migration(cursor, "003_member_telegram_user_id_nullable.sql")
                connection.commit()

                # Create the member password hash table
                execute_migration(cursor, "004_member_secure.sql")
                connection.commit()

                # Check for an empty member data table
                cursor.execute("SELECT COUNT(*) AS total_members FROM member_data;")
                rowCount = cursor.fetchone()
                if rowCount.get("total_members") == 0:
                    logger.info("No registered users. Creating default owner account from config data")

                    # Add the data from record to our new member_data table
                    ownerData = {
                        "first_name": ConfigManager().owner_info["first_name"],
                        "last_name": ConfigManager().owner_info["last_name"],
                        "email": None,
                        "roles": ["owner"],
                        "register_date": datetime.now(),
                        "community_score": 100
                    }
                    insertOwner_sql = """INSERT INTO member_data (first_name, last_name, email, roles, register_date, community_score) 
                    VALUES (%(first_name)s, %(last_name)s, %(email)s, %(roles)s, %(register_date)s, %(community_score)s)
                    RETURNING member_id;"""
                    
                    cursor.execute(insertOwner_sql, ownerData)
                    result = cursor.fetchone()
                    memberID = result.get("member_id")

                    ownerTelegramData = {
                        "member_id": memberID,
                        "first_name": ConfigManager().owner_info["first_name"],
                        "last_name": ConfigManager().owner_info["last_name"],
                        "username": ConfigManager().owner_info["username"],
                        "user_id": ConfigManager().owner_info["user_id"]
                    }
                    insertOwnerTelegram_sql = """INSERT INTO member_telegram (member_id, first_name, last_name, username, user_id) 
                    VALUES (%(member_id)s, %(first_name)s, %(last_name)s, %(username)s, %(user_id)s);"""

                    cursor.execute(insertOwnerTelegram_sql, ownerTelegramData)

                    connection.commit()
                    logger.info("Owner account added.")
                else:
                    logger.info(f"There are {rowCount.get('total_members')} registered users.")
                
                connection.commit()
                # Close the cursor
                cursor.close()
            except psycopg.Error as error:
                logger.error(f"Exception while working with psycopg and PostgreSQL:\n{error}")
            finally:
                if (connection):
                    connection.close()
                    logger.debug(f"PostgreSQL connection is closed.")
        
        return cls._instance

    def getMemberByID(self, memberID: int) -> dict:
        logger.info(f"Getting member data for member ID:  {memberID}.")

        connection = None
        response = None
        try:
            connection = connection = psycopg.connect(conninfo=ConfigManager()._instance.db_conninfo, row_factory=dict_row)
            cursor = connection.cursor()
            logger.debug(f"PostgreSQL connection established.")
            
            getMemberQuery_sql = """SELECT mem.member_id, mem.first_name, mem.last_name, mem.email, mem.roles, mem.register_date, mem.community_score, tg.username, tg.user_id
            FROM member_data AS mem
            LEFT JOIN member_telegram AS tg
            ON mem.member_id = tg.member_id
            WHERE mem.member_id = %s
            LIMIT 1;"""

            
            cursor.execute(getMemberQuery_sql, (memberID, ))
            result = cursor.fetchone()
            # Close the cursor
            cursor.close()
            
            response = result
            
        except psycopg.Error as error:
                logger.error(f"Exception while working with psycopg and PostgreSQL:\n{error}")
        finally:
            if (connection):
                connection.close()
                logger.debug(f"PostgreSQL connection is closed.")
        
        return response

    def getMemberByTelegramID(self, telegramUserID: int) -> dict:
        logger.info(f"Getting member data for telegram ID:  {telegramUserID}.")

        connection = None
        response = None
        try:
            connection = connection = psycopg.connect(conninfo=ConfigManager()._instance.db_conninfo, row_factory=dict_row)
            cursor = connection.cursor()
            logger.debug(f"PostgreSQL connection established.")
            
            getMemberQuery_sql = """SELECT mem.member_id, mem.first_name, mem.last_name, mem.email, mem.roles, mem.register_date, mem.community_score, tg.username, tg.user_id
            FROM member_data AS mem
            JOIN member_telegram AS tg
            ON mem.member_id = tg.member_id
            WHERE tg.user_id = %s
            LIMIT 1;"""

            
            cursor.execute(getMemberQuery_sql, (telegramUserID, ))
            result = cursor.fetchone()
            # Close the cursor
            cursor.close()
            
            response = result
            
        except psycopg.Error as error:
                logger.error(f"Exception while working with pyscopg and PostgreSQL:\n{error}")
        finally:
            if (connection):
                connection.close()
                logger.debug(f"PostgreSQL connection is closed.")
        
        return response

    def getMembersByRoles(self, roles: list) -> list:
        logger.info("Get members by roles.")
        if len(roles) == 0:
            logger.error("Roles list is empty.")
            return []
        
        connection = None
        response = None
        try:
            connection = psycopg.connect(conninfo=ConfigManager()._instance.db_conninfo, row_factory=dict_row)
            cursor = connection.cursor()
            logger.debug(f"PostgreSQL connection established.")

            query_sql = """SELECT mem.member_id, mem.first_name, mem.last_name, mem.email, mem.roles, mem.register_date, mem.community_score, tg.username, tg.user_id
            FROM member_data AS mem
            JOIN member_telegram AS tg
            ON mem.member_id = tg.member_id
            WHERE roles && %s;"""
            cursor.execute(query_sql, (roles, ))

            results = cursor.fetchall()
            # Close the cursor
            cursor.close()

            response = results

        except psycopg.Error as error:
                logger.error(f"Exception while working with psycopg and PostgreSQL:\n{error}")
        finally:
            if (connection):
                connection.close()
                logger.debug(f"PostgreSQL connection is closed.")
            
            return response

    def addMemberFromTelegram(self, memberData: dict) -> int | None:
        logger.info(f"Adding a new member from telegram.")
        connection = None
        memberID = None
        try:
            connection = psycopg.connect(conninfo=ConfigManager()._instance.db_conninfo, row_factory=dict_row)
            cursor = connection.cursor()
            logger.debug(f"PostgreSQL connection established.")

            # Add the data from record to our new member_data table
            insertMember_sql = """INSERT INTO member_data (first_name, last_name, email, roles, register_date) 
            VALUES (%(first_name)s, %(last_name)s, %(email)s, %(roles)s, %(register_date)s)
            RETURNING member_id;"""
            
            cursor.execute(insertMember_sql, memberData)
            result = cursor.fetchone()
            memberID = result.get("member_id") if result else None
            if memberID is None:
                cursor.close()
                return None

            memberTelegramData = {
                "member_id": memberID,
                "first_name": memberData.get("first_name"),
                "last_name": memberData.get("last_name"),
                "username": memberData.get("username"),
                "user_id": memberData.get("user_id")
            }
            insertMemberTelegram_sql = """INSERT INTO member_telegram (member_id, first_name, last_name, username, user_id) 
            VALUES (%(member_id)s, %(first_name)s, %(last_name)s, %(username)s, %(user_id)s);"""

            cursor.execute(insertMemberTelegram_sql, memberTelegramData)
            
            connection.commit()
            cursor.close()
            
        except psycopg.Error as error:
            logger.error(f"Exception while working with psycopg and PostgreSQL:\n{error}")
        finally:
            if (connection):
                connection.close()
                logger.debug(f"PostgreSQL connection is closed.")
        
        return memberID

    def registerWebMember(
        self,
        username: str,
        password: str,
        firstName: str | None = None,
        lastName: str | None = None,
        email: str | None = None,
    ) -> tuple[dict | None, str | None]:
        logger.info("Register a new member from web signup.")
        cleanUsername = str(username or "").strip().lstrip("@")
        if cleanUsername == "":
            return None, "Username is required."
        if str(password or "").strip() == "":
            return None, "Password is required."

        connection = None
        memberID = None
        try:
            connection = psycopg.connect(conninfo=ConfigManager()._instance.db_conninfo, row_factory=dict_row)
            cursor = connection.cursor()
            logger.debug("PostgreSQL connection established.")

            cursor.execute(
                """SELECT mem.member_id
                FROM member_data AS mem
                JOIN member_telegram AS tg
                ON mem.member_id = tg.member_id
                WHERE LOWER(tg.username) = LOWER(%s)
                LIMIT 1;""",
                (cleanUsername,),
            )
            existing = cursor.fetchone()
            if existing is not None:
                cursor.close()
                return None, "Username is already in use."

            memberData = {
                "first_name": str(firstName or cleanUsername).strip(),
                "last_name": str(lastName).strip() if lastName else None,
                "email": str(email).strip() if email else None,
                "roles": ["user"],
                "register_date": datetime.now(),
                "community_score": 0,
            }

            cursor.execute(
                """INSERT INTO member_data (first_name, last_name, email, roles, register_date, community_score)
                VALUES (%(first_name)s, %(last_name)s, %(email)s, %(roles)s, %(register_date)s, %(community_score)s)
                RETURNING member_id;""",
                memberData,
            )
            result = cursor.fetchone()
            memberID = result.get("member_id") if result else None
            if memberID is None:
                cursor.close()
                return None, "Unable to allocate member id."

            cursor.execute(
                """INSERT INTO member_telegram (member_id, first_name, last_name, username, user_id)
                VALUES (%s, %s, %s, %s, %s);""",
                (
                    memberID,
                    memberData["first_name"],
                    memberData["last_name"],
                    cleanUsername,
                    None,
                ),
            )
            connection.commit()
            cursor.close()
        except psycopg.Error as error:
            logger.error(f"Exception while working with psycopg and PostgreSQL:\n{error}")
            return None, "Database error during signup."
        finally:
            if connection:
                connection.close()
                logger.debug("PostgreSQL connection is closed.")

        if memberID is None:
            return None, "Signup failed."

        configuredPassword = self.setPassword(memberID, password=password)
        if configuredPassword is None:
            cleanupConnection = None
            try:
                cleanupConnection = psycopg.connect(conninfo=ConfigManager()._instance.db_conninfo)
                cleanupCursor = cleanupConnection.cursor()
                cleanupCursor.execute("DELETE FROM member_data WHERE member_id = %s;", (memberID,))
                cleanupConnection.commit()
                cleanupCursor.close()
            except psycopg.Error as error:
                logger.error(f"Exception while cleaning up failed signup:\n{error}")
            finally:
                if cleanupConnection:
                    cleanupConnection.close()
            minPasswordLength = ConfigManager().runtimeInt("security.password_min_length", 12)
            return (
                None,
                "Password does not meet policy. "
                f"Use at least {max(8, minPasswordLength)} chars with uppercase, lowercase, digits, and symbols.",
            )

        member = self.getMemberByID(memberID)
        return member, None

    def loginMember(self, username: str, password: str) -> dict:
        logger.info(f"Login member with username and password.")
        # Get primary key and register date by the username
        loginResults = None
        connection = None
        try:
            connection = psycopg.connect(conninfo=ConfigManager()._instance.db_conninfo, row_factory=dict_row)
            cursor = connection.cursor()
            logger.debug(f"PostgreSQL connection established.")

            getSaltQuery_sql = """SELECT mem.member_id, mem.register_date
            FROM member_data AS mem
            JOIN member_telegram AS tg
            ON mem.member_id = tg.member_id
            WHERE LOWER(tg.username) = LOWER(%s)
            LIMIT 1;"""

            cursor.execute(getSaltQuery_sql, (username, ))
            saltData = cursor.fetchone()
            if saltData:
                memberID = saltData["member_id"]
                registerDate: datetime = saltData["register_date"]
                unixTimestamp = time.mktime(registerDate.timetuple())
                hashInput = str(memberID) + password + str(unixTimestamp)
                securePassword = hashlib.sha256(hashInput.encode())

                getMemberQuery_sql = """SELECT mem.member_id, mem.first_name, mem.last_name, mem.email, mem.roles, mem.register_date, mem.community_score, tg.username, tg.user_id
                FROM member_data AS mem
                JOIN member_secure AS ms
                ON ms.member_id = mem.member_id
                LEFT JOIN member_telegram AS tg
                ON tg.member_id = mem.member_id
                WHERE ms.secure_hash = %s::BYTEA
                LIMIT 1;"""
                cursor.execute(getMemberQuery_sql, (securePassword.digest(), ))
                loginResults = cursor.fetchone()

            cursor.close()
        except psycopg.Error as error:
            logger.error(f"Exception while working with psycopg and PostgreSQL:\n{error}")
        finally:
            if (connection):
                connection.close()
                logger.debug(f"PostgreSQL connection is closed.")

            return loginResults

    def setPassword(self, memberID: int, password: str = None) -> str:
        logger.info("Set a member password.")

        member = self.getMemberByID(memberID)
        if member is None:
            return None
        minPasswordLength = ConfigManager().runtimeInt("security.password_min_length", 12)
        
        # Create a random password if none was sent
        if password is None:
            alphabet = string.ascii_letters + string.digits
            while True:
                password = ''.join(secrets.choice(alphabet) for i in range(max(8, minPasswordLength)))
                if (any(c.islower() for c in password)
                        and any(c.isupper() for c in password)
                        and sum(c.isdigit() for c in password) >= 2):
                    break
        else:
            # Return None if the password does not meet minimum requirements
            pattern = re.compile(
                r"(?=.*\d)(?=.*[a-z])(?=.*[A-Z])(?=.*[-+_!@#$%^&*.,?]).{"
                + str(max(8, minPasswordLength))
                + r",}"
            )
            validPassword = pattern.search(password)
            if validPassword is None:
                logger.error("Invalid password.")
                # Critical error effects remaining functionality, exit
                return None
        
        # Create a hash string
        registerDate: datetime = member["register_date"]
        unixTimestamp = time.mktime(registerDate.timetuple())
        memberID = member["member_id"]
        hashInput = str(memberID) + password + str(unixTimestamp)
        securePassword = hashlib.sha256(hashInput.encode())

        # Store the hash string in passkey table
        results = None
        connection = None
        try:
            connection = psycopg.connect(conninfo=ConfigManager()._instance.db_conninfo)
            cursor = connection.cursor()
            logger.debug(f"PostgreSQL connection established.")

            insert_sql = """INSERT INTO member_secure (member_id, secure_hash) 
            VALUES (%s, %s)
            ON CONFLICT (member_id)
            DO UPDATE SET secure_hash = %s;"""
            cursor.execute(insert_sql, (memberID, securePassword.digest(), securePassword.digest()))
            connection.commit()
            results = password
            cursor.close()
            
        except psycopg.Error as error:
            logger.error(f"Error while working with psycopg and PostgreSQL:\n{error}")
        finally:
            if (connection):
                connection.close()
                logger.debug(f"PostgreSQL connection is closed.")

            # Return password to UI
            return results

    def updateMemberEmail(self, memberID: int, email: str):
        logger.info("Update member email.")

        # Run Reg Expression to validate
        # Return None if the email does not meet minimum requirements
        pattern = re.compile(r"^[\w\-\.]+@([\w-]+\.)+[\w-]{2,72}$")
        validEmail = pattern.search(email)
        if validEmail is None:
            logger.error("Invalid email.")
            # Critical error effects remaining functionality, exit
            return None
        
        logger.debug("Valid Email")

        # Add email to database
        connection = None
        try:
            connection = psycopg.connect(conninfo=ConfigManager()._instance.db_conninfo)
            cursor = connection.cursor()
            logger.debug(f"PostgreSQL connection established.")
            
            updateEmail_sql = "UPDATE member_data SET email = %s WHERE member_id = %s;"
            valueTuple = (email, memberID)
            cursor.execute(updateEmail_sql, valueTuple)
            connection.commit()
            cursor.close()
            
        except psycopg.Error as error:
                logger.error(f"Exception while working with psycopg and PostgreSQL:\n{error}")
        finally:
            if (connection):
                connection.close()
                logger.debug(f"PostgreSQL connection is closed.")

    def updateMemberRoles(self, memberID: int, roles: list):
        logger.info("Update member roles.")

        member = self.getMemberByID(memberID)
        if member is not None:
            # TODO Validate the roles agaist the master roles list

            connection = None
            response = False
            try:
                connection = psycopg.connect(conninfo=ConfigManager()._instance.db_conninfo)
                cursor = connection.cursor()
                logger.debug(f"PostgreSQL connection established.")
                vectorDimensions = max(1, ConfigManager().runtimeInt("vectors.embedding_dimensions", 768))
                
                updateRoles_sql = "UPDATE member_data SET roles = %s WHERE member_id = %s;"
                valueTuple = (roles, memberID)
                cursor.execute(updateRoles_sql, valueTuple)
                connection.commit()
                cursor.close()
                response = True
                
            except psycopg.Error as error:
                    logger.error(f"Exception while working with psycopg and PostgreSQL:\n{error}")
            finally:
                if (connection):
                    connection.close()
                    logger.debug(f"PostgreSQL connection is closed.")
            
            return response
    
    def updateCommunityScore(self, memberID: int, newScore: float):
        logger.info("Update the users community score.")

        connection = None
        try:
            connection = psycopg.connect(conninfo=ConfigManager()._instance.db_conninfo)
            cursor = connection.cursor()
            logger.debug(f"PostgreSQL connection established.")

            update_sql = "UPDATE member_data SET community_score = %s WHERE member_id = %s;"
            cursor.execute(update_sql, (newScore, memberID))
            connection.commit()
            cursor.close()
            
        except psycopg.Error as error:
                logger.error(f"Exception while working with psycopg and PostgreSQL:\n{error}")
        finally:
            if (connection):
                connection.close()
                logger.debug(f"PostgreSQL connection is closed.")

    def validateMiniappData(self, telegramInitData: str):
        logger.info(f"Validating the data sent via telegram miniapp.")
        if not telegramInitData:
            logger.warning("Miniapp validation failed: missing init data payload.")
            return None

        # Parse the telegram initData query string
        queryDict = parse_qs(telegramInitData)
        knownHashValues = queryDict.get("hash")
        userValues = queryDict.get("user")
        if not knownHashValues or not userValues:
            logger.warning("Miniapp validation failed: required payload fields are missing.")
            return None

        knownHash = knownHashValues[0]

        try:
            telegramUserData = json.loads(userValues[0])
        except (json.JSONDecodeError, TypeError):
            logger.warning("Miniapp validation failed: malformed user payload.")
            return None

        memberTelegramID = telegramUserData.get("id")
        if memberTelegramID is None:
            logger.warning("Miniapp validation failed: user id not found in payload.")
            return None

        # Create a data check string from the query string
        # Data Check String must have the hash propoerty removed
        initDataChunks = []
        for chunk in unquote(telegramInitData).split("&"):
            if chunk[:len("hash=")] == "hash=" or "=" not in chunk:
                continue
            key, value = chunk.split("=", 1)
            initDataChunks.append((key, value))
        initDataChunks = sorted(initDataChunks, key=lambda x: x[0])
        initData = "\n".join([f"{rec[0]}={rec[1]}" for rec in initDataChunks])

        # Create the Secret Key
        key = "WebAppData".encode()
        token = ConfigManager().bot_token
        if token is None or str(token).strip() == "":
            logger.warning("Miniapp validation unavailable: bot_token is missing from config.")
            return None

        secretKey = hmac.new(key, str(token).encode(), hashlib.sha256)
        digest = hmac.new(secretKey.digest(), initData.encode(), hashlib.sha256)

        if hmac.compare_digest(knownHash, digest.hexdigest()):
            logger.info("Miniapp payload hash validated.")
            member = MemberManager().getMemberByTelegramID(memberTelegramID)
            if member is not None:
                return member["member_id"]
        
        return None


    # Define getters
    # TODO move to config and config manager
    @property
    def rolesList(self):
        return self.__rolesList


class CollaborationWorkspaceManager:
    _instance = None
    _ALLOWED_PROCESS_STATUS = {"active", "paused", "blocked", "completed", "cancelled"}
    _ALLOWED_STEP_STATUS = {"pending", "in_progress", "blocked", "completed", "skipped", "cancelled"}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(CollaborationWorkspaceManager, cls).__new__(cls)
            connection = None
            try:
                connection = psycopg.connect(conninfo=ConfigManager()._instance.db_conninfo, row_factory=dict_row)
                cursor = connection.cursor()
                logger.debug("PostgreSQL connection established for collaboration workspace.")
                execute_migration(cursor, "084_agent_process_workspace.sql")
                connection.commit()
                execute_migration(cursor, "085_member_outbox.sql")
                connection.commit()
                cursor.close()
            except psycopg.Error as error:
                logger.error(f"Exception while preparing collaboration workspace tables:\n{error}")
            finally:
                if connection:
                    connection.close()
                    logger.debug("PostgreSQL connection is closed.")
        return cls._instance

    @staticmethod
    def _as_int(value: Any, fallback: int | None = None) -> int | None:
        try:
            return int(value)
        except (TypeError, ValueError):
            return fallback

    @staticmethod
    def _as_text(value: Any, fallback: str = "") -> str:
        text = str(value if value is not None else "").strip()
        return text if text else fallback

    @staticmethod
    def _coerce_dict(value: Any) -> dict[str, Any]:
        if isinstance(value, dict):
            return dict(value)
        return {}

    @staticmethod
    def _coerce_list(value: Any) -> list[Any]:
        if isinstance(value, list):
            return list(value)
        return []

    @staticmethod
    def _timestamp_iso(value: Any) -> str | None:
        if isinstance(value, datetime):
            return value.replace(microsecond=0).isoformat()
        return None

    def _connect(self) -> psycopg.Connection:
        return psycopg.connect(conninfo=ConfigManager()._instance.db_conninfo, row_factory=dict_row)

    def _normalize_process_status(self, value: Any, default: str = "active") -> str:
        status = self._as_text(value, default).lower()
        if status not in self._ALLOWED_PROCESS_STATUS:
            return default
        return status

    def _normalize_step_status(self, value: Any, default: str = "pending") -> str:
        status = self._as_text(value, default).lower()
        if status not in self._ALLOWED_STEP_STATUS:
            return default
        return status

    def _normalize_steps(self, raw_steps: Any) -> list[dict[str, Any]]:
        steps_input: list[Any]
        if isinstance(raw_steps, str):
            cleaned = raw_steps.strip()
            if cleaned == "":
                steps_input = []
            else:
                try:
                    parsed = json.loads(cleaned)
                except json.JSONDecodeError:
                    parsed = []
                if isinstance(parsed, dict):
                    steps_input = self._coerce_list(parsed.get("steps"))
                elif isinstance(parsed, list):
                    steps_input = parsed
                else:
                    steps_input = []
        elif isinstance(raw_steps, dict):
            steps_input = self._coerce_list(raw_steps.get("steps"))
        else:
            steps_input = self._coerce_list(raw_steps)

        normalized: list[dict[str, Any]] = []
        for item in steps_input[:400]:
            if isinstance(item, str):
                label = self._as_text(item)
                details = ""
                status = "pending"
                required = True
                payload = {}
            elif isinstance(item, dict):
                label = self._as_text(
                    item.get("label")
                    or item.get("title")
                    or item.get("name")
                    or item.get("step")
                )
                details = self._as_text(item.get("details") or item.get("description"))
                status = self._normalize_step_status(item.get("status"), "pending")
                required = bool(item.get("required", True))
                payload = self._coerce_dict(item.get("payload") or item.get("metadata"))
            else:
                continue

            if label == "":
                continue
            normalized.append(
                {
                    "label": label[:240],
                    "details": details,
                    "status": status,
                    "required": required,
                    "payload": payload,
                }
            )
        return normalized

    def _refresh_process_progress(self, cursor: Any, process_id: int) -> dict[str, Any]:
        cursor.execute(
            """SELECT process_status
            FROM agent_processes
            WHERE process_id = %s
            LIMIT 1;""",
            (process_id,),
        )
        process_row = cursor.fetchone()
        if process_row is None:
            return {"steps_total": 0, "steps_completed": 0, "completion_percent": 0.0}

        current_status = self._normalize_process_status(process_row.get("process_status"), "active")
        cursor.execute(
            """SELECT
                COUNT(*)::INT AS total_steps,
                COALESCE(SUM(CASE WHEN step_status IN ('completed', 'skipped') THEN 1 ELSE 0 END), 0)::INT AS completed_steps
            FROM agent_process_steps
            WHERE process_id = %s;""",
            (process_id,),
        )
        counts = cursor.fetchone() or {}
        total_steps = max(0, self._as_int(counts.get("total_steps"), 0) or 0)
        completed_steps = max(0, self._as_int(counts.get("completed_steps"), 0) or 0)
        completion_percent = 0.0 if total_steps <= 0 else round((completed_steps / total_steps) * 100.0, 2)

        next_status = current_status
        completed_at = None
        if current_status not in {"cancelled"}:
            if total_steps > 0 and completed_steps >= total_steps:
                next_status = "completed"
                completed_at = datetime.now()
            elif current_status == "completed" and completed_steps < total_steps:
                next_status = "active"

        cursor.execute(
            """UPDATE agent_processes
            SET process_status = %s,
                steps_total = %s,
                steps_completed = %s,
                completion_percent = %s,
                updated_at = NOW(),
                completed_at = CASE
                    WHEN %s = 'completed' THEN COALESCE(completed_at, NOW())
                    WHEN %s != 'completed' THEN NULL
                    ELSE completed_at
                END
            WHERE process_id = %s;""",
            (
                next_status,
                total_steps,
                completed_steps,
                completion_percent,
                next_status,
                next_status,
                process_id,
            ),
        )
        return {
            "steps_total": total_steps,
            "steps_completed": completed_steps,
            "completion_percent": completion_percent,
            "process_status": next_status,
        }

    def listKnownUsers(
        self,
        queryString: str | None = None,
        count: int = 20,
        include_without_username: bool = False,
    ) -> list[dict[str, Any]]:
        limit = min(100, max(1, self._as_int(count, 20) or 20))
        query = self._as_text(queryString).lower()
        include_no_username = bool(include_without_username)

        connection = None
        response: list[dict[str, Any]] = []
        try:
            connection = self._connect()
            cursor = connection.cursor()
            where_parts = []
            values: list[Any] = []
            if not include_no_username:
                where_parts.append("COALESCE(tg.username, '') <> ''")

            if query:
                like = f"%{query}%"
                where_parts.append(
                    "("
                    "LOWER(COALESCE(tg.username, '')) LIKE %s "
                    "OR LOWER(COALESCE(mem.first_name, '')) LIKE %s "
                    "OR LOWER(COALESCE(mem.last_name, '')) LIKE %s "
                    "OR CAST(mem.member_id AS TEXT) = %s "
                    "OR CAST(COALESCE(tg.user_id, 0) AS TEXT) = %s"
                    ")"
                )
                values.extend([like, like, like, query, query])

            where_sql = ""
            if where_parts:
                where_sql = "WHERE " + " AND ".join(where_parts)

            cursor.execute(
                f"""SELECT mem.member_id, mem.first_name, mem.last_name, mem.community_score, mem.roles, tg.username, tg.user_id,
                contact.last_contact_at
                FROM member_data AS mem
                LEFT JOIN member_telegram AS tg
                ON mem.member_id = tg.member_id
                LEFT JOIN (
                    SELECT chat_host_id AS member_id, MAX(message_timestamp) AS last_contact_at
                    FROM chat_history
                    WHERE chat_type = 'member'
                    GROUP BY chat_host_id
                ) AS contact
                ON mem.member_id = contact.member_id
                {where_sql}
                ORDER BY LOWER(COALESCE(tg.username, '')), mem.member_id
                LIMIT %s;""",
                (*values, limit),
            )
            records = cursor.fetchall() or []
            for row in records:
                row_map = self._coerce_dict(row)
                response.append(
                    {
                        "member_id": row_map.get("member_id"),
                        "username": self._as_text(row_map.get("username")),
                        "user_id": row_map.get("user_id"),
                        "first_name": row_map.get("first_name"),
                        "last_name": row_map.get("last_name"),
                        "community_score": row_map.get("community_score"),
                        "roles": row_map.get("roles") if isinstance(row_map.get("roles"), list) else [],
                        "has_telegram_user_id": row_map.get("user_id") is not None,
                        "last_contact_at": self._timestamp_iso(row_map.get("last_contact_at")),
                    }
                )
            cursor.close()
        except psycopg.Error as error:
            logger.error(f"Exception while listing known users:\n{error}")
        finally:
            if connection:
                connection.close()
        return response

    def listKnownGroups(
        self,
        queryString: str | None = None,
        count: int = 12,
    ) -> list[dict[str, Any]]:
        limit = min(64, max(1, self._as_int(count, 12) or 12))
        query = self._as_text(queryString).lower()

        connection = None
        response: list[dict[str, Any]] = []
        try:
            connection = self._connect()
            cursor = connection.cursor()

            where_parts = []
            values: list[Any] = []
            if query:
                like = f"%{query}%"
                where_parts.append(
                    "("
                    "LOWER(COALESCE(cd.community_name, '')) LIKE %s "
                    "OR LOWER(COALESCE(tg.chat_title, '')) LIKE %s "
                    "OR LOWER(COALESCE(cd.community_link, '')) LIKE %s "
                    "OR CAST(cd.community_id AS TEXT) = %s "
                    "OR CAST(COALESCE(tg.chat_id, 0) AS TEXT) = %s"
                    ")"
                )
                values.extend([like, like, like, query, query])

            where_sql = ""
            if where_parts:
                where_sql = "WHERE " + " AND ".join(where_parts)

            cursor.execute(
                f"""SELECT
                    cd.community_id,
                    cd.community_name,
                    cd.community_link,
                    cd.roles,
                    tg.chat_id,
                    tg.chat_title,
                    tg.has_topics,
                    last_msg.message_timestamp AS last_activity_at,
                    assistant_msg.message_timestamp AS last_assistant_at,
                    assistant_msg.message_text AS last_assistant_message,
                    activity.activity_24h_count
                FROM community_data AS cd
                LEFT JOIN community_telegram AS tg
                ON cd.community_id = tg.community_id
                LEFT JOIN LATERAL (
                    SELECT ch.message_timestamp, ch.message_text
                    FROM chat_history AS ch
                    WHERE ch.community_id = cd.community_id
                    AND ch.chat_type = 'community'
                    ORDER BY ch.message_timestamp DESC
                    LIMIT 1
                ) AS last_msg
                ON TRUE
                LEFT JOIN LATERAL (
                    SELECT ch.message_timestamp, ch.message_text
                    FROM chat_history AS ch
                    WHERE ch.community_id = cd.community_id
                    AND ch.chat_type = 'community'
                    AND ch.member_id IS NULL
                    ORDER BY ch.message_timestamp DESC
                    LIMIT 1
                ) AS assistant_msg
                ON TRUE
                LEFT JOIN LATERAL (
                    SELECT COUNT(*)::INT AS activity_24h_count
                    FROM chat_history AS ch
                    WHERE ch.community_id = cd.community_id
                    AND ch.chat_type = 'community'
                    AND ch.message_timestamp > (NOW() - INTERVAL '24 HOURS')
                ) AS activity
                ON TRUE
                {where_sql}
                ORDER BY COALESCE(last_msg.message_timestamp, cd.register_date) DESC, cd.community_id
                LIMIT %s;""",
                (*values, limit),
            )
            rows = cursor.fetchall() or []
            cursor.close()
            for row in rows:
                row_map = self._coerce_dict(row)
                assistant_excerpt = self._as_text(row_map.get("last_assistant_message"))
                if len(assistant_excerpt) > 220:
                    assistant_excerpt = assistant_excerpt[:217].rstrip() + "..."
                response.append(
                    {
                        "community_id": row_map.get("community_id"),
                        "community_name": self._as_text(row_map.get("community_name")),
                        "community_link": self._as_text(row_map.get("community_link")),
                        "chat_id": row_map.get("chat_id"),
                        "chat_title": self._as_text(row_map.get("chat_title")),
                        "has_topics": bool(row_map.get("has_topics")),
                        "roles": row_map.get("roles") if isinstance(row_map.get("roles"), list) else [],
                        "activity_24h_count": self._as_int(row_map.get("activity_24h_count"), 0) or 0,
                        "last_activity_at": self._timestamp_iso(row_map.get("last_activity_at")),
                        "last_assistant_at": self._timestamp_iso(row_map.get("last_assistant_at")),
                        "last_assistant_excerpt": assistant_excerpt,
                    }
                )
        except psycopg.Error as error:
            logger.error(f"Exception while listing known groups:\n{error}")
        finally:
            if connection:
                connection.close()
        return response

    def resolveKnownUser(
        self,
        *,
        username: str | None = None,
        memberID: int | None = None,
    ) -> dict[str, Any] | None:
        clean_username = self._as_text(username).lstrip("@")
        member_id = self._as_int(memberID)

        if member_id is None and clean_username == "":
            return None

        connection = None
        response = None
        try:
            connection = self._connect()
            cursor = connection.cursor()
            if member_id is not None:
                cursor.execute(
                    """SELECT mem.member_id, mem.first_name, mem.last_name, mem.community_score, mem.roles, tg.username, tg.user_id
                    FROM member_data AS mem
                    LEFT JOIN member_telegram AS tg
                    ON mem.member_id = tg.member_id
                    WHERE mem.member_id = %s
                    LIMIT 1;""",
                    (member_id,),
                )
            else:
                cursor.execute(
                    """SELECT mem.member_id, mem.first_name, mem.last_name, mem.community_score, mem.roles, tg.username, tg.user_id
                    FROM member_data AS mem
                    LEFT JOIN member_telegram AS tg
                    ON mem.member_id = tg.member_id
                    WHERE LOWER(COALESCE(tg.username, '')) = LOWER(%s)
                    LIMIT 1;""",
                    (clean_username,),
                )
            result = cursor.fetchone()
            cursor.close()
            if result is not None:
                result_map = self._coerce_dict(result)
                response = {
                    "member_id": result_map.get("member_id"),
                    "username": self._as_text(result_map.get("username")),
                    "user_id": result_map.get("user_id"),
                    "first_name": result_map.get("first_name"),
                    "last_name": result_map.get("last_name"),
                    "community_score": result_map.get("community_score"),
                    "roles": result_map.get("roles") if isinstance(result_map.get("roles"), list) else [],
                }
        except psycopg.Error as error:
            logger.error(f"Exception while resolving known user:\n{error}")
        finally:
            if connection:
                connection.close()
        return response

    def createOrUpdateProcess(
        self,
        *,
        ownerMemberID: int,
        processLabel: str,
        processDescription: str | None = None,
        processSpec: Any = None,
        processID: int | None = None,
        processStatus: str = "active",
        replaceSteps: bool = True,
    ) -> dict[str, Any]:
        owner_id = self._as_int(ownerMemberID)
        if owner_id is None or owner_id <= 0:
            return {"status": "error", "error": "owner_member_id_required"}

        label = self._as_text(processLabel)
        if label == "":
            return {"status": "error", "error": "process_label_required"}
        description = self._as_text(processDescription)
        normalized_status = self._normalize_process_status(processStatus, "active")
        normalized_steps = self._normalize_steps(processSpec)

        parsed_spec: Any = processSpec
        if isinstance(processSpec, str):
            try:
                parsed_spec = json.loads(processSpec)
            except json.JSONDecodeError:
                parsed_spec = {"raw_text": processSpec}
        if isinstance(parsed_spec, list):
            parsed_spec = {"steps": parsed_spec}
        if not isinstance(parsed_spec, dict):
            parsed_spec = {"spec": parsed_spec}
        if "steps" not in parsed_spec and normalized_steps:
            parsed_spec["steps"] = normalized_steps

        connection = None
        try:
            connection = self._connect()
            cursor = connection.cursor()
            process_id = self._as_int(processID)
            if process_id is not None and process_id > 0:
                cursor.execute(
                    """SELECT process_id
                    FROM agent_processes
                    WHERE process_id = %s AND owner_member_id = %s
                    LIMIT 1;""",
                    (process_id, owner_id),
                )
                existing = cursor.fetchone()
                if existing is None:
                    cursor.close()
                    connection.rollback()
                    return {"status": "error", "error": "process_not_found"}

                cursor.execute(
                    """UPDATE agent_processes
                    SET process_label = %s,
                        process_description = %s,
                        process_status = %s,
                        process_payload = %s::jsonb,
                        updated_at = NOW()
                    WHERE process_id = %s
                    RETURNING process_id;""",
                    (label[:160], description, normalized_status, json.dumps(parsed_spec), process_id),
                )
                updated = cursor.fetchone()
                process_id = self._as_int(updated.get("process_id") if isinstance(updated, dict) else None)
            else:
                cursor.execute(
                    """INSERT INTO agent_processes (
                        owner_member_id,
                        process_label,
                        process_description,
                        process_status,
                        process_payload,
                        created_at,
                        updated_at
                    )
                    VALUES (%s, %s, %s, %s, %s::jsonb, NOW(), NOW())
                    RETURNING process_id;""",
                    (owner_id, label[:160], description, normalized_status, json.dumps(parsed_spec)),
                )
                inserted = cursor.fetchone()
                process_id = self._as_int(inserted.get("process_id") if isinstance(inserted, dict) else None)

            if process_id is None:
                cursor.close()
                connection.rollback()
                return {"status": "error", "error": "process_write_failed"}

            if replaceSteps:
                cursor.execute(
                    "DELETE FROM agent_process_steps WHERE process_id = %s;",
                    (process_id,),
                )

            if normalized_steps:
                for index, step in enumerate(normalized_steps, start=1):
                    step_status = self._normalize_step_status(step.get("status"), "pending")
                    cursor.execute(
                        """INSERT INTO agent_process_steps (
                            process_id,
                            step_order,
                            step_label,
                            step_details,
                            step_status,
                            is_required,
                            step_payload,
                            created_at,
                            updated_at,
                            completed_at
                        )
                        VALUES (
                            %s, %s, %s, %s, %s, %s, %s::jsonb, NOW(), NOW(),
                            CASE WHEN %s = 'completed' THEN NOW() ELSE NULL END
                        );""",
                        (
                            process_id,
                            index,
                            self._as_text(step.get("label"))[:240],
                            self._as_text(step.get("details")),
                            step_status,
                            bool(step.get("required", True)),
                            json.dumps(self._coerce_dict(step.get("payload"))),
                            step_status,
                        ),
                    )

            self._refresh_process_progress(cursor, process_id)
            connection.commit()
            cursor.close()
            result = self.getProcessByID(owner_id, process_id, include_steps=True)
            if result is None:
                return {"status": "error", "error": "process_lookup_failed"}
            return {"status": "ok", "process": result}
        except psycopg.Error as error:
            if connection:
                connection.rollback()
            logger.error(f"Exception while creating/updating process workspace:\n{error}")
            return {"status": "error", "error": "database_error", "detail": str(error)}
        finally:
            if connection:
                connection.close()

    def getProcessByID(
        self,
        ownerMemberID: int,
        processID: int,
        include_steps: bool = True,
    ) -> dict[str, Any] | None:
        owner_id = self._as_int(ownerMemberID)
        process_id = self._as_int(processID)
        if owner_id is None or process_id is None:
            return None

        connection = None
        response = None
        try:
            connection = self._connect()
            cursor = connection.cursor()
            cursor.execute(
                """SELECT
                    process_id,
                    owner_member_id,
                    process_label,
                    process_description,
                    process_status,
                    completion_percent,
                    steps_total,
                    steps_completed,
                    process_payload,
                    created_at,
                    updated_at,
                    completed_at
                FROM agent_processes
                WHERE process_id = %s
                AND owner_member_id = %s
                LIMIT 1;""",
                (process_id, owner_id),
            )
            row = cursor.fetchone()
            if row is None:
                cursor.close()
                return None
            process_row = self._coerce_dict(row)
            response = {
                "process_id": process_row.get("process_id"),
                "owner_member_id": process_row.get("owner_member_id"),
                "process_label": process_row.get("process_label"),
                "process_description": process_row.get("process_description"),
                "process_status": process_row.get("process_status"),
                "completion_percent": process_row.get("completion_percent"),
                "steps_total": process_row.get("steps_total"),
                "steps_completed": process_row.get("steps_completed"),
                "process_payload": self._coerce_dict(process_row.get("process_payload")),
                "created_at": self._timestamp_iso(process_row.get("created_at")),
                "updated_at": self._timestamp_iso(process_row.get("updated_at")),
                "completed_at": self._timestamp_iso(process_row.get("completed_at")),
            }

            steps: list[dict[str, Any]] = []
            if include_steps:
                cursor.execute(
                    """SELECT
                        step_id,
                        process_id,
                        step_order,
                        step_label,
                        step_details,
                        step_status,
                        is_required,
                        step_payload,
                        created_at,
                        updated_at,
                        completed_at
                    FROM agent_process_steps
                    WHERE process_id = %s
                    ORDER BY step_order ASC, step_id ASC;""",
                    (process_id,),
                )
                rows = cursor.fetchall() or []
                for step_row in rows:
                    step_map = self._coerce_dict(step_row)
                    steps.append(
                        {
                            "step_id": step_map.get("step_id"),
                            "step_order": step_map.get("step_order"),
                            "step_label": step_map.get("step_label"),
                            "step_details": step_map.get("step_details"),
                            "step_status": step_map.get("step_status"),
                            "is_required": bool(step_map.get("is_required", True)),
                            "step_payload": self._coerce_dict(step_map.get("step_payload")),
                            "created_at": self._timestamp_iso(step_map.get("created_at")),
                            "updated_at": self._timestamp_iso(step_map.get("updated_at")),
                            "completed_at": self._timestamp_iso(step_map.get("completed_at")),
                        }
                    )
                response["steps"] = steps
                response["next_missing_steps"] = [
                    step for step in steps if step.get("step_status") not in {"completed", "skipped", "cancelled"}
                ][:5]
            cursor.close()
        except psycopg.Error as error:
            logger.error(f"Exception while getting process workspace data:\n{error}")
        finally:
            if connection:
                connection.close()
        return response

    def listProcesses(
        self,
        ownerMemberID: int,
        processStatus: str | None = "active",
        count: int = 12,
        include_steps: bool = False,
    ) -> list[dict[str, Any]]:
        owner_id = self._as_int(ownerMemberID)
        if owner_id is None or owner_id <= 0:
            return []
        limit = min(100, max(1, self._as_int(count, 12) or 12))
        status_filter = self._as_text(processStatus, "active").lower()

        connection = None
        output: list[dict[str, Any]] = []
        try:
            connection = self._connect()
            cursor = connection.cursor()
            if status_filter in {"all", "*", ""}:
                cursor.execute(
                    """SELECT process_id
                    FROM agent_processes
                    WHERE owner_member_id = %s
                    ORDER BY updated_at DESC, process_id DESC
                    LIMIT %s;""",
                    (owner_id, limit),
                )
            else:
                cursor.execute(
                    """SELECT process_id
                    FROM agent_processes
                    WHERE owner_member_id = %s
                    AND process_status = %s
                    ORDER BY updated_at DESC, process_id DESC
                    LIMIT %s;""",
                    (owner_id, status_filter, limit),
                )
            rows = cursor.fetchall() or []
            cursor.close()
            for row in rows:
                process_id = self._as_int(self._coerce_dict(row).get("process_id"))
                if process_id is None:
                    continue
                process_data = self.getProcessByID(owner_id, process_id, include_steps=include_steps)
                if process_data is not None:
                    output.append(process_data)
        except psycopg.Error as error:
            logger.error(f"Exception while listing process workspace records:\n{error}")
        finally:
            if connection:
                connection.close()
        return output

    def updateProcessStep(
        self,
        *,
        ownerMemberID: int,
        processID: int,
        stepID: int | None = None,
        stepOrder: int | None = None,
        stepLabel: str | None = None,
        stepStatus: str = "completed",
        stepDetails: str | None = None,
        stepPayload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        owner_id = self._as_int(ownerMemberID)
        process_id = self._as_int(processID)
        if owner_id is None or process_id is None:
            return {"status": "error", "error": "invalid_process_reference"}

        normalized_status = self._normalize_step_status(stepStatus, "completed")
        normalized_label = self._as_text(stepLabel)
        payload_map = self._coerce_dict(stepPayload)

        connection = None
        try:
            connection = self._connect()
            cursor = connection.cursor()
            cursor.execute(
                """SELECT process_id
                FROM agent_processes
                WHERE process_id = %s AND owner_member_id = %s
                LIMIT 1;""",
                (process_id, owner_id),
            )
            process_row = cursor.fetchone()
            if process_row is None:
                cursor.close()
                connection.rollback()
                return {"status": "error", "error": "process_not_found"}

            target_step_id = self._as_int(stepID)
            step_row = None
            if target_step_id is not None:
                cursor.execute(
                    """SELECT step_id, step_details, step_payload
                    FROM agent_process_steps
                    WHERE process_id = %s AND step_id = %s
                    LIMIT 1;""",
                    (process_id, target_step_id),
                )
                step_row = cursor.fetchone()
            elif self._as_int(stepOrder) is not None:
                cursor.execute(
                    """SELECT step_id, step_details, step_payload
                    FROM agent_process_steps
                    WHERE process_id = %s AND step_order = %s
                    LIMIT 1;""",
                    (process_id, self._as_int(stepOrder)),
                )
                step_row = cursor.fetchone()
            elif normalized_label:
                cursor.execute(
                    """SELECT step_id, step_details, step_payload
                    FROM agent_process_steps
                    WHERE process_id = %s
                    AND LOWER(step_label) = LOWER(%s)
                    LIMIT 1;""",
                    (process_id, normalized_label),
                )
                step_row = cursor.fetchone()

            if step_row is None and normalized_label:
                cursor.execute(
                    """SELECT COALESCE(MAX(step_order), 0)::INT AS max_step_order
                    FROM agent_process_steps
                    WHERE process_id = %s;""",
                    (process_id,),
                )
                max_row = cursor.fetchone() or {}
                next_order = (self._as_int(self._coerce_dict(max_row).get("max_step_order"), 0) or 0) + 1
                cursor.execute(
                    """INSERT INTO agent_process_steps (
                        process_id,
                        step_order,
                        step_label,
                        step_details,
                        step_status,
                        is_required,
                        step_payload,
                        created_at,
                        updated_at,
                        completed_at
                    )
                    VALUES (
                        %s, %s, %s, %s, %s, TRUE, %s::jsonb, NOW(), NOW(),
                        CASE WHEN %s = 'completed' THEN NOW() ELSE NULL END
                    )
                    RETURNING step_id;""",
                    (
                        process_id,
                        next_order,
                        normalized_label[:240],
                        self._as_text(stepDetails),
                        normalized_status,
                        json.dumps(payload_map),
                        normalized_status,
                    ),
                )
                inserted = cursor.fetchone() or {}
                target_step_id = self._as_int(self._coerce_dict(inserted).get("step_id"))
            elif step_row is None:
                cursor.close()
                connection.rollback()
                return {"status": "error", "error": "step_not_found"}
            else:
                step_map = self._coerce_dict(step_row)
                target_step_id = self._as_int(step_map.get("step_id"))
                existing_details = self._as_text(step_map.get("step_details"))
                existing_payload = self._coerce_dict(step_map.get("step_payload"))
                merged_payload = dict(existing_payload)
                merged_payload.update(payload_map)
                next_details = existing_details if stepDetails is None else self._as_text(stepDetails)
                cursor.execute(
                    """UPDATE agent_process_steps
                    SET step_status = %s,
                        step_details = %s,
                        step_payload = %s::jsonb,
                        updated_at = NOW(),
                        completed_at = CASE
                            WHEN %s = 'completed' THEN NOW()
                            WHEN %s != 'completed' THEN NULL
                            ELSE completed_at
                        END
                    WHERE process_id = %s AND step_id = %s;""",
                    (
                        normalized_status,
                        next_details,
                        json.dumps(merged_payload),
                        normalized_status,
                        normalized_status,
                        process_id,
                        target_step_id,
                    ),
                )

            self._refresh_process_progress(cursor, process_id)
            connection.commit()
            cursor.close()
            process_data = self.getProcessByID(owner_id, process_id, include_steps=True)
            if process_data is None:
                return {"status": "error", "error": "process_lookup_failed"}
            return {"status": "ok", "process": process_data, "updated_step_id": target_step_id}
        except psycopg.Error as error:
            if connection:
                connection.rollback()
            logger.error(f"Exception while updating process step:\n{error}")
            return {"status": "error", "error": "database_error", "detail": str(error)}
        finally:
            if connection:
                connection.close()

    def queueOrSendUserMessage(
        self,
        *,
        senderMemberID: int | None,
        targetUsername: str | None = None,
        targetMemberID: int | None = None,
        messageText: str,
        deliveryChannel: str = "telegram",
        processID: int | None = None,
        sendNow: bool = True,
    ) -> dict[str, Any]:
        clean_message = self._as_text(messageText)
        if clean_message == "":
            return {"status": "error", "error": "message_text_required"}

        target_member = self.resolveKnownUser(username=targetUsername, memberID=targetMemberID)
        if target_member is None:
            suggestions = self.listKnownUsers(queryString=targetUsername, count=5)
            return {
                "status": "error",
                "error": "target_member_not_found",
                "target_username": self._as_text(targetUsername),
                "target_member_id": self._as_int(targetMemberID),
                "known_user_suggestions": suggestions,
            }

        sender_id = self._as_int(senderMemberID)
        target_member_id = self._as_int(target_member.get("member_id"))
        target_telegram_id = self._as_int(target_member.get("user_id"))
        channel = self._as_text(deliveryChannel, "telegram").lower()
        process_id = self._as_int(processID)

        connection = None
        outbox_row: dict[str, Any] | None = None
        try:
            connection = self._connect()
            cursor = connection.cursor()
            cursor.execute(
                """INSERT INTO member_outbox (
                    sender_member_id,
                    target_member_id,
                    target_username,
                    delivery_channel,
                    message_text,
                    delivery_status,
                    process_id,
                    metadata,
                    created_at,
                    updated_at
                )
                VALUES (%s, %s, %s, %s, %s, 'queued', %s, %s::jsonb, NOW(), NOW())
                RETURNING
                    outbox_id,
                    sender_member_id,
                    target_member_id,
                    target_username,
                    delivery_channel,
                    message_text,
                    delivery_status,
                    process_id,
                    metadata,
                    created_at,
                    updated_at,
                    delivered_at,
                    failure_reason;""",
                (
                    sender_id,
                    target_member_id,
                    self._as_text(target_member.get("username")),
                    channel,
                    clean_message,
                    process_id,
                    json.dumps({"send_now_requested": bool(sendNow)}),
                ),
            )
            inserted = cursor.fetchone()
            outbox_row = self._coerce_dict(inserted)
            outbox_id = self._as_int(outbox_row.get("outbox_id"))

            delivery_status = "queued"
            failure_reason = None
            metadata_update = self._coerce_dict(outbox_row.get("metadata"))
            delivery_result = None

            if sendNow and channel == "telegram":
                bot_token = self._as_text(ConfigManager()._instance.bot_token)
                if bot_token == "":
                    delivery_status = "failed"
                    failure_reason = "bot_token_missing"
                elif target_telegram_id is None:
                    delivery_status = "failed"
                    failure_reason = "target_user_missing_telegram_id"
                else:
                    send_url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
                    try:
                        telegram_response = requests.post(
                            send_url,
                            json={"chat_id": target_telegram_id, "text": clean_message},
                            timeout=12.0,
                        )
                        response_payload = telegram_response.json()
                        delivery_result = response_payload
                        metadata_update["telegram_http_status"] = telegram_response.status_code
                        metadata_update["telegram_response"] = response_payload
                        if telegram_response.ok and isinstance(response_payload, dict) and bool(response_payload.get("ok")):
                            delivery_status = "sent"
                        else:
                            delivery_status = "failed"
                            failure_reason = self._as_text(
                                self._coerce_dict(response_payload).get("description"),
                                f"telegram_send_http_{telegram_response.status_code}",
                            )
                    except Exception as error:  # noqa: BLE001
                        delivery_status = "failed"
                        failure_reason = f"telegram_send_exception:{error}"

            if outbox_id is not None and delivery_status != "queued":
                cursor.execute(
                    """UPDATE member_outbox
                    SET delivery_status = %s,
                        updated_at = NOW(),
                        delivered_at = CASE WHEN %s = 'sent' THEN NOW() ELSE delivered_at END,
                        failure_reason = %s,
                        metadata = %s::jsonb
                    WHERE outbox_id = %s
                    RETURNING
                        outbox_id,
                        sender_member_id,
                        target_member_id,
                        target_username,
                        delivery_channel,
                        message_text,
                        delivery_status,
                        process_id,
                        metadata,
                        created_at,
                        updated_at,
                        delivered_at,
                        failure_reason;""",
                    (
                        delivery_status,
                        delivery_status,
                        failure_reason,
                        json.dumps(metadata_update),
                        outbox_id,
                    ),
                )
                outbox_row = self._coerce_dict(cursor.fetchone())
            connection.commit()
            cursor.close()

            outbox = {
                "outbox_id": outbox_row.get("outbox_id"),
                "sender_member_id": outbox_row.get("sender_member_id"),
                "target_member_id": outbox_row.get("target_member_id"),
                "target_username": outbox_row.get("target_username"),
                "delivery_channel": outbox_row.get("delivery_channel"),
                "message_text": outbox_row.get("message_text"),
                "delivery_status": outbox_row.get("delivery_status"),
                "process_id": outbox_row.get("process_id"),
                "metadata": self._coerce_dict(outbox_row.get("metadata")),
                "created_at": self._timestamp_iso(outbox_row.get("created_at")),
                "updated_at": self._timestamp_iso(outbox_row.get("updated_at")),
                "delivered_at": self._timestamp_iso(outbox_row.get("delivered_at")),
                "failure_reason": outbox_row.get("failure_reason"),
            }
            return {
                "status": "ok",
                "target_member": target_member,
                "outbox": outbox,
                "delivery_result": delivery_result,
            }
        except psycopg.Error as error:
            if connection:
                connection.rollback()
            logger.error(f"Exception while queueing/sending user message:\n{error}")
            return {"status": "error", "error": "database_error", "detail": str(error)}
        finally:
            if connection:
                connection.close()

    def listOutboxForMember(
        self,
        *,
        memberID: int,
        count: int = 20,
        deliveryStatus: str | None = None,
    ) -> list[dict[str, Any]]:
        member_id = self._as_int(memberID)
        if member_id is None or member_id <= 0:
            return []
        limit = min(100, max(1, self._as_int(count, 20) or 20))
        status_filter = self._as_text(deliveryStatus).lower()

        connection = None
        output: list[dict[str, Any]] = []
        try:
            connection = self._connect()
            cursor = connection.cursor()
            if status_filter:
                cursor.execute(
                    """SELECT outbox_id, sender_member_id, target_member_id, target_username, delivery_channel,
                    message_text, delivery_status, process_id, metadata, created_at, updated_at, delivered_at, failure_reason
                    FROM member_outbox
                    WHERE (sender_member_id = %s OR target_member_id = %s)
                    AND delivery_status = %s
                    ORDER BY created_at DESC
                    LIMIT %s;""",
                    (member_id, member_id, status_filter, limit),
                )
            else:
                cursor.execute(
                    """SELECT outbox_id, sender_member_id, target_member_id, target_username, delivery_channel,
                    message_text, delivery_status, process_id, metadata, created_at, updated_at, delivered_at, failure_reason
                    FROM member_outbox
                    WHERE (sender_member_id = %s OR target_member_id = %s)
                    ORDER BY created_at DESC
                    LIMIT %s;""",
                    (member_id, member_id, limit),
                )
            rows = cursor.fetchall() or []
            cursor.close()
            for row in rows:
                row_map = self._coerce_dict(row)
                output.append(
                    {
                        "outbox_id": row_map.get("outbox_id"),
                        "sender_member_id": row_map.get("sender_member_id"),
                        "target_member_id": row_map.get("target_member_id"),
                        "target_username": row_map.get("target_username"),
                        "delivery_channel": row_map.get("delivery_channel"),
                        "message_text": row_map.get("message_text"),
                        "delivery_status": row_map.get("delivery_status"),
                        "process_id": row_map.get("process_id"),
                        "metadata": self._coerce_dict(row_map.get("metadata")),
                        "created_at": self._timestamp_iso(row_map.get("created_at")),
                        "updated_at": self._timestamp_iso(row_map.get("updated_at")),
                        "delivered_at": self._timestamp_iso(row_map.get("delivered_at")),
                        "failure_reason": row_map.get("failure_reason"),
                    }
                )
        except psycopg.Error as error:
            logger.error(f"Exception while listing outbox messages:\n{error}")
        finally:
            if connection:
                connection.close()
        return output


class ChatHistoryManager:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            # Intialize the new singleton
            cls._instance = super(ChatHistoryManager, cls).__new__(cls)

            connection = None
            
            try:
                connection = psycopg.connect(conninfo=ConfigManager()._instance.db_conninfo)
                cursor = connection.cursor()
                logger.debug(f"PostgreSQL connection established.")
                
                # Create the chat history table if it does not exist
                execute_migration(cursor, "010_chat_history.sql")
                connection.commit()

                # pgvector can be unavailable in some local environments. Keep
                # base chat history available even if embeddings setup fails.
                vectorDimensions = max(1, ConfigManager().runtimeInt("vectors.embedding_dimensions", 768))
                try:
                    execute_migration(cursor, "011_create_vector_extension.sql")
                    connection.commit()
                except (Exception, psycopg.DatabaseError) as error:
                    connection.rollback()
                    logger.warning(f"Unable to ensure pgvector extension. Continuing without vector features:\n{error}")

                try:
                    execute_migration(
                        cursor,
                        "012_chat_history_embeddings.sql",
                        context={"vector_dimensions": vectorDimensions},
                    )
                    connection.commit()
                except (Exception, psycopg.DatabaseError) as error:
                    connection.rollback()
                    logger.warning(f"Unable to create chat_history_embeddings table. Continuing without embeddings persistence:\n{error}")

                # close the communication with the PostgreSQL
                cursor.close()
            except (Exception, psycopg.DatabaseError) as error:
                logger.error(f"Exception while working with psycopg and PostgreSQL:\n{error}")
            finally:
                if connection is not None:
                    connection.close()
                    logger.debug(f"PostgreSQL connection is closed.")

        
        return cls._instance
    
    def addChatHistory(
        self,
        messageID: int,
        messageText: str,
        platform: str,
        memberID: int = None,
        communityID: int = None,
        chatHostID: int = None,
        topicID: int = None,
        timestamp: datetime | None = None,
    ) -> int:
        logger.info(f"Adding a new chat history record.")
        chatHostID = chatHostID if chatHostID else communityID if communityID else memberID
        if not chatHostID:
            return
        
        chatType = "community" if communityID is not None else "member"
        deferEmbeddings = _defer_embeddings_on_write_enabled()
        embedding = None if deferEmbeddings else getEmbeddings(messageText)

        if isinstance(timestamp, datetime):
            timestampValue = timestamp
        else:
            timestampValue = datetime.now(timezone.utc)
        if timestampValue.tzinfo is not None:
            # Persist UTC wall-clock without timezone to match existing TIMESTAMP column.
            timestampValue = timestampValue.astimezone(timezone.utc).replace(tzinfo=None)

        connection = None
        response = None
        try:
            connection = psycopg.connect(conninfo=ConfigManager()._instance.db_conninfo, row_factory=dict_row)
            cursor = connection.cursor()
            logger.debug(f"PostgreSQL connection established.")

            insertHistory_sql = """INSERT INTO chat_history (member_id, community_id, chat_host_id, topic_id, chat_type, platform, message_id, message_text, message_timestamp)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING history_id;"""
            cursor.execute(
                insertHistory_sql,
                (memberID, communityID, chatHostID, topicID, chatType, platform, messageID, messageText, timestampValue),
            )
            result = cursor.fetchone()
            historyID = result.get("history_id")
            response = historyID

            connection.commit()

            if deferEmbeddings:
                enqueued = _enqueue_chat_embedding(historyID, messageText)
                if not enqueued:
                    logger.warning(
                        "Deferred embedding enqueue failed for history id %s; falling back to synchronous persistence.",
                        historyID,
                    )
                    _persist_chat_embedding(historyID, messageText, source="sync-fallback")
            elif embedding is not None:
                try:
                    insertEmbeddings_sql = """INSERT INTO chat_history_embeddings (history_id, embeddings)
                    VALUES (%s, %s);"""
                    cursor.execute(insertEmbeddings_sql, (historyID, embedding))
                    connection.commit()
                except (Exception, psycopg.DatabaseError) as error:
                    connection.rollback()
                    logger.warning(f"Unable to persist chat embeddings for history id {historyID}:\n{error}")

            # close the communication with the PostgreSQL
            cursor.close()
        except (Exception, psycopg.DatabaseError) as error:
            logger.error(f"Exception while working with psycopg and PostgreSQL:\n{error}")
        finally:
            if connection is not None:
                connection.close()
                logger.debug(f"PostgreSQL connection is closed.")

            return response

    def getMemberChatHistory(self, memberID: int, platform: str = None, timeInHours: int = 1):
        logger.info(f"Getting chat history records.")
        timePeriod = datetime.now() - timedelta(hours=timeInHours)

        response = None
        connection = None
        try:
            connection = psycopg.connect(conninfo=ConfigManager()._instance.db_conninfo, row_factory=dict_row)
            cursor = connection.cursor()
            logger.debug(f"PostgreSQL connection established.")

            if platform is None:
                querySQL = """SELECT history_id, message_id, message_text, message_timestamp
                FROM chat_history
                WHERE chat_type = 'member'
                AND chat_host_id = %s 
                AND message_timestamp > %s
                ORDER BY message_timestamp;"""
                cursor.execute(querySQL, (memberID, timePeriod))
            else:
                querySQL = """SELECT history_id, message_id, message_text, message_timestamp
                FROM chat_history
                WHERE chat_type = 'member'
                AND chat_host_id = %s 
                AND platform = %s
                AND message_timestamp > %s
                ORDER BY message_timestamp;"""
                cursor.execute(querySQL, (memberID, platform, timePeriod))

            results = cursor.fetchall()
            response = results
            # close the communication with the PostgreSQL
            cursor.close()
        except (Exception, psycopg.DatabaseError) as error:
            logger.error(f"Exception while working with psycopg and PostgreSQL:\n{error}")
        finally:
            if connection is not None:
                connection.close()
                logger.debug(f"PostgreSQL connection is closed.")
            
            return response

    def getCommunityChatHistory(self, communityID: int, topicID: int = None, platform: str = None, timeInHours: int = 1):
        logger.info(f"Getting chat history records.")
        timePeriod = datetime.now() - timedelta(hours=timeInHours)

        response = None
        connection = None
        try:
            connection = psycopg.connect(conninfo=ConfigManager()._instance.db_conninfo, row_factory=dict_row)
            cursor = connection.cursor()
            logger.debug(f"PostgreSQL connection established.")

            beginQuery_sql = """SELECT history_id, message_id, message_text, message_timestamp
                FROM chat_history
                WHERE chat_type = 'community'
                AND chat_host_id = %s
                AND message_timestamp > %s"""
            
            endQuery_sql = " ORDER BY message_timestamp;"

            valueArray = [communityID, timePeriod]

            if topicID:
                beginQuery_sql = beginQuery_sql + " AND topic_id = %s"
                valueArray.append(topicID)
            else:
                beginQuery_sql = beginQuery_sql + " AND topic_id IS NULL"
            
            if platform:
                beginQuery_sql = beginQuery_sql + " AND platform = %s"
                valueArray.append(platform)

            historyQuery_sql = beginQuery_sql + endQuery_sql

            cursor.execute(historyQuery_sql, valueArray)

            results = cursor.fetchall()
            response = results
            # close the communication with the PostgreSQL
            cursor.close()
        except (Exception, psycopg.DatabaseError) as error:
            logger.error(f"Exception while working with psycopg and PostgreSQL:\n{error}")
        finally:
            if connection is not None:
                connection.close()
                logger.debug(f"PostgreSQL connection is closed.")
            
            return response

    def getChatHistory(
        self,
        chatHostID: int,
        chatType: str,
        platform: str,
        topicID: int = None,
        timeInHours: int | None = None,
        limit: int | None = None,
    ) -> list:
        logger.info(f"Getting chat history records.")
        if timeInHours is None:
            timeInHours = ConfigManager().runtimeInt("retrieval.chat_history_window_hours", 12)
        if limit is None:
            limit = ConfigManager().runtimeInt("retrieval.chat_history_default_limit", 1)
        timePeriod = datetime.now() - timedelta(hours=max(0, int(timeInHours)))
        
        connection = None
        results = None
        try:
            connection = psycopg.connect(conninfo=ConfigManager()._instance.db_conninfo, row_factory=dict_row)
            cursor = connection.cursor()
            logger.debug(f"PostgreSQL connection established.")
            
            baseQuery_sql = """SELECT history_id, member_id, message_id, message_text, message_timestamp
                FROM chat_history
                WHERE chat_host_id = %s
                AND chat_type = %s
                AND platform = %s
                AND message_timestamp > %s"""

            valueArray = [chatHostID, chatType, platform, timePeriod]

            if topicID:
                baseQuery_sql = baseQuery_sql + " AND topic_id = %s"
                valueArray.append(topicID)
            else:
                baseQuery_sql = baseQuery_sql + " AND topic_id IS NULL"

            if int(limit) > 0:
                # Fetch the most recent N records, then reorder chronologically for downstream prompt assembly.
                historyQuery_sql = (
                    "SELECT history_id, member_id, message_id, message_text, message_timestamp "
                    "FROM ("
                    + baseQuery_sql
                    + " ORDER BY message_timestamp DESC LIMIT %s"
                    + ") AS recent_history "
                    "ORDER BY message_timestamp;"
                )
                valueArray.append(int(limit))
            else:
                historyQuery_sql = baseQuery_sql + " ORDER BY message_timestamp;"
            cursor.execute(historyQuery_sql, valueArray)

            results = cursor.fetchall()
            # close the communication with the PostgreSQL
            cursor.close()
        except (Exception, psycopg.DatabaseError) as error:
            logger.error(f"Exception while working with psycopg and PostgreSQL:\n{error}")
        finally:
            if connection is not None:
                connection.close()
                logger.debug(f"PostgreSQL connection is closed.")
            
            return results

    def getChatHistoryWithSenderData(
        self,
        chatHostID: int,
        chatType: str,
        platform: str,
        topicID: int = None,
        timeInHours: int | None = None,
        limit: int | None = None,
    ) -> list:
        logger.info(f"Getting chat history records.")
        if timeInHours is None:
            timeInHours = ConfigManager().runtimeInt("retrieval.chat_history_sender_window_hours", 12)
        if limit is None:
            limit = ConfigManager().runtimeInt("retrieval.chat_history_sender_default_limit", 0)
        timePeriod = datetime.now() - timedelta(hours=max(0, int(timeInHours)))
        
        connection = None
        results = None
        try:
            connection = psycopg.connect(conninfo=ConfigManager()._instance.db_conninfo, row_factory=dict_row)
            cursor = connection.cursor()
            logger.debug(f"PostgreSQL connection established.")
            
            baseQuery_sql = """SELECT ch.history_id, ch.message_id, ch.message_text, ch.message_timestamp, mem.member_id, mem.first_name, mem.last_name
                FROM chat_history AS ch
                LEFT JOIN member_data AS mem
                ON ch.member_id = mem.member_id
                WHERE chat_host_id = %s
                AND chat_type = %s
                AND platform = %s
                AND message_timestamp > %s"""

            valueArray = [chatHostID, chatType, platform, timePeriod]

            if topicID:
                baseQuery_sql = baseQuery_sql + " AND topic_id = %s"
                valueArray.append(topicID)
            else:
                baseQuery_sql = baseQuery_sql + " AND topic_id IS NULL"

            if int(limit) > 0:
                historyQuery_sql = (
                    "SELECT history_id, message_id, message_text, message_timestamp, member_id, first_name, last_name "
                    "FROM ("
                    + baseQuery_sql
                    + " ORDER BY ch.message_timestamp DESC LIMIT %s"
                    + ") AS recent_history "
                    "ORDER BY message_timestamp;"
                )
                valueArray.append(int(limit))
            else:
                historyQuery_sql = baseQuery_sql + " ORDER BY ch.message_timestamp;"
            cursor.execute(historyQuery_sql, valueArray)

            results = cursor.fetchall()
            # close the communication with the PostgreSQL
            cursor.close()
        except (Exception, psycopg.DatabaseError) as error:
            logger.error(f"Exception while working with psycopg and PostgreSQL:\n{error}")
        finally:
            if connection is not None:
                connection.close()
                logger.debug(f"PostgreSQL connection is closed.")
            
            return results

    def getMessageByHistoryID(self, historyID: int) -> dict:
        logger.info(f"Get message from chat history using history ID.")

        connection = None
        response = None
        try:
            connection = psycopg.connect(conninfo=ConfigManager()._instance.db_conninfo, row_factory=dict_row)
            cursor = connection.cursor()
            logger.debug(f"PostgreSQL connection established.")

            querySQL = """SELECT *
            FROM chat_history
            WHERE history_id = %s 
            LIMIT 1;"""
            cursor.execute(querySQL, (historyID, ))

            results = cursor.fetchone()
            # Handle results into response
            response = results

            cursor.close()
        except (Exception, psycopg.DatabaseError) as error:
            logger.error(f"Exception while working with psycopg and PostgreSQL:\n{error}")
        finally:
            if connection is not None:
                connection.close()
                logger.debug(f"PostgreSQL connection is closed.")
            
            return response
    
    def getMessageByMessageID(self, chatHostID: int, chatType: str, platform: str, messageID: int) -> dict:
        logger.info(f"Get message from chat history.")

        connection = None
        response = None
        try:
            connection = psycopg.connect(conninfo=ConfigManager()._instance.db_conninfo, row_factory=dict_row)
            cursor = connection.cursor()
            logger.debug(f"PostgreSQL connection established.")

            querySQL = """SELECT *
            FROM chat_history
            WHERE platform = %s
            AND chat_type = %s
            AND chat_host_id = %s 
            AND message_id = %s
            LIMIT 1;"""
            cursor.execute(querySQL, (platform, chatType, chatHostID, messageID))

            results = cursor.fetchone()
            # Handle results into response
            response = results
            
            cursor.close()
        except (Exception, psycopg.DatabaseError) as error:
            logger.error(f"Exception while working with psycopg and PostgreSQL:\n{error}")
        finally:
            if connection is not None:
                connection.close()
                logger.debug(f"PostgreSQL connection is closed.")
            
            return response

    def searchChatHistory(
        self,
        text: str,
        limit: int | None = None,
        chatHostID: int | None = None,
        chatType: str | None = None,
        platform: str | None = None,
        topicID: int | None = None,
        scopeTopic: bool = False,
        timeInHours: int | None = None,
    ) -> list:
        logger.info(f"Searching chat history records.")
        embedding = getEmbeddings(text)
        if embedding is None:
            logger.warning("Skipping chat history search because embeddings are unavailable.")
            return list()
        if limit is None:
            limit = ConfigManager().runtimeInt("retrieval.chat_history_default_limit", 1)
        if timeInHours is None:
            timeInHours = ConfigManager().runtimeInt("retrieval.chat_history_window_hours", 12)
        timePeriod = datetime.now() - timedelta(hours=max(0, int(timeInHours)))
        
        connection = None
        response = None
        try:
            connection = psycopg.connect(conninfo=ConfigManager()._instance.db_conninfo, row_factory=dict_row)
            cursor = connection.cursor()
            logger.debug(f"PostgreSQL connection established.")

            whereClauses = ["ch.message_timestamp > %s"]
            values: list[Any] = [embedding, timePeriod]

            if chatHostID is not None:
                whereClauses.append("ch.chat_host_id = %s")
                values.append(chatHostID)
            if isinstance(chatType, str) and chatType.strip():
                whereClauses.append("ch.chat_type = %s")
                values.append(chatType.strip())
            if isinstance(platform, str) and platform.strip():
                whereClauses.append("ch.platform = %s")
                values.append(platform.strip())

            if scopeTopic:
                if topicID:
                    whereClauses.append("ch.topic_id = %s")
                    values.append(topicID)
                else:
                    whereClauses.append("ch.topic_id IS NULL")
            elif topicID:
                whereClauses.append("ch.topic_id = %s")
                values.append(topicID)

            querySQL = f"""SELECT ch.history_id, ch.message_id, ch.message_text, ch.message_timestamp, che.embeddings <-> %s::vector AS distance
            FROM chat_history AS ch
            JOIN chat_history_embeddings AS che
            ON ch.history_id = che.history_id
            WHERE {" AND ".join(whereClauses)}
            ORDER BY distance
            LIMIT %s"""
            values.append(max(1, int(limit)))
            cursor.execute(querySQL, values)
            results = cursor.fetchall()
            response = results
            # close the communication with the PostgreSQL
            cursor.close()
        except (Exception, psycopg.DatabaseError) as error:
            logger.error(f"Exception while working with psycopg and PostgreSQL:\n{error}")
        finally:
            if connection is not None:
                connection.close()
                logger.debug(f"PostgreSQL connection is closed.")
            
            return response


class CommunityManager:
    _instance = None
    
    # Private attributes
    # TODO place this in config
    #__rolesList = ["user","tester","marketing","admin","owner"]

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(CommunityManager, cls).__new__(cls)
            # Initialize the singleton instance.

            connection = None
            # Create the community database tables if they don't exist yet
            try:
                connection = psycopg.connect(conninfo=ConfigManager()._instance.db_conninfo)
                cursor = connection.cursor()
                logger.debug(f"PostgreSQL connection established.")

                # Create the new community data table
                execute_migration(cursor, "020_community_data.sql")
                
                # Create the community telegram table if it doesn't exist
                execute_migration(cursor, "021_community_telegram.sql")

                # Create per-community isolation metadata table.
                execute_migration(cursor, "089_community_isolation.sql")
                connection.commit()
                # Close the cursor
                cursor.close()
            except psycopg.Error as error:
                logger.error(f"Exception while working with psycopg and PostgreSQL:\n{error}")
            finally:
                if (connection):
                    connection.close()
                    logger.debug(f"PostgreSQL connection is closed.")
        
        return cls._instance

    @staticmethod
    def _default_group_storage_mode() -> str:
        mode = str(
            ConfigManager().runtimeValue("telegram.group_storage_mode", "shared_pg")
            or "shared_pg"
        ).strip().lower()
        if mode not in {"shared_pg", "dedicated_pg"}:
            mode = "shared_pg"
        return mode

    @staticmethod
    def _default_group_storage_prefix() -> str:
        prefix = str(
            ConfigManager().runtimeValue("telegram.group_storage_prefix", "community")
            or "community"
        ).strip().lower()
        if prefix == "":
            prefix = "community"
        return re.sub(r"[^a-z0-9_]+", "_", prefix)

    @staticmethod
    def _build_storage_key(communityID: int, chatID: int) -> str:
        prefix = CommunityManager._default_group_storage_prefix()
        return f"{prefix}:{int(communityID)}:{int(chatID)}"

    def getCommunityByID(self, communityID: int) -> dict:
        logger.info(f"Getting community account data for ID:  {communityID}")

        connection = None
        response = None
        try:
            connection = connection = psycopg.connect(conninfo=ConfigManager()._instance.db_conninfo, row_factory=dict_row)
            cursor = connection.cursor()
            logger.debug(f"PostgreSQL connection established.")

            getCommunityQuery_sql = """SELECT cd.community_id, cd.community_name, cd.community_link, cd.roles, cd.created_by, cd.register_date, tg.chat_id, tg.chat_title, tg.has_topics,
            iso.storage_mode, iso.storage_key, iso.storage_schema, iso.storage_database, iso.context_isolation_enabled
            FROM community_data AS cd
            LEFT JOIN community_telegram AS tg
            ON cd.community_id = tg.community_id
            LEFT JOIN community_isolation AS iso
            ON cd.community_id = iso.community_id
            WHERE cd.community_id = %s
            LIMIT 1;"""
            
            cursor.execute(getCommunityQuery_sql, (communityID, ))
            result = cursor.fetchone()
            # Close the cursor
            cursor.close()
            
            response = result
            
        except psycopg.Error as error:
                logger.error(f"Exception while working with psycopg and PostgreSQL:\n{error}")
        finally:
            if (connection):
                connection.close()
                logger.debug(f"PostgreSQL connection is closed.")
            
            return response

    def getCommunityByTelegramID(self, telegramChatID: int) -> dict:
        logger.info(f"Getting community account data for telegram ID:  {telegramChatID}")

        connection = None
        response = None
        try:
            connection = connection = psycopg.connect(conninfo=ConfigManager()._instance.db_conninfo, row_factory=dict_row)
            cursor = connection.cursor()
            logger.debug(f"PostgreSQL connection established.")

            getCommunityQuery_sql = """SELECT cd.community_id, cd.community_name, cd.community_link, cd.roles, cd.created_by, cd.register_date, tg.chat_id, tg.chat_title, tg.has_topics,
            iso.storage_mode, iso.storage_key, iso.storage_schema, iso.storage_database, iso.context_isolation_enabled
            FROM community_data AS cd
            JOIN community_telegram AS tg
            ON cd.community_id = tg.community_id
            LEFT JOIN community_isolation AS iso
            ON cd.community_id = iso.community_id
            WHERE tg.chat_id = %s
            LIMIT 1;"""
            
            cursor.execute(getCommunityQuery_sql, (telegramChatID, ))
            result = cursor.fetchone()
            # Close the cursor
            cursor.close()
            
            response = result
            
        except psycopg.Error as error:
                logger.error(f"Exception while working with psycopg and PostgreSQL:\n{error}")
        finally:
            if (connection):
                connection.close()
                logger.debug(f"PostgreSQL connection is closed.")
            
            return response

    def addCommunityFromTelegram(self, communityData: dict) -> dict | None:
        logger.info(f"Adding a community from telegram.")
        
        connection = None
        response = None
        try:
            connection = psycopg.connect(conninfo=ConfigManager()._instance.db_conninfo, row_factory=dict_row)
            cursor = connection.cursor()
            logger.debug(f"PostgreSQL connection established.")

            insertMember_sql = """INSERT INTO community_data (community_name, community_link, roles, created_by, register_date) 
            VALUES (%(community_name)s, %(community_link)s, %(roles)s, %(created_by)s, %(register_date)s)
            RETURNING community_id;"""
            
            cursor.execute(insertMember_sql, communityData)
            result = cursor.fetchone()
            communityID = result.get("community_id")

            communityTelegramData = {
                "community_id": communityID,
                "chat_id": communityData.get("chat_id"),
                "chat_title": communityData.get("chat_title"),
                "has_topics": communityData.get("has_topics")
            }
            insertMemberTelegram_sql = """INSERT INTO community_telegram (community_id, chat_id, chat_title, has_topics) 
            VALUES (%(community_id)s, %(chat_id)s, %(chat_title)s, %(has_topics)s);"""

            cursor.execute(insertMemberTelegram_sql, communityTelegramData)
            storageMode = self._default_group_storage_mode()
            isolationEnabled = bool(
                ConfigManager().runtimeBool("telegram.group_context_isolation_enabled", True)
            )
            storageKey = self._build_storage_key(communityID, communityData.get("chat_id"))
            storageSchema = f"community_{communityID}" if storageMode == "dedicated_pg" else "public"
            isolationUpsert_sql = """INSERT INTO community_isolation
            (community_id, platform, chat_id, storage_mode, storage_key, storage_schema, storage_database, context_isolation_enabled, created_at, updated_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, NOW(), NOW())
            ON CONFLICT (community_id)
            DO UPDATE SET
                chat_id = EXCLUDED.chat_id,
                storage_mode = EXCLUDED.storage_mode,
                storage_key = EXCLUDED.storage_key,
                storage_schema = EXCLUDED.storage_schema,
                storage_database = EXCLUDED.storage_database,
                context_isolation_enabled = EXCLUDED.context_isolation_enabled,
                updated_at = NOW();"""
            cursor.execute(
                isolationUpsert_sql,
                (
                    communityID,
                    "telegram",
                    communityData.get("chat_id"),
                    storageMode,
                    storageKey,
                    storageSchema,
                    None,
                    isolationEnabled,
                ),
            )
            connection.commit()
            response = self.getCommunityByID(communityID)
            
            cursor.close()
        except psycopg.Error as error:
                logger.error(f"Exception while working with psycopg and PostgreSQL:\n{error}")
        finally:
            if (connection):
                connection.close()
                logger.debug(f"PostgreSQL connection is closed.")
        return response

    def ensureCommunityIsolation(
        self,
        communityID: int,
        chatID: int,
        *,
        platform: str = "telegram",
    ) -> dict | None:
        logger.info(f"Ensuring isolation metadata for community ID: {communityID}")
        if int(communityID) <= 0:
            return None

        connection = None
        response = None
        try:
            connection = psycopg.connect(conninfo=ConfigManager()._instance.db_conninfo, row_factory=dict_row)
            cursor = connection.cursor()
            logger.debug("PostgreSQL connection established.")

            storageMode = self._default_group_storage_mode()
            isolationEnabled = bool(
                ConfigManager().runtimeBool("telegram.group_context_isolation_enabled", True)
            )
            storageKey = self._build_storage_key(int(communityID), int(chatID))
            storageSchema = f"community_{int(communityID)}" if storageMode == "dedicated_pg" else "public"

            upsert_sql = """INSERT INTO community_isolation
            (community_id, platform, chat_id, storage_mode, storage_key, storage_schema, storage_database, context_isolation_enabled, created_at, updated_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, NOW(), NOW())
            ON CONFLICT (community_id)
            DO UPDATE SET
                platform = EXCLUDED.platform,
                chat_id = EXCLUDED.chat_id,
                storage_mode = EXCLUDED.storage_mode,
                storage_key = EXCLUDED.storage_key,
                storage_schema = EXCLUDED.storage_schema,
                storage_database = EXCLUDED.storage_database,
                context_isolation_enabled = EXCLUDED.context_isolation_enabled,
                updated_at = NOW()
            RETURNING *;"""
            cursor.execute(
                upsert_sql,
                (
                    int(communityID),
                    str(platform or "telegram"),
                    int(chatID),
                    storageMode,
                    storageKey,
                    storageSchema,
                    None,
                    isolationEnabled,
                ),
            )
            response = cursor.fetchone()
            connection.commit()
            cursor.close()
        except psycopg.Error as error:
            logger.error(f"Exception while ensuring community isolation metadata:\n{error}")
        finally:
            if connection is not None:
                connection.close()
                logger.debug("PostgreSQL connection is closed.")

        return response

    def updateCommunityTelegramMetadata(
        self,
        communityID: int,
        chatID: int,
        chatTitle: str | None,
        hasTopics: bool,
    ) -> bool:
        logger.info(f"Updating telegram metadata for community ID: {communityID}")
        if int(communityID) <= 0:
            return False

        connection = None
        success = False
        try:
            connection = psycopg.connect(conninfo=ConfigManager()._instance.db_conninfo)
            cursor = connection.cursor()
            logger.debug("PostgreSQL connection established.")
            upsert_sql = """INSERT INTO community_telegram (community_id, chat_id, chat_title, has_topics)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (community_id)
            DO UPDATE SET
                chat_id = EXCLUDED.chat_id,
                chat_title = EXCLUDED.chat_title,
                has_topics = EXCLUDED.has_topics;"""
            cursor.execute(
                upsert_sql,
                (
                    int(communityID),
                    int(chatID),
                    str(chatTitle or "").strip()[:96] or None,
                    bool(hasTopics),
                ),
            )
            connection.commit()
            cursor.close()
            success = True
        except psycopg.Error as error:
            logger.error(f"Exception while updating community telegram metadata:\n{error}")
        finally:
            if connection is not None:
                connection.close()
                logger.debug("PostgreSQL connection is closed.")
        return success

    def updateCommunityRoles(self, communityID: int, roles: list):
            logger.info("Update community roles")
            
            community = self.getCommunityByID(communityID)
            if community is not None:
                # TODO Validate the roles
                    
                connection = None
                try:
                    connection = psycopg.connect(conninfo=ConfigManager()._instance.db_conninfo)
                    cursor = connection.cursor()
                    logger.debug(f"PostgreSQL connection established.")

                    updateRoles_sql = "UPDATE community_data SET roles = %s WHERE community_id = %s;"
                    valueTuple = (roles, communityID)

                    cursor.execute(updateRoles_sql, valueTuple)
                    connection.commit()
                    cursor.close()
                    
                except psycopg.Error as error:
                        logger.error(f"Exception while working with psycopg and PostgreSQL:\n{error}")
                finally:
                    if (connection):
                        connection.close()
                        logger.debug(f"PostgreSQL connection is closed.")
                        return True


class CommunityScoreManager:
    _instance = None

    # Set community score rules
    defaultCommunityScoreRules = [
        {
            "private" : {
                "min" : 50
            },
            "community" : {
                "min" : 0
            },
            "message_per_hour" : 2,
            "image_per_hour" : 0
        },
        {
            "private" : {
                "min" : 55
            },
            "community" : {
                "min" : 5
            },
            "message_per_hour" : 4,
            "image_per_hour" : 0
        },
        {
            "private" : {
                "min" : 60
            },
            "community" : {
                "min" : 10
            },
            "message_per_hour" : 6,
            "image_per_hour" : 0
        },
        {
            "private" : {
                "min" : 65
            },
            "community" : {
                "min" : 15
            },
            "message_per_hour" : 8,
            "image_per_hour" : 0
        },
        {
            "private" : {
                "min" : 70
            },
            "community" : {
                "min" : 20
            },
            "message_per_hour" : 10,
            "image_per_hour" : 1
        },
        {
            "private" : {
                "min" : 75
            },
            "community" : {
                "min" : 25
            },
            "message_per_hour" : 12,
            "image_per_hour" : 1
        }
    ]

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(CommunityScoreManager, cls).__new__(cls)
            configuredRules = ConfigManager().runtimeValue("community.score_rules", cls.defaultCommunityScoreRules)
            if isinstance(configuredRules, list) and len(configuredRules) > 0:
                cls._instance.communityScoreRules = configuredRules
            else:
                cls._instance.communityScoreRules = cls.defaultCommunityScoreRules

            connection = None
            try:
                connection = psycopg.connect(conninfo=ConfigManager()._instance.db_conninfo)
                cursor = connection.cursor()
                logger.debug(f"PostgreSQL connection established.")
                
                # Create the individual accounts table if it doesn't exist
                execute_migration(cursor, "030_community_score.sql")

                connection.commit()
                # Close the cursor
                cursor.close()
            except psycopg.Error as error:
                logger.error(f"Exception while working with psycopg and PostgreSQL:\n{error}")
            finally:
                if (connection):
                    connection.close()
                    logger.debug(f"PostgreSQL connection is closed.")

        return cls._instance
    
    def scoreMessage(self, historyID: int):
        logger.info("Score message.")

        # Get all the pertinent data from chat history
        historyMessage = ChatHistoryManager().getMessageByHistoryID(historyID)
        # Get member data from the history message member ID
        memberData = MemberManager().getMemberByID(historyMessage.get("member_id"))
        if memberData is None:
            return
        
        messageText = historyMessage.get("message_text")

        # Get current community score and account length
        currentScoreRatio = 2 if memberData["community_score"] / 50 > 2 else memberData["community_score"] / 50
        accountLength = datetime.now() - memberData["register_date"]
        accountLengthRatio = 1 if (accountLength.days / 365) > 1 else accountLength.days / 365
        finalRatio = ceil(((currentScoreRatio + accountLengthRatio) / 2) * 100) / 100

        readabilityScore = textstat.textstat.flesch_kincaid_grade(messageText)
        baselineScore = readabilityScore - 7
        if baselineScore > 0:
            # Positive score, cap at 5
            points = 5 if baselineScore > 5 else baselineScore
        else:
            # Negative score cap at -5
            points = -5 if baselineScore < -5 else baselineScore

        finalPoints = ceil(points * finalRatio * 10) / 10

        # Add the message score record
        connection = None
        try:
            connection = psycopg.connect(conninfo=ConfigManager()._instance.db_conninfo)
            cursor = connection.cursor()
            logger.debug(f"PostgreSQL connection established.")
            
            insert_sql = "INSERT INTO community_score (history_id, event, read_score, points_awarded, awarded_from_id, multiplier) VALUES (%s, 'message', %s, %s, %s, %s);"
            cursor.execute(insert_sql, (historyID, baselineScore, finalPoints, memberData["member_id"], finalRatio))
            connection.commit()
            cursor.close()
            
        except psycopg.Error as error:
            logger.error(f"Exception while working with psycopg and PostgreSQL:\n{error}")
        finally:
            if (connection):
                connection.close()
                logger.debug(f"PostgreSQL connection is closed.")

        # Call accounts manager method to calculate user's new community score
        # TODO add up each record vs just the new score and old score

        MemberManager().updateCommunityScore(memberData["member_id"], memberData["community_score"] + finalPoints)

    def scoreMessageFromReaction(self, memberID: int, historyID: int):
        logger.info(f"Adjust community score based on a reaction to a message")
        # Get the user_id of the original message sender
        historyMessage = ChatHistoryManager().getMessageByHistoryID(historyID)
        originalSenderID = historyMessage.get("member_id")
        originalSender = MemberManager().getMemberByID(originalSenderID)
        reactingMember = MemberManager().getMemberByID(memberID)
 
        if originalSender is None:
            return

        # Get factoring values from the member who gave the reaction
        # Get current community score and account length
        currentScoreRatio = 2 if reactingMember["community_score"] / 50 > 2 else reactingMember["community_score"] / 50
        accountLength = datetime.now() - reactingMember["register_date"]
        accountLengthRatio = 1 if (accountLength.days / 365) > 1 else accountLength.days / 365
        finalRatio = ceil(((currentScoreRatio + accountLengthRatio) / 2) * 100) / 100
        finalPoints = ceil(finalRatio * 10) / 10

        # Create a new record in community score with the ratio value as the score.
        # Add the message score record
        try:
            connection = psycopg.connect(conninfo=ConfigManager()._instance.db_conninfo)
            cursor = connection.cursor()
            logger.debug(f"PostgreSQL connection established.")

            insert_sql = "INSERT INTO community_score (history_id, event, points_awarded, awarded_from_id, multiplier) VALUES (%s, 'reaction', %s, %s, %s);"
            cursor.execute(insert_sql, (historyID, finalPoints, memberID, finalRatio))
            connection.commit()
            cursor.close()
            
        except psycopg.Error as error:
            logger.error(f"Exception while working with psycopg and PostgreSQL:\n{error}")
        finally:
            if (connection):
                connection.close()
                logger.debug(f"PostgreSQL connection is closed.")

        # Call accounts manager method to calculate user's new community score
        # TODO add up each record vs just the new score and old score

        MemberManager().updateCommunityScore(originalSender["member_id"], originalSender["community_score"] + finalPoints)    

    def getRateLimits(self, memberID: int, chatType: str) -> int:
        logger.info(f"Get message rate for member id:  {memberID}.")
        if isinstance(memberID, dict):
            member = memberID
        else:
            member = MemberManager().getMemberByID(memberID)

        if member is None:
            logger.warning(f"Unable to resolve member for rate-limit lookup: {memberID}")
            return {
                "message": 0,
                "image": 0
            }

        memberScore = member.get("community_score", 0) or 0
        rateLimits = {
            "message": 0,
            "image": 0
        }
        for rule in self.communityScoreRules:
            scoreRule = rule.get(chatType) if isinstance(rule.get(chatType), dict) else rule.get("community", {})
            minScore = scoreRule.get("min", 0)
            if memberScore >= minScore:
                rateLimits["message"] = rule["message_per_hour"]
                rateLimits["image"] = rule["image_per_hour"]
            else:
                break
        
        return rateLimits


class ConfigManager:
    _instance = None

    @staticmethod
    def _env_db(prefix: str) -> dict | None:
        db_name = os.getenv(f"{prefix}DB")
        user = os.getenv(f"{prefix}USER")
        password = os.getenv(f"{prefix}PASSWORD")
        host = os.getenv(f"{prefix}HOST")
        port = os.getenv(f"{prefix}PORT")

        required = (db_name, user, password, host)
        if any(value is None or value == "" for value in required):
            return None

        output = {
            "db_name": db_name,
            "user": user,
            "password": password,
            "host": host,
        }
        if port:
            output["port"] = port
        return output

    @staticmethod
    def _env_bool(name: str, default: bool = False) -> bool:
        value = os.getenv(name)
        if value is None:
            return default
        return value.strip().lower() in {"1", "true", "yes", "on"}

    @staticmethod
    def _as_non_empty_string(value: Any) -> str | None:
        if value is None:
            return None
        text = str(value).strip()
        return text if text else None

    @staticmethod
    def _hydrate_inference_config(
        inferenceConfig: dict | None,
        runtimeSettings: dict[str, Any],
    ) -> dict[str, dict[str, str]]:
        inferenceConfig = inferenceConfig if isinstance(inferenceConfig, dict) else {}
        defaultHost = ConfigManager._as_non_empty_string(
            get_runtime_setting(runtimeSettings, "inference.default_ollama_host", "http://127.0.0.1:11434")
        ) or "http://127.0.0.1:11434"
        def _section_model(section_name: str) -> str:
            section = inferenceConfig.get(section_name)
            if isinstance(section, dict):
                return ConfigManager._as_non_empty_string(section.get("model")) or ""
            return ""
        fallbackModelFromConfig = (
            _section_model("chat")
            or _section_model("tool")
            or _section_model("generate")
            or _section_model("multimodal")
            or _section_model("embedding")
            or ""
        )
        modelDefaults = {
            "embedding": ConfigManager._as_non_empty_string(
                get_runtime_setting(runtimeSettings, "inference.default_embedding_model", "")
            ) or fallbackModelFromConfig,
            "generate": ConfigManager._as_non_empty_string(
                get_runtime_setting(runtimeSettings, "inference.default_generate_model", "")
            ) or fallbackModelFromConfig,
            "chat": ConfigManager._as_non_empty_string(
                get_runtime_setting(runtimeSettings, "inference.default_chat_model", "")
            ) or fallbackModelFromConfig,
            "tool": ConfigManager._as_non_empty_string(
                get_runtime_setting(runtimeSettings, "inference.default_tool_model", "")
            ) or fallbackModelFromConfig,
            "multimodal": ConfigManager._as_non_empty_string(
                get_runtime_setting(runtimeSettings, "inference.default_multimodal_model", "")
            ) or fallbackModelFromConfig,
        }

        hydrated: dict[str, dict[str, str]] = {}
        for key, modelDefault in modelDefaults.items():
            section = inferenceConfig.get(key)
            if not isinstance(section, dict):
                section = {}
            hostValue = ConfigManager._as_non_empty_string(section.get("url")) or defaultHost
            modelValue = ConfigManager._as_non_empty_string(section.get("model")) or modelDefault
            hydrated[key] = {
                "url": hostValue.rstrip("/"),
                "model": modelValue,
            }
        return hydrated

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ConfigManager, cls).__new__(cls)
            load_dotenv_file(".env", override=False)
            configPath = os.getenv("RYO_CONFIG_PATH", "config.json")
            # Open the config file
            with open(configPath, "r", encoding="utf-8") as f:
                config_json = json.load(f)
            runtimeSettings = build_runtime_settings(config_json)

            database = config_json.get("database")
            if database is None:
                database = cls._env_db("POSTGRES_")
            if not isinstance(database, dict):
                database = {}
            if not database.get("host"):
                database["host"] = cls._as_non_empty_string(
                    get_runtime_setting(runtimeSettings, "database.default_primary_host", "127.0.0.1")
                ) or "127.0.0.1"
            if not database.get("port"):
                database["port"] = cls._as_non_empty_string(
                    get_runtime_setting(runtimeSettings, "database.default_primary_port", "5432")
                ) or "5432"

            databaseFallback = config_json.get("database_fallback")
            if databaseFallback is None:
                envFallback = cls._env_db("POSTGRES_FALLBACK_")
                if envFallback is not None:
                    envFallback["enabled"] = cls._env_bool("POSTGRES_FALLBACK_ENABLED", default=False)
                    envFallback["mode"] = os.getenv("POSTGRES_FALLBACK_MODE", "local")
                    databaseFallback = envFallback
            if not isinstance(databaseFallback, dict):
                databaseFallback = {}
            if not databaseFallback.get("host"):
                databaseFallback["host"] = cls._as_non_empty_string(
                    get_runtime_setting(runtimeSettings, "database.default_fallback_host", "127.0.0.1")
                ) or "127.0.0.1"
            if not databaseFallback.get("port"):
                databaseFallback["port"] = cls._as_non_empty_string(
                    get_runtime_setting(runtimeSettings, "database.default_fallback_port", "5433")
                ) or "5433"

            fallbackEnabled = None
            if isinstance(databaseFallback, dict) and "enabled" in databaseFallback:
                fallbackEnabled = bool(databaseFallback.get("enabled"))
            connectTimeout = get_runtime_setting(runtimeSettings, "database.connect_timeout_seconds", 2)
            try:
                connectTimeoutInt = int(connectTimeout)
            except (TypeError, ValueError):
                connectTimeoutInt = 2

            dbRouter = DatabaseRouter(
                primary_database=database,
                fallback_database=databaseFallback,
                fallback_enabled=fallbackEnabled,
                connect_timeout=connectTimeoutInt,
            )
            dbRoute = dbRouter.resolve()
            connectionString = (
                dbRoute.active_conninfo
                or dbRoute.primary_conninfo
                or dbRoute.fallback_conninfo
                or ""
            )
            defaults = config_json.get("defaults")
            if not isinstance(defaults, dict):
                defaults = {}

            inference = cls._hydrate_inference_config(config_json.get("inference"), runtimeSettings)
            apiKeys = config_json.get("api_keys")
            if not isinstance(apiKeys, dict):
                apiKeys = {}
            twitterKeys = config_json.get("twitter_keys")
            if not isinstance(twitterKeys, dict):
                twitterKeys = {}
            rolesList = config_json.get("roles_list")
            if not isinstance(rolesList, list):
                rolesList = ["user", "tester", "marketing", "admin", "owner"]

            cls._instance.bot_name = config_json.get("bot_name")
            cls._instance.bot_id = config_json.get("bot_id")
            cls._instance.bot_token = config_json.get("bot_token")
            cls._instance.web_ui_url = config_json.get("web_ui_url")
            cls._instance.owner_info = config_json.get("owner_info")
            cls._instance.config_path = configPath
            cls._instance.config_json = config_json
            cls._instance.config = config_json
            cls._instance.runtime_settings = runtimeSettings
            cls._instance.database = database
            cls._instance.database_fallback = databaseFallback
            cls._instance.knowledge_domains = None if config_json.get("knowledge") is None else config_json.get("knowledge").get("domains")
            cls._instance.roles_list = rolesList
            cls._instance.db_conninfo = connectionString
            cls._instance.database_route = dbRoute.to_dict()
            cls._instance.defaults = defaults
            cls._instance.inference = inference
            cls._instance.twitter_keys = twitterKeys
            cls._instance.brave_keys = apiKeys.get("brave_search", os.getenv("BRAVE_SEARCH_API_KEY"))

            if cls._instance.database_route["status"] == "fallback":
                logger.warning(
                    "Database router selected fallback target. "
                    f"Status={cls._instance.database_route['status']}, "
                    f"Errors={cls._instance.database_route['errors']}"
                )
            elif cls._instance.database_route["status"] == "failed_all":
                logger.error(
                    "Database router failed to validate primary/fallback connections. "
                    f"Errors={cls._instance.database_route['errors']}"
                )
        
        return cls._instance

    @staticmethod
    def _is_valid_http_url(value: str | None) -> bool:
        if value is None:
            return False
        parsed = urlparse(str(value).strip())
        return parsed.scheme in {"http", "https"} and bool(parsed.netloc)

    @staticmethod
    def _is_numeric_id(value) -> bool:
        if isinstance(value, int):
            return value > 0
        if value is None:
            return False
        text = str(value).strip()
        return text.isdigit() and int(text) > 0

    def getTelegramConfigIssues(self, require_owner: bool = True, require_web_ui_url: bool = True) -> list[str]:
        issues: list[str] = []
        if not str(self._instance.bot_name or "").strip():
            issues.append("bot_name")
        if not self._is_numeric_id(self._instance.bot_id):
            issues.append("bot_id")
        if not str(self._instance.bot_token or "").strip():
            issues.append("bot_token")
        if require_web_ui_url and not self._is_valid_http_url(self._instance.web_ui_url):
            issues.append("web_ui_url")

        if require_owner:
            owner = self._instance.owner_info if isinstance(self._instance.owner_info, dict) else {}
            if not str(owner.get("first_name", "")).strip():
                issues.append("owner_info.first_name")
            if not str(owner.get("last_name", "")).strip():
                issues.append("owner_info.last_name")
            if not self._is_numeric_id(owner.get("user_id")):
                issues.append("owner_info.user_id")
            if not str(owner.get("username", "")).strip():
                issues.append("owner_info.username")

        return issues

    def isTelegramConfigValid(self, require_owner: bool = True, require_web_ui_url: bool = True) -> bool:
        return len(self.getTelegramConfigIssues(require_owner=require_owner, require_web_ui_url=require_web_ui_url)) == 0

    def runtimeValue(self, path: str, default: Any = None) -> Any:
        return get_runtime_setting(self._instance.runtime_settings, path, default)

    def runtimeInt(self, path: str, default: int) -> int:
        value = self.runtimeValue(path, default)
        try:
            return int(value)
        except (TypeError, ValueError):
            return int(default)

    def runtimeFloat(self, path: str, default: float) -> float:
        value = self.runtimeValue(path, default)
        try:
            return float(value)
        except (TypeError, ValueError):
            return float(default)

    def runtimeBool(self, path: str, default: bool) -> bool:
        value = self.runtimeValue(path, default)
        if isinstance(value, bool):
            return value
        if value is None:
            return default
        return str(value).strip().lower() in {"1", "true", "yes", "on"}

    def updateConfig(self, key, value):
        self.config[key] = value
        # Save new config changes to JSON file

    def reloadFromDisk(self) -> bool:
        """Reload config/runtime data while preserving existing object references."""
        cls = self.__class__
        current = self
        previous_instance = cls._instance
        try:
            cls._instance = None
            refreshed = cls()
        except Exception as error:  # noqa: BLE001
            logger.error(f"Failed to reload config from disk:\n{error}")
            cls._instance = previous_instance if previous_instance is not None else current
            return False

        try:
            current.__dict__.clear()
            current.__dict__.update(refreshed.__dict__)
            cls._instance = current
            return True
        except Exception as error:  # noqa: BLE001
            logger.error(f"Failed to hydrate active config instance after reload:\n{error}")
            cls._instance = refreshed
            return False
    
    # Define getters
    @property
    def rolesList(self) -> list:
        return self._instance.roles_list
    
    @property
    def botName(self) -> str:
        return self._instance.bot_name
    
    @property
    def knowledgeDomains(self) -> list:
        return self._instance.knowledge_domains
    
    @property
    def webUIUrl(self) -> str:
        return self._instance.web_ui_url

    @property
    def databaseRoute(self) -> dict:
        return self._instance.database_route

    @property
    def runtimeSettings(self) -> dict:
        return self._instance.runtime_settings


class KnowledgeManager:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(KnowledgeManager, cls).__new__(cls)
            # Intialize the new singleton

            connection = None
            try:
                connection = psycopg.connect(conninfo=ConfigManager()._instance.db_conninfo)
                cursor = connection.cursor()
                logger.debug(f"PostgreSQL connection established.")
                vectorDimensions = max(1, ConfigManager().runtimeInt("vectors.embedding_dimensions", 768))
                
                # Create the knowledge table if it does not exist
                execute_migration(
                    cursor,
                    "040_knowledge.sql",
                    context={"vector_dimensions": vectorDimensions},
                )

                # Create the knowledge retrieval table if it does not exist
                execute_migration(cursor, "041_knowledge_retrievals.sql")

                connection.commit()

                # close the communication with the PostgreSQL
                cursor.close()
            except (Exception, psycopg.DatabaseError) as error:
                logger.error(f"Exception while working with psycopg and PostgreSQL:\n{error}")
            finally:
                if connection is not None:
                    connection.close()
                    logger.debug(f"PostgreSQL connection is closed.")

        
        return cls._instance

    def addDocument(self, document: str, domains: list = [], roles: list = [], categories: list = [], documentMetadata: dict = {}, addedBy: dict = {}):
        logger.info(f"Adding a new knowledge document.")
        embedding = getEmbeddings(document)
        connection = None
        recordID = None
        try:
            connection = psycopg.connect(conninfo=ConfigManager()._instance.db_conninfo)
            cursor = connection.cursor()
            logger.debug(f"PostgreSQL connection established.")

            insertSQL = """INSERT INTO knowledge (domains, roles, categories, knowledge_document, document_metadata, embeddings, record_timestamp, record_metadata)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING knowledge_id;"""
            cursor.execute(insertSQL, (domains, roles, categories, document, json.dumps(documentMetadata), embedding, datetime.now(), json.dumps(addedBy)))
            recordID = cursor.fetchone()[0]
            connection.commit()
            # close the communication with the PostgreSQL
            cursor.close()
        except (Exception, psycopg.DatabaseError) as error:
            logger.error(f"Exception while working with psycopg and PostgreSQL:\n{error}")
        finally:
            if connection is not None:
                connection.close()
                logger.debug(f"PostgreSQL connection is closed.")

            return recordID

    def addRetrieval(self, promptID: int, responseID: int, knowledge_id: int, distance: float, timestamp = datetime.now()):
        logger.info(f"Add a knowledge document retrieval record.")
        
        connection = None
        recordID = None
        try:
            connection = psycopg.connect(conninfo=ConfigManager()._instance.db_conninfo)
            cursor = connection.cursor()
            logger.debug(f"PostgreSQL connection established.")

            insertSQL = """INSERT INTO knowledge_retrievals (prompt_id, response_id, knowledge_id, distance, retrieval_timestamp)
            VALUES (%s, %s, %s, %s, %s)
            RETURNING retrieval_id;"""
            cursor.execute(insertSQL, (promptID, responseID, knowledge_id, distance, timestamp))
            recordID = cursor.fetchone()[0]
            connection.commit()
            # close the communication with the PostgreSQL
            cursor.close()
        except (Exception, psycopg.DatabaseError) as error:
            logger.error(f"Exception while working with psycopg and PostgreSQL:\n{error}")
        finally:
            if connection is not None:
                connection.close()
                logger.debug(f"PostgreSQL connection is closed.")

            return recordID

    def getKnowledge(self) -> list:
        logger.info(f"Get knowledge documents.")

        response = None
        connection = None
        try:
            connection = psycopg.connect(conninfo=ConfigManager()._instance.db_conninfo, row_factory=dict_row)
            cursor = connection.cursor()
            logger.debug(f"PostgreSQL connection established.")

            queryLimit = max(1, ConfigManager().runtimeInt("retrieval.knowledge_list_limit", 10))
            querySQL = """SELECT knowledge_id, domains, roles, categories, knowledge_document, document_metadata, record_timestamp, record_metadata
            FROM knowledge
            LIMIT %s"""
            cursor.execute(querySQL, (queryLimit,))
            results = cursor.fetchall()

            response = results
            # close the communication with the PostgreSQL
            cursor.close()
        except (Exception, psycopg.DatabaseError) as error:
            logger.error(f"Exception while working with psycopg and PostgreSQL:\n{error}")
        finally:
            if connection is not None:
                connection.close()
                logger.debug(f"PostgreSQL connection is closed.")
            
            return response

    def searchKnowledge(self, text: str, limit: int | None = None) -> list:
        logger.info(f"Searching knowledge documents.")
        embedding = getEmbeddings(text)
        if embedding is None:
            logger.warning("Skipping knowledge search because embeddings are unavailable.")
            return list()
        if limit is None:
            limit = ConfigManager().runtimeInt("retrieval.knowledge_search_default_limit", 1)

        connection = None
        response = None
        try:
            connection = psycopg.connect(conninfo=ConfigManager()._instance.db_conninfo, row_factory=dict_row)
            cursor = connection.cursor()
            logger.debug(f"PostgreSQL connection established.")

            querySQL = """SELECT knowledge_id, domains, roles, categories, knowledge_document, document_metadata, embeddings <-> %s::vector AS distance, record_timestamp, record_metadata
            FROM knowledge
            ORDER BY distance
            LIMIT %s"""
            cursor.execute(querySQL, (embedding, max(1, int(limit))))
            results = cursor.fetchall()

            response = results
            # close the communication with the PostgreSQL
            cursor.close()
        except (Exception, psycopg.DatabaseError) as error:
            logger.error(f"Exception while working with psycopg and PostgreSQL:\n{error}")
        finally:
            if connection is not None:
                connection.close()
                logger.debug(f"PostgreSQL connection is closed.")
            
            return response


class DocumentManager:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(DocumentManager, cls).__new__(cls)
            cls._instance._accessPolicy = DocumentAccessPolicy()

            connection = None
            try:
                connection = psycopg.connect(conninfo=ConfigManager()._instance.db_conninfo)
                cursor = connection.cursor()
                logger.debug(f"PostgreSQL connection established.")

                execute_migration(cursor, "090_document_sources.sql")
                execute_migration(cursor, "091_document_versions.sql")
                execute_migration(cursor, "092_document_nodes.sql")
                execute_migration(cursor, "093_document_chunks.sql")
                execute_migration(cursor, "094_document_retrieval_events.sql")
                execute_migration(cursor, "095_document_rls_policies.sql")
                execute_migration(cursor, "096_document_storage_objects.sql")
                execute_migration(cursor, "097_document_ingestion_jobs.sql")
                execute_migration(cursor, "098_document_ingestion_attempts.sql")
                execute_migration(cursor, "099_document_node_edges.sql")

                connection.commit()
                cursor.close()
            except (Exception, psycopg.DatabaseError) as error:
                logger.error(f"Exception while working with psycopg and PostgreSQL:\n{error}")
            finally:
                if connection is not None:
                    connection.close()
                    logger.debug(f"PostgreSQL connection is closed.")

        return cls._instance

    def _safe_limit(self, value: int | None, default: int = 25, max_limit: int = 200) -> int:
        if value is None:
            return max(1, int(default))
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            parsed = int(default)
        if parsed <= 0:
            parsed = int(default)
        return max(1, min(parsed, max_limit))

    def _resolved_scope(
        self,
        payload: dict[str, Any],
        *,
        actor_member_id: int | None = None,
        actor_roles: list[str] | tuple[str, ...] | None = None,
    ):
        scope = resolve_document_scope(payload)
        try:
            actor_id = int(actor_member_id) if actor_member_id is not None else int(scope.owner_member_id)
        except (TypeError, ValueError):
            actor_id = int(scope.owner_member_id)
        self._accessPolicy.assert_scope_access(
            actor_member_id=actor_id,
            scope=scope,
            actor_roles=actor_roles,
        )
        return scope

    def _assert_scoped_source_exists(self, cursor: Any, document_source_id: int, scope: Any) -> None:
        where_sql, where_params = build_scope_where_clause(scope)
        query_sql = (
            "SELECT document_source_id "
            "FROM document_sources "
            "WHERE document_source_id = %s AND "
            + where_sql
            + " LIMIT 1"
        )
        cursor.execute(query_sql, (int(document_source_id), *where_params))
        if cursor.fetchone() is None:
            raise DocumentScopeAccessError("Document source is out of scope or does not exist.")

    def _assert_scoped_version_exists(self, cursor: Any, document_version_id: int, scope: Any) -> None:
        where_sql, where_params = build_scope_where_clause(scope)
        query_sql = (
            "SELECT document_version_id "
            "FROM document_versions "
            "WHERE document_version_id = %s AND "
            + where_sql
            + " LIMIT 1"
        )
        cursor.execute(query_sql, (int(document_version_id), *where_params))
        if cursor.fetchone() is None:
            raise DocumentScopeAccessError("Document version is out of scope or does not exist.")

    def _assert_scoped_storage_object_exists(self, cursor: Any, storage_object_id: int, scope: Any) -> None:
        where_sql, where_params = build_scope_where_clause(scope)
        query_sql = (
            "SELECT storage_object_id "
            "FROM document_storage_objects "
            "WHERE storage_object_id = %s AND "
            + where_sql
            + " LIMIT 1"
        )
        cursor.execute(query_sql, (int(storage_object_id), *where_params))
        if cursor.fetchone() is None:
            raise DocumentScopeAccessError("Document storage object is out of scope or does not exist.")

    def _assert_scoped_ingestion_job_exists(self, cursor: Any, ingestion_job_id: int, scope: Any) -> None:
        where_sql, where_params = build_scope_where_clause(scope)
        query_sql = (
            "SELECT ingestion_job_id "
            "FROM document_ingestion_jobs "
            "WHERE ingestion_job_id = %s AND "
            + where_sql
            + " LIMIT 1"
        )
        cursor.execute(query_sql, (int(ingestion_job_id), *where_params))
        if cursor.fetchone() is None:
            raise DocumentScopeAccessError("Document ingestion job is out of scope or does not exist.")

    def _normalize_document_state(self, value: Any, default: str = "queued") -> str:
        state = str(value if value is not None else "").strip().lower()
        if not state:
            state = str(default).strip().lower()
        if state not in DOCUMENT_SOURCE_STATES:
            raise ValueError(
                f"Invalid document state '{state}'. Allowed states: {', '.join(DOCUMENT_SOURCE_STATES)}."
            )
        return state

    def _normalize_ingestion_job_status(self, value: Any, default: str = "queued") -> str:
        status = str(value if value is not None else "").strip().lower()
        if not status:
            status = str(default).strip().lower()
        if status not in DOCUMENT_INGESTION_JOB_STATES:
            raise ValueError(
                f"Invalid ingestion job status '{status}'. Allowed: {', '.join(DOCUMENT_INGESTION_JOB_STATES)}."
            )
        return status

    def _normalize_ingestion_attempt_status(self, value: Any, default: str = "running") -> str:
        status = str(value if value is not None else "").strip().lower()
        if not status:
            status = str(default).strip().lower()
        if status not in DOCUMENT_INGESTION_ATTEMPT_STATES:
            raise ValueError(
                f"Invalid ingestion attempt status '{status}'. Allowed: {', '.join(DOCUMENT_INGESTION_ATTEMPT_STATES)}."
            )
        return status

    def _apply_scope_bypass_settings(self, cursor: Any) -> None:
        cursor.execute("SELECT set_config(%s, %s, true);", ("app.scope_bypass", "1"))

    def _safe_positive_int(self, value: Any, default: int) -> int:
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            parsed = int(default)
        if parsed <= 0:
            parsed = int(default)
        return max(1, parsed)

    def _safe_non_negative_int(self, value: Any, default: int = 0) -> int:
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            parsed = int(default)
        if parsed < 0:
            parsed = int(default)
        return max(0, parsed)

    def _optional_iso_timestamp(self, value: Any) -> datetime | None:
        if value is None:
            return None
        if isinstance(value, datetime):
            return value
        text = str(value).strip()
        if not text:
            return None
        normalized = text.replace("Z", "+00:00")
        return datetime.fromisoformat(normalized)

    def _optional_dict(self, value: Any) -> dict[str, Any]:
        if isinstance(value, dict):
            return dict(value)
        return {}

    def createDocumentSource(
        self,
        payload: dict[str, Any],
        *,
        actor_member_id: int | None = None,
        actor_roles: list[str] | tuple[str, ...] | None = None,
    ) -> dict | None:
        contract = validate_document_source_contract(payload)
        scope = self._resolved_scope(
            contract.to_dict(),
            actor_member_id=actor_member_id,
            actor_roles=actor_roles,
        )
        connection = None
        response = None
        try:
            connection = psycopg.connect(conninfo=ConfigManager()._instance.db_conninfo, row_factory=dict_row)
            cursor = connection.cursor()
            logger.debug(f"PostgreSQL connection established.")
            apply_pg_scope_settings(cursor, scope)

            insert_sql = """INSERT INTO document_sources (
                schema_version, owner_member_id, chat_host_id, chat_type, community_id, topic_id, platform,
                source_external_id, source_name, source_mime, source_sha256, source_size_bytes, source_uri,
                source_state, source_metadata
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING document_source_id, schema_version, owner_member_id, chat_host_id, chat_type, community_id, topic_id,
                platform, source_external_id, source_name, source_mime, source_sha256, source_size_bytes, source_uri,
                source_state, source_metadata, created_at, updated_at;"""
            cursor.execute(
                insert_sql,
                (
                    contract.schema_version,
                    scope.owner_member_id,
                    scope.chat_host_id,
                    scope.chat_type,
                    scope.community_id,
                    scope.topic_id,
                    scope.platform,
                    contract.source_external_id or None,
                    contract.source_name,
                    contract.source_mime or None,
                    contract.source_sha256 or None,
                    contract.source_size_bytes,
                    contract.source_uri or None,
                    contract.source_state,
                    json.dumps(contract.source_metadata),
                ),
            )
            response = cursor.fetchone()
            connection.commit()
            cursor.close()
        except (Exception, psycopg.DatabaseError) as error:
            logger.error(f"Exception while working with psycopg and PostgreSQL:\n{error}")
        finally:
            if connection is not None:
                connection.close()
                logger.debug(f"PostgreSQL connection is closed.")
            return response

    def createDocumentVersion(
        self,
        payload: dict[str, Any],
        *,
        actor_member_id: int | None = None,
        actor_roles: list[str] | tuple[str, ...] | None = None,
    ) -> dict | None:
        contract = validate_document_version_contract(payload)
        scope = self._resolved_scope(
            contract.to_dict(),
            actor_member_id=actor_member_id,
            actor_roles=actor_roles,
        )
        connection = None
        response = None
        try:
            connection = psycopg.connect(conninfo=ConfigManager()._instance.db_conninfo, row_factory=dict_row)
            cursor = connection.cursor()
            logger.debug(f"PostgreSQL connection established.")
            apply_pg_scope_settings(cursor, scope)
            self._assert_scoped_source_exists(cursor, contract.document_source_id, scope)

            insert_sql = """INSERT INTO document_versions (
                document_source_id, schema_version, owner_member_id, chat_host_id, chat_type, community_id, topic_id,
                platform, version_number, source_sha256, parser_name, parser_version, parser_status, parse_artifact, record_metadata
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING document_version_id, document_source_id, schema_version, owner_member_id, chat_host_id, chat_type,
                community_id, topic_id, platform, version_number, source_sha256, parser_name, parser_version, parser_status,
                parse_artifact, record_metadata, created_at;"""
            cursor.execute(
                insert_sql,
                (
                    contract.document_source_id,
                    contract.schema_version,
                    scope.owner_member_id,
                    scope.chat_host_id,
                    scope.chat_type,
                    scope.community_id,
                    scope.topic_id,
                    scope.platform,
                    contract.version_number,
                    contract.source_sha256 or None,
                    contract.parser_name or None,
                    contract.parser_version or None,
                    contract.parser_status,
                    json.dumps(contract.parse_artifact),
                    json.dumps(contract.record_metadata),
                ),
            )
            response = cursor.fetchone()
            connection.commit()
            cursor.close()
        except (Exception, psycopg.DatabaseError) as error:
            logger.error(f"Exception while working with psycopg and PostgreSQL:\n{error}")
        finally:
            if connection is not None:
                connection.close()
                logger.debug(f"PostgreSQL connection is closed.")
            return response

    def addDocumentRetrievalEvent(
        self,
        payload: dict[str, Any],
        *,
        actor_member_id: int | None = None,
        actor_roles: list[str] | tuple[str, ...] | None = None,
    ) -> dict | None:
        contract = validate_document_retrieval_event_contract(payload)
        scope = self._resolved_scope(
            contract.to_dict(),
            actor_member_id=actor_member_id,
            actor_roles=actor_roles,
        )
        connection = None
        response = None
        try:
            connection = psycopg.connect(conninfo=ConfigManager()._instance.db_conninfo, row_factory=dict_row)
            cursor = connection.cursor()
            logger.debug(f"PostgreSQL connection established.")
            apply_pg_scope_settings(cursor, scope)
            if contract.document_source_id is not None:
                self._assert_scoped_source_exists(cursor, contract.document_source_id, scope)
            if contract.document_version_id is not None:
                self._assert_scoped_version_exists(cursor, contract.document_version_id, scope)

            insert_sql = """INSERT INTO document_retrieval_events (
                schema_version, owner_member_id, chat_host_id, chat_type, community_id, topic_id, platform, request_id,
                query_text, document_source_id, document_version_id, result_count, max_distance,
                query_metadata, retrieval_metadata, citations
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING retrieval_event_id, schema_version, owner_member_id, chat_host_id, chat_type, community_id, topic_id,
                platform, request_id, query_text, document_source_id, document_version_id, result_count,
                max_distance, query_metadata, retrieval_metadata, citations, created_at;"""
            cursor.execute(
                insert_sql,
                (
                    contract.schema_version,
                    scope.owner_member_id,
                    scope.chat_host_id,
                    scope.chat_type,
                    scope.community_id,
                    scope.topic_id,
                    scope.platform,
                    contract.request_id,
                    contract.query_text,
                    contract.document_source_id,
                    contract.document_version_id,
                    contract.result_count,
                    contract.max_distance,
                    json.dumps(contract.query_metadata),
                    json.dumps(contract.retrieval_metadata),
                    json.dumps(contract.citations),
                ),
            )
            response = cursor.fetchone()
            connection.commit()
            cursor.close()
        except (Exception, psycopg.DatabaseError) as error:
            logger.error(f"Exception while working with psycopg and PostgreSQL:\n{error}")
        finally:
            if connection is not None:
                connection.close()
                logger.debug(f"PostgreSQL connection is closed.")
            return response

    def updateDocumentVersionParserStatus(
        self,
        document_version_id: int,
        parser_status: str,
        scope_payload: dict[str, Any],
        *,
        parser_name: str | None = None,
        parser_version: str | None = None,
        parse_artifact_patch: dict[str, Any] | None = None,
        record_metadata_patch: dict[str, Any] | None = None,
        actor_member_id: int | None = None,
        actor_roles: list[str] | tuple[str, ...] | None = None,
    ) -> dict | None:
        version_id = int(document_version_id)
        if version_id <= 0:
            raise ValueError("document_version_id must be greater than zero.")
        state = self._normalize_document_state(parser_status, "queued")
        scope = self._resolved_scope(
            scope_payload,
            actor_member_id=actor_member_id,
            actor_roles=actor_roles,
        )
        artifact_patch = self._optional_dict(parse_artifact_patch)
        metadata_patch = self._optional_dict(record_metadata_patch)
        response = None
        connection = None
        try:
            connection = psycopg.connect(conninfo=ConfigManager()._instance.db_conninfo, row_factory=dict_row)
            cursor = connection.cursor()
            logger.debug(f"PostgreSQL connection established.")
            apply_pg_scope_settings(cursor, scope)
            self._assert_scoped_version_exists(cursor, version_id, scope)
            update_sql = """UPDATE document_versions
            SET parser_status = %s,
                parser_name = COALESCE(%s, parser_name),
                parser_version = COALESCE(%s, parser_version),
                parse_artifact = COALESCE(parse_artifact, '{}'::jsonb) || %s::jsonb,
                record_metadata = COALESCE(record_metadata, '{}'::jsonb) || %s::jsonb
            WHERE document_version_id = %s
            RETURNING document_version_id, document_source_id, schema_version, owner_member_id, chat_host_id, chat_type,
                community_id, topic_id, platform, version_number, source_sha256, parser_name, parser_version, parser_status,
                parse_artifact, record_metadata, created_at;"""
            cursor.execute(
                update_sql,
                (
                    state,
                    str(parser_name).strip() if parser_name is not None else None,
                    str(parser_version).strip() if parser_version is not None else None,
                    json.dumps(artifact_patch),
                    json.dumps(metadata_patch),
                    version_id,
                ),
            )
            response = cursor.fetchone()
            connection.commit()
            cursor.close()
        except (Exception, psycopg.DatabaseError) as error:
            logger.error(f"Exception while working with psycopg and PostgreSQL:\n{error}")
        finally:
            if connection is not None:
                connection.close()
                logger.debug(f"PostgreSQL connection is closed.")
            return response

    def findLatestDocumentSourceByDigest(
        self,
        scope_payload: dict[str, Any],
        source_sha256: str,
        *,
        actor_member_id: int | None = None,
        actor_roles: list[str] | tuple[str, ...] | None = None,
        source_name: str | None = None,
    ) -> dict | None:
        digest = str(source_sha256 or "").strip().lower()
        if not digest:
            return None
        scope = self._resolved_scope(
            scope_payload,
            actor_member_id=actor_member_id,
            actor_roles=actor_roles,
        )
        response = None
        connection = None
        try:
            connection = psycopg.connect(conninfo=ConfigManager()._instance.db_conninfo, row_factory=dict_row)
            cursor = connection.cursor()
            logger.debug(f"PostgreSQL connection established.")
            apply_pg_scope_settings(cursor, scope)
            where_sql, where_params = build_scope_where_clause(scope)
            query_sql = (
                "SELECT document_source_id, schema_version, owner_member_id, chat_host_id, chat_type, community_id, "
                "topic_id, platform, source_external_id, source_name, source_mime, source_sha256, source_size_bytes, "
                "source_uri, source_state, source_metadata, created_at, updated_at "
                "FROM document_sources WHERE "
                + where_sql
                + " AND source_sha256 = %s AND source_state <> 'deleted'"
            )
            params: list[Any] = [*where_params, digest]
            source_name_text = str(source_name or "").strip()
            if source_name_text:
                query_sql += " AND source_name = %s"
                params.append(source_name_text)
            query_sql += " ORDER BY created_at DESC LIMIT 1"
            cursor.execute(query_sql, tuple(params))
            response = cursor.fetchone()
            cursor.close()
        except (Exception, psycopg.DatabaseError) as error:
            logger.error(f"Exception while working with psycopg and PostgreSQL:\n{error}")
        finally:
            if connection is not None:
                connection.close()
                logger.debug(f"PostgreSQL connection is closed.")
            return response

    def nextDocumentVersionNumber(
        self,
        document_source_id: int,
        scope_payload: dict[str, Any],
        *,
        actor_member_id: int | None = None,
        actor_roles: list[str] | tuple[str, ...] | None = None,
    ) -> int:
        source_id = int(document_source_id)
        if source_id <= 0:
            raise ValueError("document_source_id must be greater than zero.")
        scope = self._resolved_scope(
            scope_payload,
            actor_member_id=actor_member_id,
            actor_roles=actor_roles,
        )
        next_version = 1
        connection = None
        try:
            connection = psycopg.connect(conninfo=ConfigManager()._instance.db_conninfo, row_factory=dict_row)
            cursor = connection.cursor()
            logger.debug(f"PostgreSQL connection established.")
            apply_pg_scope_settings(cursor, scope)
            self._assert_scoped_source_exists(cursor, source_id, scope)
            where_sql, where_params = build_scope_where_clause(scope)
            query_sql = (
                "SELECT COALESCE(MAX(version_number), 0) AS max_version "
                "FROM document_versions "
                "WHERE document_source_id = %s AND "
                + where_sql
            )
            cursor.execute(query_sql, (source_id, *where_params))
            row = cursor.fetchone()
            if isinstance(row, dict):
                next_version = int(row.get("max_version", 0) or 0) + 1
            cursor.close()
        except (Exception, psycopg.DatabaseError) as error:
            logger.error(f"Exception while working with psycopg and PostgreSQL:\n{error}")
        finally:
            if connection is not None:
                connection.close()
                logger.debug(f"PostgreSQL connection is closed.")
            return max(1, int(next_version))

    def createDocumentStorageObject(
        self,
        payload: dict[str, Any],
        *,
        actor_member_id: int | None = None,
        actor_roles: list[str] | tuple[str, ...] | None = None,
    ) -> dict | None:
        scope = self._resolved_scope(
            payload,
            actor_member_id=actor_member_id,
            actor_roles=actor_roles,
        )
        source_id = int(payload.get("document_source_id", 0) or 0)
        if source_id <= 0:
            raise ValueError("document_source_id is required.")
        storage_backend = str(payload.get("storage_backend") or "local_fs").strip().lower()
        storage_key = str(payload.get("storage_key") or "").strip()
        if not storage_key:
            raise ValueError("storage_key is required.")
        file_name = str(payload.get("file_name") or "").strip()
        if not file_name:
            raise ValueError("file_name is required.")
        file_sha256 = str(payload.get("file_sha256") or "").strip().lower()
        if not file_sha256:
            raise ValueError("file_sha256 is required.")
        try:
            file_size_bytes = int(payload.get("file_size_bytes", 0))
        except (TypeError, ValueError):
            file_size_bytes = -1
        if file_size_bytes < 0:
            raise ValueError("file_size_bytes must be zero or greater.")
        try:
            object_state = self._normalize_document_state(payload.get("object_state"), "received")
        except ValueError as error:
            raise ValueError(str(error)) from None

        record_metadata = self._optional_dict(payload.get("record_metadata"))
        retention_until = self._optional_iso_timestamp(payload.get("retention_until"))
        deleted_at = self._optional_iso_timestamp(payload.get("deleted_at"))
        schema_version = int(payload.get("schema_version", 1) or 1)

        response = None
        connection = None
        try:
            connection = psycopg.connect(conninfo=ConfigManager()._instance.db_conninfo, row_factory=dict_row)
            cursor = connection.cursor()
            logger.debug(f"PostgreSQL connection established.")
            apply_pg_scope_settings(cursor, scope)
            self._assert_scoped_source_exists(cursor, source_id, scope)
            insert_sql = """INSERT INTO document_storage_objects (
                document_source_id, schema_version, owner_member_id, chat_host_id, chat_type, community_id, topic_id,
                platform, storage_backend, storage_key, storage_path, object_state, file_name, file_mime, file_sha256,
                file_size_bytes, dedupe_status, retention_until, deleted_at, record_metadata
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING storage_object_id, document_source_id, schema_version, owner_member_id, chat_host_id, chat_type,
                community_id, topic_id, platform, storage_backend, storage_key, storage_path, object_state, file_name,
                file_mime, file_sha256, file_size_bytes, dedupe_status, retention_until, deleted_at, record_metadata,
                created_at, updated_at;"""
            cursor.execute(
                insert_sql,
                (
                    source_id,
                    schema_version,
                    scope.owner_member_id,
                    scope.chat_host_id,
                    scope.chat_type,
                    scope.community_id,
                    scope.topic_id,
                    scope.platform,
                    storage_backend,
                    storage_key,
                    str(payload.get("storage_path") or "").strip() or None,
                    object_state,
                    file_name,
                    str(payload.get("file_mime") or "").strip().lower() or None,
                    file_sha256,
                    file_size_bytes,
                    str(payload.get("dedupe_status") or "new").strip().lower(),
                    retention_until,
                    deleted_at,
                    json.dumps(record_metadata),
                ),
            )
            response = cursor.fetchone()
            connection.commit()
            cursor.close()
        except (Exception, psycopg.DatabaseError) as error:
            logger.error(f"Exception while working with psycopg and PostgreSQL:\n{error}")
        finally:
            if connection is not None:
                connection.close()
                logger.debug(f"PostgreSQL connection is closed.")
            return response

    def updateDocumentSourceState(
        self,
        document_source_id: int,
        source_state: str,
        scope_payload: dict[str, Any],
        *,
        actor_member_id: int | None = None,
        actor_roles: list[str] | tuple[str, ...] | None = None,
        metadata_patch: dict[str, Any] | None = None,
    ) -> dict | None:
        source_id = int(document_source_id)
        if source_id <= 0:
            raise ValueError("document_source_id must be greater than zero.")
        state = self._normalize_document_state(source_state, "queued")
        scope = self._resolved_scope(
            scope_payload,
            actor_member_id=actor_member_id,
            actor_roles=actor_roles,
        )
        response = None
        connection = None
        try:
            connection = psycopg.connect(conninfo=ConfigManager()._instance.db_conninfo, row_factory=dict_row)
            cursor = connection.cursor()
            logger.debug(f"PostgreSQL connection established.")
            apply_pg_scope_settings(cursor, scope)
            self._assert_scoped_source_exists(cursor, source_id, scope)
            patch = self._optional_dict(metadata_patch)
            if patch:
                update_sql = """UPDATE document_sources
                SET source_state = %s,
                    source_metadata = COALESCE(source_metadata, '{}'::jsonb) || %s::jsonb,
                    updated_at = NOW()
                WHERE document_source_id = %s
                RETURNING document_source_id, schema_version, owner_member_id, chat_host_id, chat_type, community_id,
                    topic_id, platform, source_external_id, source_name, source_mime, source_sha256, source_size_bytes,
                    source_uri, source_state, source_metadata, created_at, updated_at;"""
                cursor.execute(update_sql, (state, json.dumps(patch), source_id))
            else:
                update_sql = """UPDATE document_sources
                SET source_state = %s, updated_at = NOW()
                WHERE document_source_id = %s
                RETURNING document_source_id, schema_version, owner_member_id, chat_host_id, chat_type, community_id,
                    topic_id, platform, source_external_id, source_name, source_mime, source_sha256, source_size_bytes,
                    source_uri, source_state, source_metadata, created_at, updated_at;"""
                cursor.execute(update_sql, (state, source_id))
            response = cursor.fetchone()
            connection.commit()
            cursor.close()
        except (Exception, psycopg.DatabaseError) as error:
            logger.error(f"Exception while working with psycopg and PostgreSQL:\n{error}")
        finally:
            if connection is not None:
                connection.close()
                logger.debug(f"PostgreSQL connection is closed.")
            return response

    def updateDocumentStorageObjectState(
        self,
        storage_object_id: int,
        object_state: str,
        scope_payload: dict[str, Any],
        *,
        actor_member_id: int | None = None,
        actor_roles: list[str] | tuple[str, ...] | None = None,
    ) -> dict | None:
        object_id = int(storage_object_id)
        if object_id <= 0:
            raise ValueError("storage_object_id must be greater than zero.")
        state = self._normalize_document_state(object_state, "queued")
        scope = self._resolved_scope(
            scope_payload,
            actor_member_id=actor_member_id,
            actor_roles=actor_roles,
        )
        response = None
        connection = None
        try:
            connection = psycopg.connect(conninfo=ConfigManager()._instance.db_conninfo, row_factory=dict_row)
            cursor = connection.cursor()
            logger.debug(f"PostgreSQL connection established.")
            apply_pg_scope_settings(cursor, scope)
            self._assert_scoped_storage_object_exists(cursor, object_id, scope)
            deleted_at = datetime.now(timezone.utc) if state == "deleted" else None
            update_sql = """UPDATE document_storage_objects
            SET object_state = %s,
                deleted_at = %s,
                updated_at = NOW()
            WHERE storage_object_id = %s
            RETURNING storage_object_id, document_source_id, schema_version, owner_member_id, chat_host_id, chat_type,
                community_id, topic_id, platform, storage_backend, storage_key, storage_path, object_state, file_name,
                file_mime, file_sha256, file_size_bytes, dedupe_status, retention_until, deleted_at, record_metadata,
                created_at, updated_at;"""
            cursor.execute(update_sql, (state, deleted_at, object_id))
            response = cursor.fetchone()
            connection.commit()
            cursor.close()
        except (Exception, psycopg.DatabaseError) as error:
            logger.error(f"Exception while working with psycopg and PostgreSQL:\n{error}")
        finally:
            if connection is not None:
                connection.close()
                logger.debug(f"PostgreSQL connection is closed.")
            return response

    def enqueueDocumentIngestionJob(
        self,
        payload: dict[str, Any],
        *,
        actor_member_id: int | None = None,
        actor_roles: list[str] | tuple[str, ...] | None = None,
    ) -> dict | None:
        scope = self._resolved_scope(
            payload,
            actor_member_id=actor_member_id,
            actor_roles=actor_roles,
        )
        source_id = self._safe_positive_int(payload.get("document_source_id"), 0)
        version_id = self._safe_positive_int(payload.get("document_version_id"), 0)
        if source_id <= 0 or version_id <= 0:
            raise ValueError("document_source_id and document_version_id are required.")

        storage_object_id_raw = payload.get("storage_object_id")
        storage_object_id = None
        if storage_object_id_raw is not None and str(storage_object_id_raw).strip() != "":
            storage_object_id = self._safe_positive_int(storage_object_id_raw, 0)
            if storage_object_id <= 0:
                raise ValueError("storage_object_id must be greater than zero when provided.")

        pipeline_version = str(payload.get("pipeline_version") or "v1").strip().lower() or "v1"
        idempotency_key = str(payload.get("idempotency_key") or "").strip()
        if not idempotency_key:
            idempotency_key = f"{source_id}:{version_id}:{pipeline_version}"

        job_status = self._normalize_ingestion_job_status(payload.get("job_status"), "queued")
        priority = self._safe_non_negative_int(payload.get("priority"), 100)
        max_attempts = self._safe_positive_int(payload.get("max_attempts"), 3)
        schema_version = self._safe_positive_int(payload.get("schema_version"), 1)
        available_at = self._optional_iso_timestamp(payload.get("available_at"))
        scheduled_at = self._optional_iso_timestamp(payload.get("scheduled_at")) or datetime.now(timezone.utc)
        metadata = self._optional_dict(payload.get("record_metadata"))

        response = None
        connection = None
        try:
            connection = psycopg.connect(conninfo=ConfigManager()._instance.db_conninfo, row_factory=dict_row)
            cursor = connection.cursor()
            logger.debug("PostgreSQL connection established.")
            apply_pg_scope_settings(cursor, scope)
            self._assert_scoped_source_exists(cursor, source_id, scope)
            self._assert_scoped_version_exists(cursor, version_id, scope)
            if storage_object_id is not None:
                self._assert_scoped_storage_object_exists(cursor, storage_object_id, scope)

            insert_sql = """INSERT INTO document_ingestion_jobs (
                document_source_id, document_version_id, storage_object_id, schema_version,
                owner_member_id, chat_host_id, chat_type, community_id, topic_id, platform,
                pipeline_version, idempotency_key, job_status, priority, max_attempts,
                available_at, scheduled_at, record_metadata
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (document_source_id, document_version_id, pipeline_version)
            DO UPDATE SET
                updated_at = NOW(),
                record_metadata = COALESCE(document_ingestion_jobs.record_metadata, '{}'::jsonb) || EXCLUDED.record_metadata
            RETURNING ingestion_job_id, document_source_id, document_version_id, storage_object_id, schema_version,
                owner_member_id, chat_host_id, chat_type, community_id, topic_id, platform, pipeline_version,
                idempotency_key, job_status, priority, attempt_count, max_attempts, available_at, scheduled_at,
                lease_owner, lease_expires_at, heartbeat_at, last_error, last_error_context, dead_letter_reason,
                cancel_requested, cancelled_at, completed_at, record_metadata, created_at, updated_at;"""
            cursor.execute(
                insert_sql,
                (
                    source_id,
                    version_id,
                    storage_object_id,
                    schema_version,
                    scope.owner_member_id,
                    scope.chat_host_id,
                    scope.chat_type,
                    scope.community_id,
                    scope.topic_id,
                    scope.platform,
                    pipeline_version,
                    idempotency_key,
                    job_status,
                    priority,
                    max_attempts,
                    available_at or datetime.now(timezone.utc),
                    scheduled_at,
                    json.dumps(metadata),
                ),
            )
            response = cursor.fetchone()
            connection.commit()
            cursor.close()
        except (Exception, psycopg.DatabaseError) as error:
            logger.error(f"Exception while working with psycopg and PostgreSQL:\n{error}")
        finally:
            if connection is not None:
                connection.close()
                logger.debug("PostgreSQL connection is closed.")
            return response

    def getDocumentIngestionJobs(
        self,
        scope_payload: dict[str, Any],
        *,
        statuses: list[str] | tuple[str, ...] | None = None,
        document_source_id: int | None = None,
        actor_member_id: int | None = None,
        actor_roles: list[str] | tuple[str, ...] | None = None,
        limit: int | None = None,
    ) -> list:
        scope = self._resolved_scope(
            scope_payload,
            actor_member_id=actor_member_id,
            actor_roles=actor_roles,
        )
        query_limit = self._safe_limit(limit, default=50)
        response: list = list()
        connection = None
        try:
            connection = psycopg.connect(conninfo=ConfigManager()._instance.db_conninfo, row_factory=dict_row)
            cursor = connection.cursor()
            logger.debug("PostgreSQL connection established.")
            apply_pg_scope_settings(cursor, scope)
            where_sql, where_params = build_scope_where_clause(scope)
            query_sql = (
                "SELECT ingestion_job_id, document_source_id, document_version_id, storage_object_id, schema_version, "
                "owner_member_id, chat_host_id, chat_type, community_id, topic_id, platform, pipeline_version, idempotency_key, "
                "job_status, priority, attempt_count, max_attempts, available_at, scheduled_at, lease_owner, lease_expires_at, "
                "heartbeat_at, last_error, last_error_context, dead_letter_reason, cancel_requested, cancelled_at, completed_at, "
                "record_metadata, created_at, updated_at "
                "FROM document_ingestion_jobs WHERE "
                + where_sql
            )
            params: list[Any] = list(where_params)
            status_values = [self._normalize_ingestion_job_status(item) for item in (statuses or []) if str(item).strip()]
            if status_values:
                query_sql += " AND job_status = ANY(%s)"
                params.append(status_values)
            if document_source_id is not None:
                source_id = self._safe_positive_int(document_source_id, 0)
                if source_id > 0:
                    query_sql += " AND document_source_id = %s"
                    params.append(source_id)
            query_sql += " ORDER BY created_at DESC LIMIT %s"
            params.append(query_limit)
            cursor.execute(query_sql, tuple(params))
            results = cursor.fetchall()
            response = self._accessPolicy.filter_records(results, scope=scope)
            cursor.close()
        except (Exception, psycopg.DatabaseError) as error:
            logger.error(f"Exception while working with psycopg and PostgreSQL:\n{error}")
        finally:
            if connection is not None:
                connection.close()
                logger.debug("PostgreSQL connection is closed.")
            return response

    def getDocumentIngestionJobByID(
        self,
        ingestion_job_id: int,
        scope_payload: dict[str, Any],
        *,
        actor_member_id: int | None = None,
        actor_roles: list[str] | tuple[str, ...] | None = None,
    ) -> dict | None:
        job_id = self._safe_positive_int(ingestion_job_id, 0)
        if job_id <= 0:
            return None
        scope = self._resolved_scope(
            scope_payload,
            actor_member_id=actor_member_id,
            actor_roles=actor_roles,
        )
        response = None
        connection = None
        try:
            connection = psycopg.connect(conninfo=ConfigManager()._instance.db_conninfo, row_factory=dict_row)
            cursor = connection.cursor()
            logger.debug("PostgreSQL connection established.")
            apply_pg_scope_settings(cursor, scope)
            where_sql, where_params = build_scope_where_clause(scope)
            query_sql = (
                "SELECT ingestion_job_id, document_source_id, document_version_id, storage_object_id, schema_version, "
                "owner_member_id, chat_host_id, chat_type, community_id, topic_id, platform, pipeline_version, idempotency_key, "
                "job_status, priority, attempt_count, max_attempts, available_at, scheduled_at, lease_owner, lease_expires_at, "
                "heartbeat_at, last_error, last_error_context, dead_letter_reason, cancel_requested, cancelled_at, completed_at, "
                "record_metadata, created_at, updated_at "
                "FROM document_ingestion_jobs WHERE ingestion_job_id = %s AND "
                + where_sql
                + " LIMIT 1"
            )
            cursor.execute(query_sql, (job_id, *where_params))
            response = cursor.fetchone()
            cursor.close()
        except (Exception, psycopg.DatabaseError) as error:
            logger.error(f"Exception while working with psycopg and PostgreSQL:\n{error}")
        finally:
            if connection is not None:
                connection.close()
                logger.debug("PostgreSQL connection is closed.")
            return response

    def cancelDocumentIngestionJob(
        self,
        ingestion_job_id: int,
        scope_payload: dict[str, Any],
        *,
        actor_member_id: int | None = None,
        actor_roles: list[str] | tuple[str, ...] | None = None,
        reason: str = "",
    ) -> dict | None:
        job_id = self._safe_positive_int(ingestion_job_id, 0)
        if job_id <= 0:
            return None
        scope = self._resolved_scope(
            scope_payload,
            actor_member_id=actor_member_id,
            actor_roles=actor_roles,
        )
        response = None
        connection = None
        try:
            connection = psycopg.connect(conninfo=ConfigManager()._instance.db_conninfo, row_factory=dict_row)
            cursor = connection.cursor()
            logger.debug("PostgreSQL connection established.")
            apply_pg_scope_settings(cursor, scope)
            self._assert_scoped_ingestion_job_exists(cursor, job_id, scope)
            update_sql = """UPDATE document_ingestion_jobs
            SET cancel_requested = TRUE,
                job_status = CASE WHEN job_status = 'completed' THEN 'completed' ELSE 'cancelled' END,
                cancelled_at = CASE WHEN cancelled_at IS NULL THEN NOW() ELSE cancelled_at END,
                dead_letter_reason = CASE WHEN dead_letter_reason IS NULL OR dead_letter_reason = '' THEN %s ELSE dead_letter_reason END,
                lease_owner = NULL,
                lease_expires_at = NULL,
                heartbeat_at = NULL,
                updated_at = NOW()
            WHERE ingestion_job_id = %s
            RETURNING ingestion_job_id, document_source_id, document_version_id, storage_object_id, schema_version,
                owner_member_id, chat_host_id, chat_type, community_id, topic_id, platform, pipeline_version, idempotency_key,
                job_status, priority, attempt_count, max_attempts, available_at, scheduled_at, lease_owner, lease_expires_at,
                heartbeat_at, last_error, last_error_context, dead_letter_reason, cancel_requested, cancelled_at, completed_at,
                record_metadata, created_at, updated_at;"""
            cursor.execute(update_sql, (str(reason or "").strip() or "cancelled_by_operator", job_id))
            response = cursor.fetchone()
            connection.commit()
            cursor.close()
        except (Exception, psycopg.DatabaseError) as error:
            logger.error(f"Exception while working with psycopg and PostgreSQL:\n{error}")
        finally:
            if connection is not None:
                connection.close()
                logger.debug("PostgreSQL connection is closed.")
            return response

    def requeueDocumentIngestionJob(
        self,
        ingestion_job_id: int,
        scope_payload: dict[str, Any],
        *,
        actor_member_id: int | None = None,
        actor_roles: list[str] | tuple[str, ...] | None = None,
    ) -> dict | None:
        job_id = self._safe_positive_int(ingestion_job_id, 0)
        if job_id <= 0:
            return None
        scope = self._resolved_scope(
            scope_payload,
            actor_member_id=actor_member_id,
            actor_roles=actor_roles,
        )
        response = None
        connection = None
        try:
            connection = psycopg.connect(conninfo=ConfigManager()._instance.db_conninfo, row_factory=dict_row)
            cursor = connection.cursor()
            logger.debug("PostgreSQL connection established.")
            apply_pg_scope_settings(cursor, scope)
            self._assert_scoped_ingestion_job_exists(cursor, job_id, scope)
            update_sql = """UPDATE document_ingestion_jobs
            SET job_status = 'queued',
                available_at = NOW(),
                cancel_requested = FALSE,
                cancelled_at = NULL,
                lease_owner = NULL,
                lease_expires_at = NULL,
                heartbeat_at = NULL,
                dead_letter_reason = NULL,
                last_error = NULL,
                last_error_context = '{}'::jsonb,
                updated_at = NOW()
            WHERE ingestion_job_id = %s
            RETURNING ingestion_job_id, document_source_id, document_version_id, storage_object_id, schema_version,
                owner_member_id, chat_host_id, chat_type, community_id, topic_id, platform, pipeline_version, idempotency_key,
                job_status, priority, attempt_count, max_attempts, available_at, scheduled_at, lease_owner, lease_expires_at,
                heartbeat_at, last_error, last_error_context, dead_letter_reason, cancel_requested, cancelled_at, completed_at,
                record_metadata, created_at, updated_at;"""
            cursor.execute(update_sql, (job_id,))
            response = cursor.fetchone()
            connection.commit()
            cursor.close()
        except (Exception, psycopg.DatabaseError) as error:
            logger.error(f"Exception while working with psycopg and PostgreSQL:\n{error}")
        finally:
            if connection is not None:
                connection.close()
                logger.debug("PostgreSQL connection is closed.")
            return response

    def leaseNextDocumentIngestionJob(
        self,
        *,
        worker_id: str,
        lease_seconds: int = 45,
    ) -> dict | None:
        worker = str(worker_id or "").strip() or "worker"
        lease = max(5, int(lease_seconds))
        response = None
        connection = None
        try:
            connection = psycopg.connect(conninfo=ConfigManager()._instance.db_conninfo, row_factory=dict_row)
            cursor = connection.cursor()
            logger.debug("PostgreSQL connection established.")
            self._apply_scope_bypass_settings(cursor)
            lease_sql = """WITH candidate AS (
                SELECT ingestion_job_id
                FROM document_ingestion_jobs
                WHERE job_status IN ('queued', 'retry_wait')
                    AND cancel_requested = FALSE
                    AND available_at <= NOW()
                    AND (lease_expires_at IS NULL OR lease_expires_at < NOW())
                ORDER BY priority ASC, available_at ASC, ingestion_job_id ASC
                LIMIT 1
                FOR UPDATE SKIP LOCKED
            )
            UPDATE document_ingestion_jobs AS job
            SET job_status = 'leased',
                lease_owner = %s,
                lease_expires_at = NOW() + (%s * INTERVAL '1 second'),
                heartbeat_at = NOW(),
                updated_at = NOW()
            FROM candidate
            WHERE job.ingestion_job_id = candidate.ingestion_job_id
            RETURNING job.ingestion_job_id, job.document_source_id, job.document_version_id, job.storage_object_id,
                job.schema_version, job.owner_member_id, job.chat_host_id, job.chat_type, job.community_id, job.topic_id,
                job.platform, job.pipeline_version, job.idempotency_key, job.job_status, job.priority, job.attempt_count,
                job.max_attempts, job.available_at, job.scheduled_at, job.lease_owner, job.lease_expires_at,
                job.heartbeat_at, job.last_error, job.last_error_context, job.dead_letter_reason, job.cancel_requested,
                job.cancelled_at, job.completed_at, job.record_metadata, job.created_at, job.updated_at;"""
            cursor.execute(lease_sql, (worker, lease))
            response = cursor.fetchone()
            connection.commit()
            cursor.close()
        except (Exception, psycopg.DatabaseError) as error:
            logger.error(f"Exception while working with psycopg and PostgreSQL:\n{error}")
        finally:
            if connection is not None:
                connection.close()
                logger.debug("PostgreSQL connection is closed.")
            return response

    def heartbeatDocumentIngestionJob(
        self,
        ingestion_job_id: int,
        *,
        worker_id: str,
        lease_seconds: int = 45,
    ) -> dict | None:
        job_id = self._safe_positive_int(ingestion_job_id, 0)
        if job_id <= 0:
            return None
        worker = str(worker_id or "").strip() or "worker"
        lease = max(5, int(lease_seconds))
        response = None
        connection = None
        try:
            connection = psycopg.connect(conninfo=ConfigManager()._instance.db_conninfo, row_factory=dict_row)
            cursor = connection.cursor()
            logger.debug("PostgreSQL connection established.")
            self._apply_scope_bypass_settings(cursor)
            update_sql = """UPDATE document_ingestion_jobs
            SET heartbeat_at = NOW(),
                lease_expires_at = NOW() + (%s * INTERVAL '1 second'),
                updated_at = NOW()
            WHERE ingestion_job_id = %s
                AND lease_owner = %s
                AND job_status IN ('leased', 'running')
            RETURNING ingestion_job_id, document_source_id, document_version_id, storage_object_id, schema_version,
                owner_member_id, chat_host_id, chat_type, community_id, topic_id, platform, pipeline_version, idempotency_key,
                job_status, priority, attempt_count, max_attempts, available_at, scheduled_at, lease_owner, lease_expires_at,
                heartbeat_at, last_error, last_error_context, dead_letter_reason, cancel_requested, cancelled_at, completed_at,
                record_metadata, created_at, updated_at;"""
            cursor.execute(update_sql, (lease, job_id, worker))
            response = cursor.fetchone()
            connection.commit()
            cursor.close()
        except (Exception, psycopg.DatabaseError) as error:
            logger.error(f"Exception while working with psycopg and PostgreSQL:\n{error}")
        finally:
            if connection is not None:
                connection.close()
                logger.debug("PostgreSQL connection is closed.")
            return response

    def markDocumentIngestionJobRunning(
        self,
        ingestion_job_id: int,
        *,
        worker_id: str,
    ) -> dict | None:
        job_id = self._safe_positive_int(ingestion_job_id, 0)
        if job_id <= 0:
            return None
        worker = str(worker_id or "").strip() or "worker"
        response = None
        connection = None
        try:
            connection = psycopg.connect(conninfo=ConfigManager()._instance.db_conninfo, row_factory=dict_row)
            cursor = connection.cursor()
            logger.debug("PostgreSQL connection established.")
            self._apply_scope_bypass_settings(cursor)
            update_sql = """UPDATE document_ingestion_jobs
            SET job_status = 'running',
                attempt_count = attempt_count + 1,
                heartbeat_at = NOW(),
                updated_at = NOW()
            WHERE ingestion_job_id = %s
                AND lease_owner = %s
                AND job_status IN ('leased', 'retry_wait', 'queued')
            RETURNING ingestion_job_id, document_source_id, document_version_id, storage_object_id, schema_version,
                owner_member_id, chat_host_id, chat_type, community_id, topic_id, platform, pipeline_version, idempotency_key,
                job_status, priority, attempt_count, max_attempts, available_at, scheduled_at, lease_owner, lease_expires_at,
                heartbeat_at, last_error, last_error_context, dead_letter_reason, cancel_requested, cancelled_at, completed_at,
                record_metadata, created_at, updated_at;"""
            cursor.execute(update_sql, (job_id, worker))
            response = cursor.fetchone()
            connection.commit()
            cursor.close()
        except (Exception, psycopg.DatabaseError) as error:
            logger.error(f"Exception while working with psycopg and PostgreSQL:\n{error}")
        finally:
            if connection is not None:
                connection.close()
                logger.debug("PostgreSQL connection is closed.")
            return response

    def completeDocumentIngestionJob(
        self,
        ingestion_job_id: int,
        *,
        worker_id: str,
        metadata_patch: dict[str, Any] | None = None,
    ) -> dict | None:
        job_id = self._safe_positive_int(ingestion_job_id, 0)
        if job_id <= 0:
            return None
        worker = str(worker_id or "").strip() or "worker"
        patch = self._optional_dict(metadata_patch)
        response = None
        connection = None
        try:
            connection = psycopg.connect(conninfo=ConfigManager()._instance.db_conninfo, row_factory=dict_row)
            cursor = connection.cursor()
            logger.debug("PostgreSQL connection established.")
            self._apply_scope_bypass_settings(cursor)
            update_sql = """UPDATE document_ingestion_jobs
            SET job_status = 'completed',
                completed_at = NOW(),
                lease_owner = NULL,
                lease_expires_at = NULL,
                heartbeat_at = NULL,
                dead_letter_reason = NULL,
                cancel_requested = FALSE,
                cancelled_at = NULL,
                last_error = NULL,
                last_error_context = '{}'::jsonb,
                record_metadata = COALESCE(record_metadata, '{}'::jsonb) || %s::jsonb,
                updated_at = NOW()
            WHERE ingestion_job_id = %s
                AND lease_owner = %s
            RETURNING ingestion_job_id, document_source_id, document_version_id, storage_object_id, schema_version,
                owner_member_id, chat_host_id, chat_type, community_id, topic_id, platform, pipeline_version, idempotency_key,
                job_status, priority, attempt_count, max_attempts, available_at, scheduled_at, lease_owner, lease_expires_at,
                heartbeat_at, last_error, last_error_context, dead_letter_reason, cancel_requested, cancelled_at, completed_at,
                record_metadata, created_at, updated_at;"""
            cursor.execute(update_sql, (json.dumps(patch), job_id, worker))
            response = cursor.fetchone()
            connection.commit()
            cursor.close()
        except (Exception, psycopg.DatabaseError) as error:
            logger.error(f"Exception while working with psycopg and PostgreSQL:\n{error}")
        finally:
            if connection is not None:
                connection.close()
                logger.debug("PostgreSQL connection is closed.")
            return response

    def failDocumentIngestionJob(
        self,
        ingestion_job_id: int,
        *,
        worker_id: str,
        error_message: str,
        error_context: dict[str, Any] | None = None,
        retry_base_seconds: int = 5,
        retry_max_seconds: int = 300,
    ) -> dict | None:
        job_id = self._safe_positive_int(ingestion_job_id, 0)
        if job_id <= 0:
            return None
        worker = str(worker_id or "").strip() or "worker"
        error_text = str(error_message or "").strip() or "ingestion_failure"
        context_payload = self._optional_dict(error_context)
        base = max(1, int(retry_base_seconds))
        ceiling = max(base, int(retry_max_seconds))
        response = None
        connection = None
        try:
            connection = psycopg.connect(conninfo=ConfigManager()._instance.db_conninfo, row_factory=dict_row)
            cursor = connection.cursor()
            logger.debug("PostgreSQL connection established.")
            self._apply_scope_bypass_settings(cursor)
            cursor.execute(
                """SELECT attempt_count, max_attempts
                FROM document_ingestion_jobs
                WHERE ingestion_job_id = %s
                    AND lease_owner = %s
                LIMIT 1""",
                (job_id, worker),
            )
            current = cursor.fetchone()
            if not isinstance(current, dict):
                connection.rollback()
                cursor.close()
                return None
            attempt_count = self._safe_non_negative_int(current.get("attempt_count"), 0)
            max_attempts = self._safe_positive_int(current.get("max_attempts"), 3)
            can_retry = attempt_count < max_attempts
            backoff_seconds = min(ceiling, base * (2 ** max(0, attempt_count - 1)))
            if can_retry:
                update_sql = """UPDATE document_ingestion_jobs
                SET job_status = 'retry_wait',
                    available_at = NOW() + (%s * INTERVAL '1 second'),
                    lease_owner = NULL,
                    lease_expires_at = NULL,
                    heartbeat_at = NULL,
                    last_error = %s,
                    last_error_context = %s::jsonb,
                    updated_at = NOW()
                WHERE ingestion_job_id = %s
                RETURNING ingestion_job_id, document_source_id, document_version_id, storage_object_id, schema_version,
                    owner_member_id, chat_host_id, chat_type, community_id, topic_id, platform, pipeline_version, idempotency_key,
                    job_status, priority, attempt_count, max_attempts, available_at, scheduled_at, lease_owner, lease_expires_at,
                    heartbeat_at, last_error, last_error_context, dead_letter_reason, cancel_requested, cancelled_at, completed_at,
                    record_metadata, created_at, updated_at;"""
                cursor.execute(update_sql, (backoff_seconds, error_text, json.dumps(context_payload), job_id))
            else:
                update_sql = """UPDATE document_ingestion_jobs
                SET job_status = 'dead_letter',
                    dead_letter_reason = %s,
                    lease_owner = NULL,
                    lease_expires_at = NULL,
                    heartbeat_at = NULL,
                    last_error = %s,
                    last_error_context = %s::jsonb,
                    updated_at = NOW()
                WHERE ingestion_job_id = %s
                RETURNING ingestion_job_id, document_source_id, document_version_id, storage_object_id, schema_version,
                    owner_member_id, chat_host_id, chat_type, community_id, topic_id, platform, pipeline_version, idempotency_key,
                    job_status, priority, attempt_count, max_attempts, available_at, scheduled_at, lease_owner, lease_expires_at,
                    heartbeat_at, last_error, last_error_context, dead_letter_reason, cancel_requested, cancelled_at, completed_at,
                    record_metadata, created_at, updated_at;"""
                cursor.execute(
                    update_sql,
                    (f"max_attempts_exhausted:{error_text}", error_text, json.dumps(context_payload), job_id),
                )
            response = cursor.fetchone()
            connection.commit()
            cursor.close()
        except (Exception, psycopg.DatabaseError) as error:
            logger.error(f"Exception while working with psycopg and PostgreSQL:\n{error}")
        finally:
            if connection is not None:
                connection.close()
                logger.debug("PostgreSQL connection is closed.")
            return response

    def createDocumentIngestionAttempt(
        self,
        job_record: dict[str, Any],
        *,
        worker_id: str,
        attempt_status: str = "running",
    ) -> dict | None:
        if not isinstance(job_record, dict):
            return None
        job_id = self._safe_positive_int(job_record.get("ingestion_job_id"), 0)
        if job_id <= 0:
            return None
        attempt_number = self._safe_positive_int(job_record.get("attempt_count"), 1)
        status = self._normalize_ingestion_attempt_status(attempt_status, "running")
        worker = str(worker_id or "").strip() or "worker"
        response = None
        connection = None
        try:
            connection = psycopg.connect(conninfo=ConfigManager()._instance.db_conninfo, row_factory=dict_row)
            cursor = connection.cursor()
            logger.debug("PostgreSQL connection established.")
            self._apply_scope_bypass_settings(cursor)
            insert_sql = """INSERT INTO document_ingestion_attempts (
                ingestion_job_id, schema_version, owner_member_id, chat_host_id, chat_type, community_id, topic_id,
                platform, attempt_number, attempt_status, worker_id
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (ingestion_job_id, attempt_number)
            DO UPDATE SET
                attempt_status = EXCLUDED.attempt_status,
                worker_id = EXCLUDED.worker_id
            RETURNING ingestion_attempt_id, ingestion_job_id, schema_version, owner_member_id, chat_host_id, chat_type,
                community_id, topic_id, platform, attempt_number, attempt_status, worker_id, started_at, finished_at,
                duration_ms, error_message, error_context, stage_events, created_at;"""
            cursor.execute(
                insert_sql,
                (
                    job_id,
                    self._safe_positive_int(job_record.get("schema_version"), 1),
                    self._safe_positive_int(job_record.get("owner_member_id"), 1),
                    self._safe_positive_int(job_record.get("chat_host_id"), 1),
                    str(job_record.get("chat_type") or "member").strip().lower(),
                    self._safe_non_negative_int(job_record.get("community_id"), 0) or None,
                    self._safe_non_negative_int(job_record.get("topic_id"), 0) or None,
                    str(job_record.get("platform") or "web").strip().lower(),
                    attempt_number,
                    status,
                    worker,
                ),
            )
            response = cursor.fetchone()
            connection.commit()
            cursor.close()
        except (Exception, psycopg.DatabaseError) as error:
            logger.error(f"Exception while working with psycopg and PostgreSQL:\n{error}")
        finally:
            if connection is not None:
                connection.close()
                logger.debug("PostgreSQL connection is closed.")
            return response

    def finishDocumentIngestionAttempt(
        self,
        ingestion_attempt_id: int,
        *,
        attempt_status: str,
        error_message: str = "",
        error_context: dict[str, Any] | None = None,
        stage_events: list[dict[str, Any]] | None = None,
        duration_ms: int | None = None,
    ) -> dict | None:
        attempt_id = self._safe_positive_int(ingestion_attempt_id, 0)
        if attempt_id <= 0:
            return None
        status = self._normalize_ingestion_attempt_status(attempt_status, "failed")
        response = None
        connection = None
        try:
            connection = psycopg.connect(conninfo=ConfigManager()._instance.db_conninfo, row_factory=dict_row)
            cursor = connection.cursor()
            logger.debug("PostgreSQL connection established.")
            self._apply_scope_bypass_settings(cursor)
            update_sql = """UPDATE document_ingestion_attempts
            SET attempt_status = %s,
                finished_at = NOW(),
                duration_ms = %s,
                error_message = %s,
                error_context = %s::jsonb,
                stage_events = %s::jsonb
            WHERE ingestion_attempt_id = %s
            RETURNING ingestion_attempt_id, ingestion_job_id, schema_version, owner_member_id, chat_host_id, chat_type,
                community_id, topic_id, platform, attempt_number, attempt_status, worker_id, started_at, finished_at,
                duration_ms, error_message, error_context, stage_events, created_at;"""
            cursor.execute(
                update_sql,
                (
                    status,
                    self._safe_non_negative_int(duration_ms, 0) if duration_ms is not None else None,
                    str(error_message or "").strip() or None,
                    json.dumps(self._optional_dict(error_context)),
                    json.dumps(list(stage_events) if isinstance(stage_events, list) else []),
                    attempt_id,
                ),
            )
            response = cursor.fetchone()
            connection.commit()
            cursor.close()
        except (Exception, psycopg.DatabaseError) as error:
            logger.error(f"Exception while working with psycopg and PostgreSQL:\n{error}")
        finally:
            if connection is not None:
                connection.close()
                logger.debug("PostgreSQL connection is closed.")
            return response

    def getDocumentIngestionAttempts(
        self,
        scope_payload: dict[str, Any],
        *,
        ingestion_job_id: int,
        actor_member_id: int | None = None,
        actor_roles: list[str] | tuple[str, ...] | None = None,
        limit: int | None = None,
    ) -> list:
        job_id = self._safe_positive_int(ingestion_job_id, 0)
        if job_id <= 0:
            return list()
        scope = self._resolved_scope(
            scope_payload,
            actor_member_id=actor_member_id,
            actor_roles=actor_roles,
        )
        query_limit = self._safe_limit(limit, default=20)
        response: list = list()
        connection = None
        try:
            connection = psycopg.connect(conninfo=ConfigManager()._instance.db_conninfo, row_factory=dict_row)
            cursor = connection.cursor()
            logger.debug("PostgreSQL connection established.")
            apply_pg_scope_settings(cursor, scope)
            where_sql, where_params = build_scope_where_clause(scope)
            query_sql = (
                "SELECT ingestion_attempt_id, ingestion_job_id, schema_version, owner_member_id, chat_host_id, chat_type, "
                "community_id, topic_id, platform, attempt_number, attempt_status, worker_id, started_at, finished_at, "
                "duration_ms, error_message, error_context, stage_events, created_at "
                "FROM document_ingestion_attempts WHERE ingestion_job_id = %s AND "
                + where_sql
                + " ORDER BY attempt_number DESC LIMIT %s"
            )
            cursor.execute(query_sql, (job_id, *where_params, query_limit))
            results = cursor.fetchall()
            response = self._accessPolicy.filter_records(results, scope=scope)
            cursor.close()
        except (Exception, psycopg.DatabaseError) as error:
            logger.error(f"Exception while working with psycopg and PostgreSQL:\n{error}")
        finally:
            if connection is not None:
                connection.close()
                logger.debug("PostgreSQL connection is closed.")
            return response

    def getDocumentSources(
        self,
        scope_payload: dict[str, Any],
        *,
        actor_member_id: int | None = None,
        actor_roles: list[str] | tuple[str, ...] | None = None,
        limit: int | None = None,
    ) -> list:
        scope = self._resolved_scope(
            scope_payload,
            actor_member_id=actor_member_id,
            actor_roles=actor_roles,
        )
        query_limit = self._safe_limit(limit, default=25)
        response: list = list()
        connection = None
        try:
            connection = psycopg.connect(conninfo=ConfigManager()._instance.db_conninfo, row_factory=dict_row)
            cursor = connection.cursor()
            logger.debug(f"PostgreSQL connection established.")
            apply_pg_scope_settings(cursor, scope)
            where_sql, where_params = build_scope_where_clause(scope)
            query_sql = (
                "SELECT document_source_id, schema_version, owner_member_id, chat_host_id, chat_type, community_id, "
                "topic_id, platform, source_external_id, source_name, source_mime, source_sha256, source_size_bytes, "
                "source_uri, source_state, source_metadata, created_at, updated_at "
                "FROM document_sources WHERE "
                + where_sql
                + " ORDER BY created_at DESC LIMIT %s"
            )
            cursor.execute(query_sql, (*where_params, query_limit))
            results = cursor.fetchall()
            response = self._accessPolicy.filter_records(results, scope=scope)
            cursor.close()
        except (Exception, psycopg.DatabaseError) as error:
            logger.error(f"Exception while working with psycopg and PostgreSQL:\n{error}")
        finally:
            if connection is not None:
                connection.close()
                logger.debug(f"PostgreSQL connection is closed.")
            return response

    def getDocumentVersions(
        self,
        scope_payload: dict[str, Any],
        *,
        document_source_id: int | None = None,
        actor_member_id: int | None = None,
        actor_roles: list[str] | tuple[str, ...] | None = None,
        limit: int | None = None,
    ) -> list:
        scope = self._resolved_scope(
            scope_payload,
            actor_member_id=actor_member_id,
            actor_roles=actor_roles,
        )
        query_limit = self._safe_limit(limit, default=25)
        response: list = list()
        connection = None
        try:
            connection = psycopg.connect(conninfo=ConfigManager()._instance.db_conninfo, row_factory=dict_row)
            cursor = connection.cursor()
            logger.debug(f"PostgreSQL connection established.")
            apply_pg_scope_settings(cursor, scope)
            where_sql, where_params = build_scope_where_clause(scope)
            query_sql = (
                "SELECT document_version_id, document_source_id, schema_version, owner_member_id, chat_host_id, chat_type, "
                "community_id, topic_id, platform, version_number, source_sha256, parser_name, parser_version, parser_status, "
                "parse_artifact, record_metadata, created_at "
                "FROM document_versions WHERE "
                + where_sql
            )
            params: list[Any] = list(where_params)
            if document_source_id is not None:
                try:
                    source_id = int(document_source_id)
                except (TypeError, ValueError):
                    source_id = 0
                if source_id <= 0:
                    return list()
                query_sql += " AND document_source_id = %s"
                params.append(source_id)
            query_sql += " ORDER BY created_at DESC LIMIT %s"
            params.append(query_limit)
            cursor.execute(query_sql, tuple(params))
            results = cursor.fetchall()
            response = self._accessPolicy.filter_records(results, scope=scope)
            cursor.close()
        except (Exception, psycopg.DatabaseError) as error:
            logger.error(f"Exception while working with psycopg and PostgreSQL:\n{error}")
        finally:
            if connection is not None:
                connection.close()
                logger.debug(f"PostgreSQL connection is closed.")
            return response

    def replaceDocumentVersionTree(
        self,
        document_version_id: int,
        *,
        scope_payload: dict[str, Any],
        nodes: list[dict[str, Any]] | None,
        edges: list[dict[str, Any]] | None = None,
        actor_member_id: int | None = None,
        actor_roles: list[str] | tuple[str, ...] | None = None,
    ) -> dict[str, Any] | None:
        try:
            version_id = int(document_version_id)
        except (TypeError, ValueError):
            version_id = 0
        if version_id <= 0:
            raise ValueError("document_version_id must be greater than zero.")

        scope = self._resolved_scope(
            scope_payload,
            actor_member_id=actor_member_id,
            actor_roles=actor_roles,
        )
        scope_dict = scope.to_dict()

        prepared_nodes: list[dict[str, Any]] = []
        parent_node_key_by_key: dict[str, str | None] = {}
        seen_node_keys: set[str] = set()
        for raw in list(nodes or []):
            source = dict(raw) if isinstance(raw, dict) else {}
            parent_node_key = str(source.get("parent_node_key") or "").strip() or None
            payload = dict(source)
            payload["schema_version"] = int(payload.get("schema_version", 1) or 1)
            payload["scope"] = scope_dict
            payload["document_version_id"] = version_id
            payload["parent_node_id"] = None
            payload["path"] = str(payload.get("path") or payload.get("node_path") or "").strip()
            payload["node_metadata"] = self._optional_dict(payload.get("node_metadata"))
            contract = validate_document_node_contract(payload)
            if contract.node_key in seen_node_keys:
                raise ValueError(f"duplicate node_key in tree payload: {contract.node_key}")
            seen_node_keys.add(contract.node_key)
            parent_node_key_by_key[contract.node_key] = parent_node_key
            prepared_nodes.append(
                {
                    "contract": contract,
                    "parent_node_key": parent_node_key,
                }
            )

        prepared_edges: list[dict[str, Any]] = []
        for raw in list(edges or []):
            source = dict(raw) if isinstance(raw, dict) else {}
            source_node_key = str(source.get("source_node_key") or "").strip()
            target_node_key = str(source.get("target_node_key") or "").strip()
            edge_type = str(source.get("edge_type") or "").strip().lower()
            if not source_node_key or not target_node_key or not edge_type:
                continue
            prepared_edges.append(
                {
                    "source_node_key": source_node_key,
                    "target_node_key": target_node_key,
                    "edge_type": edge_type,
                    "ordinal": int(source.get("ordinal", 0) or 0),
                    "edge_metadata": self._optional_dict(source.get("edge_metadata")),
                }
            )

        response: dict[str, Any] = {
            "document_version_id": version_id,
            "node_count": 0,
            "edge_count": 0,
            "nodes": [],
            "edges": [],
        }
        connection = None
        try:
            connection = psycopg.connect(conninfo=ConfigManager()._instance.db_conninfo, row_factory=dict_row)
            cursor = connection.cursor()
            logger.debug("PostgreSQL connection established.")
            apply_pg_scope_settings(cursor, scope)
            self._assert_scoped_version_exists(cursor, version_id, scope)

            cursor.execute("DELETE FROM document_node_edges WHERE document_version_id = %s", (version_id,))
            cursor.execute("DELETE FROM document_nodes WHERE document_version_id = %s", (version_id,))

            if not prepared_nodes:
                connection.commit()
                cursor.close()
                return response

            node_id_by_key: dict[str, int] = {}
            insert_node_sql = """INSERT INTO document_nodes (
                document_version_id, parent_node_id, schema_version, owner_member_id, chat_host_id, chat_type,
                community_id, topic_id, platform, node_key, node_type, node_title, ordinal, token_count,
                page_start, page_end, char_start, char_end, node_path, node_metadata
            ) VALUES (%s, NULL, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING document_node_id, node_key;"""
            for item in prepared_nodes:
                contract = item["contract"]
                cursor.execute(
                    insert_node_sql,
                    (
                        version_id,
                        contract.schema_version,
                        scope.owner_member_id,
                        scope.chat_host_id,
                        scope.chat_type,
                        scope.community_id,
                        scope.topic_id,
                        scope.platform,
                        contract.node_key,
                        contract.node_type,
                        contract.node_title or None,
                        contract.ordinal,
                        contract.token_count,
                        contract.page_start,
                        contract.page_end,
                        contract.char_start,
                        contract.char_end,
                        contract.path or None,
                        json.dumps(contract.node_metadata),
                    ),
                )
                node_row = cursor.fetchone()
                if isinstance(node_row, dict):
                    node_id = int(node_row.get("document_node_id", 0) or 0)
                    if node_id > 0:
                        node_id_by_key[str(node_row.get("node_key") or contract.node_key)] = node_id

            update_parent_sql = """UPDATE document_nodes
            SET parent_node_id = %s
            WHERE document_node_id = %s;"""
            for item in prepared_nodes:
                contract = item["contract"]
                parent_key = str(item.get("parent_node_key") or "").strip()
                if not parent_key:
                    continue
                parent_id = int(node_id_by_key.get(parent_key, 0) or 0)
                child_id = int(node_id_by_key.get(contract.node_key, 0) or 0)
                if parent_id <= 0 or child_id <= 0:
                    continue
                cursor.execute(update_parent_sql, (parent_id, child_id))

            select_nodes_sql = """SELECT document_node_id, document_version_id, parent_node_id, schema_version,
                owner_member_id, chat_host_id, chat_type, community_id, topic_id, platform, node_key, node_type,
                node_title, ordinal, token_count, page_start, page_end, char_start, char_end, node_path AS path,
                node_metadata, created_at
            FROM document_nodes
            WHERE document_version_id = %s
            ORDER BY created_at ASC, document_node_id ASC;"""
            cursor.execute(select_nodes_sql, (version_id,))
            node_rows = cursor.fetchall()
            response["nodes"] = list(node_rows or [])
            response["node_count"] = len(response["nodes"])

            node_key_by_id: dict[int, str] = {}
            for row in response["nodes"]:
                if not isinstance(row, dict):
                    continue
                node_id = int(row.get("document_node_id", 0) or 0)
                node_key = str(row.get("node_key") or "").strip()
                if node_id > 0 and node_key:
                    node_key_by_id[node_id] = node_key
                    node_id_by_key[node_key] = node_id

            if not prepared_edges:
                children_by_parent: dict[int, list[dict[str, Any]]] = {}
                for row in response["nodes"]:
                    if not isinstance(row, dict):
                        continue
                    parent_id = row.get("parent_node_id")
                    if parent_id is None:
                        continue
                    try:
                        normalized_parent = int(parent_id)
                    except (TypeError, ValueError):
                        continue
                    children_by_parent.setdefault(normalized_parent, []).append(dict(row))

                for parent_id, children in children_by_parent.items():
                    sorted_children = sorted(
                        children,
                        key=lambda item: (
                            int(item.get("ordinal", 0) or 0),
                            int(item.get("document_node_id", 0) or 0),
                        ),
                    )
                    parent_key = node_key_by_id.get(parent_id, "")
                    if not parent_key:
                        continue
                    for child_index, child in enumerate(sorted_children):
                        child_key = str(child.get("node_key") or "").strip()
                        if not child_key:
                            continue
                        prepared_edges.append(
                            {
                                "source_node_key": parent_key,
                                "target_node_key": child_key,
                                "edge_type": "parent_child",
                                "ordinal": int(child.get("ordinal", child_index) or child_index),
                                "edge_metadata": {},
                            }
                        )
                    for sibling_index in range(1, len(sorted_children)):
                        prev_key = str(sorted_children[sibling_index - 1].get("node_key") or "").strip()
                        curr_key = str(sorted_children[sibling_index].get("node_key") or "").strip()
                        if not prev_key or not curr_key:
                            continue
                        prepared_edges.append(
                            {
                                "source_node_key": prev_key,
                                "target_node_key": curr_key,
                                "edge_type": "next_sibling",
                                "ordinal": sibling_index,
                                "edge_metadata": {},
                            }
                        )

            insert_edge_sql = """INSERT INTO document_node_edges (
                document_version_id, source_node_id, target_node_id, schema_version, owner_member_id, chat_host_id,
                chat_type, community_id, topic_id, platform, edge_type, ordinal, edge_metadata
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (document_version_id, source_node_id, target_node_id, edge_type)
            DO UPDATE SET
                ordinal = EXCLUDED.ordinal,
                edge_metadata = EXCLUDED.edge_metadata
            RETURNING document_node_edge_id, document_version_id, source_node_id, target_node_id, schema_version,
                owner_member_id, chat_host_id, chat_type, community_id, topic_id, platform, edge_type, ordinal,
                edge_metadata, created_at;"""
            seen_edge_keys: set[tuple[int, int, str]] = set()
            edge_rows: list[dict[str, Any]] = []
            for edge in prepared_edges:
                source_key = str(edge.get("source_node_key") or "").strip()
                target_key = str(edge.get("target_node_key") or "").strip()
                source_id = int(node_id_by_key.get(source_key, 0) or 0)
                target_id = int(node_id_by_key.get(target_key, 0) or 0)
                if source_id <= 0 or target_id <= 0:
                    continue
                edge_payload = {
                    "schema_version": int(edge.get("schema_version", 1) or 1),
                    "scope": scope_dict,
                    "document_version_id": version_id,
                    "source_node_id": source_id,
                    "target_node_id": target_id,
                    "edge_type": str(edge.get("edge_type") or "").strip().lower(),
                    "ordinal": int(edge.get("ordinal", 0) or 0),
                    "edge_metadata": self._optional_dict(edge.get("edge_metadata")),
                }
                contract = validate_document_node_edge_contract(edge_payload)
                dedupe_key = (contract.source_node_id, contract.target_node_id, contract.edge_type)
                if dedupe_key in seen_edge_keys:
                    continue
                seen_edge_keys.add(dedupe_key)
                cursor.execute(
                    insert_edge_sql,
                    (
                        version_id,
                        contract.source_node_id,
                        contract.target_node_id,
                        contract.schema_version,
                        scope.owner_member_id,
                        scope.chat_host_id,
                        scope.chat_type,
                        scope.community_id,
                        scope.topic_id,
                        scope.platform,
                        contract.edge_type,
                        contract.ordinal,
                        json.dumps(contract.edge_metadata),
                    ),
                )
                edge_row = cursor.fetchone()
                if isinstance(edge_row, dict):
                    source_edge_id = int(edge_row.get("source_node_id", 0) or 0)
                    target_edge_id = int(edge_row.get("target_node_id", 0) or 0)
                    edge_row["source_node_key"] = node_key_by_id.get(source_edge_id)
                    edge_row["target_node_key"] = node_key_by_id.get(target_edge_id)
                    edge_rows.append(edge_row)

            response["edges"] = edge_rows
            response["edge_count"] = len(edge_rows)
            connection.commit()
            cursor.close()
        except (Exception, psycopg.DatabaseError) as error:
            logger.error(f"Exception while working with psycopg and PostgreSQL:\n{error}")
            response = None
        finally:
            if connection is not None:
                connection.close()
                logger.debug("PostgreSQL connection is closed.")
            return response

    def getDocumentNodes(
        self,
        scope_payload: dict[str, Any],
        *,
        document_version_id: int,
        actor_member_id: int | None = None,
        actor_roles: list[str] | tuple[str, ...] | None = None,
        limit: int | None = None,
    ) -> list:
        try:
            version_id = int(document_version_id)
        except (TypeError, ValueError):
            version_id = 0
        if version_id <= 0:
            return list()
        scope = self._resolved_scope(
            scope_payload,
            actor_member_id=actor_member_id,
            actor_roles=actor_roles,
        )
        query_limit = self._safe_limit(limit, default=500, max_limit=5000)
        response: list = list()
        connection = None
        try:
            connection = psycopg.connect(conninfo=ConfigManager()._instance.db_conninfo, row_factory=dict_row)
            cursor = connection.cursor()
            logger.debug("PostgreSQL connection established.")
            apply_pg_scope_settings(cursor, scope)
            where_sql, where_params = build_scope_where_clause(scope)
            query_sql = (
                "SELECT document_node_id, document_version_id, parent_node_id, schema_version, owner_member_id, "
                "chat_host_id, chat_type, community_id, topic_id, platform, node_key, node_type, node_title, "
                "ordinal, token_count, page_start, page_end, char_start, char_end, node_path AS path, "
                "node_metadata, created_at "
                "FROM document_nodes WHERE document_version_id = %s AND "
                + where_sql
                + " ORDER BY created_at ASC, document_node_id ASC LIMIT %s"
            )
            cursor.execute(query_sql, (version_id, *where_params, query_limit))
            results = cursor.fetchall()
            response = self._accessPolicy.filter_records(results, scope=scope)
            cursor.close()
        except (Exception, psycopg.DatabaseError) as error:
            logger.error(f"Exception while working with psycopg and PostgreSQL:\n{error}")
        finally:
            if connection is not None:
                connection.close()
                logger.debug("PostgreSQL connection is closed.")
            return response

    def getDocumentNodeEdges(
        self,
        scope_payload: dict[str, Any],
        *,
        document_version_id: int,
        edge_types: list[str] | tuple[str, ...] | None = None,
        actor_member_id: int | None = None,
        actor_roles: list[str] | tuple[str, ...] | None = None,
        limit: int | None = None,
    ) -> list:
        try:
            version_id = int(document_version_id)
        except (TypeError, ValueError):
            version_id = 0
        if version_id <= 0:
            return list()
        scope = self._resolved_scope(
            scope_payload,
            actor_member_id=actor_member_id,
            actor_roles=actor_roles,
        )
        query_limit = self._safe_limit(limit, default=1000, max_limit=10000)
        response: list = list()
        connection = None
        try:
            connection = psycopg.connect(conninfo=ConfigManager()._instance.db_conninfo, row_factory=dict_row)
            cursor = connection.cursor()
            logger.debug("PostgreSQL connection established.")
            apply_pg_scope_settings(cursor, scope)
            where_sql, where_params = build_scope_where_clause(scope, table_alias="e")
            query_sql = (
                "SELECT e.document_node_edge_id, e.document_version_id, e.source_node_id, src.node_key AS source_node_key, "
                "e.target_node_id, tgt.node_key AS target_node_key, e.schema_version, e.owner_member_id, e.chat_host_id, "
                "e.chat_type, e.community_id, e.topic_id, e.platform, e.edge_type, e.ordinal, e.edge_metadata, e.created_at "
                "FROM document_node_edges e "
                "LEFT JOIN document_nodes src ON src.document_node_id = e.source_node_id "
                "LEFT JOIN document_nodes tgt ON tgt.document_node_id = e.target_node_id "
                "WHERE e.document_version_id = %s AND "
                + where_sql
            )
            params: list[Any] = [version_id, *where_params]
            normalized_edge_types = [str(item).strip().lower() for item in list(edge_types or []) if str(item).strip()]
            if normalized_edge_types:
                query_sql += " AND e.edge_type = ANY(%s)"
                params.append(normalized_edge_types)
            query_sql += " ORDER BY e.created_at ASC, e.document_node_edge_id ASC LIMIT %s"
            params.append(query_limit)
            cursor.execute(query_sql, tuple(params))
            results = cursor.fetchall()
            response = self._accessPolicy.filter_records(results, scope=scope)
            cursor.close()
        except (Exception, psycopg.DatabaseError) as error:
            logger.error(f"Exception while working with psycopg and PostgreSQL:\n{error}")
        finally:
            if connection is not None:
                connection.close()
                logger.debug("PostgreSQL connection is closed.")
            return response

    def getDocumentTreePreview(
        self,
        scope_payload: dict[str, Any],
        *,
        document_version_id: int,
        actor_member_id: int | None = None,
        actor_roles: list[str] | tuple[str, ...] | None = None,
        node_limit: int | None = None,
        edge_limit: int | None = None,
    ) -> dict[str, Any]:
        nodes = self.getDocumentNodes(
            scope_payload,
            document_version_id=document_version_id,
            actor_member_id=actor_member_id,
            actor_roles=actor_roles,
            limit=node_limit,
        )
        edges = self.getDocumentNodeEdges(
            scope_payload,
            document_version_id=document_version_id,
            actor_member_id=actor_member_id,
            actor_roles=actor_roles,
            limit=edge_limit,
        )
        try:
            version_id = int(document_version_id)
        except (TypeError, ValueError):
            version_id = 0
        return {
            "document_version_id": version_id,
            "nodes": nodes,
            "edges": edges,
            "node_count": len(nodes),
            "edge_count": len(edges),
        }

    def getDocumentStorageObjects(
        self,
        scope_payload: dict[str, Any],
        *,
        document_source_id: int | None = None,
        actor_member_id: int | None = None,
        actor_roles: list[str] | tuple[str, ...] | None = None,
        limit: int | None = None,
    ) -> list:
        scope = self._resolved_scope(
            scope_payload,
            actor_member_id=actor_member_id,
            actor_roles=actor_roles,
        )
        query_limit = self._safe_limit(limit, default=25)
        response: list = list()
        connection = None
        try:
            connection = psycopg.connect(conninfo=ConfigManager()._instance.db_conninfo, row_factory=dict_row)
            cursor = connection.cursor()
            logger.debug(f"PostgreSQL connection established.")
            apply_pg_scope_settings(cursor, scope)
            where_sql, where_params = build_scope_where_clause(scope)
            query_sql = (
                "SELECT storage_object_id, document_source_id, schema_version, owner_member_id, chat_host_id, chat_type, "
                "community_id, topic_id, platform, storage_backend, storage_key, storage_path, object_state, file_name, "
                "file_mime, file_sha256, file_size_bytes, dedupe_status, retention_until, deleted_at, record_metadata, "
                "created_at, updated_at "
                "FROM document_storage_objects WHERE "
                + where_sql
            )
            params: list[Any] = list(where_params)
            if document_source_id is not None:
                try:
                    source_id = int(document_source_id)
                except (TypeError, ValueError):
                    source_id = 0
                if source_id <= 0:
                    return list()
                query_sql += " AND document_source_id = %s"
                params.append(source_id)
            query_sql += " ORDER BY created_at DESC LIMIT %s"
            params.append(query_limit)
            cursor.execute(query_sql, tuple(params))
            results = cursor.fetchall()
            response = self._accessPolicy.filter_records(results, scope=scope)
            cursor.close()
        except (Exception, psycopg.DatabaseError) as error:
            logger.error(f"Exception while working with psycopg and PostgreSQL:\n{error}")
        finally:
            if connection is not None:
                connection.close()
                logger.debug(f"PostgreSQL connection is closed.")
            return response

    def getDocumentRetrievalEvents(
        self,
        scope_payload: dict[str, Any],
        *,
        actor_member_id: int | None = None,
        actor_roles: list[str] | tuple[str, ...] | None = None,
        limit: int | None = None,
    ) -> list:
        scope = self._resolved_scope(
            scope_payload,
            actor_member_id=actor_member_id,
            actor_roles=actor_roles,
        )
        query_limit = self._safe_limit(limit, default=50)
        response: list = list()
        connection = None
        try:
            connection = psycopg.connect(conninfo=ConfigManager()._instance.db_conninfo, row_factory=dict_row)
            cursor = connection.cursor()
            logger.debug(f"PostgreSQL connection established.")
            apply_pg_scope_settings(cursor, scope)
            where_sql, where_params = build_scope_where_clause(scope)
            query_sql = (
                "SELECT retrieval_event_id, schema_version, owner_member_id, chat_host_id, chat_type, community_id, "
                "topic_id, platform, request_id, query_text, document_source_id, document_version_id, result_count, "
                "max_distance, query_metadata, retrieval_metadata, citations, created_at "
                "FROM document_retrieval_events WHERE "
                + where_sql
                + " ORDER BY created_at DESC LIMIT %s"
            )
            cursor.execute(query_sql, (*where_params, query_limit))
            results = cursor.fetchall()
            response = self._accessPolicy.filter_records(results, scope=scope)
            cursor.close()
        except (Exception, psycopg.DatabaseError) as error:
            logger.error(f"Exception while working with psycopg and PostgreSQL:\n{error}")
        finally:
            if connection is not None:
                connection.close()
                logger.debug(f"PostgreSQL connection is closed.")
            return response


class ProposalManager:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ProposalManager, cls).__new__(cls)
            # Intialize the new singleton

            # Create the proposal tables if they don't exist yet
            connection = None
            try:
                connection = psycopg.connect(conninfo=ConfigManager()._instance.db_conninfo)
                cursor = connection.cursor()
                logger.debug(f"PostgreSQL connection established.")

                # Create the proposals table if it doesn't exist
                execute_migration(cursor, "050_proposals.sql")

                # Create the proposal disclosure table if it doesn't exist
                execute_migration(cursor, "051_proposal_disclosure.sql")
                
                connection.commit()
                # Close the cursor
                cursor.close()
            except psycopg.Error as error:
                logger.error(f"Exception while working with psycopg and PostgreSQL:\n{error}")
            finally:
                if (connection):
                    connection.close()
                    logger.debug(f"PostgreSQL connection is closed.")
        
        return cls._instance

    def getProposals(self) -> list:
        logger.info(f"Getting the proposals list.")

        try:
            connection = psycopg.connect(conninfo=ConfigManager()._instance.db_conninfo)
            cursor = connection.cursor()
            logger.debug(f"PostgreSQL connection established.")
            
            usageQuery_sql = "SELECT rowid, * FROM proposals;"
            cursor.execute(usageQuery_sql)
            results = cursor.fetchall()
            # Close the cursor
            cursor.close()

            proposalList = []

            for result in results:
                proposal = {
                    "project_id": result[0],
                    "submitted_from": result[1],
                    "project_title": result[2],
                    "project_description": result[3],
                    "filename": result[4],
                    "submit_date": result[5],
                }
                proposalList.append(proposal)

            return proposalList

        except psycopg.Error as error:
            logger.error(f"Exception while working with psycopg and PostgreSQL:\n{error}")
            return list()
        finally:
            if (connection):
                connection.close()
                logger.debug(f"PostgreSQL connection is closed.")

    def addDisclosureAgreement(self, userID: int, proposalID: int):
        logger.info(f"Adding a new discolure agreement for a proposal.")
          
        try:
            connection = psycopg.connect(conninfo=ConfigManager()._instance.db_conninfo)
            cursor = connection.cursor()
            logger.debug(f"PostgreSQL connection established.")
            
            addAgreement_sql = "INSERT INTO proposal_disclosure (user_id, proposal_id, agreement_date) VALUES (%s, %s, %s);"
            cursor.execute(addAgreement_sql, (userID, proposalID, datetime.now()))
            
            connection.commit()
            # Close the cursor
            cursor.close()

        except psycopg.Error as error:
            logger.error(f"Exception while working with psycopg and PostgreSQL:\n{error}")
        finally:
            if (connection):
                connection.close()
                logger.debug(f"PostgreSQL connection is closed.")


class SpamManager:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(SpamManager, cls).__new__(cls)
            # Intialize the new singleton

            connection = None
            try:
                connection = psycopg.connect(conninfo=ConfigManager()._instance.db_conninfo)
                cursor = connection.cursor()
                logger.debug(f"PostgreSQL connection established.")
                vectorDimensions = max(1, ConfigManager().runtimeInt("vectors.embedding_dimensions", 768))
                
                # Create the knowledge table if it does not exist
                execute_migration(
                    cursor,
                    "060_spam.sql",
                    context={"vector_dimensions": vectorDimensions},
                )
                connection.commit()

                # close the communication with the PostgreSQL
                cursor.close()
            except (Exception, psycopg.DatabaseError) as error:
                logger.error(f"Exception while working with psycopg and PostgreSQL:\n{error}")
            finally:
                if connection is not None:
                    connection.close()
                    logger.debug(f"PostgreSQL connection is closed.")

        return cls._instance

    def addSpamText(self, spamText: str, addedBy: int) -> int:
        logger.info(f"Adding a new spam message.")
        embedding = getEmbeddings(spamText)
        # TODO Validate member data
        member = MemberManager().getMemberByID(addedBy)
        if member is None:
            return None
        
        connection = None
        recordID = None
        try:
            connection = psycopg.connect(conninfo=ConfigManager()._instance.db_conninfo)
            cursor = connection.cursor()
            logger.debug(f"PostgreSQL connection established.")

            insertSQL = """INSERT INTO spam (spam_text, embeddings, record_timestamp, added_by)
            VALUES (%s, %s, %s, %s)
            RETURNING spam_id;"""
            cursor.execute(insertSQL, (spamText, embedding, datetime.now(), addedBy))
            recordID = cursor.fetchone()[0]
            connection.commit()
            # close the communication with the PostgreSQL
            cursor.close()
        except (Exception, psycopg.DatabaseError) as error:
            logger.error(f"Exception while working with psycopg and PostgreSQL:\n{error}")
        finally:
            if connection is not None:
                connection.close()
                logger.debug(f"PostgreSQL connection is closed.")

            return recordID

    def getSpam(self) -> list:
        logger.info(f"Get spam messages.")

        response = None
        connection = None
        try:
            connection = psycopg.connect(conninfo=ConfigManager()._instance.db_conninfo, row_factory=dict_row)
            cursor = connection.cursor()
            logger.debug(f"PostgreSQL connection established.")

            queryLimit = max(1, ConfigManager().runtimeInt("retrieval.spam_list_limit", 10))
            querySQL = """SELECT spam_id, spam_text, record_timestamp, added_by
            FROM spam
            LIMIT %s"""
            cursor.execute(querySQL, (queryLimit,))
            results = cursor.fetchall()

            response = results
            # close the communication with the PostgreSQL
            cursor.close()
        except (Exception, psycopg.DatabaseError) as error:
            logger.error(f"Exception while working with psycopg and PostgreSQL:\n{error}")
        finally:
            if connection is not None:
                connection.close()
                logger.debug(f"PostgreSQL connection is closed.")
            
            return response

    def searchSpam(self, text: str, limit: int | None = None) -> list:
        logger.info(f"Searching spam messages.")
        embedding = getEmbeddings(text)
        if embedding is None:
            logger.warning("Skipping spam search because embeddings are unavailable.")
            return list()
        if limit is None:
            limit = ConfigManager().runtimeInt("retrieval.spam_search_default_limit", 1)

        connection = None
        response = None
        try:
            connection = psycopg.connect(conninfo=ConfigManager()._instance.db_conninfo, row_factory=dict_row)
            cursor = connection.cursor()
            logger.debug(f"PostgreSQL connection established.")

            querySQL = """SELECT spam_id, spam_text, embeddings <-> %s::vector AS distance, record_timestamp, added_by
            FROM spam
            ORDER BY distance
            LIMIT %s"""
            cursor.execute(querySQL, (embedding, max(1, int(limit))))
            results = cursor.fetchall()

            response = results
            # close the communication with the PostgreSQL
            cursor.close()
        except (Exception, psycopg.DatabaseError) as error:
            logger.error(f"Exception while working with psycopg and PostgreSQL:\n{error}")
        finally:
            if connection is not None:
                connection.close()
                logger.debug(f"PostgreSQL connection is closed.")
            
            return response


class UsageManager:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(UsageManager, cls).__new__(cls)
            # Intialize the new singleton

            # Create the usage table if it doesn't exist yet
            connection = None
            try:
                connection = psycopg.connect(conninfo=ConfigManager()._instance.db_conninfo)
                cursor = connection.cursor()
                logger.debug(f"PostgreSQL connection established.")
                
                # Create the individual accounts table if it doesn't exist
                execute_migration(cursor, "070_inference_usage.sql")
                
                connection.commit()
                # Close the cursor
                cursor.close()
            except psycopg.Error as error:
                logger.error(f"Exception while working with psycopg and PostgreSQL:\n{error}")
            finally:
                if (connection):
                    connection.close()
                    logger.debug(f"PostgreSQL connection is closed.")
        
        return cls._instance

    @staticmethod
    def _usage_stat_int(stats: dict | None, key: str, default: int = 0) -> int:
        payload = stats if isinstance(stats, dict) else {}
        raw_value = payload.get(key, default)
        if raw_value is None:
            return int(default)
        try:
            value = int(raw_value)
        except (TypeError, ValueError):
            return int(default)
        return max(0, value)

    def addUsage(self, promptHistoryID: int, responseHistoryID: int, stats: dict):
        logger.info("Add usage data.")
        if promptHistoryID is None or responseHistoryID is None:
            logger.warning(
                "Skipping usage insert because prompt/response history id is missing "
                f"(prompt={promptHistoryID}, response={responseHistoryID})."
            )
            return
        
        try:
            connection = psycopg.connect(conninfo=ConfigManager()._instance.db_conninfo)
            cursor = connection.cursor()
            logger.debug(f"PostgreSQL connection established.")

            loadDuration = self._usage_stat_int(stats, "load_duration", 0)
            promptEvalCount = self._usage_stat_int(stats, "prompt_eval_count", 0)
            promptEvalDuration = self._usage_stat_int(stats, "prompt_eval_duration", 0)
            evalCount = self._usage_stat_int(stats, "eval_count", 0)
            evalDuration = self._usage_stat_int(stats, "eval_duration", 0)
            totalDuration = self._usage_stat_int(stats, "total_duration", 0)
            
            addUsage_sql = """INSERT INTO inference_usage (prompt_history_id, response_history_id, load_duration, prompt_eval_count, prompt_eval_duration, eval_count, eval_duration, total_duration) 
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s);"""
            cursor.execute(
                addUsage_sql,
                (
                    promptHistoryID,
                    responseHistoryID,
                    loadDuration,
                    promptEvalCount,
                    promptEvalDuration,
                    evalCount,
                    evalDuration,
                    totalDuration,
                ),
            )
            
            connection.commit()
            # Close the cursor
            cursor.close()

        except psycopg.Error as error:
            logger.error(f"Exception while working with psycopg and PostgreSQL:\n{error}")
        finally:
            if (connection):
                connection.close()
                logger.debug(f"PostgreSQL connection is closed.")

    def getUsageForMember(self, MemberID: int, timeInHours: int = 1) -> list:
        logger.info(f"Get a list of usage data for user.")

        connection = None
        response = list()
        if MemberID is None:
            return response
        try:
            connection = psycopg.connect(conninfo=ConfigManager()._instance.db_conninfo, row_factory=dict_row)
            cursor = connection.cursor()
            logger.debug(f"PostgreSQL connection established.")

            beginSearch_dt = datetime.now() - timedelta(hours=timeInHours)
            
            usageQuery_sql = """SELECT * 
            FROM inference_usage AS iu
            JOIN chat_history AS ch
            ON iu.prompt_history_id = ch.history_id
            WHERE ch.member_id = %s 
            AND ch.message_timestamp > %s;"""
            cursor.execute(usageQuery_sql, (MemberID, beginSearch_dt))
            results = cursor.fetchall()
            response = results
            # Close the cursor
            cursor.close()
        except psycopg.Error as error:
            logger.error(f"Exception while working with psycopg and PostgreSQL:\n{error}")
        finally:
            if (connection):
                connection.close()
                logger.debug(f"PostgreSQL connection is closed.")
            
            return response



####################
# HELPER FUNCTIONS #
####################

def getEmbeddings(text: str) -> list:
    logger.info("Getting embeddings.")
    promptText = str(text or "").strip()
    if promptText == "":
        logger.debug("Skipping embeddings request because input text is empty.")
        return None

    configManager = ConfigManager()
    embeddingConfig = configManager.inference.get("embedding", {})
    host = embeddingConfig.get("url") if embeddingConfig.get("url") else configManager.runtimeValue(
        "inference.default_ollama_host", "http://127.0.0.1:11434"
    )
    host = str(host or "http://127.0.0.1:11434").rstrip("/")

    probeTimeout = configManager.runtimeFloat("inference.probe_timeout_seconds", 3.0)
    embeddingTimeout = configManager.runtimeFloat("inference.embedding_timeout_seconds", max(3.0, probeTimeout))
    if embeddingTimeout <= 0:
        embeddingTimeout = max(3.0, probeTimeout)

    maxInputChars = configManager.runtimeInt("inference.embedding_max_input_chars", 6000)
    if maxInputChars > 0 and len(promptText) > maxInputChars:
        logger.info(
            f"Truncating embedding input from {len(promptText)} to {maxInputChars} chars."
        )
        promptText = promptText[:maxInputChars]

    configuredModel = embeddingConfig.get("model")
    fallbackModel = configManager.runtimeValue("inference.default_embedding_model", "nomic-embed-text:latest")
    fallbackModel = fallbackModel or "nomic-embed-text:latest"
    candidateModels = list()
    for modelName in (configuredModel, fallbackModel, "nomic-embed-text:latest"):
        if isinstance(modelName, str):
            cleaned = modelName.strip()
            if cleaned and cleaned not in candidateModels:
                candidateModels.append(cleaned)

    if not candidateModels:
        logger.error("Embedding model is missing from config inference settings.")
        return None

    lastError = None
    for index, model in enumerate(candidateModels):
        started = time.monotonic()
        try:
            results = Client(host=host, timeout=float(embeddingTimeout)).embeddings(
                model=model,
                prompt=promptText,
            )
            embeddingVector = getattr(results, "embedding", None)
            if embeddingVector is None and isinstance(results, dict):
                embeddingVector = results.get("embedding")
            if not isinstance(embeddingVector, list) or len(embeddingVector) == 0:
                raise ValueError("embedding response did not include a vector payload")

            elapsed = time.monotonic() - started
            if index > 0:
                logger.warning(f"Embedding model fallback succeeded with: {model} ({elapsed:.2f}s)")
            else:
                logger.debug(f"Embedding model '{model}' completed in {elapsed:.2f}s")
            return embeddingVector
        except Exception as error:  # noqa: BLE001
            elapsed = time.monotonic() - started
            lastError = error
            if index < (len(candidateModels) - 1):
                logger.warning(
                    f"Embedding model '{model}' failed after {elapsed:.2f}s "
                    f"(timeout={embeddingTimeout:.1f}s); trying fallback model."
                )
                continue
            logger.error(
                f"Exception while getting embeddings from Ollama after {elapsed:.2f}s "
                f"(timeout={embeddingTimeout:.1f}s):\n{error}"
            )

    if lastError is not None:
        logger.debug(f"Last embedding model error: {lastError}")
    return None
