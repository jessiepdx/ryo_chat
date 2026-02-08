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
import re
import secrets
import string
import textstat
import time
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from hypermindlabs.database_router import DatabaseRouter
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
    format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s (%(filename)s:%(lineno)d)"

    FORMATS = {
        logging.DEBUG: grey + format + reset,
        logging.INFO: grey + format + reset,
        logging.WARNING: yellow + format + reset,
        logging.ERROR: red + format + reset,
        logging.CRITICAL: bold_red + format + reset
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
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
                    logger.info(f"There are {rowCount.get("total_members")} registered users.")
                
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
    
    def addChatHistory(self, messageID: int, messageText: str, platform: str, memberID: int = None, communityID: int = None, chatHostID: int = None, topicID: int = None, timestamp: datetime = datetime.now()) -> int:
        logger.info(f"Adding a new chat history record.")
        chatHostID = chatHostID if chatHostID else communityID if communityID else memberID
        if not chatHostID:
            return
        
        chatType = "community" if communityID is not None else "member"
        
        embedding = getEmbeddings(messageText)

        connection = None
        response = None
        try:
            connection = psycopg.connect(conninfo=ConfigManager()._instance.db_conninfo, row_factory=dict_row)
            cursor = connection.cursor()
            logger.debug(f"PostgreSQL connection established.")

            insertHistory_sql = """INSERT INTO chat_history (member_id, community_id, chat_host_id, topic_id, chat_type, platform, message_id, message_text, message_timestamp)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING history_id;"""
            cursor.execute(insertHistory_sql, (memberID, communityID, chatHostID, topicID, chatType, platform, messageID, messageText, timestamp))
            result = cursor.fetchone()
            historyID = result.get("history_id")
            response = historyID

            connection.commit()

            if embedding is not None:
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
            
            beginQuery_sql = """SELECT history_id, member_id, message_id, message_text, message_timestamp
                FROM chat_history
                WHERE chat_host_id = %s
                AND chat_type = %s
                AND platform = %s
                AND message_timestamp > %s"""
            
            endQuery_sql = " ORDER BY message_timestamp"

            valueArray = [chatHostID, chatType, platform, timePeriod]

            if topicID:
                beginQuery_sql = beginQuery_sql + " AND topic_id = %s"
                valueArray.append(topicID)
            else:
                beginQuery_sql = beginQuery_sql + " AND topic_id IS NULL"
            if int(limit) > 0:
                endQuery_sql += " LIMIT %s"
                valueArray.append(int(limit))

            historyQuery_sql = beginQuery_sql + endQuery_sql + ";"
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
            
            beginQuery_sql = """SELECT ch.history_id, ch.message_id, ch.message_text, ch.message_timestamp, mem.member_id, mem.first_name, mem.last_name
                FROM chat_history AS ch
                LEFT JOIN member_data AS mem
                ON ch.member_id = mem.member_id
                WHERE chat_host_id = %s
                AND chat_type = %s
                AND platform = %s
                AND message_timestamp > %s"""
            
            endQuery_sql = " ORDER BY message_timestamp"

            valueArray = [chatHostID, chatType, platform, timePeriod]

            if topicID:
                beginQuery_sql = beginQuery_sql + " AND topic_id = %s"
                valueArray.append(topicID)
            else:
                beginQuery_sql = beginQuery_sql + " AND topic_id IS NULL"
            if int(limit) > 0:
                endQuery_sql += " LIMIT %s"
                valueArray.append(int(limit))

            historyQuery_sql = beginQuery_sql + endQuery_sql + ";"
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

    def searchChatHistory(self, text: str, limit: int | None = None) -> list:
        logger.info(f"Searching chat history records.")
        embedding = getEmbeddings(text)
        if limit is None:
            limit = ConfigManager().runtimeInt("retrieval.chat_history_default_limit", 1)
        
        connection = None
        response = None
        try:
            connection = psycopg.connect(conninfo=ConfigManager()._instance.db_conninfo, row_factory=dict_row)
            cursor = connection.cursor()
            logger.debug(f"PostgreSQL connection established.")
            
            querySQL = """SELECT ch.history_id, ch.message_id, ch.message_text, ch.message_timestamp, che.embeddings <-> %s::vector AS distance
            FROM chat_history AS ch
            JOIN chat_history_embeddings AS che
            ON ch.history_id = che.history_id
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

    def getCommunityByID(self, communityID: int) -> dict:
        logger.info(f"Getting community account data for ID:  {communityID}")

        connection = None
        response = None
        try:
            connection = connection = psycopg.connect(conninfo=ConfigManager()._instance.db_conninfo, row_factory=dict_row)
            cursor = connection.cursor()
            logger.debug(f"PostgreSQL connection established.")

            getCommunityQuery_sql = """SELECT cd.community_id, cd.community_name, cd.community_link, cd.roles, cd.created_by, cd.register_date, tg.chat_id, tg.chat_title, tg.has_topics
            FROM community_data AS cd
            LEFT JOIN community_telegram AS tg
            ON cd.community_id = tg.community_id
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

            getCommunityQuery_sql = """SELECT cd.community_id, cd.community_name, cd.community_link, cd.roles, cd.created_by, cd.register_date, tg.chat_id, tg.chat_title, tg.has_topics
            FROM community_data AS cd
            JOIN community_telegram AS tg
            ON cd.community_id = tg.community_id
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

    def addCommunityFromTelegram(self, communityData: dict):
        logger.info(f"Adding a community from telegram.")
        
        connection = None
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
            print(communityID)

            communityTelegramData = {
                "community_id": communityID,
                "chat_id": communityData.get("chat_id"),
                "chat_title": communityData.get("chat_title"),
                "has_topics": communityData.get("has_topics")
            }
            insertMemberTelegram_sql = """INSERT INTO community_telegram (community_id, chat_id, chat_title, has_topics) 
            VALUES (%(community_id)s, %(chat_id)s, %(chat_title)s, %(has_topics)s);"""

            cursor.execute(insertMemberTelegram_sql, communityTelegramData)
            connection.commit()
            
            cursor.close()
        except psycopg.Error as error:
                logger.error(f"Exception while working with psycopg and PostgreSQL:\n{error}")
        finally:
            if (connection):
                connection.close()
                logger.debug(f"PostgreSQL connection is closed.")

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
        modelDefaults = {
            "embedding": ConfigManager._as_non_empty_string(
                get_runtime_setting(runtimeSettings, "inference.default_embedding_model", "nomic-embed-text:latest")
            ) or "nomic-embed-text:latest",
            "generate": ConfigManager._as_non_empty_string(
                get_runtime_setting(runtimeSettings, "inference.default_generate_model", "llama3.2:latest")
            ) or "llama3.2:latest",
            "chat": ConfigManager._as_non_empty_string(
                get_runtime_setting(runtimeSettings, "inference.default_chat_model", "llama3.2:latest")
            ) or "llama3.2:latest",
            "tool": ConfigManager._as_non_empty_string(
                get_runtime_setting(runtimeSettings, "inference.default_tool_model", "llama3.2:latest")
            ) or "llama3.2:latest",
            "multimodal": ConfigManager._as_non_empty_string(
                get_runtime_setting(runtimeSettings, "inference.default_multimodal_model", "llama3.2-vision:latest")
            ) or "llama3.2-vision:latest",
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

    def addUsage(self, promptHistoryID: int, responseHistoryID: int, stats: dict):
        logger.info("Add usage data.")
        
        try:
            connection = psycopg.connect(conninfo=ConfigManager()._instance.db_conninfo)
            cursor = connection.cursor()
            logger.debug(f"PostgreSQL connection established.")
            
            addUsage_sql = """INSERT INTO inference_usage (prompt_history_id, response_history_id, load_duration, prompt_eval_count, prompt_eval_duration, eval_count, eval_duration, total_duration) 
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s);"""
            cursor.execute(addUsage_sql, (promptHistoryID, responseHistoryID, stats["load_duration"], stats["prompt_eval_count"], stats["prompt_eval_duration"], stats["eval_count"], stats["eval_duration"], stats["total_duration"]))
            
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
    configManager = ConfigManager()
    embeddingConfig = configManager.inference.get("embedding", {})
    host = embeddingConfig.get("url") if embeddingConfig.get("url") else configManager.runtimeValue(
        "inference.default_ollama_host", "http://127.0.0.1:11434"
    )
    host = host or "http://127.0.0.1:11434"
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
        try:
            results = Client(host=host).embeddings(
                model=model,
                prompt=text
            )
            if index > 0:
                logger.warning(f"Embedding model fallback succeeded with: {model}")
            return results.embedding
        except Exception as error:  # noqa: BLE001
            lastError = error
            if index < (len(candidateModels) - 1):
                logger.warning(f"Embedding model '{model}' failed; trying fallback model.")
                continue
            logger.error(f"Exception while getting embeddings from Ollama:\n{error}")

    if lastError is not None:
        logger.debug(f"Last embedding model error: {lastError}")
    return None
