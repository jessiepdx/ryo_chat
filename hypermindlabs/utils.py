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
import psycopg
import re
import secrets
import string
import textstat
import time
import uuid
from datetime import datetime, timedelta, timezone
from math import ceil
from ollama import Client
from psycopg.rows import dict_row
from urllib.parse import parse_qs, unquote



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
                memberData_sql = """CREATE TABLE IF NOT EXISTS member_data (
                    member_id SERIAL PRIMARY KEY,
                    first_name VARCHAR(96),
                    last_name VARCHAR(96),
                    email VARCHAR(72),
                    roles VARCHAR(32)[],
                    register_date TIMESTAMP NOT NULL,
                    community_score REAL NOT NULL DEFAULT 0
                );"""
                cursor.execute(memberData_sql)
                connection.commit()
                
                # Create the member telegram table if it doesn't exist
                memberTelegramTable_sql = """CREATE TABLE IF NOT EXISTS member_telegram (
                    record_id SERIAL PRIMARY KEY,
                    member_id INT UNIQUE NOT NULL,
                    first_name VARCHAR(96),
                    last_name VARCHAR(96),
                    username VARCHAR(96),
                    user_id BIGINT UNIQUE NOT NULL,
                    CONSTRAINT member_link
                        FOREIGN KEY(member_id)
                        REFERENCES member_data(member_id)
                        ON DELETE CASCADE
                );"""
                cursor.execute(memberTelegramTable_sql)
                connection.commit()

                # Create the member password hash table
                memberSecureTable_sql = """CREATE TABLE IF NOT EXISTS member_secure (
                    secure_id SERIAL PRIMARY KEY,
                    member_id INTEGER UNIQUE NOT NULL,
                    secure_hash BYTEA,
                    CONSTRAINT member_link
                        FOREIGN KEY(member_id)
                        REFERENCES member_data(member_id)
                        ON DELETE CASCADE
                );"""
                cursor.execute(memberSecureTable_sql)
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

    def addMemberFromTelegram(self, memberData: dict):
        logger.info(f"Adding a new member from telegram.")
        connection = None
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
            memberID = result.get("member_id")

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
            WHERE tg.username = %s
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
        
        # Create a random password if none was sent
        if password is None:
            alphabet = string.ascii_letters + string.digits
            while True:
                password = ''.join(secrets.choice(alphabet) for i in range(12))
                if (any(c.islower() for c in password)
                        and any(c.isupper() for c in password)
                        and sum(c.isdigit() for c in password) >= 2):
                    break
        else:
            # Return None if the password does not meet minimum requirements
            pattern = re.compile(r"(?=.*\d)(?=.*[a-z])(?=.*[A-Z])(?=.*[-+_!@#$%^&*.,?]).{12,}")
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
        # Parse the telegram initData query string
        queryDict = parse_qs(telegramInitData)
        knownHash = queryDict["hash"][0]

        telegramUserData = json.loads(queryDict["user"][0])
        memberTelegramID = telegramUserData["id"]

        # Create a data check string from the query string
        # Data Check String must have the hash propoerty removed
        initData = sorted([chunk.split("=") for chunk in unquote(telegramInitData).split("&") if chunk[:len("hash=")] != "hash="], key=lambda x: x[0])
        initData = "\n".join([f"{rec[0]}={rec[1]}" for rec in initData])

        # Create the Secret Key
        key = "WebAppData".encode()
        token = ConfigManager().bot_token.encode()
        secretKey = hmac.new(key, token, hashlib.sha256)
        digest = hmac.new(secretKey.digest(), initData.encode(), hashlib.sha256)

        if hmac.compare_digest(knownHash, digest.hexdigest()):
            print("data validated!")
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
                memberHistory_sql = """CREATE TABLE IF NOT EXISTS chat_history (
                    history_id SERIAL PRIMARY KEY,
                    member_id INT,
                    community_id INT,
                    chat_host_id INT,
                    topic_id INT,
                    chat_type VARCHAR(16),
                    platform VARCHAR(24),
                    message_id INT NOT NULL,
                    message_text TEXT,
                    message_timestamp TIMESTAMP
                );"""
                cursor.execute(memberHistory_sql)

                createHistorySQL = """CREATE TABLE IF NOT EXISTS chat_history_embeddings (
                    embedding_id SERIAL PRIMARY KEY,
                    history_id INT NOT NULL,
                    embeddings vector(768),
                    CONSTRAINT message_link
                        FOREIGN KEY(history_id)
                        REFERENCES chat_history(history_id)
                        ON DELETE CASCADE
                );"""
                cursor.execute(createHistorySQL)
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
            
            insertEmbeddings_sql = """INSERT INTO chat_history_embeddings (history_id, embeddings)
            VALUES (%s, %s);"""
            cursor.execute(insertEmbeddings_sql, (historyID, embedding))

            response = historyID

            connection.commit()
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

    def getChatHistory(self, chatHostID: int, chatType: str, platform: str, topicID: int = None, timeInHours: int = 12, limit: int = 1) -> list:
        logger.info(f"Getting chat history records.")
        timePeriod = datetime.now() - timedelta(hours=timeInHours)
        
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
            
            endQuery_sql = " ORDER BY message_timestamp;"

            valueArray = [chatHostID, chatType, platform, timePeriod]

            if topicID:
                beginQuery_sql = beginQuery_sql + " AND topic_id = %s"
                valueArray.append(topicID)
            else:
                beginQuery_sql = beginQuery_sql + " AND topic_id IS NULL"
            

            historyQuery_sql = beginQuery_sql + endQuery_sql
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

    def getChatHistoryWithSenderData(self, chatHostID: int, chatType: str, platform: str, topicID: int = None, timeInHours: int = 12, limit: int = 0) -> list:
        logger.info(f"Getting chat history records.")
        timePeriod = datetime.now() - timedelta(hours=timeInHours)
        
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
            
            endQuery_sql = " ORDER BY message_timestamp;"

            valueArray = [chatHostID, chatType, platform, timePeriod]

            if topicID:
                beginQuery_sql = beginQuery_sql + " AND topic_id = %s"
                valueArray.append(topicID)
            else:
                beginQuery_sql = beginQuery_sql + " AND topic_id IS NULL"
            

            historyQuery_sql = beginQuery_sql + endQuery_sql
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

    def searchChatHistory(self, text: str, limit: int=1) -> list:
        logger.info(f"Searching chat history records.")
        embedding = getEmbeddings(text)
        
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
            cursor.execute(querySQL, (embedding, limit))
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
                communityData_sql = """CREATE TABLE IF NOT EXISTS community_data (
                    community_id SERIAL PRIMARY KEY,
                    community_name VARCHAR(96),
                    community_link VARCHAR(256),
                    roles VARCHAR(32)[],
                    created_by INT,
                    register_date TIMESTAMP NOT NULL,
                    CONSTRAINT member_link
                        FOREIGN KEY(created_by)
                        REFERENCES member_data(member_id)
                        ON DELETE SET NULL
                );"""
                cursor.execute(communityData_sql)
                
                # Create the community telegram table if it doesn't exist
                communityTelegramTable_sql = """CREATE TABLE IF NOT EXISTS community_telegram (
                    record_id SERIAL PRIMARY KEY,
                    community_id INT UNIQUE NOT NULL,
                    chat_id BIGINT UNIQUE NOT NULL,
                    chat_title VARCHAR(96),
                    has_topics BOOL,
                    CONSTRAINT community_link
                        FOREIGN KEY(community_id)
                        REFERENCES community_data(community_id)
                        ON DELETE CASCADE
                );"""
                cursor.execute(communityTelegramTable_sql)
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
    communityScoreRules = [
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

            connection = None
            try:
                connection = psycopg.connect(conninfo=ConfigManager()._instance.db_conninfo)
                cursor = connection.cursor()
                logger.debug(f"PostgreSQL connection established.")
                
                # Create the individual accounts table if it doesn't exist
                communityScoreTable_sql = """CREATE TABLE IF NOT EXISTS community_score (
                    score_id SERIAL PRIMARY KEY,
                    history_id INTEGER NOT NULL,
                    event TEXT NOT NULL,
                    read_score REAL,
                    points_awarded REAL NOT NULL,
                    awarded_from_id INTEGER NOT NULL,
                    multiplier REAL NOT NULL,
                    CONSTRAINT history_link
                        FOREIGN KEY(history_id)
                        REFERENCES chat_history(history_id)
                        ON DELETE SET NULL
                );"""
                cursor.execute(communityScoreTable_sql)

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
        member = MemberManager().getMemberByID(memberID)

        memberScore = member.get("community_score")
        rateLimits = {
            "message": 0,
            "image": 0
        }
        for rule in self.communityScoreRules:
            if memberScore >= rule[chatType]["min"]:
                rateLimits["message"] = rule["message_per_hour"]
                rateLimits["image"] = rule["image_per_hour"]
            else:
                break
        
        return rateLimits


class ConfigManager:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ConfigManager, cls).__new__(cls)
            # Open the config file
            f = open("config.json", "r")
            config_json = json.load(f)
            database = config_json.get("database")
            # Create Database connection string
            connectionString = f"dbname={database.get('db_name')} user={database.get('user')} password={database.get('password')} host={database.get('host')}"
            if database.get("port") is not None:
                connectionString = connectionString + f" port={database.get('port')}"

            cls._instance.bot_name = config_json.get("bot_name")
            cls._instance.bot_id = config_json.get("bot_id")
            cls._instance.bot_token = config_json.get("bot_token")
            cls._instance.web_ui_url = config_json.get("web_ui_url")
            cls._instance.owner_info = config_json.get("owner_info")
            cls._instance.database = database
            cls._instance.knowledge_domains = None if config_json.get("knowledge") is None else config_json.get("knowledge").get("domains")
            cls._instance.db_conninfo = connectionString
            cls._instance.defaults = config_json.get("defaults")
            cls._instance.inference = config_json.get("inference")
            cls._instance.twitter_keys = config_json.get("twitter_keys")
            cls._instance.brave_keys = config_json["api_keys"].get("brave_search")
        
        return cls._instance

    def updateConfig(self, key, value):
        self.config[key] = value
        # Save new config changes to JSON file
    
    # Define getters

    @property
    def botName(self):
        return self._instance.bot_name
    
    @property
    def knowledgeDomains(self):
        return self._instance.knowledge_domains
    
    @property
    def webUIUrl(self):
        return self._instance.web_ui_url


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
                
                # Create the knowledge table if it does not exist
                createKnowledgeSQL = """CREATE TABLE IF NOT EXISTS knowledge (
                    knowledge_id SERIAL PRIMARY KEY,
                    domains TEXT[],
                    roles TEXT[],
                    categories TEXT[],
                    knowledge_document TEXT,
                    document_metadata JSON,
                    embeddings vector(768),
                    record_timestamp TIMESTAMP,
                    record_metadata JSON
                );"""
                cursor.execute(createKnowledgeSQL)

                # Create the knowledge retrieval table if it does not exist
                createRetrievalsSQL = """CREATE TABLE IF NOT EXISTS knowledge_retrievals (
                    retrieval_id SERIAL PRIMARY KEY,
                    prompt_id INT,
                    response_id INT,
                    knowledge_id INT,
                    distance DOUBLE PRECISION,
                    retrieval_timestamp TIMESTAMP,
                    CONSTRAINT prompt_link
                        FOREIGN KEY(prompt_id)
                        REFERENCES chat_history(history_id)
                        ON DELETE SET NULL,
                    CONSTRAINT response_link
                        FOREIGN KEY(response_id)
                        REFERENCES chat_history(history_id)
                        ON DELETE SET NULL,
                    CONSTRAINT knowledge_link
                        FOREIGN KEY(knowledge_id)
                        REFERENCES knowledge(knowledge_id)
                        ON DELETE CASCADE
                );"""
                cursor.execute(createRetrievalsSQL)

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

            querySQL = """SELECT knowledge_id, domains, roles, categories, knowledge_document, document_metadata, record_timestamp, record_metadata
            FROM knowledge
            LIMIT 10"""
            cursor.execute(querySQL)
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

    def searchKnowledge(self, text: str, limit: int=1) -> list:
        logger.info(f"Searching knowledge documents.")
        embedding = getEmbeddings(text)

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
            cursor.execute(querySQL, (embedding, limit))
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
                proposalsTable_sql = """CREATE TABLE IF NOT EXISTS proposals (
                    proposal_id SERIAL PRIMARY KEY,
                    submitted_from TEXT,
                    project_title TEXT,
                    project_description TEXT,
                    filename TEXT,
                    submit_date timestamp
                );"""
                cursor.execute(proposalsTable_sql)

                # Create the proposal disclosure table if it doesn't exist
                proposalDisclosure_sql = """CREATE TABLE IF NOT EXISTS proposal_disclosure (
                    disclosure_id SERIAL PRIMARY KEY,
                    user_id INT,
                    proposal_id INT,
                    agreement_date timestamp
                );"""
                cursor.execute(proposalDisclosure_sql)
                
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
                usageTable_sql = """CREATE TABLE IF NOT EXISTS inference_usage (
                    usage_id SERIAL PRIMARY KEY,
                    prompt_history_id INT NOT NULL,
                    response_history_id INT NOT NULL,
                    load_duration BIGINT NOT NULL,
                    prompt_eval_count INTEGER NOT NULL,
                    prompt_eval_duration BIGINT NOT NULL,
                    eval_count INTEGER NOT NULL,
                    eval_duration BIGINT NOT NULL,
                    total_duration BIGINT NOT NULL,
                    CONSTRAINT prompt_history
                        FOREIGN KEY(prompt_history_id)
                        REFERENCES chat_history(history_id)
                        ON DELETE CASCADE,
                    CONSTRAINT response_history
                        FOREIGN KEY(response_history_id)
                        REFERENCES chat_history(history_id)
                        ON DELETE CASCADE
                );"""
                cursor.execute(usageTable_sql)
                
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
        response = None
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
    try:
        results = Client(host=ConfigManager().inference["embedding"]["url"]).embeddings(
            model=ConfigManager().inference["embedding"]["model"],
            prompt=text
        )
        return results.embedding
    except Exception as error:
        logger.error(f"Exception while getting embeddings from Ollama:\n{error}")
        return None

