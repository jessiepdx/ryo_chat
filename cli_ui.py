##########################################################################
#                                                                        #
#  This file (cli_ui.py) contains the command line interface for         #
#  project ryo (run your own) chat                                       #
#                                                                        #
#  Created by:  Jessie W                                                 #
#  Github: jessiepdx                                                     #
#  Contributors:                                                         #
#      Robit                                                             #
#  Created: March 1st, 2025                                              #
#  Modified: April 3rd, 2025                                             #
#                                                                        #
##########################################################################


###########
# IMPORTS #
###########

import asyncio
import json
import logging
import sys
import requests
import uuid
from datetime import datetime, timedelta, timezone
from hypermindlabs.utils import ChatHistoryManager, ConfigManager, ConsoleColors, MemberManager, SpamManager
from hypermindlabs.agents import ConversationOrchestrator, MessageAnalysisAgent, DevTestAgent, ToolCallingAgent, braveSearch
from ollama import Client, ChatResponse, ListResponse, Message, ProgressResponse



###########
# LOGGING #
###########

# Enable logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)



###########
# GLOBALS #
###########

chats = ChatHistoryManager()
config = ConfigManager()
memberData = None
members = MemberManager()
messageHistory = list()
spam = SpamManager()
toolInference = config._instance.inference.get("tool")
ollamaClient = Client(host=toolInference.get("url"))


# Think of this as UI or user policy
settings = {
    "show_stats": False,
    "model": config.inference.get("chat", {}).get(
        "model",
        config.runtimeValue("inference.default_chat_model", "llama3.2:latest"),
    ),
}

logger.info(f"Database route status: {config.databaseRoute}")



####################
# HELPER FUNCTIONS #
####################

def clearMessages():
    messageHistory.clear()
    print("chat history cleared")

def listModels():
    modelList: ListResponse = ollamaClient.list()

    return modelList.models

def ollamaProgress():
    progressReport: ProgressResponse = ollamaClient.ps()

    return progressReport.models

def toggleStats():
    currentValue = settings.get("show_stats")
    settings["show_stats"] = not currentValue
    print("Show statistics:  " + str(settings.get("show_stats")))


def prompt_input(prompt: str) -> str | None:
    try:
        return input(prompt)
    except EOFError:
        logger.info("CLI UI received EOF on stdin; shutting down cleanly.")
        return None
    except KeyboardInterrupt:
        logger.info("CLI UI interrupted by user.")
        return None


def _login_exit_requested(value: str | None) -> bool:
    text = str(value or "").strip().lower()
    return text in {"q", "quit", "exit", "/bye", "/quit"}


def _cli_login_attempt_limit() -> int:
    configured = config.runtimeInt("cli.login_max_attempts", 5)
    if configured <= 0:
        return 5
    return configured


def _database_available_for_cli() -> tuple[bool, str]:
    route = config.databaseRoute if isinstance(config.databaseRoute, dict) else {}
    status = str(route.get("status") or "").strip().lower()
    conninfo = str(getattr(config, "_instance", object()).db_conninfo or "").strip()

    if status == "failed_all":
        return (
            False,
            "Database is unavailable (primary and fallback failed). "
            "Run `python3 app.py` and complete DB setup first.",
        )
    if not conninfo:
        return (
            False,
            "Database connection string is missing. "
            "Run `python3 app.py` and review database settings.",
        )
    return True, ""


def _transient_history_limit() -> int:
    configured = config.runtimeInt("cli.guest_history_messages", 24)
    return max(4, configured)


def _role_content_from_message(item: object) -> tuple[str, str]:
    if isinstance(item, dict):
        role = str(item.get("role") or "").strip()
        content = str(item.get("content") or "").strip()
        return role, content
    role = str(getattr(item, "role", "") or "").strip()
    content = str(getattr(item, "content", "") or "").strip()
    return role, content


def _trim_transient_messages(messages: list) -> list[dict]:
    limit = _transient_history_limit()
    output = []
    for item in messages:
        role, content = _role_content_from_message(item)
        if role not in {"user", "assistant"} or not content:
            continue
        output.append({"role": role, "content": content})
    if len(output) > limit:
        output = output[-limit:]
    return output
    


async def main():
    if not sys.stdin.isatty():
        logger.info("CLI UI requires an interactive terminal. Launch via app.py dashboard 'open interface'.")
        return

    guestMode = False
    guestSessionMessages: list[dict] = []
    guestTag = uuid.uuid4().hex[:8]

    # TODO Prompt user for host, then display list of models and prompt user to select a model
    loginAttempts = 0
    maxLoginAttempts = _cli_login_attempt_limit()
    while(True):
        # Get username
        tg_username = prompt_input(f"{ConsoleColors['green']}Telegram Username > {ConsoleColors['default']}")
        if tg_username is None:
            return
        if _login_exit_requested(tg_username):
            return
        tg_username = str(tg_username).strip()

        password = prompt_input(f"{ConsoleColors['green']}Password > {ConsoleColors['default']}")
        if password is None:
            return
        if _login_exit_requested(password):
            return
        password = str(password)

        # Optional guest mode: blank username + blank password.
        if not tg_username and password.strip() == "":
            guestMode = True
            memberData = {
                "member_id": 0,
                "first_name": "Guest",
                "username": f"guest-{guestTag}",
            }
            print(
                f"{ConsoleColors['yellow']}Entering guest mode.{ConsoleColors['default']} "
                "Session history is transient and erased when this CLI process exits."
            )
            break

        if not tg_username:
            print(f"{ConsoleColors['yellow']}Username cannot be empty unless using guest mode.{ConsoleColors['default']}")
            continue

        db_ok, db_error = _database_available_for_cli()
        if not db_ok:
            print(
                f"{ConsoleColors['red']}[cli] {db_error}{ConsoleColors['default']} "
                f"{ConsoleColors['yellow']}Press Enter on both prompts to continue in guest mode.{ConsoleColors['default']}"
            )
            continue

        memberData = members.loginMember(tg_username, password)
        if memberData:
            print(f"{ConsoleColors['green']}Login successful. Welcome {memberData.get('first_name') or tg_username}.{ConsoleColors['default']}")
            break

        loginAttempts += 1
        print(
            f"{ConsoleColors['red']}Login failed.{ConsoleColors['default']} "
            f"Check username/password and DB status. "
            f"Attempt {loginAttempts}/{maxLoginAttempts}."
        )
        if loginAttempts >= maxLoginAttempts:
            print(
                f"{ConsoleColors['yellow']}Too many failed login attempts. Exiting CLI route.{ConsoleColors['default']}"
            )
            return
    
    # Logged in member
    memberID = int(memberData.get("member_id") or 0)
    lastMessageID = 0
    # Load in some chat history
    if (not guestMode) and len(messageHistory) == 0:
        shortHistoryLimit = config.runtimeInt("retrieval.conversation_short_history_limit", 20)
        shortHistory = chats.getChatHistory(memberID, "member", "cli", limit=shortHistoryLimit)
        for historyMessage in shortHistory:
            role = "assistant" if historyMessage.get("member_id") is None else "user"
            content = historyMessage.get("message_text")
            messageHistory.append(Message(role=role, content=content))
            lastMessageID = historyMessage.get("message_id")
    
    while(True):
        userInput = prompt_input(f"{ConsoleColors['dark_green']}User > {ConsoleColors['default']}")
        if userInput is None:
            return
        # Check for and handle commands
        if (userInput[:1] == "/"):
            command = userInput.split(" ")[0]
            match command:
                
                case "/bye":
                    break
                case "/clear":
                    clearMessages()
                    continue
                case "/list":
                    models = listModels()
                    for model in models:
                        if model.model == settings.get("model"):
                            print(ConsoleColors["green"] + model.model + ConsoleColors["default"])
                        else:
                            print(model.model)

                    continue
                case "/model":
                    # Get the available models
                    models = listModels()
                    availableModelNames = list()
                    for model in models:
                        availableModelNames.append(model.model)

                    # Get the user's input for choosing a model
                    modelRequested = prompt_input("Enter the model you wish to use:  ")
                    if modelRequested is None:
                        return
                    # Check the requested model against the model list
                    if modelRequested in availableModelNames:
                        settings["model"] = modelRequested
                        print("Now using " + modelRequested)
                    else:
                        print("That model is not available")
                    
                    continue
                case "/ps":
                    print("Currently running models:")
                    models = ollamaProgress()
                    for model in models:
                        if model.model == settings.get("model"):
                            print(ConsoleColors["green"] + model.model + ConsoleColors["default"])
                        else:
                            print(model.model)

                    continue
                case "/search":
                    query = prompt_input("Search query:  ")
                    if query is None:
                        return
                    braveWebResults = braveSearch(queryString=query)
                    print(braveWebResults.get("results"))
                    continue
                case "/spam":
                    if guestMode:
                        print(f"{ConsoleColors['yellow']}Spam commands are disabled in guest mode.{ConsoleColors['default']}")
                        continue
                    # Check the arguments
                    argument = userInput.split(" ")[1]
                    match argument:
                        case "add":
                            logger.info("Adding new spam text.")
                            query = prompt_input("Spam text to add:  ")
                            if query is None:
                                return
                            newRecord = spam.addSpamText(query, memberID)
                            print(newRecord)
                            continue
                        case "search":
                            logger.info("Searching the spam table.")
                            query = prompt_input("Text to search spam:  ")
                            if query is None:
                                return
                            result = spam.searchSpam(query)
                            print(result)
                            continue
                case "/stats":
                    toggleStats()
                    continue
                case _:
                    print("unknown command")
                    continue

        cliContext = {
            "platform": "cli_guest" if guestMode else "cli",
            "guest_mode": guestMode,
            "member_first_name": memberData.get("first_name"),
            "telegram_username": memberData.get("username"),
        }
        conversationOptions = {}
        if guestMode:
            conversationOptions = {
                "transient_session": True,
                "transient_messages": guestSessionMessages,
                "message_id": (sum(1 for item in guestSessionMessages if item.get("role") == "user") + 1),
            }

        conversation = ConversationOrchestrator(userInput, memberID, cliContext, options=conversationOptions)
        # Set the Conversational Orchestrator streaming response method
        def cliStreamingHandler(streamingChunk: str):
            print(streamingChunk, end="", flush=True)
        
        conversation.streamingResponse = cliStreamingHandler
        response = await conversation.runAgents()
        if guestMode:
            guestSessionMessages = _trim_transient_messages(conversation.messages)
        
    if guestMode:
        guestSessionMessages.clear()
        messageHistory.clear()

if __name__ == "__main__":
    logger.info("RYO - begin cli ui application.")
    # Create an event loop object
    loop = asyncio.new_event_loop()
    loop.run_until_complete(main())
    loop.close()
