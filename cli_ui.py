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
import requests
import uuid
from datetime import datetime, timedelta, timezone
from hypermindlabs.utils import ChatHistoryManager, ConfigManager, ConsoleColors, MemberManager
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

toolInference = config._instance.inference.get("tool")
ollamaClient = Client(host=toolInference.get("url"))


# Think of this as UI or user policy
settings = {
    "show_stats": False,
    "model": "llama3.2:latest"
}



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
    


async def main():
    # TODO Prompt user for host, then display list of models and prompt user to select a model
    while(True):
        # Get username
        tg_username = input(f"{ConsoleColors['green']}Telegram Username > {ConsoleColors['default']}")
        password = input(f"{ConsoleColors['green']}Password > {ConsoleColors['default']}")

        memberData = members.loginMember(tg_username, password)
        if memberData:
            print(memberData)
            break
    
    # Logged in member
    memberID = memberData.get("member_id")
    lastMessageID = 0
    # Load in some chat history
    if len(messageHistory) == 0:
        shortHistory = chats.getChatHistory(memberID, "member", "cli", limit=20)
        for historyMessage in shortHistory:
            role = "assistant" if historyMessage.get("member_id") is None else "user"
            content = historyMessage.get("message_text")
            messageHistory.append(Message(role=role, content=content))
            lastMessageID = historyMessage.get("message_id")
    
    while(True):
        userInput = input(f"{ConsoleColors['dark_green']}User > {ConsoleColors['default']}")
        # Check for and handle commands
        if (userInput[:1] == "/"):
            command = userInput.split(" ")[0]
            match command:
                case "/search":
                    query = input("Search query:  ")
                    braveWebResults = braveSearch(queryString=query)
                    print(braveWebResults.get("results"))
                    continue
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
                    modelRequested = input("Enter the model you wish to use:  ")
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
                case "/stats":
                    toggleStats()
                    continue
                case _:
                    print("unknown command")
                    continue

        # All of this should move from the UI elements to an Orchestrator
        # How does a UI element separate from the Agent calling handle streaming outputs though...
        # Perhaps by assigning it to a property in the orchestrator that can be accessed by the UI
        cliContext = {
            "platform": "cli"
        }
        conversation = ConversationOrchestrator(userInput, memberID, cliContext)
        # Set the Conversational Orchestrator streaming response method
        def cliStreamingHandler(streamingChunk: str):
            print(streamingChunk, end="", flush=True)
        
        conversation.streamingResponse = cliStreamingHandler
        response = await conversation.runAgents()
        

if __name__ == "__main__":
    logger.info("RYO - begin cli ui application.")
    # Create an event loop object
    loop = asyncio.new_event_loop()
    loop.run_until_complete(main())
    loop.close()