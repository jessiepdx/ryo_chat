##########################################################################
#                                                                        #
#  This file (agents.py) contains the agents modules for Hypermind Labs  #
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

from datetime import datetime, timedelta, timezone
import inspect
import json
import logging
import requests
from types import SimpleNamespace
from typing import Any, AsyncIterator
from hypermindlabs.model_router import ModelExecutionError, ModelRouter
from hypermindlabs.policy_manager import PolicyManager, PolicyValidationError
from hypermindlabs.tool_registry import (
    build_tool_specs,
    model_tool_definitions,
    register_runtime_tools,
)
from hypermindlabs.tool_runtime import ToolRuntime
from hypermindlabs.utils import (
    ChatHistoryManager, 
    ConfigManager, 
    ConsoleColors, 
    KnowledgeManager, 
    UsageManager, 
    MemberManager
)
from ollama import AsyncClient, ChatResponse, Message
from pydantic import BaseModel

# Tweepy logic eventual goes into the Twitter / X UI code
import tweepy
import tweepy.asynchronous



###########
# GLOBALS #
###########

chatHistory = ChatHistoryManager()
config = ConfigManager()
knowledge = KnowledgeManager()
members = MemberManager()
usage = UsageManager()

# Enable logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)

logger = logging.getLogger(__name__)

# Set timezone for time
timezone(-timedelta(hours=7), "Pacific")


def _runtime_int(path: str, default: int) -> int:
    return config.runtimeInt(path, default)


def _runtime_float(path: str, default: float) -> float:
    return config.runtimeFloat(path, default)


def _runtime_value(path: str, default: Any = None) -> Any:
    return config.runtimeValue(path, default)


def _fallback_stream(message_text: str) -> AsyncIterator[Any]:
    async def _stream():
        yield SimpleNamespace(
            message=SimpleNamespace(content=message_text),
            done=True,
            total_duration=0,
            load_duration=0,
            prompt_eval_count=0,
            prompt_eval_duration=0,
            eval_count=0,
            eval_duration=0,
        )

    return _stream()


def _routing_from_error(error: Exception) -> dict[str, Any]:
    if isinstance(error, ModelExecutionError):
        metadata = getattr(error, "metadata", None)
        if isinstance(metadata, dict):
            return metadata
    return {"errors": [str(error)], "status": "failed_all_candidates"}



################
# ORCHESTRATOR #
################

class ConversationOrchestrator:
    _messages = list()
    
    def __init__(self, message: str, memberID: int, context: dict=None, messageID: int=None, options: dict=None):
        
        # Get member data to 
        ## A. Make sure they exist in the DB
        ## B. Use member information in context
        self._memberData = members.getMemberByID(memberID)
        self._message = message
        self._messageID = messageID
        #self._context = context
        self._options = options if isinstance(options, dict) else {}
        stageCallback = self._options.get("stage_callback")
        self._stage_callback = stageCallback if callable(stageCallback) else None

        # Check the context and get a short collection of recent chat history into the orchestrator's messages list
        self._chatHostID = None if context is None else context.get("chat_host_id")
        self._chatType = None if context is None else context.get("chat_type")
        self._communityID = None if context is None else context.get("community_id")
        self._platform = None if context is None else context.get("platform")
        self._topicID = None if context is None else context.get("topic_id")

        if self._chatHostID is None:
            if self._communityID is None:
                self._chatHostID = memberID
                self._chatType = "member"
            else:
                self._chatHostID = self._communityID
                self._chatType = "community"

        # Get the most recent chat history records for the given context
        availablePlatforms = ["cli", "telegram"]
        availableChatTypes = ["member", "community"]
        if self._platform in availablePlatforms and self._chatType in availableChatTypes:
            shortHistoryLimit = _runtime_int("retrieval.conversation_short_history_limit", 20)
            shortHistory = chatHistory.getChatHistory(
                self._chatHostID,
                self._chatType,
                self._platform,
                self._topicID,
                limit=shortHistoryLimit,
            )
            for historyMessage in shortHistory:
                role = "assistant" if historyMessage.get("member_id") is None else "user"
                content = historyMessage.get("message_text")
                self._messages.append(Message(role=role, content=content))

            # If no message ID provided, increment one from the last message's id
            if self._messageID is None:
                self._messageID = 1 if not shortHistory else shortHistory[-1].get("message_id")
                self._responseID = self._messageID + 1
                
        # Add the newest message to local list and the database
        newMessage = Message(role="user", content=message)
        self._messages.append(newMessage)
        # Need to update this to pass all the available context, such as community_id, topic_id... even if None
        self._promptHistoryID = chatHistory.addChatHistory(
            messageID=self._messageID, 
            messageText=self._message, 
            platform=self._platform, 
            memberID=memberID, 
            communityID=self._communityID,
            chatHostID=self._chatHostID,
            topicID=self._topicID
        )

        # TODO Check options and pass to agents if necessary
        
    
    async def _emit_stage(self, stage: str, detail: str = "", **meta: Any) -> None:
        if not callable(self._stage_callback):
            return
        event = {
            "stage": str(stage),
            "detail": str(detail or ""),
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "meta": meta if isinstance(meta, dict) else {},
        }
        try:
            maybeAwaitable = self._stage_callback(event)
            if inspect.isawaitable(maybeAwaitable):
                await maybeAwaitable
        except Exception as error:  # noqa: BLE001
            logger.warning(f"Stage callback failed [{stage}]: {error}")


    async def runAgents(self):
        await self._emit_stage("orchestrator.start", "Accepted request and preparing context.")
        # Create all the agent calls in the flow as methods to call, each handles the messages passed to the actual agent
        # Make a copy of the local messages list and add the known context data. 
        # "Known Context is only in the message history for the analysis agent"
        analysisMessages = self._messages.copy()
        # Create a pseudo tool call with the known context
        knownContext = {
            "tool_name": "Known Context",
            "tool_results": {
                "user_interface": self._platform,
                "member_first_name": self._memberData.get("first_name"),
                "telegram_username": self._memberData.get("username"),
                "chat_type": self._chatType,
                "timestamp": datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
            }
        }
        
        analysisMessages.append(Message(role="tool", content=json.dumps(knownContext)))
        await self._emit_stage("analysis.start", "Running message analysis policy and model selection.")
        self._analysisAgent = MessageAnalysisAgent(analysisMessages, options=self._options)
        self._analysisResponse = await self._analysisAgent.generateResponse()

        analysisResponseMessage = ""
        
        #print(f"{ConsoleColors["purple"]}Analysis Agent > ", end="")
        chunk: ChatResponse
        async for chunk in self._analysisResponse:
            # Call the streaming response method. This is intended to be over written by the UI for cutom handling
            self.streamingResponse(streamingChunk=chunk.message.content)
            analysisResponseMessage = analysisResponseMessage + chunk.message.content
            if chunk.done:
                self._analysisStats = {
                    "total_duration": chunk.total_duration,
                    "load_duration": chunk.load_duration,
                    "prompt_eval_count": chunk.prompt_eval_count,
                    "prompt_eval_duration": chunk.prompt_eval_duration,
                    "eval_count": chunk.eval_count,
                    "eval_duration": chunk.eval_duration,
                }
        await self._emit_stage(
            "analysis.complete",
            "Analysis stage complete.",
            model=getattr(self._analysisAgent, "_model", None),
        )

        #print(ConsoleColors["default"])
        
        # TODO Just send the last USER message to the tools followed by the thoughts from analysis agent
        await self._emit_stage("tools.start", "Evaluating tool calls.")
        self._toolsAgent = ToolCallingAgent(self._messages, options=self._options)
        toolResponses = await self._toolsAgent.generateResponse()
        await self._emit_stage(
            "tools.complete",
            "Tool execution stage complete.",
            tool_calls=len(toolResponses),
        )
        # Non Streaming results
        # Waited to add the analysis agents response to chat history because it throws off the tool calling agent
        analysisMessage = Message(role="tool",content=analysisResponseMessage)
        self._messages.append(analysisMessage)

        for toolResponse in toolResponses:
            self._messages.append(Message(role="tool", content=json.dumps(toolResponse)))
        
        # TODO Tools to thoughts "thinking" agent next, will produce thoughts based analysis and tool responses, outputs thoughts followed by a prompt
        
        # TODO Passes only the prompt to the response agent
        # Pass options=options to override the langauge model

        self._chatConversationAgent = ChatConversationAgent(messages=self._messages, options=self._options)
        await self._emit_stage(
            "response.start",
            "Generating final response.",
            model=getattr(self._chatConversationAgent, "_model", None),
        )
        response = await self._chatConversationAgent.generateResponse()

        responseMessage = ""
        
        #print(f"{ConsoleColors["blue"]}Assistant > ", end="")
        chunk: ChatResponse
        async for chunk in response:
            self.streamingResponse(streamingChunk=chunk.message.content)
            responseMessage = responseMessage + chunk.message.content
            if chunk.done:
                self._devStats = {
                    "total_duration": chunk.total_duration,
                    "load_duration": chunk.load_duration,
                    "prompt_eval_count": chunk.prompt_eval_count,
                    "prompt_eval_duration": chunk.prompt_eval_duration,
                    "eval_count": chunk.eval_count,
                    "eval_duration": chunk.eval_duration,
                }
        await self._emit_stage("response.complete", "Final response generated.")

        self._chatResponseMessage = responseMessage
        #print(ConsoleColors["default"])

        if hasattr(self, "_responseID"):
            print("autogen response ID, store message")
            self.storeResponse(self._responseID)
        
        assistantMessage = Message(role="assistant", content=responseMessage)
        # Add the final response to the overall chat history (role ASSISTANT)
        self._messages.append(assistantMessage)
        await self._emit_stage("orchestrator.complete", "Completed end-to-end orchestration.")

        return responseMessage

    def storeResponse(self, messageID: int=None):
        responseID = messageID if messageID is not None else self._messageID + 1
        self._responseHistoryID = chatHistory.addChatHistory(
            messageID=responseID, 
            messageText=self._chatResponseMessage, 
            platform=self._platform, 
            memberID=None, 
            communityID=self._communityID,
            chatHostID=self._chatHostID, 
            topicID=self._topicID
        )
    
    def streamingResponse(self, streamingChunk: str):
        return

    @property
    def messages(self):
        return self._messages
    
    @property
    def messageID(self):
        return self._messageID
    
    @property
    def promptHistoryID(self):
        return self._promptHistoryID
    
    @property
    def responseHistoryID(self):
        return self._responseHistoryID
    
    @property
    def stats(self):
        # Eventually add up the stats from all agents
        return self._devStats



##################
# POLICY MANAGER #
##################

# TODO Need to create a Policy Manager that will load the agent's policy and allow for edits and save edits to file
# Perhaps policy manager belongs in utils?



###############
# AGENT TOOLS #
###############

def braveSearch(queryString: str, count: int = 5) -> list:
    """
    Search the web using Brave search API

    Args:
        queryString (str): The search query to look up
        count (int): (Optional) The number of search results to return. Defaults to 5

    Returns:
        list: A list of JSON structures containing the search result details
    """
    brave_key = config._instance.brave_keys
    braveUrl = "https://api.search.brave.com/res/v1/web/search"
    braveHeaders = {
        "Accept": "application/json",
        "Accept-Encoding": "gzip",
        "X-Subscription-Token": brave_key
    }

    queryParams = {
        "q": queryString,
        "count": count,
        "result_filter": "web"
    }
    results = requests.get(url=braveUrl, params=queryParams, headers=braveHeaders)
    braveResults = results.json()

    if braveWebResults := braveResults.get("web"):
        webResultsList = list()
        for webResult in braveWebResults.get("results"):
            webSearchData = {k: webResult[k] for k in webResult.keys() if k in ["title", "url", "description", "extra_snippets"]}
            webResultsList.append(webSearchData)
    
        return webResultsList
    
    return None


# Need to limit this only search within the context of the current conversation
def chatHistorySearch(queryString: str, count: int = 2) -> list:
    """
    Search the chat history database for a vector match on text. 

    Args:
        queryString (str): The search query to look up related messages in the chat history
        count (int): (Optional) The number of search results to return. Defaults to 1

    Returns:
        list: A list of search results. Each search result is a JSON structure
    """

    # Need to limit this only search within the context of the current conversation
    results = chatHistory.searchChatHistory(text=queryString, limit=1)
    logger.debug(f"Chat history tool results:\n{results}")
    
    convertedResults = list()
    for result in results:
        chatHistoryRecord = dict()
        for key, value in result.items():
            chatHistoryRecord[key] = value.strftime("%Y-%m-%d %H:%M:%S") if key == "message_timestamp" else value

        convertedResults.append(chatHistoryRecord)

    return convertedResults


def knowledgeSearch(queryString: str, count: int = 2) -> list:
    """
    Search the knowledge database for a vector match on text. 
    Knowledge database contains information specific to the following topics: 
    Hypermind Labs, Dropbear Robot, the Egg and mini Egg project.

    Args:
        queryString (str): The search query to look up related documents
        count (int): (Optional) The number of search results to return. Defaults to 2

    Returns:
        list: A list of search results. Each search result is a JSON structure
    """

    results = knowledge.searchKnowledge(text=queryString, limit=count)
    convertedResults = list()
    for result in results:
        knowledgeRecord = dict()
        for key, value in result.items():
            knowledgeRecord[key] = value.strftime("%Y-%m-%d %H:%M:%S") if key == "record_timestamp" else value

        convertedResults.append(knowledgeRecord)
    
    return convertedResults


def skipTools() -> dict:
    """Tool response used when the model intentionally skips tool usage."""
    return {
        "skipped": True,
        "message": "Tool usage skipped by model.",
    }


################
# BEGIN AGENTS #
################

class ToolCallingAgent():
    def __init__(self, messages: list, options: dict=None):
        logger.info(f"New instance of the tool calling agent.")
        
        # Over write defaults with loaded policy
        agentName = "tool_calling"
        endpointOverride = None if options is None else options.get("ollama_host")
        # TODO have defaults to use if policy fails to load or missing key values
        policy = loadAgentPolicy(agentName, endpointOverride=endpointOverride)
        self._systemPrompt = loadAgentSystemPrompt(agentName)
        self._allowCustomSystemPrompt = policy.get("allow_custom_system_prompt")
        self._allowed_models = resolveAllowedModels(agentName, policy)
        self._model = self._allowed_models[0]

        toolRuntimePolicy = policy.get("tool_runtime", {})
        if not isinstance(toolRuntimePolicy, dict):
            toolRuntimePolicy = {}

        toolPolicies = toolRuntimePolicy.get("tools", {})
        if not isinstance(toolPolicies, dict):
            toolPolicies = {}

        def _float_value(value: Any, default: float) -> float:
            try:
                return float(value)
            except (TypeError, ValueError):
                return default

        def _int_value(value: Any, default: int) -> int:
            try:
                return int(value)
            except (TypeError, ValueError):
                return default

        defaultTimeout = _float_value(
            toolRuntimePolicy.get("default_timeout_seconds"),
            _runtime_float("tool_runtime.default_timeout_seconds", 8.0),
        )
        defaultRetries = _int_value(
            toolRuntimePolicy.get("default_max_retries"),
            _runtime_int("tool_runtime.default_max_retries", 1),
        )
        rejectUnknownArgs = bool(toolRuntimePolicy.get("reject_unknown_args", False))
        unknownToolBehavior = str(toolRuntimePolicy.get("unknown_tool_behavior", "structured_error")).strip().lower()
        if unknownToolBehavior not in {"structured_error", "ignore"}:
            unknownToolBehavior = "structured_error"
        self._unknownToolBehavior = unknownToolBehavior

        # Check for options passed and if policy allows for those options
        if options:
            modelRequested = options.get("model_requested")
            if modelRequested in self._allowed_models:
                self._model = modelRequested

        self._modelRouter = ModelRouter(
            inference_config=config._instance.inference,
            endpoint_override=endpointOverride,
        )

        self._toolRuntime = ToolRuntime(
            api_keys={
                "brave_search": config._instance.brave_keys,
            }
        )
        self._toolSpecs = build_tool_specs(
            brave_search_fn=braveSearch,
            chat_history_search_fn=chatHistorySearch,
            knowledge_search_fn=knowledgeSearch,
            skip_tools_fn=skipTools,
            knowledge_domains=config.knowledgeDomains,
        )
        self._modelTools = model_tool_definitions(self._toolSpecs)
        register_runtime_tools(
            runtime=self._toolRuntime,
            specs=self._toolSpecs,
            tool_policy=toolPolicies,
            default_timeout_seconds=defaultTimeout,
            default_max_retries=defaultRetries,
            reject_unknown_args=rejectUnknownArgs,
        )

        self._messages = list()
        systemPrompt = Message(role="system", content=self._systemPrompt)
        self._messages.append(systemPrompt)
        # Only allow system messages in passed messages container if the policy allows for a system prompt override
        if self._allowCustomSystemPrompt:
            self._messages += messages
        else:
            histMessage: Message
            self._messages += [histMessage for histMessage in messages if histMessage.role != "system"]   

    async def generateResponse(self):
        logger.info(f"Generate a response for the tool calling agent.")

        try:
            self._response, self._routing = await self._modelRouter.chat_with_fallback(
                capability="tool",
                requested_model=self._model,
                allowed_models=self._allowed_models,
                messages=self._messages,
                stream=False,
                tools=self._modelTools,
            )
        except ModelExecutionError as error:
            logger.error(f"Tool calling model execution failed:\n{error}")
            logger.error(f"Routing metadata:\n{getattr(error, 'metadata', {})}")
            return list()
        except Exception as error:  # noqa: BLE001
            logger.error(f"Tool calling model execution failed unexpectedly:\n{error}")
            return list()

        logger.info(f"Tool calling route metadata:\n{self._routing}")
        toolResults = list()
        
        if self._response.message.tool_calls:
            for tool in self._response.message.tool_calls:
                toolName = None
                toolArgs = None
                if hasattr(tool, "function"):
                    toolName = getattr(tool.function, "name", None)
                    toolArgs = getattr(tool.function, "arguments", None)
                elif isinstance(tool, dict):
                    functionData = tool.get("function")
                    if isinstance(functionData, dict):
                        toolName = functionData.get("name")
                        toolArgs = functionData.get("arguments")
                    else:
                        toolName = tool.get("name")
                        toolArgs = tool.get("arguments")

                logger.debug(
                    "Tool calling agent execution request:\n"
                    f"Tool: {toolName}\n"
                    f"Arguments: {toolArgs}"
                )
                toolResult = self._toolRuntime.execute_tool_call(tool)
                if (
                    self._unknownToolBehavior == "ignore"
                    and toolResult.get("status") == "error"
                    and isinstance(toolResult.get("error"), dict)
                    and toolResult.get("error", {}).get("code") == "tool_not_registered"
                ):
                    logger.warning(
                        "Ignoring unknown tool call per policy.\n"
                        f"Tool: {toolResult.get('tool_name')}"
                    )
                    continue
                toolResults.append(toolResult)
                if toolResult.get("status") == "error":
                    logger.warning(
                        "Tool execution degraded gracefully.\n"
                        f"Tool: {toolResult.get('tool_name')}\n"
                        f"Error: {toolResult.get('error')}"
                    )
        
        return toolResults
    
    @property
    def messages(self):
        return self._messages


class MessageAnalysisAgent():
    def __init__(self, messages: list, options: dict=None):
        logger.info(f"New instance of the message analysis agent.")
        
        # Over write defaults with loaded policy
        agentName = "message_analysis"
        endpointOverride = None if options is None else options.get("ollama_host")
        # TODO have defaults to use if policy fails to load or missing key values
        policy = loadAgentPolicy(agentName, endpointOverride=endpointOverride)
        self._systemPrompt = loadAgentSystemPrompt(agentName)
        self._allowCustomSystemPrompt = policy.get("allow_custom_system_prompt")
        self._allowed_models = resolveAllowedModels(agentName, policy)
        self._model = self._allowed_models[0]

        # Check for options passed and if policy allows for those options
        if options:
            modelRequested = options.get("model_requested")
            if modelRequested in self._allowed_models:
                self._model = modelRequested

        self._modelRouter = ModelRouter(
            inference_config=config._instance.inference,
            endpoint_override=endpointOverride,
        )

        self._messages = list()
        systemPrompt = Message(role="system", content=self._systemPrompt)
        self._messages.append(systemPrompt)
        # Only allow system messages in passed messages container if the policy allows for a system prompt override
        if self._allowCustomSystemPrompt:
            self._messages += messages
        else:
            histMessage: Message
            self._messages += [histMessage for histMessage in messages if histMessage.role != "system"]
        
    async def generateResponse(self):
        logger.info(f"Generate a response for the message analysis agent.")
        try:
            self._response, self._routing = await self._modelRouter.chat_with_fallback(
                capability="analysis",
                requested_model=self._model,
                allowed_models=self._allowed_models,
                messages=self._messages,
                stream=True,
                format="json",
            )
        except Exception as error:  # noqa: BLE001
            self._routing = _routing_from_error(error)
            logger.error(f"Message analysis model execution failed:\n{error}")
            logger.error(f"Routing metadata:\n{self._routing}")
            self._response = _fallback_stream(
                '{"analysis":"unavailable","reason":"all_candidate_models_failed"}'
            )
        logger.info(f"Message analysis route metadata:\n{self._routing}")
        return self._response
    
    @property
    def messages(self):
        return self._messages


class DevTestAgent():
    def __init__(self, messages: list, options: dict=None):
        logger.info(f"New instance of the dev test agent.")
        
        # Over write defaults with loaded policy
        agentName = "dev_test"
        endpointOverride = None if options is None else options.get("ollama_host")
        # TODO have defaults to use if policy fails to load or missing key values
        policy = loadAgentPolicy(agentName, endpointOverride=endpointOverride)
        self._systemPrompt = loadAgentSystemPrompt(agentName)
        self._allowCustomSystemPrompt = policy.get("allow_custom_system_prompt")
        self._allowed_models = resolveAllowedModels(agentName, policy)
        self._model = self._allowed_models[0]

        # Check for options passed and if policy allows for those options
        if options:
            modelRequested = options.get("model_requested")
            if modelRequested in self._allowed_models:
                self._model = modelRequested

        self._modelRouter = ModelRouter(
            inference_config=config._instance.inference,
            endpoint_override=endpointOverride,
        )

        # TODO Check to see if policy allows for system prompt overrides

        self._messages = list()
        systemPrompt = Message(role="system", content=self._systemPrompt)
        self._messages.append(systemPrompt)
        # Only allow system messages in passed messages container if the policy allows for a system prompt override
        if self._allowCustomSystemPrompt:
            self._messages += messages
        else:
            histMessage: Message
            self._messages += [histMessage for histMessage in messages if histMessage.role != "system"]
        
    async def generateResponse(self):
        logger.info(f"Generate a response for the dev test agent.")
        try:
            self._response, self._routing = await self._modelRouter.chat_with_fallback(
                capability="dev_test",
                requested_model=self._model,
                allowed_models=self._allowed_models,
                messages=self._messages,
                stream=True,
            )
        except Exception as error:  # noqa: BLE001
            self._routing = _routing_from_error(error)
            logger.error(f"Dev test model execution failed:\n{error}")
            logger.error(f"Routing metadata:\n{self._routing}")
            self._response = _fallback_stream(
                "I could not complete the dev-test response because all configured models failed."
            )
        logger.info(f"Dev test route metadata:\n{self._routing}")
        return self._response
    
    @property
    def messages(self):
        return self._messages


class ChatConversationAgent():
    def __init__(self, messages: list, options: dict=None):
        logger.info(f"New instance of the chat conversation agent.")
        
        # Over write defaults with loaded policy
        agentName = "chat_conversation"
        endpointOverride = None if options is None else options.get("ollama_host")
        # TODO have defaults to use if policy fails to load or missing key values
        policy = loadAgentPolicy(agentName, endpointOverride=endpointOverride)
        self._systemPrompt = loadAgentSystemPrompt(agentName)
        self._allowCustomSystemPrompt = policy.get("allow_custom_system_prompt")
        self._allowed_models = resolveAllowedModels(agentName, policy)
        self._model = self._allowed_models[0]

        # Check for options passed and if policy allows for those options
        if options:
            modelRequested = options.get("model_requested")
            if modelRequested in self._allowed_models:
                self._model = modelRequested

        self._modelRouter = ModelRouter(
            inference_config=config._instance.inference,
            endpoint_override=endpointOverride,
        )

        # TODO Check to see if policy allows for system prompt overrides

        self._messages = list()
        systemPrompt = Message(role="system", content=self._systemPrompt)
        self._messages.append(systemPrompt)
        # Only allow system messages in passed messages container if the policy allows for a system prompt override
        if self._allowCustomSystemPrompt:
            self._messages += messages
        else:
            histMessage: Message
            self._messages += [histMessage for histMessage in messages if histMessage.role != "system"]
        
    async def generateResponse(self):
        logger.info(f"Generate a response for the chat conversation agent.")
        try:
            self._response, self._routing = await self._modelRouter.chat_with_fallback(
                capability="chat",
                requested_model=self._model,
                allowed_models=self._allowed_models,
                messages=self._messages,
                stream=True,
            )
        except Exception as error:  # noqa: BLE001
            self._routing = _routing_from_error(error)
            logger.error(f"Chat conversation model execution failed:\n{error}")
            logger.error(f"Routing metadata:\n{self._routing}")
            self._response = _fallback_stream(
                "I could not generate a full response because all configured models are unavailable right now."
            )
        logger.info(f"Chat conversation route metadata:\n{self._routing}")
        return self._response
    
    @property
    def messages(self):
        return self._messages



####################
# ORINGINAL AGENTS #
####################

class ConversationalAgent():
    _documents = list()

    def __init__(self, message_data: str, memberID: int):
        logger.info(f"New instance of the conversational agent.")
        self.messageData = message_data
        self.fromUser = MemberManager().getMemberByID(memberID)

    async def generateResponse(self):
        logger.info(f"Generate a response for the conversational agent via async Ollama call.")
        # This is passed to ollama for the messages array
        messageHistory = []
        # First add the system messages to messageHistory
        messageHistory.append({
            "role" : "system",
            "content" : config.defaults["system_prompt"] + config.defaults["chat_sys_prompt"]
        })
        # Get chat history from new DB

        communityID = self.messageData.get("community_id")
        memberID = self.messageData.get("member_id")
        chatHostID = communityID if communityID else memberID
        if not chatHostID:
            return
        chatType = "community" if communityID else "member"
        
        platform = self.messageData.get("platform")
        topicID = self.messageData.get("topic_id")
        
        chatHistoryResults = chatHistory.getChatHistoryWithSenderData(chatHostID, chatType, platform, topicID)
        for history in chatHistoryResults:
            contextJson = {
                "sent_from" : {
                    "first_name" : history.get("first_name"),
                    "last_name" : history.get("last_name")
                },
                "sent_at" : history.get("message_timestamp").strftime("%Y-%m-%d %H:%M:%S")
            }
            messageContext = {
                "role" : "tool",
                "content" : json.dumps(contextJson)
            }
            messageHistory.append(messageContext)

            historyMessage = {
                "role" : "assistant" if history.get("member_id") is None else "user",
                "content" : history.get("message_text")
            }
            messageHistory.append(historyMessage)

        # Create new message entity

        contextJson = {
            "sent_from" : {
                "username" : self.fromUser.get("username"),
                "first_name" : self.fromUser.get("first_name"),
                "last_name" : self.fromUser.get("last_name")
            },
            "sent_at" : datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

        # Search the vectorDB if there is a knowledge db and add to contextJson
        message = self.messageData.get("message_text")
        wordCount = 0 if message is None else len(message.split(" "))
        if wordCount > _runtime_int("conversation.knowledge_lookup_word_threshold", 6):
            # Add the documents to the agent instance so the UI can access them and store retrieval records
            self._documents = knowledge.searchKnowledge(
                message,
                limit=_runtime_int("conversation.knowledge_lookup_result_limit", 2),
            )
            knowledgeDocuments = list()
            for doc in self._documents:
                knowledgeDocuments.append(doc.get("knowledge_document"))

            contextJson["knowledge_documents"] = knowledgeDocuments
        
        # Add search results to the new message prompt
        newContext = {
            "role" : "tool",
            "content" : json.dumps(contextJson)
        }
        messageHistory.append(newContext)
        
        newMessage = {
            "role" : "user", 
            "content" : message
        }
        # Add the new message to the messageHistory
        messageHistory.append(newMessage)

        logger.info(f"Message sent to Ollama:\n\n{newMessage}\n")

        # Set additional Ollama options
        ollamaOptions = {
            "num_ctx" : _runtime_int("inference.model_context_window", 4096)
        }
        
        # Call the Ollama CHAT API
        output = await AsyncClient(host=config.inference["chat"]["url"]).chat(
            messages=messageHistory,
            model=config.inference["chat"]["model"],
            options=ollamaOptions,
            stream=False
        )

        # Update the chat history database with the newest message
        self.promptHistoryID = chatHistory.addChatHistory(
            messageID=self.messageData.get("message_id"), 
            messageText=self.messageData.get("message_text"), 
            platform=platform, 
            memberID=self.fromUser.get("member_id"), 
            communityID=self.messageData.get("community_id"), 
            topicID=self.messageData.get("topic_id"), 
            timestamp=self.messageData.get("message_timestamp")
        )

        # Store the response
        self.response = output["message"]["content"]
        # Add the response to the chat history

        # Store the statistics from Ollama
        self.stats = {k: output[k] for k in ("total_duration", "load_duration", "prompt_eval_count", "prompt_eval_duration", "eval_count", "eval_duration", "created_at")}
        
        logger.info(f"Response from Ollama:\n\n{self.response}\n")
        return self.response
#good

# TODO Add the prompt to chat history
class ImageAgent():

    def __init__(self, message_data: str, memberID):
        logger.info(f"New instance of the image agent.")
        self.fromUser = MemberManager().getMemberByID(memberID)
        self.messageData = message_data
        self.images = message_data.get("message_images")
        self.text = message_data.get("message_text")

    async def generateResponse(self):
        logger.info(f"Generating a response for the image agent using Ollama async client.")
        # This is passed to ollama for the messages array
        messageHistory = []
        # First add the system messages to messageHistory
        messageHistory.append({
            "role" : "system",
            "content" : config.defaults["system_prompt"] + config.defaults["chat_sys_prompt"]
        })

        # Create new message entity
        contextJson = {
            "sent_from" : {
                "username" : self.fromUser.get("username"),
                "first_name" : self.fromUser.get("first_name"),
                "last_name" : self.fromUser.get("last_name")
            },
            "sent_at" : datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # Add search results to the new message prompt
        newContext = {
            "role" : "tool",
            "content" : json.dumps(contextJson)
        }

        messageHistory.append(newContext)
        
        newMessage = {
            "role" : "user", 
            "content" : self.text,
            "images" : self.images
        }
        
        # Add the new message to the messageHistory
        messageHistory.append(newMessage)

        logger.info(f"Image sent to Ollama with the follwoing prompt:\n\n{self.text}\n")

        output = await AsyncClient(host=config.inference["multimodal"]["url"]).chat(
            model=config.inference["multimodal"]["model"], 
            stream=False,
            messages=messageHistory
        )

        # Store the response
        self.response = output["message"]["content"]
        # Store the statistics from Ollama
        self.stats = {k: output[k] for k in ("total_duration", "load_duration", "prompt_eval_count", "prompt_eval_duration", "eval_count", "eval_duration", "created_at")}
        
        logger.info(f"Response from Ollama:\n\n{self.response}\n")

        return self.response
#good

class TweetAgent():
    systemPrompt = "You are an AI Agent that writes tweets for the X platform (formerly known as Twitter). You are using a premium account which has a charater limit of 25,000 characters so you can write tweets at longer than the standard 256 character limit! Your response should only be the tweet text and nothing else. "
    systemPromptROKO = """You are the human voice of the ROKO Network, a cryptic, futuristic, and community-driven decentralized AI and robotics project focused on building networks for human-robot interaction and edge computing. Engage the Twitter community with cryptic, thought-provoking language infused with technical jargon, futuristic themes, and a hint of casual internet culture. Highlight the project's advancements in decentralized AI, blockchain, robotics, and edge computing while maintaining an air of mystery and exclusivity.

Stylization

Visionary and Ambitious: Speak with authority as a pioneer in decentralized AI and robotics. Inspire imagination while grounding ideas in technical relevance.
Cryptic Yet Informative: Balance mystery with meaningful updates. Keep followers intrigued while ensuring they stay informed about project milestones.
Tech-Savvy and Relatable: Use technical jargon strategically, paired with approachable internet slang.
Exclusive and Community-Focused: Make the audience feel like insiders in a groundbreaking movement.
Emojis as Symbols: Use 🕳️, 🐇, 🦾, 🔮, and others creatively to add intrigue and emphasize key ideas.


Core Practices

Stay Balanced: Every post should intrigue and inform. Tease cryptic ideas while ensuring relevant updates give followers a sense of progress.
Encourage Engagement: Invite followers to participate actively in votes, discussions, or milestones.
Keep Replies Short and Engaging: Spark curiosity and maintain a consistent tone.

Key Objectives

Keep followers feeling like insiders to the project's bold vision.
Balance cryptic elements with relevant, transparent updates to maintain trust and engagement.
Foster curiosity and enthusiasm, encouraging interaction without cluttered visuals or hashtags.

DO NOT respond in JSON.
DO NOT put quotes around the tweet."""

    tweetText = None
    messageHistory = []
    
    def __init__(self, message_data: str, from_user: dict):
        logger.info(f"New instance of the tweet Agent.")
        self.messageData = message_data
        self.fromUser = from_user
        self.tweetPrompt = message_data.get("tweet_prompt")
        
        # Start the message history with the system prompts
        self.messageHistory.append({
            "role" : "system",
            "content" : self.systemPrompt + self.systemPromptROKO
        })

        
    async def ComposeTweet(self) -> str:
        logger.info(f"Composing Tweet.")
        # Load the message history if there is a chat history
        
        chatID = self.messageData.get("chat_id")
        topicID = self.messageData.get("topic_id")
        ch = chatHistory.getChatHistory(chatID, topicID=topicID)
        for record in ch:
            contextJson = {
                "sent_from" : {
                    "username" : record.get("from_user").get("username"),
                    "first_name" : record.get("from_user").get("first_name"),
                    "last_name" : record.get("from_user").get("last_name")
                },
                "sent_at" : record.get("message_timestamp").strftime("%Y-%m-%d %H:%M:%S")
            }
            messageContext = {
                "role" : "tool",
                "content" : json.dumps(contextJson)
            }
            self.messageHistory.append(messageContext)
            
            historyMessage = {
                "role" : record.get("from_user").get("role"),
                "content" : record.get("message_text")
            }
            self.messageHistory.append(historyMessage)

        # Search the vectorDB if there is a knowledge db and add to contextJson
        prompt = self.messageData.get("tweet_prompt")
        wordCount = 0 if prompt is None else len(prompt.split(" "))

        if wordCount > _runtime_int("conversation.knowledge_lookup_word_threshold", 6):
            documents = knowledge.searchKnowledge(
                prompt,
                limit=_runtime_int("conversation.knowledge_lookup_result_limit", 2),
            )
            knowledgeDocuments = list()
            for doc in documents:
                knowledgeDocuments.append(doc.get("knowledge_document"))

            contextJson = {
                "knowledge_documents": knowledgeDocuments
            }

            # Add search results to the new message prompt
            newContext = {
                "role" : "tool",
                "content" : json.dumps(contextJson)
            }

            self.messageHistory.append(newContext)

        newMessage = {
            "role" : "user", 
            "content" : self.tweetPrompt
        }
        
        # Add the new message to the messageHistory
        self.messageHistory.append(newMessage)

        logger.info(f"Prompt sent to Ollama:\n\n{newMessage}\n")
        
        # Set additional Ollama options
        ollamaOptions = {
            "num_ctx" : _runtime_int("inference.model_context_window", 4096)
        }

        output = await AsyncClient(host=config.inference["chat"]["url"]).chat(
            keep_alive="15m",
            messages=self.messageHistory,
            model=config.inference["chat"]["model"],
            options=ollamaOptions,
            stream=False
        )
        responseText = output["message"]["content"]
        logger.info(f"Response from Ollama:\n\n{responseText}\n")

        responseMessage = {
            "role" : "assistant",
            "content" : responseText
        }
        self.messageHistory.append(responseMessage)

        self.tweetText = responseText

        return responseText
    
    async def ModifyTweet(self, newPrompt: str) -> str:
        logger.info(f"Modify tweet.")

        newMessage = {
            "role" : "user", 
            "content" : newPrompt
        }
        
        # Add the new message to the messageHistory
        self.messageHistory.append(newMessage)

        logger.info(f"Prompt sent to Ollama:\n\n{newMessage}\n")
        
        output = await AsyncClient(host=config.inference["chat"]["url"]).chat(
            model=config.inference["chat"]["model"], 
            stream=False,
            messages=self.messageHistory,
            keep_alive="15m"
        )
        responseText = output["message"]["content"]
        logger.info(f"Response from Ollama:\n\n{responseText}\n")

        responseMessage = {
            "role" : "assistant",
            "content" : responseText
        }
        self.messageHistory.append(responseMessage)

        self.tweetText = responseText

        return responseText
    
    async def SendTweet(self):
        logger.info(f"Send the tweet.")
        print(self.tweetText)
        
        try:
            client = tweepy.asynchronous.AsyncClient(
                consumer_key=config.twitter_keys["consumer_key"],
                consumer_secret=config.twitter_keys["consumer_secret"],
                access_token=config.twitter_keys["access_token"],
                access_token_secret=config.twitter_keys["access_token_secret"]
            )
            t = await client.create_tweet(text=self.tweetText)
            print(t)
            return t
        except tweepy.errors.Forbidden:
            logger.warning("Not authorized")
        


####################
# HELPER FUNCTIONS #
####################

def _policyManager(endpointOverride: str | None = None) -> PolicyManager:
    inference = config._instance.inference if hasattr(config, "_instance") else {}
    return PolicyManager(
        inference_config=inference,
        endpoint_override=endpointOverride,
    )


def resolveAllowedModels(policyName: str, policy: dict) -> list[str]:
    allowed = policy.get("allowed_models")
    models: list[str] = []
    if isinstance(allowed, list):
        for modelName in allowed:
            if isinstance(modelName, str):
                cleaned = modelName.strip()
                if cleaned and cleaned not in models:
                    models.append(cleaned)

    if models:
        return models

    fallbackPolicy = _policyManager().default_policy(policyName)
    fallbackModels = fallbackPolicy.get("allowed_models", [])
    if isinstance(fallbackModels, list):
        for modelName in fallbackModels:
            if isinstance(modelName, str) and modelName.strip():
                models.append(modelName.strip())

    if models:
        return models

    return [str(_runtime_value("inference.default_chat_model", "llama3.2:latest"))]


def loadAgentPolicy(policyName: str, endpointOverride: str | None = None) -> dict:
    logger.info(f"Loading agent policy for: {policyName}")
    manager = _policyManager(endpointOverride=endpointOverride)
    report = manager.validate_policy(policy_name=policyName, strict_model_check=False)

    for warning in report.warnings:
        logger.warning(f"Policy validation warning [{policyName}]: {warning}")
    for error in report.errors:
        logger.error(f"Policy validation error [{policyName}]: {error}")

    if report.errors:
        return manager.default_policy(policyName)
    if isinstance(report.normalized_policy, dict):
        return report.normalized_policy
    return manager.default_policy(policyName)


def loadAgentSystemPrompt(policyName: str) -> str:
    logger.info(f"Loading agent system prompt for: {policyName}")
    manager = _policyManager()
    try:
        return manager.load_system_prompt(policy_name=policyName, strict=True)
    except PolicyValidationError as error:
        logger.error(f"System prompt load failed [{policyName}]: {error}")
        return manager.load_system_prompt(policy_name=policyName, strict=False)


def toolCaller(toolName: str, toolArgs: dict) -> dict:
    """Validates the tool name and arguments generated by a tool calling agent.
    Once validated, runs the tool with areguments and returns the results."""
    logger.info("Tool Validator called")

    # TODO Validate the tool name and arguments
    # Look up tool DEFINITION
    # Get the property names list and make a copy dict from passed ARGS with only the keys that exist in the tool definition
    # Check new dict for the required properties are present

    # run the tool with args if VALID and return the results

    return dict()
