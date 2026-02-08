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
import re
import requests
from types import SimpleNamespace
from typing import Any, AsyncIterator
from hypermindlabs.approval_manager import ApprovalManager
from hypermindlabs.model_router import ModelExecutionError, ModelRouter
from hypermindlabs.policy_manager import PolicyManager, PolicyValidationError
from hypermindlabs.temporal_context import (
    build_temporal_context,
    coerce_datetime_utc,
    utc_now_iso,
)
from hypermindlabs.tool_registry import (
    ToolRegistryStore,
    build_tool_specs,
    model_tool_definitions,
    normalize_custom_tool_payload,
    register_runtime_tools,
)
from hypermindlabs.tool_sandbox import (
    ToolSandboxEnforcer,
    ToolSandboxPolicyStore,
    merge_sandbox_policies,
    normalize_tool_sandbox_policy,
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


def _runtime_bool(path: str, default: bool) -> bool:
    return config.runtimeBool(path, default)


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


_ALLOWED_TOOL_HINTS = {"braveSearch", "chatHistorySearch", "knowledgeSearch", "skipTools"}
_DIAGNOSTIC_REQUEST_PATTERNS = (
    r"\bdebug\b",
    r"\btrace\b",
    r"\binternal\b",
    r"\borchestrat(?:e|ion)\b",
    r"\bstage(?:s)?\b",
    r"\btool call(?:s)?\b",
    r"\bshow .*reasoning\b",
)
_META_LEAK_PATTERNS = (
    r"\bone agent of many agents\b",
    r"\bfuture agents?\b",
    r"\bmessage analysis\b",
    r"\bknown context\b",
    r"\btool results?\b",
    r"\binternal (?:state|notes|thoughts|reasoning)\b",
    r"\bchain[- ]of[- ]thought\b",
)


def _parse_json_like(text: str) -> dict[str, Any] | None:
    raw = str(text or "").strip()
    if not raw:
        return None
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        parsed = json.loads(raw[start : end + 1])
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _as_text(value: Any, fallback: str = "") -> str:
    cleaned = str(value if value is not None else "").strip()
    return cleaned if cleaned else fallback


def _as_bool(value: Any, fallback: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "on"}:
            return True
        if normalized in {"false", "0", "no", "off"}:
            return False
    return fallback


def _as_string_list(value: Any) -> list[str]:
    output: list[str] = []
    if isinstance(value, list):
        source = value
    elif isinstance(value, str):
        source = [value]
    else:
        return output
    for item in source:
        cleaned = _as_text(item)
        if cleaned and cleaned not in output:
            output.append(cleaned)
    return output


def _coerce_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    return {}


def _normalize_analysis_payload(raw_analysis_text: str, known_context: dict[str, Any] | None = None) -> dict[str, Any]:
    parsed = _parse_json_like(raw_analysis_text)
    payload = parsed if isinstance(parsed, dict) else {}
    tool_results = payload.get("tool_results")
    if isinstance(tool_results, dict):
        payload = tool_results

    context_data = known_context if isinstance(known_context, dict) else {}
    context_summary = _as_text(payload.get("context_summary"))
    if not context_summary:
        member_name = _as_text(context_data.get("member_first_name"), "member")
        interface_name = _as_text(context_data.get("user_interface"), "unknown")
        chat_type = _as_text(context_data.get("chat_type"), "member")
        context_summary = (
            f"Interface: {interface_name}; chat type: {chat_type}; "
            f"latest message from: {member_name}."
        )

    style_raw = payload.get("response_style")
    style = style_raw if isinstance(style_raw, dict) else {}
    tone = _as_text(style.get("tone"), "friendly")
    length = _as_text(style.get("length"), "concise")
    if length not in {"very_short", "short", "concise", "medium", "detailed"}:
        length = "concise"

    tool_hints = []
    for hint in _as_string_list(payload.get("tool_hints")):
        if hint in _ALLOWED_TOOL_HINTS and hint not in tool_hints:
            tool_hints.append(hint)

    risk_flags = _as_string_list(payload.get("risk_flags"))
    topic = _as_text(payload.get("topic"), "general")
    intent = _as_text(payload.get("intent"), "answer_user")

    return {
        "topic": topic,
        "intent": intent,
        "needs_tools": _as_bool(payload.get("needs_tools"), fallback=False),
        "tool_hints": tool_hints,
        "risk_flags": risk_flags,
        "response_style": {
            "tone": tone,
            "length": length,
        },
        "context_summary": context_summary,
    }


def _user_requested_diagnostics(message_text: str) -> bool:
    text = _as_text(message_text).lower()
    if not text:
        return False
    return any(re.search(pattern, text) for pattern in _DIAGNOSTIC_REQUEST_PATTERNS)


def _line_has_meta_leak(text: str) -> bool:
    lowered = _as_text(text).lower()
    if not lowered:
        return False
    return any(re.search(pattern, lowered) for pattern in _META_LEAK_PATTERNS)


def _sanitize_final_response(
    text: str,
    *,
    user_message: str = "",
    allow_internal_diagnostics: bool = False,
) -> str:
    raw = str(text or "")
    if not raw.strip():
        return raw
    if allow_internal_diagnostics or _user_requested_diagnostics(user_message):
        return raw.strip()

    lines = [line for line in raw.splitlines() if line.strip()]
    filtered = [line for line in lines if not _line_has_meta_leak(line)]
    cleaned = "\n".join(filtered).strip()
    if cleaned:
        return cleaned

    # If every line was meta/internal, return a safe user-facing fallback.
    return "I can help with that. Could you restate what you want me to focus on?"



################
# ORCHESTRATOR #
################

class ConversationOrchestrator:

    def __init__(self, message: str, memberID: int, context: dict=None, messageID: int=None, options: dict=None):
        self._messages: list[Message] = []
        self._analysisStats: dict[str, Any] = {}
        self._devStats: dict[str, Any] = {}
        self._chatResponseMessage = ""

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
        self._context = context if isinstance(context, dict) else {}
        self._chatHostID = self._context.get("chat_host_id")
        self._chatType = self._context.get("chat_type")
        self._communityID = self._context.get("community_id")
        self._platform = self._context.get("platform")
        self._topicID = self._context.get("topic_id")
        self._messageTimestamp = coerce_datetime_utc(
            self._context.get("message_timestamp"),
            assume_tz=timezone.utc,
        )
        self._messageReceivedTimestamp = coerce_datetime_utc(
            self._context.get("message_received_timestamp"),
            assume_tz=timezone.utc,
        )
        if self._messageReceivedTimestamp is None:
            self._messageReceivedTimestamp = datetime.now(timezone.utc)
        if self._messageTimestamp is None:
            self._messageTimestamp = self._messageReceivedTimestamp
        self._shortHistory: list[dict[str, Any]] = []

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
            self._shortHistory = shortHistory if isinstance(shortHistory, list) else []
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
            topicID=self._topicID,
            timestamp=self._messageTimestamp,
        )

        # TODO Check options and pass to agents if necessary
        
    
    async def _emit_stage(self, stage: str, detail: str = "", **meta: Any) -> None:
        if not callable(self._stage_callback):
            return
        event = {
            "stage": str(stage),
            "detail": str(detail or ""),
            "timestamp": utc_now_iso(),
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
        # Build and append machine-readable known context for all downstream stages.
        temporalEnabled = _runtime_bool("temporal.enabled", True)
        temporalHistoryLimit = _runtime_int(
            "temporal.history_limit",
            _runtime_int("retrieval.conversation_short_history_limit", 20),
        )
        temporalExcerptMaxChars = _runtime_int("temporal.excerpt_max_chars", 160)
        temporalTimezone = str(_runtime_value("temporal.default_timezone", "UTC") or "UTC")
        temporalContext = (
            build_temporal_context(
                platform=self._platform,
                chat_type=self._chatType,
                chat_host_id=self._chatHostID,
                topic_id=self._topicID,
                timezone_name=temporalTimezone,
                now_utc=datetime.now(timezone.utc),
                inbound_sent_at=self._messageTimestamp,
                inbound_received_at=self._messageReceivedTimestamp,
                history_messages=self._shortHistory,
                history_limit=max(0, temporalHistoryLimit),
                excerpt_max_chars=max(0, temporalExcerptMaxChars),
            )
            if temporalEnabled
            else {
                "schema": "ryo.temporal_context.v1",
                "enabled": False,
                "reason": "runtime.temporal.enabled=false",
                "clock": {"now_utc": utc_now_iso()},
            }
        )

        knownContext = {
            "tool_name": "Known Context",
            "tool_results": {
                "user_interface": self._platform,
                "member_first_name": self._memberData.get("first_name"),
                "telegram_username": self._memberData.get("username"),
                "chat_type": self._chatType,
                "timestamp_utc": temporalContext.get("clock", {}).get("now_utc", utc_now_iso()),
                "temporal_context": temporalContext,
            }
        }
        self._messages.append(Message(role="tool", content=json.dumps(knownContext)))
        analysisMessages = list(self._messages)
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

        normalizedAnalysis = _normalize_analysis_payload(
            analysisResponseMessage,
            knownContext.get("tool_results"),
        )
        analysisMessagePayload = {
            "tool_name": "Message Analysis",
            "tool_results": normalizedAnalysis,
        }

        #print(ConsoleColors["default"])

        # Feed only a sanitized analysis handoff to the tool stage.
        await self._emit_stage("tools.start", "Evaluating tool calls.")
        toolStageMessages = list(self._messages)
        toolStageMessages.append(Message(role="tool", content=json.dumps(analysisMessagePayload)))
        self._toolsAgent = ToolCallingAgent(toolStageMessages, options=self._options)
        toolResponses = await self._toolsAgent.generateResponse()
        await self._emit_stage(
            "tools.complete",
            "Tool execution stage complete.",
            tool_calls=len(toolResponses),
        )
        self._messages.append(Message(role="tool", content=json.dumps(analysisMessagePayload)))

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

        allowDiagnostics = bool(self._options.get("allow_internal_diagnostics", False))
        sanitizedResponseMessage = _sanitize_final_response(
            responseMessage,
            user_message=self._message,
            allow_internal_diagnostics=allowDiagnostics,
        )
        if sanitizedResponseMessage != responseMessage:
            await self._emit_stage("response.sanitized", "Removed internal orchestration artifacts.")
        self._chatResponseMessage = sanitizedResponseMessage
        #print(ConsoleColors["default"])

        if hasattr(self, "_responseID"):
            self.storeResponse(self._responseID)
        
        assistantMessage = Message(role="assistant", content=sanitizedResponseMessage)
        # Add the final response to the overall chat history (role ASSISTANT)
        self._messages.append(assistantMessage)
        await self._emit_stage("orchestrator.complete", "Completed end-to-end orchestration.")

        return sanitizedResponseMessage

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
        else:
            toolPolicies = _coerce_dict(toolPolicies)

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
        optionMap = options if isinstance(options, dict) else {}
        stageCallback = optionMap.get("stage_callback")
        self._stageCallback = stageCallback if callable(stageCallback) else None
        runContext = _coerce_dict(optionMap.get("run_context"))
        self._runContext = {
            "run_id": str(runContext.get("run_id") or "").strip() or None,
            "member_id": runContext.get("member_id"),
        }
        if options:
            modelRequested = options.get("model_requested")
            if modelRequested in self._allowed_models:
                self._model = modelRequested
        optionToolPolicy = optionMap.get("tool_policy")
        if isinstance(optionToolPolicy, dict):
            for toolName, overrides in optionToolPolicy.items():
                if not isinstance(overrides, dict):
                    continue
                existing = toolPolicies.get(toolName)
                existingMap = _coerce_dict(existing)
                existingMap.update(_coerce_dict(overrides))
                toolPolicies[toolName] = existingMap

        sandboxPolicyMap: dict[str, dict[str, Any]] = {}
        try:
            sandboxStore = ToolSandboxPolicyStore()
            sandboxPolicyMap = sandboxStore.policy_map()
        except Exception as error:  # noqa: BLE001
            logger.warning(f"Unable to load persisted tool sandbox policies: {error}")

        optionSandboxPolicy = optionMap.get("tool_sandbox_policy")
        if isinstance(optionSandboxPolicy, dict):
            for toolName, rawPolicy in optionSandboxPolicy.items():
                if not isinstance(rawPolicy, dict):
                    continue
                existingPolicy = _coerce_dict(sandboxPolicyMap.get(str(toolName)))
                mergedPolicy = merge_sandbox_policies(existingPolicy, rawPolicy)
                try:
                    sandboxPolicyMap[str(toolName)] = normalize_tool_sandbox_policy(
                        mergedPolicy,
                        default_tool_name=str(toolName),
                    )
                except Exception as error:  # noqa: BLE001
                    logger.warning(f"Ignoring invalid sandbox policy override for {toolName}: {error}")

        for toolName, sandboxPolicy in sandboxPolicyMap.items():
            existing = _coerce_dict(toolPolicies.get(toolName))
            existingSandbox = _coerce_dict(existing.get("sandbox_policy"))
            existing["sandbox_policy"] = merge_sandbox_policies(sandboxPolicy, existingSandbox)
            if "side_effect_class" in sandboxPolicy and "side_effect_class" not in existing:
                existing["side_effect_class"] = sandboxPolicy.get("side_effect_class")
            if "require_approval" in sandboxPolicy and "require_approval" not in existing:
                existing["require_approval"] = sandboxPolicy.get("require_approval")
            if "dry_run" in sandboxPolicy and "dry_run" not in existing:
                existing["dry_run"] = sandboxPolicy.get("dry_run")
            toolPolicies[toolName] = existing

        enabledTools = set(_as_string_list(optionMap.get("enabled_tools")))
        deniedTools = set(_as_string_list(optionMap.get("denied_tools")))

        self._modelRouter = ModelRouter(
            inference_config=config._instance.inference,
            endpoint_override=endpointOverride,
        )

        sandboxDefaultsRaw = _runtime_value("tool_runtime.sandbox", {})
        sandboxDefaults = sandboxDefaultsRaw if isinstance(sandboxDefaultsRaw, dict) else {}
        runtimeDryRun = bool(
            optionMap.get(
                "tool_dry_run",
                _runtime_bool("tool_runtime.default_dry_run", False),
            )
        )
        approvalEnabled = bool(
            optionMap.get(
                "tool_human_approval",
                _runtime_bool("tool_runtime.enable_human_approval", True),
            )
        )
        approvalTimeoutSeconds = _float_value(
            optionMap.get("tool_approval_timeout_seconds"),
            _runtime_float("tool_runtime.default_approval_timeout_seconds", 45.0),
        )
        approvalPollIntervalSeconds = _float_value(
            optionMap.get("tool_approval_poll_interval_seconds"),
            _runtime_float("tool_runtime.approval_poll_interval_seconds", 0.25),
        )
        runtimeContext = {
            "run_id": self._runContext.get("run_id"),
            "member_id": self._runContext.get("member_id"),
        }
        sandboxEnforcer = ToolSandboxEnforcer(default_policy=sandboxDefaults)
        approvalManager = ApprovalManager()

        self._toolRuntime = ToolRuntime(
            api_keys={
                "brave_search": config._instance.brave_keys,
            },
            sandbox_enforcer=sandboxEnforcer,
            approval_manager=approvalManager,
            runtime_context=runtimeContext,
            enable_human_approval=approvalEnabled,
            default_approval_timeout_seconds=approvalTimeoutSeconds,
            approval_poll_interval_seconds=approvalPollIntervalSeconds,
            default_dry_run=runtimeDryRun,
        )
        self._toolSpecs = build_tool_specs(
            brave_search_fn=braveSearch,
            chat_history_search_fn=chatHistorySearch,
            knowledge_search_fn=knowledgeSearch,
            skip_tools_fn=skipTools,
            knowledge_domains=config.knowledgeDomains,
            custom_tool_entries=[],
        )
        customToolIndex: dict[str, dict[str, Any]] = {}
        try:
            customStore = ToolRegistryStore()
            for customTool in customStore.list_custom_tools(include_disabled=False):
                customToolIndex[str(customTool.get("name"))] = customTool
        except Exception as error:  # noqa: BLE001
            logger.warning(f"Unable to load custom tool registry entries: {error}")

        optionCustomTools = optionMap.get("custom_tools")
        if isinstance(optionCustomTools, list):
            for rawCustomTool in optionCustomTools:
                try:
                    normalized = normalize_custom_tool_payload(_coerce_dict(rawCustomTool))
                except Exception as error:  # noqa: BLE001
                    logger.warning(f"Ignoring invalid custom tool in options: {error}")
                    continue
                customToolIndex[normalized["name"]] = normalized

        if customToolIndex:
            self._toolSpecs = build_tool_specs(
                brave_search_fn=braveSearch,
                chat_history_search_fn=chatHistorySearch,
                knowledge_search_fn=knowledgeSearch,
                skip_tools_fn=skipTools,
                knowledge_domains=config.knowledgeDomains,
                custom_tool_entries=list(customToolIndex.values()),
            )

        if enabledTools:
            self._toolSpecs = {name: spec for name, spec in self._toolSpecs.items() if name in enabledTools}
        if deniedTools:
            self._toolSpecs = {name: spec for name, spec in self._toolSpecs.items() if name not in deniedTools}

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

    async def _emit_tool_runtime_event(self, event: dict[str, Any] | None) -> None:
        if not callable(self._stageCallback):
            return
        payload = _coerce_dict(event)
        if not payload:
            return
        record = {
            "event_type": str(payload.get("event_type") or "run.stage"),
            "stage": str(payload.get("stage") or "tools.runtime"),
            "status": str(payload.get("status") or "info"),
            "detail": str(payload.get("detail") or ""),
            "meta": _coerce_dict(payload.get("meta")),
            "timestamp": utc_now_iso(),
        }
        try:
            maybeAwaitable = self._stageCallback(record)
            if inspect.isawaitable(maybeAwaitable):
                await maybeAwaitable
        except Exception as error:  # noqa: BLE001
            logger.warning(f"Tool runtime stage callback failed [{record['stage']}]: {error}")

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
                auditEvents = toolResult.get("audit")
                if isinstance(auditEvents, list):
                    for auditEvent in auditEvents:
                        await self._emit_tool_runtime_event(_coerce_dict(auditEvent))
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
    def __init__(self, message_data: str, memberID: int):
        logger.info(f"New instance of the conversational agent.")
        self.messageData = message_data
        self.fromUser = MemberManager().getMemberByID(memberID)
        self._documents = list()

    async def generateResponse(self):
        logger.info(f"Generate a response for the conversational agent via async Ollama call.")
        # This is passed to ollama for the messages array
        messageHistory = []
        basePrompt = loadAgentSystemPrompt("chat_conversation")
        # First add the system messages to messageHistory
        messageHistory.append({
            "role" : "system",
            "content" : basePrompt
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
        basePrompt = loadAgentSystemPrompt("chat_conversation")
        # First add the system messages to messageHistory
        messageHistory.append({
            "role" : "system",
            "content" : basePrompt
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

    def __init__(self, message_data: str, from_user: dict):
        logger.info(f"New instance of the tweet Agent.")
        self.messageData = message_data
        self.fromUser = from_user
        self.tweetPrompt = message_data.get("tweet_prompt")
        self.tweetText = None
        self.messageHistory = []
        
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
