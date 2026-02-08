##########################################################################
#                                                                        #
#  This file (web_ui.py) contains the web-based user interface for       #
#  project ryo (run your own) chat, including the telegram miniapp       #
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

import json
import hashlib
import logging
import os
import re
import secrets
import socket
import time
from datetime import datetime
from typing import Any
from urllib import error as urllib_error
from urllib import request as urllib_request
from flask import (
    Flask,
    Response,
    abort,
    g,
    jsonify,
    redirect,
    render_template,
    request,
    session,
    stream_with_context,
    url_for,
)
from hypermindlabs.approval_manager import ApprovalManager, ApprovalValidationError
from hypermindlabs.agent_definitions import (
    AgentDefinitionStore,
    AgentDefinitionValidationError,
    normalize_agent_definition,
)
from hypermindlabs.capability_manifest import build_capability_manifest, find_capability
from hypermindlabs.replay_manager import ReplayManager
from hypermindlabs.run_manager import RunManager
from hypermindlabs.run_mode_handlers import normalize_run_mode, run_modes_manifest
from hypermindlabs.tool_registry import (
    ToolRegistryStore,
    ToolRegistryValidationError,
    build_tool_specs,
    register_runtime_tools,
    tool_catalog_entries,
)
from hypermindlabs.tool_sandbox import (
    ToolSandboxPolicyStore,
    ToolSandboxPolicyValidationError,
    normalize_tool_sandbox_policy,
)
from hypermindlabs.tool_runtime import ToolRuntime
from hypermindlabs.tool_test_harness import (
    ToolHarnessValidationError,
    ToolTestHarnessStore,
    build_contract_snapshot,
    compare_contract_snapshots,
    compare_golden_outputs,
)
from hypermindlabs.utils import ConfigManager, CustomFormatter, MemberManager, KnowledgeManager
from urllib.parse import parse_qs



###########
# LOGGING #
###########

# Clear any previous logging handlers
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

# Set the basic config to append logging data to a file
logPath = "logs/"
logFilename = "web_ui_log_" + datetime.now().strftime("%Y%m%d-%H%M%S") + ".txt"
print(logPath + logFilename)
logging.basicConfig(
    filename=logPath+logFilename,
    filemode="a",
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s (%(filename)s:%(lineno)d)", 
    level=logging.DEBUG
)

# Create a stream handler for cli output
console = logging.StreamHandler()
console.setLevel(logging.INFO)
console.setFormatter(CustomFormatter())
# add the handler to the root logger
logging.getLogger().addHandler(console)

# set higher logging level for httpx to avoid all GET and POST requests being logged
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("telegram").setLevel(logging.INFO)

logger = logging.getLogger(__name__)



###########
# GLOBALS #
###########

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

config = ConfigManager()
members = MemberManager()
playgroundRuns = RunManager(enable_db=True)
playgroundReplay = ReplayManager(playgroundRuns)
agentDefinitions = AgentDefinitionStore()
toolRegistry = ToolRegistryStore()
toolSandboxPolicies = ToolSandboxPolicyStore()
toolApprovals = ApprovalManager()
toolHarness = ToolTestHarnessStore()

logger.info(f"Database route status: {config.databaseRoute}")
miniappConfigIssues = config.getTelegramConfigIssues(require_owner=False, require_web_ui_url=False)
if miniappConfigIssues:
    logger.warning(
        "Telegram miniapp login is not fully configured. Missing/invalid values: %s",
        ", ".join(miniappConfigIssues),
    )


def _runtime_int(path: str, default: int) -> int:
    return config.runtimeInt(path, default)


def _runtime_bool(path: str, default: bool) -> bool:
    return config.runtimeBool(path, default)


def _runtime_str(path: str, default: str) -> str:
    value = config.runtimeValue(path, default)
    text = default if value is None else str(value).strip()
    return text if text else default


def _normalize_bind_host(value: str | None) -> str:
    host = str(value if value is not None else "").strip()
    if host == "::":
        return "0.0.0.0"
    return host or "127.0.0.1"


def _coerce_port(value: str | int | None, default: int) -> int:
    try:
        port = int(str(value).strip())
    except (TypeError, ValueError):
        port = int(default)
    if port <= 0:
        port = int(default)
    if port > 65535:
        port = 65535
    return port


def _display_host(bind_host: str) -> str:
    cleaned = _normalize_bind_host(bind_host)
    if cleaned == "0.0.0.0":
        return "127.0.0.1"
    return cleaned


def _port_available(bind_host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            probe.bind((bind_host, port))
        except OSError:
            return False
    return True


def _select_bind_port(bind_host: str, starting_port: int, scan_limit: int) -> int:
    max_attempts = max(0, int(scan_limit))
    current = _coerce_port(starting_port, 4747)
    for _ in range(max_attempts + 1):
        if _port_available(bind_host, current):
            return current
        if current >= 65535:
            break
        current += 1
    raise RuntimeError(
        f"No available local port found for host {bind_host}. "
        f"Start={starting_port}, scan_limit={scan_limit}"
    )


USERNAME_PATTERN = re.compile(r"^[A-Za-z0-9_]{3,96}$")


def _set_auth_message(message: str, kind: str = "error", active_tab: str | None = None) -> None:
    session["auth_message"] = str(message)
    session["auth_message_kind"] = str(kind)
    if active_tab in {"login", "signup"}:
        session["auth_active_tab"] = active_tab


def _database_unavailable() -> bool:
    route = config.databaseRoute if isinstance(config.databaseRoute, dict) else {}
    return str(route.get("status", "")).strip().lower() == "failed_all"


def _database_unavailable_message() -> str:
    return (
        "Account authentication is unavailable while PostgreSQL is unreachable. "
        "Run app.py setup to repair database connection/auth."
    )


def _normalize_username(value: str | None) -> str:
    return str(value or "").strip().lstrip("@")


def _username_is_valid(value: str) -> bool:
    return USERNAME_PATTERN.fullmatch(value) is not None


def _password_min_length() -> int:
    return max(8, _runtime_int("security.password_min_length", 12))


def _password_policy_hint() -> str:
    min_length = _password_min_length()
    return (
        "Password policy: "
        f"at least {min_length} characters, including uppercase, lowercase, a number, and a symbol."
    )


def _password_policy_error(password: str) -> str | None:
    min_length = _password_min_length()
    pattern = re.compile(
        r"(?=.*\d)(?=.*[a-z])(?=.*[A-Z])(?=.*[-+_!@#$%^&*.,?]).{"
        + str(min_length)
        + r",}"
    )
    if pattern.search(str(password or "")) is None:
        return _password_policy_hint()
    return None


def _api_auth_error(message: str = "Authentication required.") -> tuple[Response, int]:
    return jsonify({"status": "error", "message": message}), 401


def _require_member_api() -> dict | None:
    if not g.memberData:
        return None
    return g.memberData


def _member_roles(member: dict | None) -> list[str]:
    if not isinstance(member, dict):
        return []
    roles = member.get("roles")
    if not isinstance(roles, list):
        return []
    return [str(role).strip() for role in roles if str(role).strip()]


def _member_can_write_playground_registry(member: dict | None) -> bool:
    roles = set(_member_roles(member))
    return "admin" in roles or "owner" in roles


def _member_can_manage_tool_approvals(member: dict | None, approval_record: dict | None = None) -> bool:
    if _member_can_write_playground_registry(member):
        return True
    if not isinstance(member, dict) or not isinstance(approval_record, dict):
        return False
    member_id = int(member.get("member_id", 0))
    owner_id = approval_record.get("run_owner_member_id")
    requester_id = approval_record.get("requested_by_member_id")
    try:
        if owner_id is not None and int(owner_id) == member_id:
            return True
    except (TypeError, ValueError):
        pass
    try:
        if requester_id is not None and int(requester_id) == member_id:
            return True
    except (TypeError, ValueError):
        pass
    return False


def _api_forbidden(message: str = "Forbidden.") -> tuple[Response, int]:
    return jsonify({"status": "error", "message": message}), 403


def _request_json_dict() -> dict:
    payload = request.get_json(silent=True)
    return payload if isinstance(payload, dict) else {}


def _payload_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return bool(default)
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return bool(default)


def _tool_sandbox_items() -> list[dict[str, Any]]:
    policies_by_tool = {
        str(item.get("tool_name")): item
        for item in toolSandboxPolicies.list_policies()
        if str(item.get("tool_name") or "").strip()
    }
    catalog = _tool_catalog_items()
    output: list[dict[str, Any]] = []
    for tool in catalog:
        name = str(tool.get("name") or "").strip()
        persisted = policies_by_tool.get(name)
        policy = normalize_tool_sandbox_policy(
            persisted if isinstance(persisted, dict) else tool.get("sandbox_policy"),
            default_tool_name=name,
        )
        output.append(
            {
                "tool_name": name,
                "source": tool.get("source"),
                "side_effect_class": str(tool.get("side_effect_class") or policy.get("side_effect_class") or "read_only"),
                "approval_required": bool(tool.get("approval_required", policy.get("require_approval", False))),
                "dry_run": bool(tool.get("dry_run", policy.get("dry_run", False))),
                "sandbox_policy": policy,
            }
        )
    return output


def _lineage_from_request(payload: dict) -> dict[str, Any]:
    lineage = payload.get("lineage")
    if isinstance(lineage, dict):
        return lineage
    return {}


def _configured_inference_models() -> dict[str, str]:
    configured: dict[str, str] = {}
    inference = config.inference if isinstance(config.inference, dict) else {}
    for capability in ("chat", "tool", "generate", "embedding", "multimodal"):
        section = inference.get(capability)
        if not isinstance(section, dict):
            continue
        model_name = str(section.get("model") or "").strip()
        if model_name:
            configured[capability] = model_name
    return configured


def _resolve_ollama_host() -> str:
    host = _runtime_str("inference.default_ollama_host", "http://127.0.0.1:11434")
    return host.rstrip("/")


def _fetch_ollama_models(host: str, timeout_seconds: float = 3.0) -> tuple[list[str], str | None]:
    endpoint = f"{host}/api/tags"
    request_obj = urllib_request.Request(endpoint, method="GET")
    try:
        with urllib_request.urlopen(request_obj, timeout=float(timeout_seconds)) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib_error.HTTPError as error:
        return [], f"Ollama endpoint returned HTTP {error.code}"
    except urllib_error.URLError as error:
        return [], f"Failed to reach Ollama endpoint: {error.reason}"
    except TimeoutError:
        return [], "Timed out while reaching Ollama endpoint"
    except Exception as error:  # noqa: BLE001
        return [], str(error)

    discovered: list[str] = []
    for entry in payload.get("models", []) if isinstance(payload, dict) else []:
        model_name: str | None = None
        if isinstance(entry, dict):
            model_name = entry.get("model") or entry.get("name")
        cleaned = str(model_name or "").strip()
        if cleaned and cleaned not in discovered:
            discovered.append(cleaned)
    return discovered, None


def _noop_tool(**_: Any) -> dict[str, Any]:
    return {"status": "noop"}


def _tool_catalog_items() -> list[dict[str, Any]]:
    custom_tools = toolRegistry.list_custom_tools(include_disabled=True)
    specs = build_tool_specs(
        brave_search_fn=_noop_tool,
        chat_history_search_fn=_noop_tool,
        knowledge_search_fn=_noop_tool,
        skip_tools_fn=_noop_tool,
        knowledge_domains=config.knowledgeDomains,
        custom_tool_entries=custom_tools,
    )
    return tool_catalog_entries(specs, custom_entries=custom_tools)


def _builtin_tool_names() -> set[str]:
    specs = build_tool_specs(
        brave_search_fn=_noop_tool,
        chat_history_search_fn=_noop_tool,
        knowledge_search_fn=_noop_tool,
        skip_tools_fn=_noop_tool,
        knowledge_domains=config.knowledgeDomains,
        custom_tool_entries=[],
    )
    return set(specs.keys())


def _harness_brave_search(queryString: str, count: int = 5) -> list[dict[str, Any]]:
    return [
        {
            "title": "Harness stub result",
            "url": "https://example.com/harness",
            "description": f"braveSearch stubbed in web harness mode for query='{queryString}'.",
            "score": 0.0,
        }
        for _ in range(max(1, min(int(count), 5)))
    ]


def _harness_chat_history_search(
    queryString: str,
    count: int = 2,
    runtime_context: dict | None = None,
) -> list[dict[str, Any]]:
    context = runtime_context if isinstance(runtime_context, dict) else {}
    return [
        {
            "message": f"chatHistorySearch stub for '{queryString}'",
            "score": 0.0,
            "run_id": context.get("run_id"),
        }
        for _ in range(max(1, min(int(count), 5)))
    ]


def _harness_knowledge_search(queryString: str, count: int = 2) -> list[dict[str, Any]]:
    return [
        {
            "domain": "harness",
            "title": "Knowledge search stub",
            "excerpt": f"knowledgeSearch stub for '{queryString}'.",
            "score": 0.0,
        }
        for _ in range(max(1, min(int(count), 5)))
    ]


def _harness_skip_tools() -> dict[str, Any]:
    return {"status": "skipped", "source": "tool_harness"}


def _harness_tool_specs() -> dict[str, Any]:
    custom_tools = toolRegistry.list_custom_tools(include_disabled=False)
    return build_tool_specs(
        brave_search_fn=_harness_brave_search,
        chat_history_search_fn=_harness_chat_history_search,
        knowledge_search_fn=_harness_knowledge_search,
        skip_tools_fn=_harness_skip_tools,
        knowledge_domains=config.knowledgeDomains,
        custom_tool_entries=custom_tools,
    )


def _tool_policy_overrides() -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for row in _tool_sandbox_items():
        tool_name = str(row.get("tool_name") or "").strip()
        if not tool_name:
            continue
        policy = row.get("sandbox_policy")
        if not isinstance(policy, dict):
            policy = {}
        output[tool_name] = {
            "side_effect_class": str(row.get("side_effect_class") or "read_only"),
            "sandbox_policy": policy,
            "require_approval": bool(row.get("approval_required", False)),
            "dry_run": bool(row.get("dry_run", False)),
        }
    return output


def _tool_catalog_index() -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for row in _tool_catalog_items():
        tool_name = str(row.get("name") or "").strip()
        if tool_name:
            output[tool_name] = row
    return output


availableMenu = [
    {
        "title": "Agent Playground",
        "route": "/agent-playground",
        "roles": ["user", "admin", "owner"]
    },
    {
        "title": "Community Engagement",
        "route": "/community-engagement",
        "roles": ["user", "admin", "owner"]
    },
    {
        "title": "Hydra Network",
        "route": "/hydra-network",
        "roles": ["user", "admin", "owner"]
    },
    {
        "title": "Admin Tools",
        "route": "/admin-tools",
        "roles": ["admin", "owner"]
    },
    {
        "title": "Knowledge Tools",
        "route": "/knowledge-tools",
        "roles": ["admin", "owner"]
    },
    {
        "title": "Newsletter",
        "route": "/newsletter",
        "roles": ["admin", "owner"]
    }
]

app = Flask(__name__)


def _resolve_web_secret_key() -> str:
    explicit = str(os.getenv("RYO_WEB_SECRET_KEY") or os.getenv("FLASK_SECRET_KEY") or "").strip()
    if explicit:
        return explicit

    # Deterministic fallback prevents session invalidation across watchdog restarts.
    route = config.databaseRoute if isinstance(config.databaseRoute, dict) else {}
    seed = "|".join(
        [
            str(config.botName or ""),
            str(config.bot_id or ""),
            str(config.bot_token or ""),
            str(route.get("primary_conninfo") or ""),
        ]
    )
    if not seed.strip("|"):
        seed = secrets.token_hex(32)
    logger.warning("No RYO_WEB_SECRET_KEY/FLASK_SECRET_KEY configured; using deterministic fallback secret.")
    return hashlib.sha256(seed.encode("utf-8")).hexdigest()


app.secret_key = _resolve_web_secret_key()
app.config["SESSION_COOKIE_HTTPONLY"] = True
app.config["SESSION_COOKIE_SAMESITE"] = "Lax"


@app.before_request
def loadUser():
    # Build user data and menu data
    g.authMessage = session.pop("auth_message", None)
    g.authMessageKind = session.pop("auth_message_kind", "error")
    g.authActiveTab = session.pop("auth_active_tab", "login")
    if "member_id" not in session:
        g.memberData = None
        g.menuData = None
    else:
        member = members.getMemberByID(session["member_id"])
        if member is None:
            logger.warning("Session member_id=%s not found. Clearing stale auth session.", session.get("member_id"))
            session.pop("member_id", None)
            g.memberData = None
            g.menuData = None
            return

        g.memberData = member
        # Create the menu data based on user roles
        memberRoles = member.get("roles")
        g.menuData = [menuItem for menuItem in availableMenu if any(role in memberRoles for role in menuItem["roles"])]


# This injects jijna context into every render template call
@app.context_processor
def baseTemplateData():
    baseTemplateContext = {
        "memberData": g.memberData,
        "menuData": g.menuData,
        "authMessage": g.authMessage,
        "authMessageKind": g.authMessageKind,
        "authActiveTab": g.authActiveTab,
        "passwordPolicyHint": _password_policy_hint(),
        "passwordPolicyMinLength": _password_min_length(),
    }
    return baseTemplateContext



######################
# BEGIN FLASK ROUTES #
######################

@app.route("/")
def index():
    print("A visit to the homepage")
    contentData = {
        "title": "Homepage",
    }

    if g.memberData:
        print("User is logged in")
        return render_template("base-html.html", 
            contentData=contentData
        )
    else:
        print("User isn't logged in")
        return render_template("base-html.html")


@app.route("/base")
def test():
    print("Base page")
    
    return render_template("base-html.html")


@app.route("/admin-tools")
def adminTools():
    print("A visit to Admin Tools page")
    
    # Get the member and make sure they have the roles to access
    if g.memberData:
        linkDetails = next(linkData for linkData in availableMenu if linkData.get("route") == "/admin-tools")
        availableRoles = g.memberData.get("roles")
        rolesAllowed = linkDetails.get("roles")

        if (any(role in availableRoles for role in rolesAllowed)):
            print("Member is authorized")

            contentData = {
                "title": "Admin Tools",
                "sections": ["spam control","user data"]
            }
            
            return render_template("admin-tools.html", 
                contentData=contentData
            )

    print("Not authorized")
    return redirect(url_for("index"))


@app.route("/knowledge-tools")
def knowledgeTools():
    print("A visit to Admin Tools page")
    
    # Get the member and make sure they have the roles to access
    if g.memberData:
        linkDetails = next(linkData for linkData in availableMenu if linkData.get("route") == "/knowledge-tools")
        availableRoles = g.memberData.get("roles")
        rolesAllowed = linkDetails.get("roles")

        if (any(role in availableRoles for role in rolesAllowed)):
            print("Member is authorized")

            contentData = {
                "title": "Knowledge Tools",
                "sections": ["Domain 1","Domain 2"]
            }

            knowledgeData = KnowledgeManager().getKnowledge()
            #print(knowledgeData[0]["document_metadata"])
            
            return render_template("knowledge-tools.html", 
                contentData=contentData,
                knowledgeData=list() if knowledgeData is None else knowledgeData
            )

    print("Not authorized")
    return redirect(url_for("index"))


@app.route("/agent-playground")
def agentPlayground():
    print("A visit to Agent Playground page")
    contentData = {
        "title": "Agent Playground",
        "sections": ["agent chat","my agents"]
    }

    if g.memberData:
        print("User is logged in")
        return render_template("agent-playground.html", 
            contentData=contentData
        )
    else:
        print("User isn't logged in")
        return redirect(url_for("index"))


@app.route("/community-engagement")
def communityEngagement():
    print("A visit to Community Engagement page")
    contentData = {
        "title": "Community Engagement",
        "sections": ["overview","score records"]
    }

    if g.memberData:
        print("User is logged in")
        return render_template("community-engagement.html", 
            contentData=contentData
        )
    else:
        print("User isn't logged in")
        return redirect(url_for("index"))


@app.route("/hydra-network")
def hydraNetwork():
    print("A visit to Hydra Network page")
    contentData = {
        "title": "Hydra Network",
        "sections": ["overview","my nodes"]
    }

    if g.memberData:
        print("User is logged in")
        return render_template("hydra-network.html", 
            contentData=contentData
        )
    else:
        print("User isn't logged in")
        return redirect(url_for("index"))

    
@app.route("/logout")
def logout():
    print("Logging out the user")
    # remove the username from the session if it's there
    session.pop('member_id', None)
    return redirect(url_for('index'))


@app.route("/profile")
def profile():
    contentData = {
        "title": "Member Profile",
        "sections": ["basics", "defaults", "chat history"]
    }

    if g.memberData:
        print("User is logged in")
        return render_template("profile.html", 
            contentData=contentData
        )
    else:
        print("User isn't logged in")
        return redirect(url_for("index"))


@app.route("/profile/<endpoint>", methods=["GET", "POST"])
def profileAPI(endpoint = None):
    # First check for a logged in user
    if g.memberData:
        memberID = g.memberData.get("member_id")
        if endpoint == "email":
            if request.method == "POST":
                response = {
                    "endpoint": endpoint
                }
                email = request.form.get("member-email")
                members.updateMemberEmail(memberID, email)
                response["member_email"] = email

                return jsonify(response)
    
@app.route("/miniapp/dashboard/")
def miniAppDashboard():
    print("Mini App Dashboard")

    return redirect(url_for("static", filename="miniapp/index.html"))



######################
## POST ONLY ROUTES ##
#--------------------#
    
@app.route("/login", methods=['POST'])
def login():
    logger.info("Web login requested.")
    if _database_unavailable():
        logger.error("Web login blocked: database route unavailable.")
        _set_auth_message(_database_unavailable_message(), kind="error", active_tab="login")
        return redirect(url_for("index"))

    username = _normalize_username(request.form.get("username"))
    password = str(request.form.get("userpass") or "").strip()

    if not username or not password:
        _set_auth_message("Username and password are required.", kind="error", active_tab="login")
        return redirect(url_for("index"))

    member = members.loginMember(username, password)
    if member is None:
        logger.warning("Web login failed for username='%s'.", username)
        _set_auth_message("Login failed. Check your username/password.", kind="error", active_tab="login")
        return redirect(url_for("index"))

    session["member_id"] = member["member_id"]
    logger.info("Web login success for member_id=%s username='%s'.", member.get("member_id"), username)
    _set_auth_message(f"Welcome back {member.get('first_name', username)}.", kind="success", active_tab="login")
    return redirect(url_for("index"))


@app.route("/signup", methods=["POST"])
def signup():
    logger.info("Web signup requested.")
    if _database_unavailable():
        logger.error("Web signup blocked: database route unavailable.")
        _set_auth_message(_database_unavailable_message(), kind="error", active_tab="signup")
        return redirect(url_for("index"))

    username = _normalize_username(request.form.get("username"))
    password = str(request.form.get("userpass") or "").strip()
    passwordConfirm = str(request.form.get("userpass-confirm") or "").strip()
    firstName = str(request.form.get("first_name") or "").strip()
    lastName = str(request.form.get("last_name") or "").strip()

    if not username or not password:
        _set_auth_message("Username and password are required to sign up.", kind="error", active_tab="signup")
        return redirect(url_for("index"))
    if not _username_is_valid(username):
        _set_auth_message(
            "Username must be 3-96 chars: letters, numbers, underscore.",
            kind="error",
            active_tab="signup",
        )
        return redirect(url_for("index"))
    if password != passwordConfirm:
        _set_auth_message("Password confirmation does not match.", kind="error", active_tab="signup")
        return redirect(url_for("index"))
    passwordError = _password_policy_error(password)
    if passwordError is not None:
        _set_auth_message(passwordError, kind="error", active_tab="signup")
        return redirect(url_for("index"))

    member, errorMessage = members.registerWebMember(
        username=username,
        password=password,
        firstName=firstName or username,
        lastName=lastName or None,
    )
    if member is None:
        logger.warning("Web signup failed for username='%s': %s", username, errorMessage)
        _set_auth_message(
            errorMessage or "Signup failed. Please try a different username.",
            kind="error",
            active_tab="signup",
        )
        return redirect(url_for("index"))

    session["member_id"] = member["member_id"]
    logger.info("Web signup success for member_id=%s username='%s'.", member.get("member_id"), username)
    _set_auth_message(
        f"Signup complete. Welcome {member.get('first_name', username)}.",
        kind="success",
        active_tab="signup",
    )
    return redirect(url_for("index"))


@app.post("/miniapp-login")
def miniappLogin():
    logger.info(f"{ConsoleColors['yellow']}Validate and login Telegram miniapp user.{ConsoleColors['default']}")
    miniappConfigIssues = config.getTelegramConfigIssues(require_owner=False, require_web_ui_url=False)
    if miniappConfigIssues:
        logger.error(
            "Miniapp login unavailable: missing/invalid Telegram config values: %s",
            ", ".join(miniappConfigIssues),
        )
        return jsonify(
            {
                "status": "error",
                "message": "Telegram miniapp login is not configured on this host.",
                "missing": miniappConfigIssues,
            }
        ), 503

    queryString = request.form.get("query-string")
    if not queryString:
        logger.warning("Miniapp login rejected: missing query-string payload.")
        return jsonify({"status": "error", "message": "Missing miniapp query-string payload."}), 400

    # Generate a hash with the data check string and secret key (from utils)
    memberID = members.validateMiniappData(queryString)

    if memberID is not None:
        session["member_id"] = memberID

    return redirect(url_for("index"))


@app.get("/api/agent-playground/bootstrap")
def playgroundBootstrapAPI():
    member = _require_member_api()
    if member is None:
        return _api_auth_error()

    memberID = int(member.get("member_id", 0))
    roles = member.get("roles") if isinstance(member.get("roles"), list) else []
    manifest = build_capability_manifest(member_roles=roles)
    response = {
        "status": "ok",
        "member": {
            "member_id": memberID,
            "username": member.get("username"),
            "first_name": member.get("first_name"),
            "last_name": member.get("last_name"),
            "roles": roles,
        },
        "run_modes": run_modes_manifest(),
        "manifest": manifest,
        "runs": playgroundRuns.list_runs(limit=15, member_id=memberID),
        "metrics": playgroundRuns.metrics_summary(),
        "agent_definitions": agentDefinitions.list_definitions(owner_member_id=memberID),
        "tool_registry": _tool_catalog_items(),
        "tool_sandbox_policies": _tool_sandbox_items(),
        "tool_harness_cases": toolHarness.list_cases(),
        "approval_queue": toolApprovals.list_requests(
            status="pending",
            member_id=None if _member_can_write_playground_registry(member) else memberID,
            limit=100,
        ),
        "can_write_registry": _member_can_write_playground_registry(member),
        "can_manage_approvals": _member_can_write_playground_registry(member),
        "can_write_harness": _member_can_write_playground_registry(member),
    }
    return jsonify(response)


@app.get("/api/agent-playground/capabilities")
def playgroundCapabilitiesAPI():
    member = _require_member_api()
    if member is None:
        return _api_auth_error()

    roles = member.get("roles") if isinstance(member.get("roles"), list) else []
    manifest = build_capability_manifest(member_roles=roles)
    return jsonify({"status": "ok", "manifest": manifest})


@app.get("/api/agent-playground/capabilities/<capability_id>")
def playgroundCapabilityByIDAPI(capability_id: str):
    member = _require_member_api()
    if member is None:
        return _api_auth_error()

    roles = member.get("roles") if isinstance(member.get("roles"), list) else []
    manifest = build_capability_manifest(member_roles=roles)
    capability = find_capability(manifest, capability_id)
    if capability is None:
        return jsonify({"status": "error", "message": "Capability not found."}), 404
    return jsonify({"status": "ok", "capability": capability})


@app.get("/api/agent-playground/models")
def playgroundModelsAPI():
    member = _require_member_api()
    if member is None:
        return _api_auth_error()

    host = _resolve_ollama_host()
    models, error = _fetch_ollama_models(host, timeout_seconds=3.5)
    configured = _configured_inference_models()
    configured_values = [value for value in configured.values() if value]

    default_requested = (
        configured.get("chat")
        or configured.get("tool")
        or configured.get("generate")
        or (configured_values[0] if configured_values else "")
    )

    return jsonify(
        {
            "status": "ok",
            "ollama_host": host,
            "available_models": models,
            "available_count": len(models),
            "configured_models": configured,
            "default_model_requested": default_requested,
            "error": error,
        }
    )


def _can_access_definition(member: dict, detail: dict) -> bool:
    memberID = int(member.get("member_id", 0))
    ownerID = int(detail.get("owner_member_id", -1))
    if ownerID == memberID:
        return True
    return _member_can_write_playground_registry(member)


@app.get("/api/agent-playground/agent-definitions")
def playgroundAgentDefinitionsListAPI():
    member = _require_member_api()
    if member is None:
        return _api_auth_error()

    memberID = int(member.get("member_id", 0))
    includeAll = str(request.args.get("all", "")).strip().lower() in {"1", "true", "yes"}
    ownerFilter = None if includeAll and _member_can_write_playground_registry(member) else memberID
    definitions = agentDefinitions.list_definitions(owner_member_id=ownerFilter)
    return jsonify(
        {
            "status": "ok",
            "definitions": definitions,
            "can_write": True,
        }
    )


@app.get("/api/agent-playground/agent-definitions/<definition_id>")
def playgroundAgentDefinitionsGetAPI(definition_id: str):
    member = _require_member_api()
    if member is None:
        return _api_auth_error()

    versionRaw = request.args.get("version")
    versionValue: int | None
    try:
        versionValue = int(versionRaw) if versionRaw is not None else None
    except (TypeError, ValueError):
        versionValue = None

    detail = agentDefinitions.get_definition(definition_id, version=versionValue)
    if detail is None:
        return jsonify({"status": "error", "message": "Agent definition not found."}), 404
    if not _can_access_definition(member, detail):
        return _api_forbidden("Access denied for this agent definition.")
    return jsonify({"status": "ok", "definition": detail})


@app.post("/api/agent-playground/agent-definitions")
def playgroundAgentDefinitionsCreateAPI():
    member = _require_member_api()
    if member is None:
        return _api_auth_error()

    payload = _request_json_dict()
    definitionPayload = payload.get("definition")
    if not isinstance(definitionPayload, dict):
        definitionPayload = payload
    changeSummary = str(payload.get("change_summary") or "Initial definition").strip()
    try:
        detail = agentDefinitions.create_definition(
            definitionPayload,
            author_member_id=int(member.get("member_id", 0)),
            change_summary=changeSummary,
        )
    except AgentDefinitionValidationError as error:
        return jsonify({"status": "error", "message": str(error)}), 400

    return jsonify({"status": "ok", "definition": detail}), 201


@app.post("/api/agent-playground/agent-definitions/<definition_id>/versions")
def playgroundAgentDefinitionsVersionCreateAPI(definition_id: str):
    member = _require_member_api()
    if member is None:
        return _api_auth_error()

    current = agentDefinitions.get_definition(definition_id)
    if current is None:
        return jsonify({"status": "error", "message": "Agent definition not found."}), 404
    if not _can_access_definition(member, current):
        return _api_forbidden("Access denied for this agent definition.")

    payload = _request_json_dict()
    definitionPayload = payload.get("definition")
    if not isinstance(definitionPayload, dict):
        return jsonify({"status": "error", "message": "definition object is required."}), 400
    changeSummary = str(payload.get("change_summary") or "Updated definition").strip()

    try:
        detail = agentDefinitions.create_version(
            definition_id,
            definitionPayload,
            author_member_id=int(member.get("member_id", 0)),
            change_summary=changeSummary,
        )
    except AgentDefinitionValidationError as error:
        return jsonify({"status": "error", "message": str(error)}), 400

    return jsonify({"status": "ok", "definition": detail})


@app.post("/api/agent-playground/agent-definitions/<definition_id>/rollback")
def playgroundAgentDefinitionsRollbackAPI(definition_id: str):
    member = _require_member_api()
    if member is None:
        return _api_auth_error()

    current = agentDefinitions.get_definition(definition_id)
    if current is None:
        return jsonify({"status": "error", "message": "Agent definition not found."}), 404
    if not _can_access_definition(member, current):
        return _api_forbidden("Access denied for this agent definition.")

    payload = _request_json_dict()
    targetVersionRaw = payload.get("target_version")
    try:
        targetVersion = int(targetVersionRaw)
    except (TypeError, ValueError):
        targetVersion = 0
    if targetVersion <= 0:
        return jsonify({"status": "error", "message": "target_version must be a positive integer."}), 400

    changeSummary = str(payload.get("change_summary") or f"Rollback to v{targetVersion}").strip()
    try:
        detail = agentDefinitions.rollback(
            definition_id,
            target_version=targetVersion,
            author_member_id=int(member.get("member_id", 0)),
            change_summary=changeSummary,
        )
    except AgentDefinitionValidationError as error:
        return jsonify({"status": "error", "message": str(error)}), 400
    return jsonify({"status": "ok", "definition": detail})


@app.get("/api/agent-playground/agent-definitions/<definition_id>/export")
def playgroundAgentDefinitionsExportAPI(definition_id: str):
    member = _require_member_api()
    if member is None:
        return _api_auth_error()

    detail = agentDefinitions.get_definition(definition_id)
    if detail is None:
        return jsonify({"status": "error", "message": "Agent definition not found."}), 404
    if not _can_access_definition(member, detail):
        return _api_forbidden("Access denied for this agent definition.")

    fmt = str(request.args.get("format") or "json").strip().lower()
    versionRaw = request.args.get("version")
    try:
        versionValue = int(versionRaw) if versionRaw is not None else None
    except (TypeError, ValueError):
        versionValue = None
    try:
        text = agentDefinitions.export_definition(definition_id, version=versionValue, fmt=fmt)
    except AgentDefinitionValidationError as error:
        return jsonify({"status": "error", "message": str(error)}), 400

    return jsonify(
        {
            "status": "ok",
            "definition_id": definition_id,
            "format": fmt,
            "payload": text,
        }
    )


@app.post("/api/agent-playground/agent-definitions/import")
def playgroundAgentDefinitionsImportAPI():
    member = _require_member_api()
    if member is None:
        return _api_auth_error()

    payload = _request_json_dict()
    rawPayload = str(payload.get("raw_payload") or payload.get("payload") or "").strip()
    fmt = str(payload.get("format") or "json").strip().lower()
    changeSummary = str(payload.get("change_summary") or "Imported definition").strip()
    try:
        detail = agentDefinitions.import_definition(
            raw_payload=rawPayload,
            fmt=fmt,
            author_member_id=int(member.get("member_id", 0)),
            change_summary=changeSummary,
        )
    except AgentDefinitionValidationError as error:
        return jsonify({"status": "error", "message": str(error)}), 400
    return jsonify({"status": "ok", "definition": detail}), 201


@app.get("/api/agent-playground/tools")
def playgroundToolsListAPI():
    member = _require_member_api()
    if member is None:
        return _api_auth_error()
    return jsonify(
        {
            "status": "ok",
            "tools": _tool_catalog_items(),
            "can_write": _member_can_write_playground_registry(member),
        }
    )


@app.post("/api/agent-playground/tools")
def playgroundToolsUpsertAPI():
    member = _require_member_api()
    if member is None:
        return _api_auth_error()
    if not _member_can_write_playground_registry(member):
        return _api_forbidden("Tool registry writes require admin or owner role.")

    payload = _request_json_dict()
    toolPayload = payload.get("tool")
    if not isinstance(toolPayload, dict):
        toolPayload = payload
    toolName = str(toolPayload.get("name") or "").strip()
    if toolName in _builtin_tool_names():
        return jsonify({"status": "error", "message": "Builtin tool names cannot be overridden."}), 400

    try:
        normalized = toolRegistry.upsert_custom_tool(
            toolPayload,
            actor_member_id=int(member.get("member_id", 0)),
        )
    except ToolRegistryValidationError as error:
        return jsonify({"status": "error", "message": str(error)}), 400
    return jsonify({"status": "ok", "tool": normalized, "tools": _tool_catalog_items()})


@app.post("/api/agent-playground/tools/<tool_name>")
def playgroundToolsUpdateAPI(tool_name: str):
    member = _require_member_api()
    if member is None:
        return _api_auth_error()
    if not _member_can_write_playground_registry(member):
        return _api_forbidden("Tool registry writes require admin or owner role.")

    payload = _request_json_dict()
    toolPayload = payload.get("tool")
    if not isinstance(toolPayload, dict):
        toolPayload = payload
    toolPayload = dict(toolPayload)
    toolPayload["name"] = tool_name
    if tool_name in _builtin_tool_names():
        return jsonify({"status": "error", "message": "Builtin tool names cannot be edited here."}), 400

    try:
        normalized = toolRegistry.upsert_custom_tool(
            toolPayload,
            actor_member_id=int(member.get("member_id", 0)),
        )
    except ToolRegistryValidationError as error:
        return jsonify({"status": "error", "message": str(error)}), 400
    return jsonify({"status": "ok", "tool": normalized, "tools": _tool_catalog_items()})


@app.delete("/api/agent-playground/tools/<tool_name>")
def playgroundToolsDeleteAPI(tool_name: str):
    member = _require_member_api()
    if member is None:
        return _api_auth_error()
    if not _member_can_write_playground_registry(member):
        return _api_forbidden("Tool registry writes require admin or owner role.")
    if tool_name in _builtin_tool_names():
        return jsonify({"status": "error", "message": "Builtin tools cannot be removed."}), 400
    try:
        removed = toolRegistry.remove_custom_tool(tool_name)
    except ToolRegistryValidationError as error:
        return jsonify({"status": "error", "message": str(error)}), 400
    if not removed:
        return jsonify({"status": "error", "message": "Custom tool not found."}), 404
    return jsonify({"status": "ok", "tools": _tool_catalog_items()})


@app.get("/api/agent-playground/tools/sandbox-policies")
def playgroundToolSandboxPoliciesListAPI():
    member = _require_member_api()
    if member is None:
        return _api_auth_error()
    return jsonify(
        {
            "status": "ok",
            "policies": _tool_sandbox_items(),
            "can_write": _member_can_write_playground_registry(member),
        }
    )


@app.get("/api/agent-playground/tools/sandbox-policies/<tool_name>")
def playgroundToolSandboxPolicyGetAPI(tool_name: str):
    member = _require_member_api()
    if member is None:
        return _api_auth_error()

    policy = toolSandboxPolicies.get_policy(tool_name)
    if policy is None:
        return jsonify({"status": "error", "message": "Sandbox policy not found."}), 404
    return jsonify({"status": "ok", "policy": policy})


@app.post("/api/agent-playground/tools/sandbox-policies")
def playgroundToolSandboxPolicyUpsertAPI():
    member = _require_member_api()
    if member is None:
        return _api_auth_error()
    if not _member_can_write_playground_registry(member):
        return _api_forbidden("Tool sandbox writes require admin or owner role.")

    payload = _request_json_dict()
    policyPayload = payload.get("policy")
    if not isinstance(policyPayload, dict):
        policyPayload = payload

    try:
        policy = toolSandboxPolicies.upsert_policy(
            policyPayload,
            actor_member_id=int(member.get("member_id", 0)),
        )
    except ToolSandboxPolicyValidationError as error:
        return jsonify({"status": "error", "message": str(error)}), 400
    return jsonify({"status": "ok", "policy": policy, "policies": _tool_sandbox_items()})


@app.post("/api/agent-playground/tools/sandbox-policies/<tool_name>")
def playgroundToolSandboxPolicyUpdateAPI(tool_name: str):
    member = _require_member_api()
    if member is None:
        return _api_auth_error()
    if not _member_can_write_playground_registry(member):
        return _api_forbidden("Tool sandbox writes require admin or owner role.")

    payload = _request_json_dict()
    policyPayload = payload.get("policy")
    if not isinstance(policyPayload, dict):
        policyPayload = payload
    policyPayload = dict(policyPayload)
    policyPayload["tool_name"] = tool_name

    try:
        policy = toolSandboxPolicies.upsert_policy(
            policyPayload,
            actor_member_id=int(member.get("member_id", 0)),
        )
    except ToolSandboxPolicyValidationError as error:
        return jsonify({"status": "error", "message": str(error)}), 400
    return jsonify({"status": "ok", "policy": policy, "policies": _tool_sandbox_items()})


@app.delete("/api/agent-playground/tools/sandbox-policies/<tool_name>")
def playgroundToolSandboxPolicyDeleteAPI(tool_name: str):
    member = _require_member_api()
    if member is None:
        return _api_auth_error()
    if not _member_can_write_playground_registry(member):
        return _api_forbidden("Tool sandbox writes require admin or owner role.")
    try:
        removed = toolSandboxPolicies.remove_policy(tool_name)
    except ToolSandboxPolicyValidationError as error:
        return jsonify({"status": "error", "message": str(error)}), 400
    if not removed:
        return jsonify({"status": "error", "message": "Sandbox policy not found."}), 404
    return jsonify({"status": "ok", "policies": _tool_sandbox_items()})


@app.get("/api/agent-playground/tools/harness/cases")
def playgroundToolHarnessCasesListAPI():
    member = _require_member_api()
    if member is None:
        return _api_auth_error()
    return jsonify(
        {
            "status": "ok",
            "cases": toolHarness.list_cases(),
            "tools": _tool_catalog_items(),
            "can_write": _member_can_write_playground_registry(member),
        }
    )


@app.post("/api/agent-playground/tools/harness/cases")
def playgroundToolHarnessCaseUpsertAPI():
    member = _require_member_api()
    if member is None:
        return _api_auth_error()
    if not _member_can_write_playground_registry(member):
        return _api_forbidden("Tool harness writes require admin or owner role.")

    payload = _request_json_dict()
    casePayload = payload.get("case")
    if not isinstance(casePayload, dict):
        casePayload = payload

    catalog = _tool_catalog_index()
    tool_name = str(casePayload.get("tool_name") or "").strip()
    if not tool_name:
        return jsonify({"status": "error", "message": "tool_name is required."}), 400
    if tool_name not in catalog:
        return jsonify({"status": "error", "message": "Tool is not available in catalog."}), 400

    try:
        case = toolHarness.upsert_case(
            casePayload,
            actor_member_id=int(member.get("member_id", 0)),
        )
    except ToolHarnessValidationError as error:
        return jsonify({"status": "error", "message": str(error)}), 400

    return jsonify(
        {
            "status": "ok",
            "case": case,
            "cases": toolHarness.list_cases(),
        }
    )


@app.delete("/api/agent-playground/tools/harness/cases/<case_id>")
def playgroundToolHarnessCaseDeleteAPI(case_id: str):
    member = _require_member_api()
    if member is None:
        return _api_auth_error()
    if not _member_can_write_playground_registry(member):
        return _api_forbidden("Tool harness writes require admin or owner role.")

    removed = toolHarness.remove_case(case_id)
    if not removed:
        return jsonify({"status": "error", "message": "Tool harness case not found."}), 404
    return jsonify({"status": "ok", "cases": toolHarness.list_cases()})


@app.post("/api/agent-playground/tools/harness/cases/<case_id>/golden")
def playgroundToolHarnessCaseGoldenAPI(case_id: str):
    member = _require_member_api()
    if member is None:
        return _api_auth_error()
    if not _member_can_write_playground_registry(member):
        return _api_forbidden("Tool harness writes require admin or owner role.")

    case = toolHarness.get_case(case_id)
    if case is None:
        return jsonify({"status": "error", "message": "Tool harness case not found."}), 404

    payload = _request_json_dict()
    has_explicit = "golden_output" in payload
    goldenOutput = payload.get("golden_output")
    if not has_explicit:
        lastReport = case.get("last_report")
        if isinstance(lastReport, dict):
            result = lastReport.get("result")
            if isinstance(result, dict):
                goldenOutput = result.get("tool_results")
    if goldenOutput is None:
        return jsonify({"status": "error", "message": "No golden output available to save."}), 400

    try:
        updated = toolHarness.set_golden_output(
            case_id,
            goldenOutput,
            actor_member_id=int(member.get("member_id", 0)),
        )
    except ToolHarnessValidationError as error:
        return jsonify({"status": "error", "message": str(error)}), 400

    return jsonify({"status": "ok", "case": updated})


@app.post("/api/agent-playground/tools/harness/cases/<case_id>/contract")
def playgroundToolHarnessCaseContractAPI(case_id: str):
    member = _require_member_api()
    if member is None:
        return _api_auth_error()
    if not _member_can_write_playground_registry(member):
        return _api_forbidden("Tool harness writes require admin or owner role.")

    case = toolHarness.get_case(case_id)
    if case is None:
        return jsonify({"status": "error", "message": "Tool harness case not found."}), 404

    payload = _request_json_dict()
    contractSnapshot = payload.get("contract_snapshot")
    if not isinstance(contractSnapshot, dict):
        tool_name = str(case.get("tool_name") or "").strip()
        catalog = _tool_catalog_index()
        tool_row = catalog.get(tool_name)
        if not isinstance(tool_row, dict):
            return jsonify({"status": "error", "message": "Tool contract source is unavailable."}), 400
        contractSnapshot = build_contract_snapshot(tool_name, tool_row.get("input_schema"))

    try:
        updated = toolHarness.set_contract_snapshot(
            case_id,
            contractSnapshot,
            actor_member_id=int(member.get("member_id", 0)),
        )
    except ToolHarnessValidationError as error:
        return jsonify({"status": "error", "message": str(error)}), 400

    return jsonify({"status": "ok", "case": updated, "contract_snapshot": contractSnapshot})


@app.post("/api/agent-playground/tools/harness/run")
def playgroundToolHarnessRunAPI():
    member = _require_member_api()
    if member is None:
        return _api_auth_error()
    if not _member_can_write_playground_registry(member):
        return _api_forbidden("Tool harness execution requires admin or owner role.")

    payload = _request_json_dict()
    case_id = str(payload.get("case_id") or "").strip()
    if not case_id:
        return jsonify({"status": "error", "message": "case_id is required."}), 400

    case = toolHarness.get_case(case_id)
    if case is None:
        return jsonify({"status": "error", "message": "Tool harness case not found."}), 404
    if not bool(case.get("enabled", True)):
        return jsonify({"status": "error", "message": "Tool harness case is disabled."}), 400

    tool_name = str(case.get("tool_name") or "").strip()
    catalog = _tool_catalog_index()
    tool_row = catalog.get(tool_name)
    if not isinstance(tool_row, dict):
        return jsonify({"status": "error", "message": "Tool is no longer available in the catalog."}), 400

    execution_mode = str(payload.get("execution_mode") or case.get("execution_mode") or "real").strip().lower()
    if execution_mode not in {"real", "mock"}:
        execution_mode = "real"

    specs = _harness_tool_specs()
    if tool_name not in specs:
        return jsonify({"status": "error", "message": "Tool is unavailable for harness execution."}), 400

    runtime = ToolRuntime(
        api_keys={"brave_search": str(config.brave_keys or "").strip()},
        enable_human_approval=False,
        default_dry_run=(execution_mode == "mock"),
    )
    runtime.set_runtime_context(
        {
            "run_id": f"tool-harness-{case_id}-{int(time.time())}",
            "member_id": int(member.get("member_id", 0)),
            "source": "agent_playground.tool_harness",
        }
    )
    register_runtime_tools(
        runtime,
        specs,
        tool_policy=_tool_policy_overrides(),
    )

    started = time.perf_counter()
    result = runtime.execute(tool_name, case.get("input_args", {}))
    duration_ms = int((time.perf_counter() - started) * 1000)

    current_contract = build_contract_snapshot(tool_name, tool_row.get("input_schema"))
    contract_result = compare_contract_snapshots(case.get("contract_snapshot"), current_contract)
    regression_result = compare_golden_outputs(case.get("golden_output"), result.get("tool_results"))

    report = {
        "schema": "ryo.tool_harness_report.v1",
        "case_id": case_id,
        "tool_name": tool_name,
        "execution_mode": execution_mode,
        "run_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "duration_ms": duration_ms,
        "result": result,
        "contract": contract_result,
        "regression": regression_result,
    }

    member_id = int(member.get("member_id", 0))
    persist_run = _payload_bool(payload.get("persist_run"), True)
    save_contract = _payload_bool(payload.get("save_contract"), False)
    save_golden = _payload_bool(payload.get("save_golden"), False)

    updated_case = case
    try:
        if persist_run:
            updated_case = toolHarness.record_run(case_id, report, actor_member_id=member_id)
        if save_contract:
            updated_case = toolHarness.set_contract_snapshot(
                case_id,
                current_contract,
                actor_member_id=member_id,
            )
        if save_golden:
            if result.get("status") != "success":
                return jsonify({"status": "error", "message": "Cannot save golden output for failed tool run."}), 400
            updated_case = toolHarness.set_golden_output(
                case_id,
                result.get("tool_results"),
                actor_member_id=member_id,
            )
    except ToolHarnessValidationError as error:
        return jsonify({"status": "error", "message": str(error)}), 400

    return jsonify(
        {
            "status": "ok",
            "report": report,
            "case": updated_case,
        }
    )


@app.get("/api/agent-playground/tool-approvals")
def playgroundToolApprovalsListAPI():
    member = _require_member_api()
    if member is None:
        return _api_auth_error()

    statusFilter = str(request.args.get("status") or "").strip().lower() or None
    runID = str(request.args.get("run_id") or "").strip() or None
    try:
        limit = int(request.args.get("limit", "100"))
    except ValueError:
        limit = 100
    memberID = int(member.get("member_id", 0))

    approvals = toolApprovals.list_requests(
        status=statusFilter,
        run_id=runID,
        member_id=None if _member_can_write_playground_registry(member) else memberID,
        limit=max(1, min(limit, 500)),
    )
    return jsonify(
        {
            "status": "ok",
            "approvals": approvals,
            "can_decide": _member_can_write_playground_registry(member),
        }
    )


@app.get("/api/agent-playground/tool-approvals/<request_id>")
def playgroundToolApprovalGetAPI(request_id: str):
    member = _require_member_api()
    if member is None:
        return _api_auth_error()

    approval = toolApprovals.get_request(request_id)
    if approval is None:
        return jsonify({"status": "error", "message": "Approval request not found."}), 404
    if not _member_can_manage_tool_approvals(member, approval):
        return _api_forbidden("Access denied for this approval request.")
    return jsonify({"status": "ok", "approval": approval})


@app.post("/api/agent-playground/tool-approvals/<request_id>/decision")
def playgroundToolApprovalDecisionAPI(request_id: str):
    member = _require_member_api()
    if member is None:
        return _api_auth_error()

    existing = toolApprovals.get_request(request_id)
    if existing is None:
        return jsonify({"status": "error", "message": "Approval request not found."}), 404
    if not _member_can_manage_tool_approvals(member, existing):
        return _api_forbidden("You cannot decide this approval request.")

    payload = _request_json_dict()
    decision = str(payload.get("decision") or "").strip().lower()
    reason = str(payload.get("reason") or "").strip()
    try:
        updated = toolApprovals.decide_request(
            request_id,
            decision=decision,
            actor_member_id=int(member.get("member_id", 0)),
            reason=reason,
            meta={"source": "agent_playground"},
        )
    except ApprovalValidationError as error:
        return jsonify({"status": "error", "message": str(error)}), 400
    return jsonify({"status": "ok", "approval": updated})


@app.get("/api/agent-playground/runs")
def playgroundRunsListAPI():
    member = _require_member_api()
    if member is None:
        return _api_auth_error()

    memberID = int(member.get("member_id", 0))
    statusFilter = request.args.get("status")
    try:
        limit = int(request.args.get("limit", "30"))
    except ValueError:
        limit = 30
    runs = playgroundRuns.list_runs(
        limit=max(1, min(limit, 200)),
        status=statusFilter if statusFilter else None,
        member_id=memberID,
    )
    return jsonify({"status": "ok", "runs": runs})


@app.post("/api/agent-playground/runs")
def playgroundRunsCreateAPI():
    member = _require_member_api()
    if member is None:
        return _api_auth_error()

    payload = _request_json_dict()
    mode = normalize_run_mode(payload.get("mode", "chat"))
    message = str(payload.get("message") or payload.get("prompt") or "").strip()
    batchInputs = payload.get("batch_inputs")
    compareModels = payload.get("compare_models")

    if mode in {"chat", "workflow", "compare"} and not message:
        return jsonify({"status": "error", "message": "message is required for this mode."}), 400
    if mode == "batch":
        if not isinstance(batchInputs, list):
            if message:
                batchInputs = [line.strip() for line in message.splitlines() if line.strip()]
            else:
                batchInputs = []
        if not batchInputs:
            return jsonify({"status": "error", "message": "batch_inputs is required for batch mode."}), 400
    if mode == "compare":
        if not isinstance(compareModels, list) or len(compareModels) < 2:
            return jsonify({"status": "error", "message": "compare_models requires at least two models."}), 400

    memberID = int(member.get("member_id", 0))
    memberCanWrite = _member_can_write_playground_registry(member)
    contextPayload = payload.get("context")
    if not isinstance(contextPayload, dict):
        contextPayload = {}
    contextPayload.setdefault("community_id", None)
    contextPayload.setdefault("chat_host_id", memberID)
    contextPayload.setdefault("platform", "web")
    contextPayload.setdefault("topic_id", None)
    contextPayload.setdefault("chat_type", "member")
    contextPayload.setdefault("message_timestamp", datetime.now().isoformat(timespec="seconds"))

    agentDefinitionPayload = payload.get("agent_definition") if isinstance(payload.get("agent_definition"), dict) else {}
    agentDefinitionRef: dict[str, Any] = {}
    agentDefinitionID = str(payload.get("agent_definition_id") or "").strip()
    agentDefinitionVersionRaw = payload.get("agent_definition_version")
    try:
        agentDefinitionVersion = int(agentDefinitionVersionRaw) if agentDefinitionVersionRaw is not None else None
    except (TypeError, ValueError):
        agentDefinitionVersion = None

    if agentDefinitionID:
        resolvedDefinition = agentDefinitions.get_definition(agentDefinitionID, version=agentDefinitionVersion)
        if resolvedDefinition is None:
            return jsonify({"status": "error", "message": "Selected agent definition was not found."}), 404
        ownerID = int(resolvedDefinition.get("owner_member_id", -1))
        if ownerID != memberID and not memberCanWrite:
            return _api_forbidden("Access denied for selected agent definition.")
        if isinstance(resolvedDefinition.get("definition"), dict):
            agentDefinitionPayload = resolvedDefinition["definition"]
            agentDefinitionRef = {
                "definition_id": resolvedDefinition.get("definition_id"),
                "version": resolvedDefinition.get("selected_version"),
                "name": resolvedDefinition.get("name"),
            }
    elif agentDefinitionPayload:
        agentDefinitionPayload = normalize_agent_definition(agentDefinitionPayload)
        agentDefinitionRef = {
            "definition_id": str(payload.get("agent_definition_id") or "").strip() or None,
            "version": agentDefinitionVersion,
            "name": agentDefinitionPayload.get("identity", {}).get("name"),
        }

    requestPayload = {
        "mode": mode,
        "message": message,
        "prompt": message,
        "context": contextPayload,
        "options": payload.get("options") if isinstance(payload.get("options"), dict) else {},
        "batch_inputs": batchInputs if isinstance(batchInputs, list) else [],
        "compare_models": compareModels if isinstance(compareModels, list) else [],
        "workflow_steps": payload.get("workflow_steps") if isinstance(payload.get("workflow_steps"), list) else [],
        "source_run_id": payload.get("source_run_id"),
        "replay_from_seq": payload.get("replay_from_seq"),
        "state_overrides": payload.get("state_overrides") if isinstance(payload.get("state_overrides"), dict) else {},
        "agent_definition": agentDefinitionPayload if isinstance(agentDefinitionPayload, dict) else {},
        "agent_definition_ref": agentDefinitionRef,
    }

    autoStart = bool(payload.get("auto_start", True))
    run = playgroundRuns.create_run(
        member_id=memberID,
        mode=mode,
        request_payload=requestPayload,
        lineage=_lineage_from_request(payload),
        auto_start=autoStart,
    )
    return jsonify({"status": "ok", "run": run}), 201


@app.get("/api/agent-playground/runs/<run_id>")
def playgroundRunDetailAPI(run_id: str):
    member = _require_member_api()
    if member is None:
        return _api_auth_error()

    run = playgroundRuns.get_run(run_id)
    if run is None:
        return jsonify({"status": "error", "message": "Run not found."}), 404
    if int(run.get("member_id", -1)) != int(member.get("member_id", 0)):
        return jsonify({"status": "error", "message": "Access denied for this run."}), 403

    return jsonify(
        {
            "status": "ok",
            "run": run,
            "events": playgroundRuns.get_events(run_id, after_seq=0, limit=500),
            "snapshots": playgroundRuns.get_snapshots(run_id, limit=500),
            "artifacts": playgroundRuns.get_artifacts(run_id, limit=500),
        }
    )


@app.get("/api/agent-playground/runs/<run_id>/events")
def playgroundRunEventsAPI(run_id: str):
    member = _require_member_api()
    if member is None:
        return _api_auth_error()

    run = playgroundRuns.get_run(run_id)
    if run is None:
        return jsonify({"status": "error", "message": "Run not found."}), 404
    if int(run.get("member_id", -1)) != int(member.get("member_id", 0)):
        return jsonify({"status": "error", "message": "Access denied for this run."}), 403

    try:
        afterSeq = int(request.args.get("after_seq", "0"))
    except ValueError:
        afterSeq = 0
    try:
        limit = int(request.args.get("limit", "250"))
    except ValueError:
        limit = 250

    events = playgroundRuns.get_events(run_id, after_seq=max(0, afterSeq), limit=max(1, min(limit, 1000)))
    return jsonify({"status": "ok", "events": events})


@app.get("/api/agent-playground/runs/<run_id>/stream")
def playgroundRunEventsStreamAPI(run_id: str):
    member = _require_member_api()
    if member is None:
        return _api_auth_error()

    run = playgroundRuns.get_run(run_id)
    if run is None:
        return jsonify({"status": "error", "message": "Run not found."}), 404
    if int(run.get("member_id", -1)) != int(member.get("member_id", 0)):
        return jsonify({"status": "error", "message": "Access denied for this run."}), 403

    try:
        afterSeq = int(request.args.get("after_seq", "0"))
    except ValueError:
        afterSeq = 0

    @stream_with_context
    def stream() -> Any:
        cursorSeq = max(0, afterSeq)
        idleCounter = 0
        while True:
            events = playgroundRuns.get_events(run_id, after_seq=cursorSeq, limit=300)
            if events:
                for event in events:
                    cursorSeq = int(event.get("seq", cursorSeq))
                    yield f"data: {json.dumps(event)}\\n\\n"
                idleCounter = 0
            else:
                idleCounter += 1
                if idleCounter % 8 == 0:
                    yield ": keepalive\\n\\n"

            latest = playgroundRuns.get_run(run_id)
            if latest is None:
                yield "event: done\\ndata: {\"status\":\"missing\"}\\n\\n"
                break
            if str(latest.get("status")) in {"completed", "failed", "cancelled"}:
                terminalEvents = playgroundRuns.get_events(run_id, after_seq=cursorSeq, limit=100)
                if terminalEvents:
                    for event in terminalEvents:
                        cursorSeq = int(event.get("seq", cursorSeq))
                        yield f"data: {json.dumps(event)}\\n\\n"
                yield f"event: done\\ndata: {json.dumps({'run_id': run_id, 'status': latest.get('status')})}\\n\\n"
                break
            time.sleep(0.35)

    return Response(
        stream(),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/api/agent-playground/runs/<run_id>/cancel")
def playgroundRunCancelAPI(run_id: str):
    member = _require_member_api()
    if member is None:
        return _api_auth_error()

    run = playgroundRuns.get_run(run_id)
    if run is None:
        return jsonify({"status": "error", "message": "Run not found."}), 404
    if int(run.get("member_id", -1)) != int(member.get("member_id", 0)):
        return jsonify({"status": "error", "message": "Access denied for this run."}), 403

    updated = playgroundRuns.cancel_run(run_id)
    return jsonify({"status": "ok", "run": updated})


@app.post("/api/agent-playground/runs/<run_id>/resume")
def playgroundRunResumeAPI(run_id: str):
    member = _require_member_api()
    if member is None:
        return _api_auth_error()

    run = playgroundRuns.get_run(run_id)
    if run is None:
        return jsonify({"status": "error", "message": "Run not found."}), 404
    if int(run.get("member_id", -1)) != int(member.get("member_id", 0)):
        return jsonify({"status": "error", "message": "Access denied for this run."}), 403

    resumed = playgroundRuns.resume_run(run_id)
    return jsonify({"status": "ok", "run": resumed})


@app.post("/api/agent-playground/runs/<run_id>/replay")
def playgroundRunReplayAPI(run_id: str):
    member = _require_member_api()
    if member is None:
        return _api_auth_error()

    run = playgroundRuns.get_run(run_id)
    if run is None:
        return jsonify({"status": "error", "message": "Run not found."}), 404
    if int(run.get("member_id", -1)) != int(member.get("member_id", 0)):
        return jsonify({"status": "error", "message": "Access denied for this run."}), 403

    payload = _request_json_dict()
    replayFromSeq = payload.get("replay_from_seq")
    if replayFromSeq is None:
        replayFromSeq = payload.get("step_seq")
    try:
        replayFromSeq = int(replayFromSeq) if replayFromSeq is not None else None
    except (TypeError, ValueError):
        replayFromSeq = None
    stateOverrides = payload.get("state_overrides") if isinstance(payload.get("state_overrides"), dict) else None

    replayRun = playgroundReplay.replay_with_state(
        run_id,
        step_seq=replayFromSeq,
        state_overrides=stateOverrides,
        auto_start=bool(payload.get("auto_start", True)),
    )
    return jsonify({"status": "ok", "run": replayRun})


@app.get("/api/agent-playground/metrics")
def playgroundMetricsAPI():
    member = _require_member_api()
    if member is None:
        return _api_auth_error()
    return jsonify({"status": "ok", "metrics": playgroundRuns.metrics_summary()})



if __name__ == "__main__":
    logger.info("RYO - begin web ui application.")
    bindHost = _normalize_bind_host(os.getenv("RYO_WEB_RESOLVED_HOST", _runtime_str("web.host", "127.0.0.1")))
    startPort = _coerce_port(
        os.getenv("RYO_WEB_RESOLVED_PORT"),
        _runtime_int("web.port", 4747),
    )
    scanLimit = max(0, _runtime_int("web.port_scan_limit", 100))
    debugMode = _runtime_bool("web.debug", False)
    useReloader = _runtime_bool("web.use_reloader", False) and debugMode

    bindPort = _select_bind_port(bindHost, startPort, scanLimit)
    publicHost = _display_host(bindHost)
    publicURL = f"http://{publicHost}:{bindPort}/"
    logger.info(
        "Web UI runtime endpoint: %s (bind_host=%s, start_port=%s, scan_limit=%s, debug=%s, reloader=%s)",
        publicURL,
        bindHost,
        startPort,
        scanLimit,
        debugMode,
        useReloader,
    )

    app.run(
        host=bindHost,
        port=bindPort,
        debug=debugMode,
        use_reloader=useReloader,
    )
