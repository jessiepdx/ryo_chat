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
import logging
import os
import re
import secrets
import socket
from datetime import datetime
from flask import Flask, request, abort, url_for, redirect, session, render_template, g, jsonify
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


def _set_auth_message(message: str, kind: str = "error") -> None:
    session["auth_message"] = str(message)
    session["auth_message_kind"] = str(kind)


def _normalize_username(value: str | None) -> str:
    return str(value or "").strip().lstrip("@")


def _username_is_valid(value: str) -> bool:
    return USERNAME_PATTERN.fullmatch(value) is not None

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
app.secret_key = secrets.token_hex()


@app.before_request
def loadUser():
    # Build user data and menu data
    g.authMessage = session.pop("auth_message", None)
    g.authMessageKind = session.pop("auth_message_kind", "error")
    if "member_id" not in session:
        g.memberData = None
        g.menuData = None
    else:
        member = members.getMemberByID(session["member_id"])
        print(member)
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
    username = _normalize_username(request.form.get("username"))
    password = str(request.form.get("userpass") or "").strip()

    if not username or not password:
        _set_auth_message("Username and password are required.", kind="error")
        return redirect(url_for("index"))

    member = members.loginMember(username, password)
    if member is None:
        _set_auth_message("Login failed. Check your username/password.", kind="error")
        return redirect(url_for("index"))

    session["member_id"] = member["member_id"]
    _set_auth_message(f"Welcome back {member.get('first_name', username)}.", kind="success")
    return redirect(url_for("index"))


@app.route("/signup", methods=["POST"])
def signup():
    logger.info("Web signup requested.")
    username = _normalize_username(request.form.get("username"))
    password = str(request.form.get("userpass") or "").strip()
    passwordConfirm = str(request.form.get("userpass-confirm") or "").strip()
    firstName = str(request.form.get("first_name") or "").strip()
    lastName = str(request.form.get("last_name") or "").strip()

    if not username or not password:
        _set_auth_message("Username and password are required to sign up.", kind="error")
        return redirect(url_for("index"))
    if not _username_is_valid(username):
        _set_auth_message("Username must be 3-96 chars: letters, numbers, underscore.", kind="error")
        return redirect(url_for("index"))
    if password != passwordConfirm:
        _set_auth_message("Password confirmation does not match.", kind="error")
        return redirect(url_for("index"))

    member, errorMessage = members.registerWebMember(
        username=username,
        password=password,
        firstName=firstName or username,
        lastName=lastName or None,
    )
    if member is None:
        _set_auth_message(errorMessage or "Signup failed. Please try a different username.", kind="error")
        return redirect(url_for("index"))

    session["member_id"] = member["member_id"]
    _set_auth_message(f"Signup complete. Welcome {member.get('first_name', username)}.", kind="success")
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
