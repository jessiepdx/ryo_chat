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
import secrets
from datetime import datetime
from flask import Flask, request, abort, url_for, redirect, session, render_template, g, jsonify
from hypermindlabs.utils import CustomFormatter, MemberManager, KnowledgeManager
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

members = MemberManager()

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
        "menuData": g.menuData
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
    print("Login")
    # redirect to home page if not logged in
    if request.method == 'POST':
        print("POST")
        # get the record for the username
        username = request.form["username"]
        password = request.form.get("userpass")
        member = members.loginMember(username, password)
        print(member)
        
        if member is not None:
            session["member_id"] = member["member_id"]
            
        return redirect(url_for("index"))


@app.post("/miniapp-login")
def miniappLogin():
    logger.info(f"{ConsoleColors['yellow']}Validate and login Telegram miniapp user.{ConsoleColors['default']}")
    queryString = request.form.get("query-string")

    # Generate a hash with the data check string and secret key (from utils)
    memberID = members.validateMiniappData(queryString)
    print(memberID)

    if memberID is not None:
        session["member_id"] = memberID

    return redirect(url_for("index"))



if __name__ == "__main__":
    #print("Begin Flask Application")
    logger.info("RYO - begin web ui application.")

    app.run(
        host="0.0.0.0",
        port=4747,
        debug=True
    )
