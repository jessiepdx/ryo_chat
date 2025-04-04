##########################################################################
#                                                                        #
#  This file (telegram_ui.py) contains the telegram user interface for   #
#  project ryo (run your own) chat                                       #
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
from datetime import datetime, timedelta, timezone
import json
import logging
from ollama import AsyncClient
import os
import re
from telegram import (
    ChatMember,
    ChatMemberUpdated,
    constants, 
    ForceReply,
    helpers,
    InlineKeyboardButton, 
    InlineKeyboardMarkup,
    ReplyKeyboardMarkup, 
    ReplyKeyboardRemove, 
    Update,
    WebAppInfo,
    ReactionType
)
from telegram.ext import (
    Application,
    CallbackQueryHandler,
    ChatMemberHandler,
    CommandHandler,
    ContextTypes,
    ConversationHandler,
    MessageHandler,
    MessageReactionHandler,
    filters
)
from hypermindlabs.utils import (
    ChatHistoryManager, 
    CommunityManager, 
    CommunityScoreManager, 
    ConfigManager, 
    CustomFormatter,
    KnowledgeManager, 
    MemberManager, 
    ProposalManager, 
    UsageManager
)
from hypermindlabs.agents import ConversationOrchestrator, ConversationalAgent, ImageAgent, TweetAgent



###########
# LOGGING #
###########

# Clear any previous logging handlers
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

# Set the basic config to append logging data to a file
logPath = "logs/"
logFilename = "telegram_log_" + datetime.now().strftime("%Y%m%d-%H%M%S") + ".txt"
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

# Set timezone for time
timezone(-timedelta(hours=7), "Pacific")



###########
# GLOBALS #
###########

chatHistory = ChatHistoryManager()
communities = CommunityManager()
communityScore = CommunityScoreManager()
config = ConfigManager()
knowledge = KnowledgeManager()
members = MemberManager()
proposals = ProposalManager()
usage = UsageManager()



###########################
# DEFINE COMMAND HANDLERS #
###########################

# Define the startBot function to handle "/start commands for PRIVATE chats"
async def startBot(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handles the start command. Registers this private chat with the chatbot. 
    This is required for the bot to be able to send DMs"""
    
    message = update.effective_message
    user = update.effective_user

    logger.info(f"Start command issued by {user.name} (user_id: {user.id}).")

    # Check if the user is already registered
    member = members.getMemberByTelegramID(user.id)

    if member is None:
        logger.info(f"New user {user.name} (user_id: {user.id}) being registered with the chatbot.")
        # There is no account for this user, begin registration process
        newAccount = {
            "username": user.username,
            "user_id": user.id,
            "first_name": user.first_name,
            "last_name": user.last_name,
            "email": None,
            "roles": ["user"],
            "register_date": datetime.now()
        }
        members.addMemberFromTelegram(newAccount)

        try:
            # TODO get official community chat links from config and insert into the welcome message
            welcomeMessage = f"""Welcome {user.name}, I am the {config.botName} chatbot. 
You will need to have a minimum community score of 50 to chat with me in private. 
Engage with the community in one of our group chats to increase your community score. 
            
Use the /help command for more information."""
            
            await message.reply_text(
                text=welcomeMessage
            )
        except Exception as err:
            logger.error("Exception while sending a telegram message", exc_info=err)
            #logger.warning(f"The following error occurred while sending a telegram message:  {err}.")

    else:
        # A user with this id is already registered
        logger.info(f"User {user.name} (user_id: {user.id}) is already registered.")
        try:
            await message.reply_text(f"Welcome back {user.name}, you are already registered with the {config.botName} chatbot.")
        except Exception as err:
            logger.warning(f"The following error occurred while sending a telegram message:  {err}.")


# Launch the Miniapp Dashboard.
async def dashboard(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send a message with a button that opens a the mini app."""
    
    user = update.effective_user
    if config.webUIUrl is None:
        # TODO message the user / owner account that the config for the dashboard is missing
        return
    
    logger.info(f"Dashboard command issued by {user.name} (user_id: {user.id}).")
    
    keyboard = [
        [
            InlineKeyboardButton("OPEN", web_app=WebAppInfo(url=config.webUIUrl + "miniapp/dashboard"))
        ]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    # Send message with text and appended InlineKeyboard
    await update.message.reply_text(
        text="Click OPEN to open the dashboard.",
        reply_markup=reply_markup
    )


# Display a help menu for the user
async def help(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Help menu displays the available commands"""

    chat = update.effective_chat
    message = update.effective_message
    user = update.effective_user

    logger.info(f"Help command issued by {user.name} (user_id: {user.id}).")

    helpMsg = """The following commands are available:\n\n"""

    if chat.type == "private":
        member = members.getMemberByTelegramID(user.id)

        if member is not None:
            helpMsg = helpMsg + """/info to display your user account info
-------------------------------
Send a message to begin chatting with the chatbot."""
        else:
            helpMsg = helpMsg + "Use the /start command to get started.\n"
    elif chat.type == "group" or chat.type == "supergroup":
        community = communities.getCommunityByTelegramID(chat.id)

        if community is not None:
            helpMsg = helpMsg + f"""/info to display group account info 
-------------------------------
Tag @{config.botName} in your message to get a response from the chatbot. The chatbot will also response if you reply to it's message.
"""
        else:
            helpMsg = helpMsg + "The chatbot needs to be added to a group by an owner or admin to register it.\n"
    else:
        # Channels are not supported
        return
    
    try:
        await message.reply_text(text=helpMsg)
    except Exception as err:
            logger.warning(f"The following error occurred while sending a telegram message:  {err}.")


# Display user and chat group data
async def info(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Display account information."""
    chat = update.effective_chat
    message = update.effective_message
    user = update.effective_user
    
    logger.info(f"Info command issued by {user.name} (user_id: {user.id}).")

    if chat.type == "private":
        member = members.getMemberByTelegramID(user.id)
        # IMPORTANT info command replies in markdown, make sure to escape markdown characters in all dynamic values
        
        if member is not None:
            infoMsg = f"""*Member ID:*  {member.get("member_id")}
*Telegram User ID:*  {user.id}
*Username:*  {'' if user.username is None else helpers.escape_markdown(user.username)}
*Name:*  {helpers.escape_markdown(member.get('first_name'))} {'' if member.get('last_name') is None else helpers.escape_markdown(member.get('last_name'))}
*Email:*  {'' if member.get('email') is None else helpers.escape_markdown(member.get('email'))}
*Roles:*  {", ".join(member.get('roles'))}
*Created:*  {member.get('register_date')}
*Community Score:*  {member.get('community_score')}"""
        else:
            infoMsg = "User is not registered."
    
    elif chat.type == "group" or chat.type == "supergroup":
        community = communities.getCommunityByTelegramID(chat.id)
        
        # temporary code to leave group chat if the group isn't registered. This is mainly used for development
        if community is None:
            try:
                # Leave the group chat
                await context.bot.leave_chat(chat.id)
            except Exception as err:
                logger.warning(f"The following error occurred while leaving the group chat:  {err}.")
            finally:
                # Exit the function
                return
        
        if community is not None:
            infoMsg = f"""*Community ID:*  {community.get("community_id")}
*Telegram Group chat ID:*  {chat.id}
*Chat type:*  {chat.type}
*Community name:*  {helpers.escape_markdown(community.get("community_name"))}
*Roles:*  {'None' if community.get('roles') is None else ', '.join(community.get('roles'))}
*Created:*  {community.get('register_date')}"""     
   
    try:
        await message.reply_markdown(
            text=infoMsg
        )
    except Exception as err:
            logger.error(msg="Exception while replying to a telegram message", exc_info=err, stack_info=False)
            #logger.warning(f"The following error occurred while sending a telegram message:  {err}.")



########################
# BEGIN COMMAND CHAINS #
########################

# Cancel command is used to cancel most of the command chains
async def cancel(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Cancels and ends the command chain."""
    message = update.effective_message
    user = update.message.from_user

    logger.info(f"User {user.name} (user_id: {user.id}) canceled the command chain.")

    try:
        await message.reply_text(
            text="Command chain canceled.", 
            reply_markup=ReplyKeyboardRemove()
        )
    except Exception as err:
        logger.warning(f"The following error occurred while sending a telegram message:  {err}.")
    finally:
        return ConversationHandler.END



#####################################
# Begin the /generate command chain #
#-----------------------------------#

# Define command chain states
SET_SYSTEM_PROMPT, SET_PROMPT = range(2)

# This is the entry point for the /generate command
async def beginGenerate(update: Update, context: ContextTypes.DEFAULT_TYPE):
    message = update.effective_message
    user = update.effective_user

    logger.info(f"Generate command chain initiated by {user.name} (user_id: {user.id}).")

    member = members.getMemberByTelegramID(user.id)
    if member is None:
        logger.info(f"An unregistered user attempted to use the /generate command.")
        try:
            await message.reply_text(
                text=f"Only members are able to use the /generate command. To register your telegram account with the {config.botName} chatbot, open a private chat with @{config.botName} and send the /start command."
            )
        except Exception as err:
            logger.warning(f"The following error occurred:  {err}.")
        finally:
            return ConversationHandler.END
    else:
        logger.info(f"User {user.name} (user_id: {user.id}) is approved to use the /generate command.")
        
        # Clear any previous temporary generate data
        context.chat_data["generate_data"] = {}
        try:
            await message.reply_text(
                text="Enter your system prompt or /skip to use the default system prompt. /cancel to cancel this command", 
                reply_markup=ForceReply(selective=True)
            )

            return SET_SYSTEM_PROMPT
        except Exception as err:
            logger.warning(f"The following error occurred while sending a telegram message:  {err}.")
            context.chat_data["generate_data"] = None

            return ConversationHandler.END


async def setSystemPrompt(update: Update, context: ContextTypes.DEFAULT_TYPE):
    logger.info(f"Store the provided system prompt.")
    message = update.effective_message
    
    # Store the system prompt in telegram storage
    gd = context.chat_data.get("generate_data")
    gd["system_prompt"] = message.text

    try:
        # Get the prompt
        await message.reply_text(
            "Enter your prompt", 
            reply_markup=ForceReply(selective=True)
        )
        
        return SET_PROMPT
    except Exception as err:
        logger.warning(f"The following error occurred while sending a telegram message:  {err}.")
        # Clear the generate data from telegram storage
        gd = None

        return ConversationHandler.END


async def skip_systemPrompt(update: Update, context: ContextTypes.DEFAULT_TYPE):
    logger.info(f"Set system prompt to default value.")
    message = update.effective_message
    
    # Store the system prompt
    gd = context.chat_data.get("generate_data")
    gd["system_prompt"] = config.defaults["generate_sys_prompt"]

    try:
        # Get the prompt
        await message.reply_text(
            "Enter your prompt", 
            reply_markup=ForceReply(selective=True)
        )
        
        return SET_PROMPT
    except Exception as err:
        logger.warning(f"The following error occurred while sending a telegram message:  {err}.")
        gd = None

        return ConversationHandler.END


async def setPrompt(update: Update, context: ContextTypes.DEFAULT_TYPE):
    logger.info(f"Prompt received, generate a response.")
    chat = update.effective_chat
    message = update.effective_message
    topicID = message.message_thread_id if message.is_topic_message else None

    gd = context.chat_data.get("generate_data")
    systemPromptText = config.defaults["system_prompt"] + gd["system_prompt"]
    
    generateClient = AsyncClient(host=config.inference["generate"]["url"])
    # TODO Wrap in try except block
    output = await generateClient.generate(
        model=config.inference["generate"]["model"], 
        stream=False, 
        system=systemPromptText, 
        prompt=message.text
    )
    responseText = output["response"]

    # Delete the generate data from telegram bot storage
    context.chat_data["generate_data"] = None
    
    try:
        await context.bot.send_message(
            chat_id=chat.id, 
            message_thread_id=topicID,
            reply_markup=ReplyKeyboardRemove(),
            text=responseText
        )
    except Exception as err:
        logger.warning(f"The following error occurred while sending a telegram message:  {err}.")
        gd = None
    finally:
        return ConversationHandler.END



#####################
# Knowledge Manager #
#-------------------#

# Define command chain states
HANDLE_KNOWLEDGE_TYPE, HANDLE_KNOWLEDGE_TEXT, HANDLE_KNOWLEDGE_SOURCE, HANDLE_KNOWLEDGE_CATEGORY, STORE_KNOWLEDGE = range(5)

# This is the entry point for the /knowledge command chain
async def knowledgeManger(update: Update, context: ContextTypes.DEFAULT_TYPE):
    message = update.effective_message
    user = update.effective_user

    logger.info(f"Knowledge command chain initiated by {user.name} (user_id: {user.id}), begin knowledge manager.")

    member = members.getMemberByTelegramID(user.id)

    if member is None:
        logger.info(f"User is not registered with the chatbot.")
        return ConversationHandler.END
    
    allowedRoles = ["admin", "owner"]
    rolesAvailable = member["roles"]

    if (any(role in rolesAvailable for role in allowedRoles)):
        logger.info(f"Member is authorized to use knowledge command.")
        # Clear any previous data for new knowledge from the telegram bot storage
        context.chat_data["new_knowledge"] = dict()

        keyboard = [
            [
                InlineKeyboardButton("Public", callback_data="public"),
                InlineKeyboardButton("Private", callback_data="private")
            ]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        try:
            # Send message with text and appended InlineKeyboard
            await message.reply_text(
                text="Let's add some knowledge to our database. First I need to know, is this public or private (confidential) information?\n\nType /cancel to cancel.",
                reply_markup=reply_markup
            )

            return HANDLE_KNOWLEDGE_TYPE
        except Exception as err:
            logger.warning(f"The following error occurred while sending a telegram message:  {err}.")
            context.chat_data["new_knowledge"] = None

            return ConversationHandler.END
        
    # The follwoing code only runs if the user is not authorized above
    logger.info(f"User {user.name} (user_id: {user.id}) is not authorized to use knowledge command.")
    try:
        await message.reply_text(text="Only admins are authorized to use the knowledge command.")
    except Exception as err:
        logger.warning(f"The following error occurred while sending a telegram message:  {err}.")
    finally:
        return ConversationHandler.END


async def setKnowledgeType(update: Update, context: ContextTypes.DEFAULT_TYPE):
    logger.info(f"Knowledge visibity received. Get knowledge text.")

    query = update.callback_query
    try:
        await query.answer()
    except Exception as err:
        logger.warning(f"The following error occurred while receiving a telegram query response:  {err}.")
        return
    
    nk = context.chat_data.get("new_knowledge")
    nk["visibility"] = query.data

    try:
        await query.edit_message_text(
            text=f"Knowledge data will be {query.data}. Enter knowledge data text:"
        )

        return HANDLE_KNOWLEDGE_TEXT
    except Exception as err:
        logger.warning(f"The following error occurred while sending a telegram message:  {err}.")
        nk = None

        return ConversationHandler.END


async def setKnowledgeText(update: Update, context: ContextTypes.DEFAULT_TYPE):
    logger.info(f"Knowledge text received., get source.")
    message = update.effective_message

    nk = context.chat_data.get("new_knowledge")
    nk["text"] = message.text

    try:
        await message.reply_text(
            text=f"Ok, heres your document:\n\n{message.text}\n\nDo you want to add any sources?\n/skip this step"
        )

        return HANDLE_KNOWLEDGE_SOURCE
    except Exception as err:
        logger.warning(f"The following error occurred while sending a telegram message:  {err}.")
        nk = None

        return ConversationHandler.END


async def setKnowledgeSource(update: Update, context: ContextTypes.DEFAULT_TYPE):
    logger.info(f"Knowledge source sent, get category tag.")
    message = update.effective_message

    nk = context.chat_data.get("new_knowledge")
    nk["source"] = message.text

    try:
        await message.reply_text(
            text=f"Ok, heres your source:\n\n{message.text}\n\nDo you want to add a category tag? \n/skip this step"
        )

        return HANDLE_KNOWLEDGE_CATEGORY
    except Exception as err:
        logger.warning(f"The following error occurred while sending a telegram message:  {err}.")
        nk = None

        return ConversationHandler.END


async def skip_knowledgeSource(update: Update, context: ContextTypes.DEFAULT_TYPE):
    logger.info(f"Skip knowledge source.")
    message = update.effective_message
    user = update.effective_user

    nk = context.chat_data.get("new_knowledge")
    nk["source"] = user.name

    try:
        await message.reply_text(
            text=f"Ok, we will just use your username or full name as the source.\n\nDo you want to add any category tags? Use a comma to separate categroies\n/skip this step"
        )

        return HANDLE_KNOWLEDGE_CATEGORY
    except Exception as err:
        logger.warning(f"The following error occurred while sending a telegram message:  {err}.")
        nk = None

        return ConversationHandler.END


async def setKnowledgeCategories(update: Update, context: ContextTypes.DEFAULT_TYPE):
    logger.info(f"Knowledge category tag received. Call final method to save.")
    message = update.effective_message

    nk = context.chat_data.get("new_knowledge")
    nk["categories"] = message.text.split(",")

    keyboard = [
        [
            InlineKeyboardButton("Yes", callback_data="yes"),
            InlineKeyboardButton("No", callback_data="no")
        ]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)

    responseText = f"""Let's do a final review.

{nk['text']}

Source:  {nk['source']}

Categories:  {message.text}

Do you wish to save?
"""

    try:
        await message.reply_text(
            text=responseText,
            reply_markup=reply_markup
        )

        return STORE_KNOWLEDGE
    except Exception as err:
        logger.warning(f"The following error occurred while sending a telegram message:  {err}.")
        nk = None
        
        return ConversationHandler.END


async def skip_knowledgeCategories(update: Update, context: ContextTypes.DEFAULT_TYPE):
    logger.info(f"Skip knowledge category.")
    message = update.effective_message

    nk = context.chat_data.get("new_knowledge")
    nk["categories"] = None

    keyboard = [
        [
            InlineKeyboardButton("Yes", callback_data="yes"),
            InlineKeyboardButton("No", callback_data="no")
        ]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)

    responseText = f"""Ok, no category tag to add.

Let's do a final review.

{nk['text']}

Source:  {nk['source']}

Category:  None

Do you wish to save?
"""

    try:
        await message.reply_text(
            text=responseText,
            reply_markup=reply_markup
        )

        return STORE_KNOWLEDGE
    except Exception as err:
        logger.warning(f"The following error occurred while sending a telegram message:  {err}.")
        nk = None

        return ConversationHandler.END


async def finalizeKnowledge(update: Update, context: ContextTypes.DEFAULT_TYPE):
    logger.info(f"Finalize the add knowledge process.")
    message = update.effective_message
    user = update.effective_user

    query = update.callback_query
    try:
        await query.answer()
    except Exception as err:
        logger.warning(f"The following error occurred:  {err}.")
        return
    
    nk = context.chat_data.get("new_knowledge")
    
    if query.data == "yes":
        # Get account information
        member = members.getMemberByTelegramID(user.id)

        # If somehow an unregistered user made it to this point
        if member is None:
            logger.info(f"An unregistered user {user.name} (user_id: {user.id}) attempted to save data to the knowledge database.")
            return ConversationHandler.END
        
        allowedRoles = ["admin", "owner"]
        rolesAvailable = member["roles"]

        if (any(role in rolesAvailable for role in allowedRoles)):
            logger.info(f"User {user.name} (user_id: {user.id}) is authorized to save documents to the knowledge database.")

            document = nk.get("text")
            documentMetadata = dict()
            categories = [] if nk.get("categories") is None else nk.get("categories")
            print(categories)
            source = nk.get("source")
            if source:
                documentMetadata["source"] = source
            
            record_id = knowledge.addDocument(document, categories=categories, documentMetadata=documentMetadata)
            print(record_id)

            try:
                # Edit telegram message
                await query.edit_message_text(
                    text=f"Document stored. Record ID:  {record_id}"
                )
            except Exception as err:
                logger.warning(f"The following error occurred while editing a telegram message:  {err}.")
            finally:
                # Delete the newKnowledge property in chat_data
                nk = None

                return ConversationHandler.END
        
        # The follwoing code only runs if the user is not authorized in the above for loop
        logger.info(f"User {user.name} (user_id: {user.id}) is not authorized to use knowledge command.")
        try:
            await message.reply_text(
                text="Only admins are authorized to use the knowledge command."
            )
        except Exception as err:
            logger.warning(f"The following error occurred while sending a telegram message:  {err}.")
        finally:
            return ConversationHandler.END

    else:
        # Text not approved, start over
        logger.info(f"User selected to not store knowledge data, process ended.")
        try:
            await query.edit_message_text(
                text="You've selected to not save the knowledge data. Process ended."
            )
        except Exception as err:
            logger.warning(f"The following error occurred while editing a telegram message:  {err}.")
        finally:
            return ConversationHandler.END



#####################
# Promotion Manager #
#-------------------#

# Define command chain states
GET_ACCOUNT, VERIFY_ACCOUNT, VERIFY_PROMOTE = range(3)

async def promoteAccount(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat = update.effective_chat
    message = update.effective_message
    user = update.effective_user
    logger.info(f"Promote account command chain initiated by {user.name} (user_id: {user.id}).")

    # Check if the user that issued the promote command is authorized to promote
    member = members.getMemberByTelegramID(user.id)

    if member is None:
        logger.info(f"A non-registered user {user.name} (user_id: {user.id}) attempted to use the promote command.")
        return

    # First check if user has valid roles. 
    allowedRoles = ["admin", "owner"]
    rolesAvailable = member["roles"]

    if (any(role in rolesAvailable for role in allowedRoles)):
        # Check if message is a reply to another message and get user account from message data
        if message.reply_to_message is not None:
            # Promote command issued as a reply to a user. Get the telegram user information
            userToPromote = message.reply_to_message.from_user
            # Get member from telegram user
            memberToPromote = members.getMemberByTelegramID(userToPromote.id)
            if memberToPromote is not None:
                logger.info(f"User {user.name} (user_id: {user.id}) is authorized to promote user {userToPromote.name} (user_id: {userToPromote.id}).")
                context.chat_data["member_to_promote"] = memberToPromote
                keyboard = [
                    [
                        InlineKeyboardButton("Administrator", callback_data="admin"),
                        InlineKeyboardButton("Marketing", callback_data="marketing")
                    ],
                    [
                        InlineKeyboardButton("Tester", callback_data="tester")
                    ]
                ]
                reply_markup = InlineKeyboardMarkup(keyboard)
                # Send message with text and appended InlineKeyboard
                try:
                    await message.reply_text(
                        text=f"What role would you like to promote {userToPromote.name} to?\n\nType /cancel to cancel.", 
                        reply_markup=reply_markup
                    )
                except Exception as err:
                    logger.warning(f"The following error occurred while sending a telegram message:  {err}.")
                finally:
                    return VERIFY_PROMOTE
                
            else:
                logger.info(f"User {user.name} (user_id: {user.id}) is authorized to promote, but user {userToPromote.name} (user_id: {userToPromote.id}) is not registered.")
                try:
                    await message.reply_text(
                        text=f"User {userToPromote.name} is not a registered user.", 
                    )
                except Exception as err:
                    logger.warning(f"The following error occurred while sending a telegram message:  {err}.")
                finally:
                    return ConversationHandler.END
                
        elif chat.type == "group" or chat.type == "supergroup":
            logger.info(f"User {user.name} (user_id: {user.id}) is authorized to promote group accounts.")
            # Promote command issued in a group chat. Get that group's account information
            communityToPromote = communities.getCommunityByTelegramID(chat.id)
            if communityToPromote is not None:
                context.chat_data["community_to_promote"] = communityToPromote
                keyboard = [
                    [
                        InlineKeyboardButton("Administrator", callback_data="admin"),
                        InlineKeyboardButton("Marketing", callback_data="marketing")
                    ],
                    [
                        InlineKeyboardButton("Tester", callback_data="tester")
                    ]
                ]
                reply_markup = InlineKeyboardMarkup(keyboard)
                # Send message with text and appended InlineKeyboard
                try:
                    await message.reply_text(
                        text=f"What role would you like to promote the group {communityToPromote['community_name']} to?\n\nType /cancel to cancel.", 
                        reply_markup=reply_markup
                    )
                except Exception as err:
                    logger.warning(f"The following error occurred while sending a telegram message:  {err}.")
                finally:
                    return VERIFY_PROMOTE
        else:
            # TODO look up user by username passed as argument after promote command
            print("who do you want to promote?")

            return ConversationHandler.END

    # Member is registered but not authorized
    else:
        logger.info(f"A non-authorized member {user.name} (user_id: {user.id}) attempted to use the promote command.")

        return ConversationHandler.END


async def setNewRole(update: Update, context: ContextTypes.DEFAULT_TYPE):
    logger.info(f"Update the account with the new role.")
    query = update.callback_query
    chat = update.effective_chat
    user = update.effective_user
    await query.answer()
    memberToPromote = context.chat_data.get("member_to_promote")

    if memberToPromote:
        # Get the telegram user from member data
        userToPromote = await context.bot.get_chat_member(chat_id=chat.id, user_id=memberToPromote["user_id"])
        logger.info(f"User {user.name} (user_id: {user.id}) is promoting {userToPromote.user.name} ({userToPromote.user.id}) to the role of {query.data}.")

        #userToPromote = context.chat_data["user_to_promote"]
        memberToPromote["roles"].append(query.data)
        #results = accounts.updateRoles(userToPromoteData["roles"], userToPromoteData["user_id"])
        results = members.updateMemberRoles(memberToPromote.get("member_id"), memberToPromote.get("roles"))
        if results:
            responseText = f"{userToPromote.user.name} has been promoted to {query.data} role."
        else:
            responseText = "An error occured promoting the user."

        # Clear the temporary user to promote value
        context.chat_data['member_to_promote'] = None

    elif "community_to_promote" in context.chat_data:
        logger.info(f"User {user.name} (user_id: {user.id}) is promoting the group {context.chat_data['group_to_promote']['chat_title']} to the role of {query.data}.")

        communityToPromote = context.chat_data["community_to_promote"]
        communityToPromote["roles"].append(query.data)
        #results = accounts.updateRoles(groupToPromote["roles"], groupToPromote["chat_id"], accountType=chat.type)
        results = communities.updateCommunityRoles(communityToPromote.get("community_id"), communityToPromote.get("roles"))
        if results:
            responseText = f"{communityToPromote['chat_title']} has been promoted to {query.data} role."
        else:
            responseText = "An error occured promoting the community chat."

        # Clear the temporary user to promote value
        context.chat_data['community_to_promote'] = None
    else:
        return ConversationHandler.END

    try:
        await query.edit_message_text(
            text=responseText
        )
    except Exception as err:
        logger.warning(f"The following error occurred while sending a telegram message:  {err}.")
    finally:
        return ConversationHandler.END



#############################
# Tweet agent command chain #
#---------------------------#

# Define tweet states
CONFIRM_TWEET, MODIFY_TWEET = range(2)

async def tweetStart(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    logger.info(f"The tweet command has been issued.")
    chat = update.effective_chat
    message = update.effective_message
    user = update.effective_user
    topicID = message.message_thread_id if message.is_topic_message else None

    # Need user account for all chat types
    member = members.getMemberByTelegramID(user.id)

    if member is None:
        logger.info(f"An unregistered user {user.name} (user_id: {user.id}) attempted to use the tweet command.")
        # Exit the function if there is no user account
        return ConversationHandler.END

    # Set the allowed roles
    allowedRoles = ["admin", "owner"]
    
    if chat.type == "private":
        rolesAvailable = member["roles"]

        if (any(role in rolesAvailable for role in allowedRoles)):
            logger.info(f"User {user.name} (user_id: {user.id}) is authorized to use tweet command in private chat.")
            # Check for "arguments" passed with the tweet command. This will act as a custom prompt for the Twitter Agent
            if len(context.args) > 0:
                tweetPrompt = " ".join(context.args)
            else:
                # No arguments passed check if the command was given as a reply to another message
                if message.reply_to_message is not None:
                    if message.reply_to_message.text is None:
                        # Can only handle replies to text currently
                        return ConversationHandler.END
                    
                    tweetPrompt = f"Create a tweet based on the following message:\n\n{message.reply_to_message.text}"
                else:
                    # Command was sent without arguments and not as a reply to a message
                    tweetPrompt = f"Create a tweet based on something you find interesting from our conversation so far."
            # call the TwitterAgent
            
            fromUser = {
                "user_id": user.id,
                "username": user.username,
                "first_name": user.first_name,
                "last_name": user.last_name,
                "role": "user",
            }

            messageData = {
                "chat_id": chat.id,
                "topic_id" : topicID,
                "message_id": message.message_id,
                "tweet_prompt": tweetPrompt
            }
            
            ta = TweetAgent(message_data=messageData, from_user=fromUser)
            tweet = await ta.ComposeTweet()

            context.chat_data["tweet_agent"] = ta

            keyboard = [
                [
                    InlineKeyboardButton("Confirm", callback_data="confirm"),
                    InlineKeyboardButton("Reject", callback_data="reject")
                ],
                [
                    InlineKeyboardButton("Modify", callback_data="modify")
                ]
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)

            await message.reply_text(
                text=tweet,
                reply_markup=reply_markup
            )

            return CONFIRM_TWEET
        else:
            logger.info(f"User {user.name} (user_id: {user.id}) is not authorized to use tweet command.")
            return ConversationHandler.END
        
    elif chat.type == "group":
        logger.info(f"Tweet command issued in a group chat.")

        # Get group account data
        #groupAcct = accounts.getAccountByTelegramID(chat.id, chat.type)
        community = communities.getCommunityByTelegramID(chat.id)
        if community is None:
            logger.info(f"User {user.name} (user_id: {user.id}) attempted to use the tweet command in an unregistered group chat.")
            # Exit the function, there is no group account
            return ConversationHandler.END
        
        # Combine user and group roles
        rolesAvailable = set(member["roles"] + community["roles"])

        if (any(role in rolesAvailable for role in allowedRoles)):
            logger.info(f"User {user.name} (user_id: {user.id}) is authorized to use tweet command in the {community['community_name']} group chat.")
            # Check for "arguments" passed with the tweet command. This will act as a custom prompt for the Twitter Agent
            if len(context.args) > 0:
                tweetPrompt = " ".join(context.args)
                knowledge = config.database + "_knowledge"
            else:
                # No arguments passed check if the command was given as a reply to another message
                if message.reply_to_message is not None:
                    if message.reply_to_message.text is None:
                        # Can only handle replies to text currently
                        return ConversationHandler.END
                    
                    tweetPrompt = f"Create a tweet based on the following message:\n\n{message.reply_to_message.text}"
                    knowledge = config.database + "_knowledge"
                else:
                    # Command was sent without arguments and not as a reply to a message
                    tweetPrompt = f"Create a tweet based on something you find interesting from our conversation so far."
                    knowledge = None
            # call the TwitterAgent
            # Need to update this to the new agent code
            ta = TweetAgent(message=tweetPrompt, chatHistory_db=f"group{community['chat_id']}_chat_history", knowledge_db=knowledge)
            tweet = await ta.ComposeTweet()

            context.chat_data["tweet_agent"] = ta

            keyboard = [
                [
                    InlineKeyboardButton("Confirm", callback_data="confirm"),
                    InlineKeyboardButton("Reject", callback_data="reject")
                ],
                [
                    InlineKeyboardButton("Modify", callback_data="modify")
                ]
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)

            await message.reply_text(
                text=tweet,
                reply_markup=reply_markup
            )

            return CONFIRM_TWEET
        else:
            logger.info(f"User {user.name} (user_id: {user.id}) is not authorized to use tweet command.")
            return ConversationHandler.END

    elif chat.type == "supergroup":
        logger.info(f"Tweet command issued in a supergroup chat.")

        # NOTE in a supergroup, normal messages sent in the general topic 
        # will have message.chat.is_forum return True, but message.is_topic_message return False
        # also will not have a message.reply_to_message value which also means no message.reply_to_message.message_thread_id
        # replies in the general topic area contain a message.reply_to_message that contains a text value of the message being replied to, just like in a normal group message

        # NOTE in a supergroup, normal messages sent in a topic thread 
        # will have message.chat.is_forum return True as well message.is_topic_message return True
        # will have a message.reply_to_message that contains a message_thread_id but does not contain a text property
        # replies sent in a topic thread will contain the above, but will have a text property

        # Get group account data
        community = communities.getCommunityByTelegramID(chat.id)
        if community is None:
            logger.info(f"User {user.name} (user_id: {user.id}) attempted to use the tweet command in an unregistered group chat.")
            # Exit the function, there is no group account
            return ConversationHandler.END
        
        # Combine user and group roles
        rolesAvailable = set(community["roles"] + community["roles"])

        if (any(role in rolesAvailable for role in allowedRoles)):
            logger.info(f"Tweet command issued in a supergroup chat.")
            
            # Check if command sent in a topic thread
            if message.is_topic_message:
                topicID = message.reply_to_message.message_thread_id
            else:
                topicID = None
            
            # Check for "arguments" passed with the tweet command. This will act as a custom prompt for the Twitter Agent
            if len(context.args) > 0:
                tweetPrompt = " ".join(context.args)
                knowledge = config.database + "_knowledge"
            elif message.is_topic_message and message.reply_to_message.text is not None:
                # Command in a topic thread and a reply to a message
                # Messages in a topic thread always have a message.reply_to_message value
                logger.info("Command in a topic thread and a reply to a message")
                tweetPrompt = f"Create a tweet based on the following message:\n\n{message.reply_to_message.text}"
                knowledge = config.database + "_knowledge"
            elif not message.is_topic_message and message.reply_to_message is not None:
                # Command in General topic and a reply to a message
                logger.info("Command in a general thread and a reply to a message")
                if message.reply_to_message.text is None:
                    # Can only handle replies to text currently
                    return ConversationHandler.END
                
                tweetPrompt = f"Create a tweet based on the following message:\n\n{message.reply_to_message.text}"
                knowledge = config.database + "_knowledge"
            else:
                # Command was sent without arguments and not as a reply to a message
                tweetPrompt = f"Create a tweet based on something you find interesting from our conversation so far."
                knowledge = None
            

            # call the TwitterAgent
            ta = TweetAgent(message=tweetPrompt, chatHistory_db=f"group{community['chat_id']}_chat_history", knowledge_db=knowledge, topicID=topicID)
            tweet = await ta.ComposeTweet()
            context.chat_data["tweet_agent"] = ta

            keyboard = [
                [
                    InlineKeyboardButton("Confirm", callback_data="confirm"),
                    InlineKeyboardButton("Reject", callback_data="reject")
                ],
                [
                    InlineKeyboardButton("Modify", callback_data="modify")
                ]
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)

            await message.reply_text(
                text=tweet,
                reply_markup=reply_markup
            )

            return CONFIRM_TWEET
        else:
            logger.info(f"User {user.name} (user_id: {user.id}) is not authorized to use tweet command.")
            return ConversationHandler.END
    
    else:
        return ConversationHandler.END


async def modifyTweet(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    logger.info(f"Modify the tweet agent response with a new prompt.")
    message = update.effective_message

    if "tweet_agent" in context.chat_data:
        tweetAgent = context.chat_data["tweet_agent"]
    else:
        # There is no tweet_agent object, exit the function
        logger.warning(f"tweet_agent value missing from context chat_data.")
        return ConversationHandler.END
    
    tweet = await tweetAgent.ModifyTweet(message.text)

    keyboard = [
        [
            InlineKeyboardButton("Confirm", callback_data="confirm"),
            InlineKeyboardButton("Reject", callback_data="reject")
        ],
        [
            InlineKeyboardButton("Modify", callback_data="modify")
        ]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)

    await message.reply_text(
        text=tweet,
        reply_markup=reply_markup
    )

    return CONFIRM_TWEET


async def confirmTweet(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    logger.info(f"Confirm tweet.")
    user = update.effective_user
    # Temporary area to mock sending tweets
    testChatID = -1002177698730
    testTopicID = 3824
    
    if "tweet_agent" in context.chat_data:
        tweetAgent = context.chat_data["tweet_agent"]
    else:
        # There is no tweet_agent object, exit the function
        logger.warning(f"tweet_agent value missing from context chat_data.")
        return ConversationHandler.END

    query = update.callback_query
    await query.answer()

    if query.data == "confirm":
        logger.info(f"User confirmed the tweet. Sending...")

        tweetResults = await tweetAgent.SendTweet()
        # TODO add if else condition on the result of the tweet agent
        # Temporarily send the tweet to a specific topic in supergroup chat
        await context.bot.send_message(
            chat_id=testChatID,
            message_thread_id=testTopicID,
            text=f"The following tweet was approved to be sent by {user.name}\n\n{tweetAgent.tweetText}"
        )

        # Edit telegram message
        await query.edit_message_text(
            text=f"The following tweet was sent:\n\n{tweetAgent.tweetText}"
        )

        return ConversationHandler.END
    elif query.data == "modify":
        # Edit telegram message
        await query.edit_message_text(
            text=f"Here is the tweet so far:\n\n{tweetAgent.tweetText}\n\nEnter a new prompt to modify this tweet with."
        )
        return MODIFY_TWEET
    else:
        logger.info(f"User rejected the tweet.")
        # Edit telegram message
        await query.edit_message_text(
            text=f"The following tweet was rejected:\n\n{tweetAgent.tweetText}"
        )
        return ConversationHandler.END



############################
# Newsletter command chain #
#--------------------------#

# Define newsletter states
ROLE_SELECTION, PHOTO_OPTION, ADD_PHOTO, COMPOSE_NEWLETTER, CONFIRM_NEWSLETTER = range(5)

async def newsletterStart(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    logger.info(f"Newsletter command issued.")
    chat = update.effective_chat
    message = update.effective_message
    user = update.effective_user

    # Need user account for all chat types
    member = members.getMemberByTelegramID(user.id)

    if member is None:
        logger.info(f"An unregistered user {user.name} (user_id: {user.id}) attempted to use the newsletter command.")
        # Exit the function if there is no user account
        return ConversationHandler.END

    # Set the allowed roles
    allowedRoles = ["admin", "owner"]

    # This section allows for this function to inherit roles from the community's roles
    if chat.type == "private":
        rolesAvailable = member["roles"]
    elif chat.type == "group" or chat.type == "supergroup":
        community = communities.getCommunityByTelegramID(chat.id)

        if community is None:
            logger.info(f"User {user.name} (user_id: {user.id}) attempted to use the newsletter command in an unregistered group chat.")
            # Exit the function, there is no group account
            return ConversationHandler.END
        
        rolesAvailable = set(member["roles"] + community["roles"])

    else:
        # Exit the function, no other chat types allowed
        return ConversationHandler.END
    
    if (any(role in rolesAvailable for role in allowedRoles)):
        logStr = f"User {user.name} (user_id: {user.id}) (roles:  {', '.join(member['roles'])}) is authorized to use the newsletter command"
        if chat.type != "private":
            logStr = logStr + f" in the {community['chat_title']} group chat (roles:  {', '.join(community['roles'])})."
        else:
            logStr = logStr + f" in private chat."
        logger.info(logStr)

        # Authorized

        # Init the temporary storage for holding newsletter data
        nd = context.chat_data["newsletter_data"] = {
            "text": None,
            "roles": []
        }

        # Check if arguments were passed with the command
        if len(context.args) > 0:
            # Create the text property
            nd["text"] = " ".join(context.args)

        # Check if the command was a reply to a message
        if ((chat.type == "private" or chat.type == "group" or (chat.type == "supergroup" and not message.is_topic_message)) and message.reply_to_message is not None) or (chat.type == "supergroup" and message.is_topic_message and message.reply_to_message.text is not None):
            # Create or add to the text property with the reply text
            nd["text"] = message.reply_to_message.text if nd["text"] is None else nd["text"] + "\n\n" + message.reply_to_message.text

        # Display role selection
        keyboard = [
            [
                InlineKeyboardButton("Users", callback_data="user"),
                InlineKeyboardButton("Beta Testers", callback_data="tester")
            ],
            [
                InlineKeyboardButton("Marketing", callback_data="marketing"),
                InlineKeyboardButton("Administrators", callback_data="admin")
            ],
            [
                InlineKeyboardButton("Done", callback_data="done")
            ]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)

        try:
            await message.reply_text(
                text="Select roles to recieve newsletter.",
                reply_markup=reply_markup
            )
        except Exception as err:
            logger.warning(f"The following error occurred while sending a telegram message:  {err}.")
        finally:
            return ROLE_SELECTION
            


    else:
        logStr = f"User {user.name} (user_id: {user.id}) (roles:  {', '.join(member['roles'])}) is NOT authorized to use the newsletter command"
        if chat.type != "private":
            logStr = logStr + f" in the {community['chat_title']} group chat (roles:  {', '.join(community['roles'])})."
        else:
            logStr = logStr + f" in private chat."
        logger.info(logStr)

        # Unauthorized

        return ConversationHandler.END


async def selectRole(update: Update, context: ContextTypes.DEFAULT_TYPE):
    logger.info("select role")
    message = update.effective_message
    query = update.callback_query
    try:
        await query.answer()
    except Exception as err:
        logger.warning(f"The following error occurred while receiving a telegram query response:  {err}.")
        return

    nd = context.chat_data["newsletter_data"]

    prettyTitles = {
        "user": "Users",
        "tester": "Testers",
        "marketing": "Marketing",
        "admin": "Administrators",
        "owner": "Owner"
    }

    if query.data in members.rolesList:
        nd["roles"].append(query.data)

    buttonList = []
    remainingRoles = [role for role in members.rolesList if role not in nd["roles"]]

    for role in remainingRoles:
        buttonText = role if role not in prettyTitles else prettyTitles[role]
        buttonList.append(InlineKeyboardButton(buttonText, callback_data=role))

    buttonList.append(InlineKeyboardButton("Done", callback_data="done"))
    
    # Display role selection
    def pairs(l):
        for i in range(0, len(l), 2):
            yield l[i:i + 2]

    keyboard = list(pairs(buttonList))
    reply_markup = InlineKeyboardMarkup(keyboard)

    try:
        await query.edit_message_text(
            text=f"Select roles to recieve newsletter. Roles selected: {', '.join(nd['roles'])}",
            reply_markup=reply_markup
        )
        return ROLE_SELECTION
    
    except Exception as err:
        logger.warning(f"The following error occurred while sending a telegram message:  {err}.")
        return


async def roleSelectionDone(update: Update, context: ContextTypes.DEFAULT_TYPE):
    logger.info("Role selection done")
    query = update.callback_query
    try:
        await query.answer()
    except Exception as err:
        logger.warning(f"The following error occurred while receiving a telegram query response:  {err}.")
        return

    nd = context.chat_data["newsletter_data"]

    if query.data == "done" and len(nd["roles"]) > 0:
        if nd["text"] is None:
            # Ask if they wish to add a photo.

            keyboard = [
                [
                    InlineKeyboardButton("Yes", callback_data="yes"),
                    InlineKeyboardButton("No", callback_data="no")
                ]
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)
            
            try:
                await query.edit_message_text(
                    text=f"Do you want to add an image?",
                    reply_markup=reply_markup
                )
                return PHOTO_OPTION
            except Exception as err:
                logger.warning(f"The following error occurred while editing a telegram query message:  {err}.")
                return

        else:
            # Already have text body, get confirmation
            keyboard = [
                [
                    InlineKeyboardButton("Yes", callback_data="yes"),
                    InlineKeyboardButton("No", callback_data="no")
                ]
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)

            try:
                await query.edit_message_text(
                    text=f"Here's your newsletter:\n\n{nd['text']}\n\nSending to the following roles: {', '.join(nd['roles'])}",
                    reply_markup=reply_markup
                )
                return CONFIRM_NEWSLETTER
            except Exception as err:
                logger.warning(f"The following error occurred while editing a telegram query message:  {err}.")
                return
    
    return ConversationHandler.END


async def photoOption(update: Update, context: ContextTypes.DEFAULT_TYPE):
    logger.info("Add a photo or continue to newsletter text")
    query = update.callback_query
    try:
        await query.answer()
    except Exception as err:
        logger.warning(f"The following error occurred while receiving a telegram query response:  {err}.")
        return
    
    nd = context.chat_data["newsletter_data"]

    if query.data == "yes":
        try:
            await query.edit_message_text(
                text="Send the image you wish to add to the newsletter"
            )
            return ADD_PHOTO
        except Exception as err:
            logger.warning(f"The following error occurred while editing a telegram query message:  {err}.")
            return
    else:
        # Get newsletter body text
        try:
            await query.edit_message_text(
                text=f"Compose the text of the newsletter. Newsletter will be sent to the following roles: {', '.join(nd['roles'])}"
            )
            return COMPOSE_NEWLETTER
        except Exception as err:
            logger.warning(f"The following error occurred while editing a telegram query message:  {err}.")
            return


async def addNewsletterPhoto(update: Update, context: ContextTypes.DEFAULT_TYPE):
    logger.info("Add the newsletter photo")
    message = update.effective_message

    nd = context.chat_data["newsletter_data"]

    photo = update.message.photo[-1].file_id

    if photo:
        nd["photo"] = photo

        try:
            await message.reply_text(
                text=f"Compose the text of the newsletter. Newsletter will be sent to the following roles: {', '.join(nd['roles'])}"
            )
            return COMPOSE_NEWLETTER
        except Exception as err:
            logger.warning(f"The following error occurred while sending a telegram message:  {err}.")
            return
    else:
        logger.info("There was an issue getting the photo.")
        return COMPOSE_NEWLETTER


async def addNewsletterText(update: Update, context: ContextTypes.DEFAULT_TYPE):
    logger.info("Add the newsletter text")
    message = update.effective_message

    nd = context.chat_data["newsletter_data"]

    if len(message.text) > 0:
        nd["text"] = message.text

        keyboard = [
            [
                InlineKeyboardButton("Yes", callback_data="yes"),
                InlineKeyboardButton("No", callback_data="no")
            ]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)

        # Check if a photo was added and reply with photo and caption if so
        newsletterPhoto = nd.get("photo")
        if newsletterPhoto is not None:
            logger.info("Newsletter has a photo.")
            try:
                await message.reply_photo(
                    photo=newsletterPhoto,
                    caption=f"Here's your newsletter:\n\n{nd['text']}\n\nSending to the following roles: {', '.join(nd['roles'])}",
                    reply_markup=reply_markup
                )
            except Exception as err:
                logger.warning(f"The following error occurred while sending a telegram message:  {err}.")
                return
        else:
            logger.info("Newsletter has no photo.")
            try:
                await message.reply_text(
                    text=f"Here's your newsletter:\n\n{nd['text']}\n\nSending to the following roles: {', '.join(nd['roles'])}",
                    reply_markup=reply_markup
                )
            except Exception as err:
                logger.warning(f"The following error occurred while sending a telegram message:  {err}.")
                return

        return CONFIRM_NEWSLETTER
    else:
        # User didn't send text
        return ConversationHandler.END


async def confirmNewsletter(update: Update, context: ContextTypes.DEFAULT_TYPE):
    logger.info("Confirm sending the newsletter")
    query = update.callback_query
    try:
        await query.answer()
    except Exception as err:
        logger.warning(f"The following error occurred while receiving a telegram query response:  {err}.")
        return
    
    nd = context.chat_data["newsletter_data"]

    if query.data == "yes":
        try:
            if query.message.caption is not None:
                await query.edit_message_caption(
                    caption="Sending..."
                )
            else:
                await query.edit_message_text(
                    text="Sending..."
                )
        except Exception as err:
            logger.warning(f"The following error occurred while editing a telegram query message:  {err}.")
            return
        
        membersInRoles = members.getMembersByRoles(nd["roles"])
        newsletterPhoto = nd.get("photo")
        for member in membersInRoles:
            try:
                if newsletterPhoto is not None:
                    await context.bot.send_photo(
                        chat_id=member["user_id"],
                        caption=nd["text"],
                        photo=newsletterPhoto
                    )
                else:
                    await context.bot.send_message(
                        chat_id=member["user_id"],
                        text=nd["text"]
                    )
                print(f"sent to {member['username']} ({member['user_id']}).")
            except Exception:
                print(f"Unable to send to {member['username']} ({member['user_id']}).")
        
        try:
            if query.message.caption is not None:
                await query.edit_message_caption(
                    caption=f"Sent:\n\n{nd['text']}\n\nto the following roles: {', '.join(nd['roles'])}"
                )
            else:
                await query.edit_message_text(
                    text=f"Sent:\n\n{nd['text']}\n\nto the following roles: {', '.join(nd['roles'])}"
                )
        except Exception as err:
            logger.warning(f"The following error occurred while editing a telegram query message:  {err}.")
    else:
        try:
            if query.message.caption is not None:
                await query.edit_message_caption(
                    caption=f"Newsletter not sent"
                )
            else:
                await query.edit_message_text(
                    text=f"Newsletter not sent"
                )
        except Exception as err:
            logger.warning(f"The following error occurred while editing a telegram query message:  {err}.")
            return
    
    return ConversationHandler.END



SELECT_PROPOSAL, PROPOSAL_NDA, SHOW_PROPOSAL = range(3)

async def proposalsManager(update: Update, context: ContextTypes.DEFAULT_TYPE):
    logger.info(f"Proposals Manager started.")
    chat = update.effective_chat
    message = update.effective_message
    topicID = message.message_thread_id if message.is_topic_message else None
    user = update.effective_user

    member = members.getMemberByTelegramID(user.id)

    if member is None:
        logger.info(f"An unregistered user {user.name} (user_id: {user.id}) attempted to use the proposals command.")
        # Exit the function if there is no user account
        return ConversationHandler.END

    # Set the allowed roles
    allowedRoles = ["admin", "owner"]
    rolesAvailable = member["roles"]

    if (any(role in rolesAvailable for role in allowedRoles)):
        context.chat_data["proposals"] = proposals = proposals.getProposals()

        if len(proposals) > 0 and len(proposals) < 6:
            logger.info(f"Show proposal list.")
            # Loop and build proposal list
            responseText = "The following proposals are available:\n"
            responseKeyboard = []
            for p in proposals:
                responseText += f"\n*{p['project_title']}* - _{p['submitted_from']}_\n*Description:*  {p['project_description']}"
                # Build reply inline keyboard for each proposal
                btn = [InlineKeyboardButton(p["project_title"], callback_data=p["project_id"])]
                responseKeyboard.append(btn)
            
            reply_markup = InlineKeyboardMarkup(responseKeyboard)
            
            await message.reply_markdown(
                text=responseText,
                reply_markup=reply_markup
            )

            return SELECT_PROPOSAL
        else:
            logger.info(f"There are no proposals to show.")

            await message.reply_text(text="There are no proposals to show.")
    else:
        logger.info(f"An unauthorized user {user.name} (user_id: {user.id}) attempted to use the proposals command.")
    
    return ConversationHandler.END


async def agreeNDA(update: Update, context: ContextTypes.DEFAULT_TYPE):
    logger.info(f"Proposal selected, get NDA confirmation.")
    message = update.effective_message
    query = update.callback_query
    await query.answer()
    proposals = context.chat_data.get("proposals")

    if proposals is not None:
        context.chat_data["proposal"] = proposal = next((p for p in proposals if p["project_id"] == int(query.data)), None)

        if proposal is None:
            return ConversationHandler.END

        # Proposal selected, display NDA
        keyboard = [
            [
                InlineKeyboardButton("Confirm", callback_data="confirm"),
                InlineKeyboardButton("Reject", callback_data="reject")
            ]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)

        confidentialAgreementResponse = f"""*The following proposal ({context.chat_data["proposal"]["project_title"]}) is a confidential document.*
CONFIDENTIALITY AGREEMENT

I acknowledge that the document and its contents provided by Hypermind Labs are confidential. I agree to:

Keep the document and all its contents strictly confidential.

Not share, distribute, or disclose the document or its contents to any third party without prior written consent from Hypermind Labs.

Use the document solely for the intended purpose of evaluating the proposal.

Take all reasonable measures to protect the confidentiality of the document and its contents.

Do you agree to these terms?"""

        await query.edit_message_text(
            text=confidentialAgreementResponse,
            reply_markup=reply_markup,
            parse_mode=constants.ParseMode.MARKDOWN
        )

        return PROPOSAL_NDA
    else:
        return ConversationHandler.END


async def openProposal(update: Update, context: ContextTypes.DEFAULT_TYPE):
    logger.info(f"NDA responded to, handle response.")
    message = update.effective_message
    query = update.callback_query
    user = update._effective_user
    await query.answer()

    if query.data != "confirm":
        await query.edit_message_text(
            text="You must agree to the NDA to view the proposal."
        )
        return ConversationHandler.END

    # User selected confirm, load the proposal
    proposal = context.chat_data.get("proposal")
    # Store the NDA acceptance
    proposals.addDisclosureAgreement(user.id, proposal.get("project_id"))

    proposalsPath = "assets/proposals/"
    proposalFile = proposal.get("filename")

    script_dir = os.path.dirname(__file__)
    rel_path = proposalsPath + proposalFile
    abs_file_path = os.path.join(script_dir, rel_path)


    await query.edit_message_text(
        text="Getting the proposal..."
    )

    await message.reply_document(
        document=open(abs_file_path, "rb"),
        protect_content=True
    )
    return ConversationHandler.END



####################
# Message Handlers #
####################

async def catchAllMessages(update: Update, context: ContextTypes.DEFAULT_TYPE):
    logger.info(f"A message was captured by the catch all function.")

# TODO when incoming message, check if the model is loaded and if not preload the main chat model before doing embeddings and db lookup. This may increase response time.


async def directChatGroup(update: Update, context: ContextTypes.DEFAULT_TYPE):
    logger.info(f"Bot messaged in group chat.")
    chat = update.effective_chat
    message = update.effective_message
    topicID = message.message_thread_id if message.is_topic_message else None
    user = update.effective_user

    # Get community data for telegram group
    community = communities.getCommunityByTelegramID(chat.id)

    if community is None:
        logger.info(f"User {user.name} (user_id: {user.id}) attempted to message the chatbot in an unregistered group chat.")
        # No community account, exit the function
        return

    # Get member data
    member = members.getMemberByTelegramID(user.id)

    if member is None:
        logger.info(f"An unregistered user {user.name} (user_id: {user.id}) attempted to message the chatbot in {community['chat_title']} group chat.")
        try:
            await message.reply_text(text=f"To message {config.botName}, open a private message to the chatbot @{config.botName} and click start.")
        except Exception as err:
            logger.warning(f"The following error occurred while sending a telegram message:  {err}.")
        finally:
            # No member account, exit the function
            return  
    
    memberID = member.get("member_id")
    # Set the allowed roles
    allowedRoles = ["tester", "marketing", "admin", "owner"]
    rolesAvailable = set(member["roles"] + community["roles"])
    
    # Check user usage
    memberUsage = usage.getUsageForMember(memberID)
    rateLimits = communityScore.getRateLimits(memberID, "community")

    if (not any(role in rolesAvailable for role in allowedRoles)):
        # User does not have permission for non rate limited chat
        # Check if user has exceeded hourly rate
        if len(memberUsage) >= rateLimits["message"]:
            logger.info(f"User {user.name} (user_id: {user.id}) has reached their hourly message rate in {community['chat_title']} group chat.")
            try:
                await message.reply_text(text=f"You have reached your hourly rate limit.")
            except Exception as err:
                logger.warning(f"The following error occurred while sending a telegram message:  {err}.")
            finally:
                # User has reach rate limit, exit function
                return

    messageContext = {
        "community_id": community.get("community_id"),
        "platform": "telegram",
        "topic_id" : topicID,
        "message_timestamp": datetime.now()
    }
    
    # Create the conversational agent instance
    conversation = ConversationOrchestrator(message.text, memberID, messageContext, message.message_id)

    try:
        # Shows the bot as "typing"
        await context.bot.send_chat_action(
            chat_id=chat.id, 
            action=constants.ChatAction.TYPING,
            message_thread_id=topicID
        )
    except Exception as err:
        logger.error(msg="Exception while sending a chat action for typing:  ", exc_info=err, stack_info=False)
        #logger.warning(f"The following error occurred while sending chat action to telegram:  {err}.")

    response = await conversation.runAgents()
    
    try:
        # Send the conversational agent response
        responseMessage = await context.bot.send_message(
            chat_id=chat.id,
            message_thread_id=topicID,
            text=f"{response}\n\n*Disclaimer*:  Test chatbots are prone to hallucination. Responses may or may not be factually correct."
        )
    except Exception as err:
        logger.error(msg="Exception while sending a telegram message:  ", exc_info=err, stack_info=False)
        #logger.warning(f"The following error occurred while sending a telegram message:  {err}.")
        return

    # Score the message if the message is greater than 20 words
    if len(message.text.split(" ")) > 20:
        communityScore.scoreMessage(conversation.promptHistoryID)

    # Add the response to the chat history
    conversation.storeResponse(responseMessage.message_id)

    # Add the retrieved documents to the retrieval table
    # This had to wait for the prompt and response chat history records to be created
    # TODO Move this code to the conversational orchestrator
    """
    docs = chatAgent._documents
    for doc in docs:
        retrievalID = knowledge.addRetrieval(chatAgent.promptHistoryID, responseHistoryID, doc.get("knowledge_id"), doc.get("distance"))
    """

    # Add the stats to the usage manager
    usage.addUsage(conversation.promptHistoryID, conversation.responseHistoryID, conversation.stats)

    # Check and send if send stats is enabled
    userDefaults = context.user_data["defaults"] = {} if "defaults" not in context.user_data else context.user_data["defaults"]

    if userDefaults.get("send_stats"):
        try:
            await sendStats(stats=conversation.stats, chatID=chat.id, context=context, threadID=topicID)
        except Exception as err:
            logger.warning(f"The following error occurred while sending a telegram message:  {err}.")


async def directChatPrivate(update: Update, context: ContextTypes.DEFAULT_TYPE):
    logger.info(f"Bot messaged in private chat.")
    chat = update.effective_chat
    message = update.effective_message
    user = update.effective_user

    minimumCommunityScore = 50

    # Get account information
    member = members.getMemberByTelegramID(user.id)

    if member is None:
        logger.info(f"Unregistered user {user.name} (user_id: {user.id}) messaged the bot in a private message.")
        try:
            await message.reply_text("Use the /start command to begin chatting with the chatbot.")
        except Exception as err:
            logger.warning(f"The following error occurred while sending a telegram message:  {err}.")
        finally:
            # No member account, exit the function
            return
    
    memberID = member.get("member_id")
    
    # Set the allowed roles
    allowedRoles = ["tester", "marketing", "admin", "owner"]
    rolesAvailable = member.get("roles")
    
    # Check user usage
    memberUsage = usage.getUsageForMember(memberID)
    rateLimits = communityScore.getRateLimits(memberID, chat.type)

    if (not any(role in rolesAvailable for role in allowedRoles)):
        # User does not have permission for non rate limited chat
        # Check if user has exceeded hourly rate
        if len(memberUsage) >= rateLimits["message"]:
            try:
                if member["community_score"] < minimumCommunityScore:
                    logger.info(f"User {user.name} (user_id: {user.id}) does not meet the minimum community score to use private chat with the chatbot. Community score:  {member['community_score']}/{minimumCommunityScore}")
                    await message.reply_text(f"You need a minimum community score of {minimumCommunityScore} to use the private chat features. Please join one of our community chats to build your community score.")
                else:
                    logger.info(f"User {user.name} (user_id: {user.id}) has reached their hourly message rate in private chat.")
                    await message.reply_text(text=f"You have reached your hourly rate limit.")
            except Exception as err:
                logger.warning(f"The following error occurred while sending a telegram message:  {err}.")
            finally:
                # User has reach rate limit, exit function
                return

    messageData = {
        "community_id": None,
        "member_id": member.get("member_id"),
        "platform": "telegram",
        "topic_id" : None,
        "message_timestamp": datetime.now()
    }

    conversation = ConversationOrchestrator(message.text, memberID, messageData, message.message_id)

    try:
        # Shows the bot as "typing"
        await context.bot.send_chat_action(
            chat_id=chat.id, 
            action=constants.ChatAction.TYPING
        )
    except Exception as err:
        logger.error(msg="Exception while sending a chat action for typing.", exc_info=err, stack_info=False)
        #logger.warning(f"The following error occurred while sending chat action to telegram:  {err}.")
    
    response = await conversation.runAgents()
    
    try:
        responseMessage = await message.reply_text(
            text=response
        )
    except Exception as err:
        logger.error(msg="Exception while replying to a telegram message", exc_info=err, stack_info=False)
        #logger.warning(f"The following error occurred while sending a telegram message:  {err}.")
        return
    
    # Score the message if the message is greater than 20 words
    if len(message.text.split(" ")) > 20:
        communityScore.scoreMessage(conversation.promptHistoryID)

    conversation.storeResponse(responseMessage.message_id)

    # Add the stats to the usage manager
    usage.addUsage(conversation.promptHistoryID, conversation.responseHistoryID, conversation.stats)

    # Check and send if send stats is enabled
    userDefaults = context.user_data["defaults"] = {} if "defaults" not in context.user_data else context.user_data["defaults"]

    if userDefaults.get("send_stats"):
        try:
            await sendStats(stats=conversation.stats, chatID=chat.id, context=context)
        except Exception as err:
            logger.warning(f"The following error occurred while sending a telegram message:  {err}.")


# TODO Still need to handle chat history and usage storage for images
async def handleImage(update: Update, context: ContextTypes.DEFAULT_TYPE):
    logger.info(f"Image received.")
    chat = update.effective_chat
    message = update.effective_message
    user = update.effective_user
    topicID = message.message_thread_id if message.is_topic_message else None

    # Need user account regardless of chat type
    member = members.getMemberByTelegramID(user.id)
    if member is None:
        logger.info(f"Unregistered user {user.name} (user_id: {user.id}) sent an image in a {chat.type} chat.")
        await message.delete()
        await context.bot.send_message(
            chat_id=chat.id,
            message_thread_id=topicID,
            text=f"Start a private chat with the @{config.botName} to send images in this chat."
        )
        
        # Exit the function
        return
    
    memberID = member.get("member_id")
    minimumCommunityScore = 70 if chat.type == "private" else 20
    # Set the allowed roles
    allowedRoles = ["tester", "marketing", "admin", "owner"]
    
    # Get account information
    if chat.type == "private":
        # Set the rolesAvailable
        rolesAvailable = member["roles"]
        # Get rate limits for private chat
        rateLimits = communityScore.getRateLimits(memberID, chat.type)

    elif chat.type == "group" or chat.type == "supergroup":
        community = communities.getCommunityByTelegramID(chat.id)
        if community is None:
            logger.info(f"User {user.name} (user_id: {user.id}) attempted to send an image in an unregistered group chat.")
            await message.delete()
            # Exit the function
            return
        
        # Combine the user and group roles into rolesAvailable
        rolesAvailable = set(member["roles"] + community["roles"])    
        # Get rate limits for group chat
        rateLimits = communityScore.getRateLimits(memberID, "community") 
    else:
        # Only chat types allowed are private, group, and supergroup
        return
    
    if (not any(role in rolesAvailable for role in allowedRoles)):
        # User does not have permission for non rate limited chat
        # Check if user has exceeded hourly rate
        memberUsage = usage.getUsageForMember(memberID)
        if len(memberUsage) >= rateLimits["image"]:
            if member["community_score"] < minimumCommunityScore:
                logger.info(f"User {user.name} (user_id: {user.id}) does not meet the minimum community score to send images in a {chat.type} chat.")
                await message.delete()
                await context.bot.send_message(
                    chat_id=chat.id,
                    message_thread_id=topicID,
                    text=f"You need a minimum community score of {minimumCommunityScore} to send images in a {chat.type} chat."
                )
            else:
                logger.info(f"User {user.name} (user_id: {user.id}) has reached their hourly message rate in {community['chat_title']} group chat.")
                await message.reply_text(text=f"You have reached your hourly rate limit.")
            
            # User has reach rate limit, exit function
            return

    # Convert images to base64
    photoFile = await message.effective_attachment[-1].get_file()
    photoBytes = await photoFile.download_as_bytearray()
    b64_photo = base64.b64encode(photoBytes)
    b64_string = b64_photo.decode()
    imageList = [b64_string]
    imagePrompt = "Describe this image." if not message.caption else message.caption

    messageData = {
        "chat_id": chat.id,
        "topic_id" : topicID,
        "message_id": message.message_id,
        "message_images": imageList,
        "message_text": imagePrompt
    }
    
    # Create the conversational agent instance
    imageAgent = ImageAgent(messageData, memberID)
    
    # Shows the bot as "typing"
    await context.bot.send_chat_action(
        chat_id=chat.id, 
        action=constants.ChatAction.TYPING,
        message_thread_id=topicID
    )
    
    response = await imageAgent.generateResponse()

    # Send the conversational agent response
    responseMessage = await context.bot.send_message(
        chat_id=chat.id,
        message_thread_id=topicID,
        text=f"{response}\n\n*Disclaimer*:  Test chatbots are prone to hallucination. Responses may or may not be factually correct."
    )

    # TODO Handle adding prompt and response into the chat history collection.
#good

async def otherGroupChat(update: Update, context: ContextTypes.DEFAULT_TYPE):
    logger.info(f"Other messages (not directed at the chatbot) from group chats.")
    # Add messages to chat history
    chat = update.effective_chat
    message = update.effective_message
    user = update.effective_user
    topicID = message.message_thread_id if message.is_topic_message else None

    community = communities.getCommunityByTelegramID(chat.id)
    member = members.getMemberByTelegramID(user.id)
    
    if community and member:
        # Update the chat history database with the newest message
        messageHistoryID = chatHistory.addChatHistory(
            messageID=message.message_id, 
            messageText=message.text, 
            platform="telegram", 
            memberID=member.get("member_id"), 
            communityID=community.get("community_id"), 
            topicID=topicID, 
            timestamp=datetime.now()
        )

        # Score the message if the user account exist and the message is greater than 20 words
        if len(message.text.split(" ")) > 20:
            communityScore.scoreMessage(messageHistoryID)
#good

# TODO only include chat history from the reply chain
async def replyToBot(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """This function gets called when a user replies to a messages in a group (or supergroup) chat. 
    Replies are filtered to handle replies to the chatbot."""
    logger.info(f"Handle replies in a group (or supergroup) chat.")
    chat = update.effective_chat
    message = update.effective_message
    user = update.effective_user
    topicID = message.message_thread_id if message.is_topic_message else None

    # Get community information
    community = communities.getCommunityByTelegramID(chat.id)

    # Group chat must be registered
    if community is None:
        # This can only occur after an authorizzed user has began adding the chatbot to group chat, but before they are able to finish the registration process
        logger.info(f"User {user.name} (user_id: {user.id}) attempted to message the chatbot in an unregistered group chat.")
        try:
            await message.reply_text(text="Chatbot is not registered with this group chat.")
        except Exception as err:
            logger.warning(f"The following error occurred while sending a telegram message:  {err}.")
        finally:
            return
    
    communityID = community.get("community_id")
    # Determine if the message is an actual reply or a regular message sent in a topic thread in a supergroup
    if (topicID and not message.reply_to_message.text) or (message.reply_to_message.text and message.reply_to_message.from_user.id != config.bot_id):
        logger.info(f"Message is not a reply to the bot but a message sent in a supergroup. Forward message to proper handler function.")
        # If the message is text, forward to other group chat
        try:
            if message.text:
                logger.info(f"Message is text type. Forward message to otherGroupChat function.")
                await otherGroupChat(update=update, context=context)
            elif message.effective_attachment is not None and type(message.effective_attachment) is tuple:
                logger.info(f"Message is image type. Forward message to handleImage function.")
                await handleImage(update=update, context=context)
        except Exception as err:
            logger.warning(f"The following error occurred while sending a telegram message:  {err}.")
        finally:
            return

    # Message is a reply to the chatbot
    
    # Get user account data
    member = members.getMemberByTelegramID(user.id)

    if member is None:
        logger.info(f"An unregistered user {user.name} (user_id: {user.id}) attempted to reply to a message from the chatbot in {community['chat_title']} group chat.")
        try:
            await message.reply_text(text=f"To reply to {config.botName}, open a private message to the chatbot @{config.botName} and click start.")
        except Exception as err:
            logger.warning(f"The following error occurred while sending a telegram message:  {err}.")
        finally:
            # No user account, exit the function
            return
    
    memberID = member.get("member_id")
    
    # Set the allowed roles
    allowedRoles = ["tester", "marketing", "admin", "owner"]
    rolesAvailable = set(member["roles"] + community["roles"])
    
    if (not any(role in rolesAvailable for role in allowedRoles)):
        # User does not have permission for non rate limited chat
        # Check if user has exceeded hourly rate
        memberUsage = usage.getUsageForMember(memberID)
        rateLimits = communityScore.getRateLimits(member, "community")
        if len(memberUsage) >= rateLimits["message"]:
            logger.info(f"User {user.name} (user_id: {user.id}) has reached their hourly message rate in {community['chat_title']} group chat.")
            try:
                await message.reply_text(text=f"You have reached your hourly rate limit.")
            except Exception as err:
                logger.warning(f"The following error occurred while sending a telegram message:  {err}.")
            finally:
                # User has reach rate limit, exit function
                return

    messageData = {
        "community_id": communityID,
        "member_id": memberID,
        "platform": "telegram",
        "topic_id" : topicID,
        "message_id": message.message_id,
        "message_text": message.text,
        "message_timestamp": datetime.now()
    }
    
    # Create the conversational agent instance
    conversation = ConversationOrchestrator(message.text, memberID, messageData, message.message_id)
    try:
    # Shows the bot as "typing"
        await context.bot.send_chat_action(
            chat_id=chat.id, 
            action=constants.ChatAction.TYPING,
            message_thread_id=topicID
        )
    except Exception as err:
        logger.error("Exception while sending a telegram message", stack_info=False, exc_info=err)
        #logger.warning(f"The following error occurred while sending chat action to telegram:  {err}.")

    response = await conversation.runAgents()

    try:
        # Send the conversational agent response
        responseMessage = await context.bot.send_message(
            chat_id=chat.id,
            message_thread_id=topicID,
            text=f"{response}\n\n*Disclaimer*:  Test chatbots are prone to hallucination. Responses may or may not be factually correct."
        )
    except Exception as err:
        logger.error("Exception while sending a telegram message", stack_info=False, exc_info=err)
        #logger.warning(f"The following error occurred while sending a telegram message:  {err}.")
        return


    # Score the message if the message is greater than 20 words
    if len(message.text.split(" ")) > 20:
        communityScore.scoreMessage(conversation.promptHistoryID)

    conversation.storeResponse(responseMessage.message_id)

    # Add the stats to the usage manager
    usage.addUsage(conversation.promptHistoryID, conversation.responseHistoryID, conversation.stats)

    # Check and send if send stats is enabled
    userDefaults = context.user_data["defaults"] = {} if "defaults" not in context.user_data else context.user_data["defaults"]

    if userDefaults.get("send_stats"):
        try:
            await sendStats(stats=conversation.stats, chatID=chat.id, context=context, threadID=topicID)
        except Exception as err:
            logger.warning(f"The following error occurred while sending a telegram message:  {err}.")



#########################
# Other Update Handlers #
#########################

async def botStatusChanged(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Update handler for when the bot's status has changed.
    Use to register group chat accounts when bot is added to a group."""
    logger.info(f"The bot's status has been updated.")
    chat = update.effective_chat
    fromUser = update.my_chat_member.from_user
    newStatus = update.my_chat_member.new_chat_member

    # Get the group account information
    community = communities.getCommunityByTelegramID(chat.id)
    
    # Check if status change is being added to a group and if the group is not registered with the chatbot
    if newStatus.status == constants.ChatMemberStatus.MEMBER and community is None:
        logger.info(f"Chatbot has been added to a new group chat.")
        # Get the account for the user adding the chatbot to the group
        member = members.getMemberByTelegramID(fromUser.id)

        # Set the allowed roles
        allowedRoles = ["admin", "owner"]
        rolesAvailable = member.get("roles") if member else []

        if (not any(role in rolesAvailable for role in allowedRoles)):
            # User does not have permission to add the bot to the chat
            # Check if user exist
            if member is None:
                logger.info(f"A non-registered user {fromUser.name} ({fromUser.id}) attempted to add the chatbot to the {chat.title} group chat.")
                responseText="Non registered user's are not authorized to add the chatbot to group chats. To register, start a private conversation with the chatbot."
            else:
                logger.info(f"A non-authorized user {fromUser.name} ({fromUser.id}) attempted to add the chatbot to the {chat.title} group chat.")
                responseText = "You are not authorized to add the chatbot to group chats."
                
            # TODO Wrap this is try except block
            # Send response
            await context.bot.send_message(
                chat_id=chat.id,
                text=responseText
            )
            # Exit the group chat
            await context.bot.leave_chat(chat.id)

            # Exit the function
            return

        # Chat isn't registered and user is authorized
        logger.info(f"User {fromUser.name} ({fromUser.id}) is authorized to add chatbot to group chats.")

        newCommunityData = {
            "community_name": chat.title,
            "community_link": None,
            "roles": ["user"],
            "created_by": member.get("member_id"),
            "chat_id": chat.id,
            "chat_title": chat.title,
            "has_topics": True if (chat.type) == "supergroup" else False,
            "register_date": datetime.now()
        }

        # Add new individual account via accounts manager
        communities.addCommunityFromTelegram(newCommunityData)

        await context.bot.send_message(
            chat_id=chat.id,
            text=f"Hello, I am the {config.botName} chatbot. Use the /help command for more information."
        )
        # Exit the function
        return
        
    elif newStatus.status != constants.ChatMemberStatus.MEMBER:
        logger.info(f"The chatbot's status change is:  {newStatus.status}.")
        return
        
    elif community is not None:
        logger.info(f"The chatbot has already been registered for this group.")
        return
    else:
        return
#good

####################
# Helper Functions #
####################

async def setPassword(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    logger.info(f"User is requesting an new access key.")
    chat = update.effective_chat
    message = update.effective_message
    user = update.effective_user
    topicID = message.message_thread_id if message.is_topic_message else None

    member = members.getMemberByTelegramID(user.id)
    if member is not None:
        memberID = member.get("member_id")
        if len(context.args) == 1:
            password = context.args[0]
            # Do regular expression
            pattern = re.compile(r"(?=.*\d)(?=.*[a-z])(?=.*[A-Z])(?=.*[-+_!@#$%^&*.,?]).{12,}")
            validPassword = pattern.search(password)
            if validPassword:
                #password = accounts.setPassword(user.id, password=password)
                password = members.setPassword(memberID, password)
                response = f"Your password {password} has been stored"
            else:
                response = "You must enter a valid password.\nHint:  passwords must by a minimum of 12 characters, contain at least one lowercase, uppercase, digit, and symbol."

        else:
            # Generate a random password
            password = members.setPassword(memberID)
            if password:
                response = f"Your randomly generated password is:  {password}"
            else:
                response = "Something went wrong."
        try:
            await message.reply_text(text=response)
        except Exception as err:
            logger.warning(f"The following error occurred while sending a telegram message:  {err}.")

        # TODO Set up auto delete of message response
#good

async def statisticsManager(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    # For now this function will just toggle on or off the sending of statistics
    logger.info("Toggling the statistics message")
    chat = update.effective_chat
    message = update.effective_message
    user = update.effective_user

    member = members.getMemberByTelegramID(user.id)

    if member is not None:
        userDefaults = context.user_data["defaults"] = {} if "defaults" not in context.user_data else context.user_data["defaults"]
        userDefaults["send_stats"] = True if "send_stats" not in userDefaults else not userDefaults["send_stats"]

        await context.bot.send_message(
            chat_id=chat.id,
            message_thread_id=message.message_thread_id if message.is_topic_message else None,
            text=f"Show statistics:  {userDefaults['send_stats']}"
        )
#good

async def sendStats(chatID: int, context: ContextTypes.DEFAULT_TYPE, stats: dict, threadID=None) -> None:
    logger.info(f"Send stats for previously generated message.")
    messageText = f"""_Statistics for previous message:_
*Load duration:*  {(stats['load_duration'] / 1000000000):.2f} seconds

*Prompt tokens:*  {stats['prompt_eval_count']} 
*Prompt tokens per second:*  {(stats['prompt_eval_count'] / (stats['prompt_eval_duration'] / 1000000000)):.2f} 

*Response tokens:*  {stats['eval_count']} 
*Response tokens per second:*  {(stats['eval_count'] / (stats['eval_duration'] / 1000000000)):.2f} 

*Total tokens per second:*  {((stats['prompt_eval_count'] + stats['eval_count']) / ((stats['prompt_eval_duration'] + stats['eval_duration']) / 1000000000)):.2f} 
*Total duration:*  {(stats['total_duration'] / 1000000000):.2f} seconds """
    
    await context.bot.send_message(
        chat_id=chatID,
        message_thread_id=threadID,
        parse_mode=constants.ParseMode.MARKDOWN,
        text=messageText
    )
#good

async def reactionsHandler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    logger.info(f"Handling message reactions.")
    chat = update.effective_chat
    message = update.effective_message
    user = update.effective_user

    # Temporarily only accepting 👍 emoji
    reaction = update.message_reaction.new_reaction[0]
    member = members.getMemberByTelegramID(user.id)
    community = communities.getCommunityByTelegramID(chat.id)
    if member is None or reaction is None or community is None:
        return
    
    memberID = member.get("member_id")
    communityID = community.get("community_id")

    if reaction.emoji == "👍":
        logger.info(f"Registered user {user.name} (user_id: {user.id}) reacted to a message with the 👍 emoji. Process community score.")

        # Get the history ID
        originalMessage = chatHistory.getMessageByMessageID(communityID, "community", "telegram", update.message_reaction.message_id)
        if originalMessage:
            communityScore.scoreMessageFromReaction(memberID, originalMessage.get("history_id"))
#good

async def handleForwardedMessage(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """This function is for spam prevention. 
    It is called when forwarded messages are sent in a group chat."""
    logger.info(f"Handle forwarded messages.")
    chat = update.effective_chat
    message = update.effective_message
    user = update.effective_user
    topicID = message.message_thread_id if message.is_topic_message else None

    # Prevent spam abuse
    member = members.getMemberByTelegramID(user.id)

    if member is None:
        # Delete the message and give the user information to message the chatbot
        logger.info(f"An unregistered user {user.name} (user_id: {user.id}) attempted to forward a message into the group chat.")

        await message.delete()
        await context.bot.send_message(
            chat_id=chat.id,
            message_thread_id=topicID,
            text=f"Start a private chat with the @{config.botName} to forward messages to this chat."
        )

        return
    
    logger.info(f"Registered user {user.name} (user_id: {user.id}) forwarded a message.")
#good

async def errorHandler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    logger.error("Exception while handling an update:", exc_info=context.error)



#################
# Main Function #
#################

def main() -> None:
    # Run the bot
    # Create the Application and pass it your bot's token.
    # .get_updates_write_timeout(100)
    application = Application.builder().token(config.bot_token).concurrent_updates(True).get_updates_write_timeout(500).build()

    # Generate command chain
    generateHandler = ConversationHandler(
        entry_points=[CommandHandler("generate", beginGenerate)],
        states={
            SET_SYSTEM_PROMPT : [
                MessageHandler(filters.TEXT & ~filters.COMMAND, setSystemPrompt),
                CommandHandler("skip", skip_systemPrompt)
            ],
            SET_PROMPT : [MessageHandler(filters.TEXT & ~filters.COMMAND, setPrompt)]
        },
        fallbacks=[CommandHandler("cancel", cancel)]
    )

    # Knowledge command chain
    knowledgeHandler = ConversationHandler(
        entry_points=[CommandHandler("knowledge", knowledgeManger)],
        states={
            HANDLE_KNOWLEDGE_TYPE : [
                CallbackQueryHandler(setKnowledgeType, pattern="^(private|public)$")
            ],
            HANDLE_KNOWLEDGE_TEXT : [
                MessageHandler(filters.TEXT & ~filters.COMMAND, setKnowledgeText)
            ],
            HANDLE_KNOWLEDGE_SOURCE : [
                MessageHandler(filters.TEXT & ~filters.COMMAND, setKnowledgeSource),
                CommandHandler("skip", skip_knowledgeSource)
            ],
            HANDLE_KNOWLEDGE_CATEGORY : [
                MessageHandler(filters.TEXT & ~filters.COMMAND, setKnowledgeCategories),
                CommandHandler("skip", skip_knowledgeCategories)
            ],
            STORE_KNOWLEDGE : [
                CallbackQueryHandler(finalizeKnowledge, pattern="^(yes|no)$")
            ]
        },
        fallbacks=[CommandHandler("cancel", cancel)]
    )

    # Promote account command chain
    promoteHandler = ConversationHandler(
        entry_points=[
            CommandHandler("promote", promoteAccount)
        ],
        states={
            VERIFY_PROMOTE : [
                CallbackQueryHandler(setNewRole, pattern="^(admin|marketing|tester)$")
            ]
        },
        fallbacks=[CommandHandler("cancel", cancel)]
    )

    # Tweet command chain
    tweetHandler = ConversationHandler(
        entry_points=[
            CommandHandler("tweet", tweetStart)
        ],
        states={
            CONFIRM_TWEET : [
                CallbackQueryHandler(confirmTweet, pattern="^(confirm|modify|reject)$")
            ],
            MODIFY_TWEET : [
                MessageHandler(filters.TEXT & ~filters.COMMAND, modifyTweet)
            ]
        },
        fallbacks=[CommandHandler("cancel", cancel)]
    )

    # Newsletter command chain
    # Get list of account types from account manager
    rolesList = members.rolesList
    newsletterHandler = ConversationHandler(
        entry_points=[
            CommandHandler("newsletter", newsletterStart)
        ],
        states={
            ROLE_SELECTION : [
                CallbackQueryHandler(selectRole, pattern=f"^({'|'.join(rolesList)})$"),
                CallbackQueryHandler(roleSelectionDone, pattern=f"^done$")
            ],
            PHOTO_OPTION : [
                CallbackQueryHandler(photoOption, pattern=f"^(yes|no)$")
            ],
            ADD_PHOTO : [
                MessageHandler(filters.PHOTO, addNewsletterPhoto)
            ],
            COMPOSE_NEWLETTER : [
                MessageHandler(filters.TEXT & ~filters.COMMAND, addNewsletterText)
            ],
            CONFIRM_NEWSLETTER : [
                CallbackQueryHandler(confirmNewsletter, pattern=f"^(yes|no)$")
            ]
        },
        fallbacks=[CommandHandler("cancel", cancel)]
    )

    # Proposals command chain
    proposalsHandler = ConversationHandler(
        entry_points=[
            CommandHandler("proposals", proposalsManager, filters=filters.ChatType.PRIVATE)
        ],
        states={
            SELECT_PROPOSAL : [
                CallbackQueryHandler(agreeNDA)
            ],
            PROPOSAL_NDA : [
                CallbackQueryHandler(openProposal, pattern="^(confirm|reject)$")
            ],
            SHOW_PROPOSAL : [
                CallbackQueryHandler(openProposal, pattern="^(done)$")
            ]
        },
        fallbacks=[CommandHandler("cancel", cancel)]
    )

    # Add command handlers
    application.add_handler(CommandHandler("start", startBot, filters=filters.ChatType.PRIVATE))
    application.add_handler(CommandHandler("dashboard", dashboard, filters=filters.ChatType.PRIVATE))
    application.add_handler(CommandHandler("help", help))
    application.add_handler(CommandHandler("info", info))
    application.add_handler(CommandHandler("statistics", statisticsManager))
    application.add_handler(CommandHandler("password", setPassword, filters=filters.ChatType.PRIVATE))

    application.add_handler(MessageReactionHandler(reactionsHandler))
    application.add_handler(MessageHandler(filters.TEXT & filters.ChatType.GROUPS & filters.FORWARDED, handleForwardedMessage))
    
    # Add conversational chains
    application.add_handlers([generateHandler, knowledgeHandler, newsletterHandler, promoteHandler, proposalsHandler, tweetHandler])

    # Add message handlers
    application.add_handler(MessageHandler(filters.Mention(config.botName) & filters.ChatType.GROUPS & ~filters.COMMAND, directChatGroup))
    application.add_handler(MessageHandler(filters.REPLY & filters.ChatType.GROUPS & ~filters.COMMAND, replyToBot))
    application.add_handler(MessageHandler(filters.TEXT & filters.ChatType.PRIVATE & ~filters.COMMAND, directChatPrivate))
    application.add_handler(MessageHandler(filters.TEXT & filters.ChatType.GROUPS & ~filters.COMMAND, otherGroupChat))
    application.add_handler(MessageHandler(filters.PHOTO, handleImage))
    application.add_handler(MessageHandler(filters.ALL, catchAllMessages))

    # Other update type handlers
    application.add_handler(ChatMemberHandler(botStatusChanged, chat_member_types=ChatMemberHandler.MY_CHAT_MEMBER))

    application.add_error_handler(errorHandler)

    # Run the bot until the user presses Ctrl-C
    application.run_polling(allowed_updates=Update.ALL_TYPES, poll_interval=5, bootstrap_retries=3, timeout=50)


if __name__ == "__main__":
    main()
    