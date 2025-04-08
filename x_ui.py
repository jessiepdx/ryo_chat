##########################################################################
#                                                                        #
#  This file (x_ui.py) contains the x (formally twitter) interface for   #
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

import datetime
import logging
import tweepy
import tweepy.asynchronous

from hypermindlabs.utils import CustomFormatter, ConfigManager
from hypermindlabs.agents import TweetAgent



###########
# LOGGING #
###########

# Clear any previous logging handlers
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

# Set the basic config to append logging data to a file
logPath = "logs/"
logFilename = "x_ui_log_" + datetime.now().strftime("%Y%m%d-%H%M%S") + ".txt"
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

config = ConfigManager()

# TODO Check if the twitter keys exist at all and throw critical if not

client = tweepy.asynchronous.AsyncClient(
    consumer_key=config.twitter_keys["consumer_key"],
    consumer_secret=config.twitter_keys["consumer_secret"],
    access_token=config.twitter_keys["access_token"],
    access_token_secret=config.twitter_keys["access_token_secret"]
)

if __name__ == "__main__":
    logger.info("RYO - begin x ui application.")