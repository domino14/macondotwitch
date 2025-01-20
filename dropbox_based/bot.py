import requests
import schedule
import time
import threading
import subprocess
from twitchio.ext import commands

# Global variables
analyzing = False
latest_file_content = None
analysis_result = None
analysis_details = None

# List of authorized users
AUTHORIZED_USERS = ["14domino", "cannnnik"]


# Function to check for updates and analyze the file
def check_for_updates(dropbox_link):
    global analyzing, latest_file_content

    response = requests.get(dropbox_link, allow_redirects=True)
    if response.status_code == 200:
        new_content = response.content
        if new_content != latest_file_content:
            print("File changed, analyzing...", flush=True)
            latest_file_content = new_content
            analyzing = True
            run_analysis(new_content)


def run_analysis(file_content):
    global analyzing, analysis_result, analysis_details

    # Save the downloaded file to /tmp/livegame.gcg
    with open("/tmp/livegame.gcg", "wb") as file:
        file.write(file_content)

    try:
        # Call the command line script
        process = subprocess.Popen(
            [
                "/home/cesar/code/macondo/bin/shell",
                "script /home/cesar/code/macondo/scripts/elite_cgp.lua",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,  # Ensure the output is text, not bytes
            cwd="/home/cesar/code/macondo",
        )
        stdout, stderr = process.communicate()
        print("stderr")
        print(stderr)
        # Check for errors
        if process.returncode != 0:
            print(f"Error during analysis: {stderr}", flush=True)
            analysis_result = "Error during analysis."
        else:
            # Parse the output for the best move
            lines = stdout.split("\n")
            print("lines", lines)
            analysis_result = "No best move found."
            analysis_details = "No analysis details."
            for line in lines:
                if line.startswith("BEST:"):
                    analysis_result = line[len("BEST:") :]
                if line.startswith("DETAILS:"):
                    analysis_details = line[len("DETAILS:") :]

            print("analysis: ", analysis_result, analysis_details, flush=True)

    except Exception as e:
        print(f"Exception during analysis: {e}", flush=True)
        analysis_result = "Exception during analysis."
        analysis_details = "Exception during analysis."

    print("Analysis complete.", flush=True)
    analyzing = False


# Schedule the update check every 5 seconds
with open("link.txt") as f:
    dropbox_link = f.read().strip()

with open("token.txt") as f:
    token = f.read().strip()

schedule.every(5).seconds.do(check_for_updates, dropbox_link)


# Twitch bot setup
class Bot(commands.Bot):

    def __init__(self):
        super().__init__(token=token, prefix="!", initial_channels=["#austinho9"])

    async def event_ready(self):
        print(f"Logged in as {self.nick}", flush=True)

    async def event_message(self, message):
        if message.echo:
            return

        await self.handle_commands(message)

    @commands.command(name="analyze")
    async def analyze_command(self, ctx):
        global analyzing, analysis_result

        # Check if the user is authorized
        if ctx.author.name not in AUTHORIZED_USERS:
            # await ctx.send(f"Sorry, {ctx.author.name}, you are not authorized to use this command.")
            return

        if analyzing:
            await ctx.send("The latest move is still being analyzed. Please wait.")
        else:
            await ctx.send(
                f"The analysis is complete. The best move is: {analysis_result}"
            )

    @commands.command(name="details")
    async def details_command(self, ctx):
        global analyzing, analysis_details

        # Check if the user is authorized
        if ctx.author.name not in AUTHORIZED_USERS:
            # await ctx.send(f"Sorry, {ctx.author.name}, you are not authorized to use this command.")
            return

        if analyzing:
            await ctx.send("The latest move is still being analyzed. Please wait.")
        else:
            await ctx.send(f"{analysis_details}")


bot = Bot()


def run_bot():
    bot.run()


# Run the bot in a separate thread
bot_thread = threading.Thread(target=run_bot)
bot_thread.start()

# Run the scheduled tasks
while True:
    schedule.run_pending()
    time.sleep(1)
