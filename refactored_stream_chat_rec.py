import os
import asyncio
import pytz
from datetime import datetime
from twitchio.ext import commands
from dotenv import load_dotenv
import streamlink

# Load environment variables
load_dotenv()

TWITCH_NICK = os.getenv("TWITCH_NICK")
TWITCH_TOKEN = os.getenv("TWITCH_TOKEN")
TWITCH_CHANNEL = os.getenv("TWITCH_CHANNEL")
CLIENT_ID = os.getenv("TWITCH_CLIENT_ID")  # Needed if using Twitch API features

EASTERN_TIMEZONE = pytz.timezone('US/Eastern')
CHAT_LOG_FILE = f"{TWITCH_CHANNEL}_chat_log.csv"
VIDEO_OUTPUT_FILE = f"{TWITCH_CHANNEL}_recorded_stream.mp4"
VIDEO_START_TIME = datetime.now(EASTERN_TIMEZONE)


# -----------------------
# Chat Bot using TwitchIO
# -----------------------
class ChatLoggerBot(commands.Bot):
    def __init__(self):
        super().__init__(
            token=TWITCH_TOKEN,
            prefix='!',
            initial_channels=[TWITCH_CHANNEL]
        )
        self.chat_log_file = open(CHAT_LOG_FILE, 'w')
        self.chat_log_file.write("timestamp,time_in_vid,message\n")

    async def event_message(self, message):
        if message.echo:
            return

        timestamp = datetime.now(EASTERN_TIMEZONE)
        time_in_vid = (timestamp - VIDEO_START_TIME).total_seconds()
        clean_content = message.content.replace(',', ' ').replace('  ', ' ')
        log_line = f"{timestamp},{time_in_vid},{clean_content}\n"
        self.chat_log_file.write(log_line)
        self.chat_log_file.flush()

    async def shutdown(self):
        self.chat_log_file.close()


# -----------------------
# Stream Recorder
# -----------------------
async def record_stream(channel_url):
    streams = streamlink.streams(channel_url)
    stream = streams.get('720p60') or streams.get('best')

    if not stream:
        print("No valid stream found.")
        return

    print(f"Recording stream from {channel_url}...")
    with stream.open() as fd, open(VIDEO_OUTPUT_FILE, "wb") as out_file:
        try:
            while True:
                data = fd.read(1024)
                if not data:
                    break
                out_file.write(data)
        except Exception as e:
            print(f"Stream recording error: {e}")

    print("Stream recording complete.")


# -----------------------
# Main Execution
# -----------------------
async def main():
    channel_url = f"https://www.twitch.tv/{TWITCH_CHANNEL}"
    bot = ChatLoggerBot()

    # Run chat logger and stream recorder concurrently
    await asyncio.gather(
        bot.start(),
        record_stream(channel_url)
    )

    await bot.shutdown()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Interrupted. Shutting down...")
