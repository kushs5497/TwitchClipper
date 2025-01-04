import os
import pytz
import socket
import subprocess
import threading
from queue import Queue
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()
TWITCH_NICK = os.getenv("TWITCH_NICK")  # Twitch username
TWITCH_TOKEN = os.getenv("TWITCH_TOKEN")  # OAuth token from Twitch
TWITCH_CHANNEL = os.getenv("TWITCH_CHANNEL")  # Channel to join (e.g., "marvelrivals")

VIDEO_OUTPUT_FILE = "recorded_stream.mp4"
CHAT_LOG_FILE = f"{TWITCH_CHANNEL}_chat_log.csv"
HIGHLIGHTS_DIR = "highlights"
VIDEO_START_TIME = None  # Will be set dynamically
EASTERN_TIMEZONE = pytz.timezone('US/Eastern')
CHAT_THREAD_STARTED = False
chat_thread = None

SERVER = "irc.chat.twitch.tv"
PORT = 6667

CHAT_QUEUE = Queue()

def record_stream():
    global VIDEO_START_TIME
    VIDEO_START_TIME = datetime.now(EASTERN_TIMEZONE)

    print(f"Recording stream from https://www.twitch.tv/{TWITCH_CHANNEL}...")
    command = f"streamlink https://www.twitch.tv/{TWITCH_CHANNEL} 720p60 -o {TWITCH_CHANNEL}_{VIDEO_OUTPUT_FILE} -f"
    subprocess.run(command, shell=True)

    print("Stream recording stopped.")

    if CHAT_THREAD_STARTED:
        chat_thread.join()

def connect_to_twitch():
    # Connects to the Twitch IRC server and joins a channel
    try:
        sock = socket.socket()
        sock.connect((SERVER, PORT))
        sock.send(f"PASS {TWITCH_TOKEN}\r\n".encode("utf-8"))
        sock.send(f"NICK {TWITCH_NICK}\r\n".encode("utf-8"))
        sock.send(f"JOIN #{TWITCH_CHANNEL}\r\n".encode("utf-8"))
        print(f"Connected to {TWITCH_CHANNEL} Twitch chat.")
        return sock
    except Exception as e:
        print(f"Error connecting to Twitch: {e}")
        return None

def save_chat(sock):
    # Reads messages from the Twitch chat and saves them to a CSV file
    print("Starting chat logging...")
    while True:
        try:
            response = sock.recv(2048).decode("utf-8")
            print(response)
            if response.startswith("PING"):
                sock.send("PONG :tmi.twitch.tv\r\n".encode("utf-8"))
            elif len(response) > 0:
                timestamp = datetime.now(EASTERN_TIMEZONE)
                CHAT_QUEUE.put((timestamp, response))
        except Exception as e:
            print(f"\n\n\n\nError reading chat messages: {e}\n\n\n\n")
            sock = connect_to_twitch()
    
    print(f"\n\n\n\nChat saving stopped.\n\n\n\n")

def chat_logging():
    while True:
        if not CHAT_QUEUE.empty():
            timestamp, response = CHAT_QUEUE.get()
            time_in_video = (timestamp - VIDEO_START_TIME).total_seconds()
            lines = response.strip().split("\n")  # Split by newline and clean up
            
            for line in response.strip().split("\n"):
                if line.strip():  # Skip empty lines
                    try:
                        sanitized_line = line.replace(',', ' ').replace('  ', ' ')  # Clean up the message
                        formatted_line = f"{timestamp},{time_in_video},{sanitized_line}\n"
                        
                        # Write to file with thread-safe lock
                        with open(CHAT_LOG_FILE, "a") as f:
                            f.write(formatted_line)
                    except Exception as e:
                        print(f"Error logging chat message: {e}")
    print(f"\n\n\n\nChat logging stopped.\n\n\n\n")

def main():
    record_thread = threading.Thread(target=record_stream)
    record_thread.start()

    sock = connect_to_twitch()
    if sock:
        with open(CHAT_LOG_FILE, "w") as f:
            f.write("timestamp,time_in_vid,message\n")
        
        global chat_thread
        chat_thread = threading.Thread(target=save_chat, args=(sock,))
        chat_thread.start()
        global CHAT_THREAD_STARTED
        CHAT_THREAD_STARTED = True

        chat_logging_thread = threading.Thread(target=chat_logging)
        chat_logging_thread.start()

        record_thread.join()
        chat_thread.join()
        sock.close()
        print("Stream Finished")
    else:
        print("Failed to connect to Twitch chat. Exiting...")

if __name__ == "__main__":
    main()
