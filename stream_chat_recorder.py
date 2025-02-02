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
EASTERN_TIMEZONE = pytz.timezone('US/Eastern')
VIDEO_START_TIME = None
VIDEO_LEN = None

SERVER = "irc.chat.twitch.tv"
PORT = 6667

CHAT_QUEUE = Queue()
stop_event = threading.Event()  # Shared stop event for thread coordination


def record_stream():
    """Records the Twitch stream using streamlink."""
    global VIDEO_START_TIME
    VIDEO_START_TIME = datetime.now(EASTERN_TIMEZONE)

    print(f"Recording stream from https://www.twitch.tv/{TWITCH_CHANNEL}...")
    command = f"streamlink https://www.twitch.tv/{TWITCH_CHANNEL} 720p60 -o {TWITCH_CHANNEL}_{VIDEO_OUTPUT_FILE} -f"
    subprocess.run(command, shell=True)  # Block until streamlink exits

    print("Stream recording stopped.")
    global VIDEO_LEN
    VIDEO_LEN = (datetime.now(EASTERN_TIMEZONE) - VIDEO_START_TIME).total_seconds()
    stop_event.set()  # Signal all threads to stop


def connect_to_twitch():
    """Connects to the Twitch IRC server and joins a channel."""
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
    """Reads messages from Twitch chat and adds them to the queue."""
    print("Starting chat logging...")
    try:
        while not stop_event.is_set():
            response = sock.recv(2048).decode("utf-8", 'ignore')
            if response.startswith("PING"):
                sock.send("PONG :tmi.twitch.tv\r\n".encode("utf-8"))
            elif response.strip():
                timestamp = datetime.now(EASTERN_TIMEZONE)
                CHAT_QUEUE.put((timestamp, response.strip()))
            else:
                sock.send(":tmi.twitch.tv RECONNECT\r\n".encode("utf-8"))
    except Exception as e:
        print(f"Error reading chat messages: {e}")
    finally:
        print("Chat logging stopped.")


def chat_logging():
    """Logs chat messages to a CSV file."""
    with open(CHAT_LOG_FILE, "w") as f:
        f.write("timestamp,time_in_vid,message\n")

    while not stop_event.is_set():
        if not CHAT_QUEUE.empty():
            timestamp, response = CHAT_QUEUE.get()
            time_in_video = (timestamp - VIDEO_START_TIME).total_seconds()
            try:
                sanitized_line = response.replace(',', ' ').replace('  ', ' ')
                formatted_line = f"{timestamp},{time_in_video},{sanitized_line}\n"
                with open(CHAT_LOG_FILE, "a") as f:
                    f.write(formatted_line)
            except Exception as e:
                print(f"Error logging chat message: {e}")


def main():
    """Main function to start and manage threads."""
    record_thread = threading.Thread(target=record_stream)
    record_thread.start()

    sock = connect_to_twitch()
    if sock:
        chat_thread = threading.Thread(target=save_chat, args=(sock,))
        chat_thread.start()

        chat_logging_thread = threading.Thread(target=chat_logging)
        chat_logging_thread.start()

        # Wait for the recording thread to finish
        record_thread.join()
        stop_event.set()  # Signal all threads to stop
        chat_thread.join()
        chat_logging_thread.join()

        sock.close()
        print("Stream and chat logging finished.")

        # Run clip_maker.py if VIDEO_LEN is not None and VIDEO_LEN > 60
        if VIDEO_LEN and VIDEO_LEN > 60:
            
            print("Running clip_maker.py...")
            subprocess.run(["python", "clip_maker.py"])
        else:
            print("Stream duration is less than 60 seconds. Skipping clip_maker.py.")

        print("Exiting...")
    else:
        print("Failed to connect to Twitch chat. Exiting...")


if __name__ == "__main__":
    main()
