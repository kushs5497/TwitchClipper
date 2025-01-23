import os
import pytz
import socket
import threading
from queue import Queue
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()
TWITCH_NICK = os.getenv("TWITCH_NICK")  # Twitch username
TWITCH_TOKEN = os.getenv("TWITCH_TOKEN")  # OAuth token from Twitch
TWITCH_CHANNEL = os.getenv("TWITCH_CHANNEL")  # Channel to join (e.g., "marvelrivals")

SERVER = "irc.chat.twitch.tv"
PORT = 6667

CHAT_QUEUE = Queue()
stop_event = threading.Event()  # Shared stop event for thread coordination


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
                print("===============SENDING PONG===============")
                sock.send("PONG :tmi.twitch.tv\r\n".encode("utf-8"))
            elif response.strip():
                print(response)
            else:
                print("===============RECONNECTING===============")
                sock.send(":tmi.twitch.tv RECONNECT\r\n".encode("utf-8"))
            
    except Exception as e:
        print(f"Error reading chat messages: {e}")
    finally:
        print("Chat logging stopped.")


def main():

    sock = connect_to_twitch()
    if sock:
        chat_thread = threading.Thread(target=save_chat, args=(sock,))
        chat_thread.start()
    else:
        print("Failed to connect to Twitch chat. Exiting...")

if __name__ == "__main__":
    main()
