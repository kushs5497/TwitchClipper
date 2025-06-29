# twitch_oauth_login.py

import os
import webbrowser
import requests
from flask import Flask, request
from dotenv import set_key, load_dotenv

load_dotenv()
CLIENT_ID = os.getenv("TWITCH_CLIENT_ID")
CLIENT_SECRET = os.getenv("TWITCH_CLIENT_SECRET")
REDIRECT_URI = "http://localhost:5000/callback"
SCOPES = "chat:read chat:edit user:read:email"

app = Flask(__name__)
access_token = None

@app.route("/callback")
def callback():
    global access_token
    code = request.args.get("code")
    if not code:
        return "No code provided", 400

    # Exchange auth code for access token
    token_url = "https://id.twitch.tv/oauth2/token"
    data = {
        "client_id": CLIENT_ID,
        "client_secret": CLIENT_SECRET,
        "code": code,
        "grant_type": "authorization_code",
        "redirect_uri": REDIRECT_URI,
    }
    resp = requests.post(token_url, data=data).json()
    access_token = resp.get("access_token")

    # Get user info
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Client-Id": CLIENT_ID,
    }
    user_data = requests.get("https://api.twitch.tv/helix/users", headers=headers).json()
    user_login = user_data["data"][0]["login"]

    # Save to .env
    set_key(".env", "USER_TWITCH_NICK", user_login)
    set_key(".env", "USER_TWITCH_TOKEN", f"oauth:{access_token}")
    set_key(".env", "USER_TWITCH_CLIENT_ID", CLIENT_ID)

    return f"✅ Success! You are logged in as: {user_login}. You can close this tab."

def login():
    url = (
        f"https://id.twitch.tv/oauth2/authorize"
        f"?client_id={CLIENT_ID}"
        f"&redirect_uri={REDIRECT_URI}"
        f"&response_type=code"
        f"&scope={SCOPES}"
    )
    print("Opening browser for Twitch login...")
    webbrowser.open(url)
    app.run(port=5000)

if __name__ == "__main__":
    login()
