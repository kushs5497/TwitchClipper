# TwitchClipper

Code to Automatically Clip Twitch Streams and Upload them to Youtube.

How does it work?

1. Using Streamlink Command line arguments, a connection to a Twitch stream is established and the twitch stream is recorded.
2. Alongside the recording in a second thread, and IRC connection is used to gather chat activity and log it in a csv file.
3. Once the stream ends, or recording is interupted, the chat activity is analyzed by frequency and sentiment, and peaks of high chat activity or high sentiment are found.
4. Using these peaks, the entire recording is cut and using DeepSeek ai and OpenAI's Whisper, a title and descrition is added and a blurred Minecraft Gameplay is added to the background.
5. Finally, the final short reel is uploaded to Youtube using Youtube Data API v3
