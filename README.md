# TwitchClipper

TwitchClipper is an automated tool that records Twitch streams, analyzes chat activity to identify engaging moments, creates short highlight clips, and uploads them to YouTube as Shorts. It leverages natural language processing, sentiment analysis, and AI-powered content generation to create compelling short-form content with minimal human intervention.

## Features

- **Automated Stream Recording**: Records live Twitch streams using Streamlink
- **Real-time Chat Analysis**: Captures and logs chat messages during streams
- **Highlight Detection**: Identifies peak moments based on chat frequency and sentiment analysis
- **AI-Powered Content Generation**: Uses DeepSeek AI and OpenAI's Whisper for title and description generation
- **Professional Video Production**: Creates vertical format videos suitable for YouTube Shorts
- **Automated YouTube Upload**: Uploads clips directly to YouTube using the YouTube Data API v3

## Prerequisites

- Python 3.8+ installed on your system
- ffmpeg installed (for video processing)
- Streamlink installed (for Twitch stream recording)
- A Twitch account with required OAuth token
- A DeepSeek API key
- YouTube API credentials (client_secrets.json file)

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/TwitchClipper.git
   cd TwitchClipper
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   
   # On Windows
   venv\Scripts\activate
   
   # On macOS/Linux
   source venv/bin/activate
   ```

3. Install required Python packages:
   ```bash
   pip install -r requirements.txt
   ```

4. Install NLTK data:
   ```bash
   python -c "import nltk; nltk.download('vader_lexicon')"
   ```

5. Install Streamlink globally (if not already installed):
   ```bash
   pip install streamlink
   ```

## Configuration

1. Copy the environment template file:
   ```bash
   cp .env_template.txt .env
   ```

2. Edit the `.env` file with your credentials:
   ```
   TWITCH_NICK=your_twitch_username
   TWITCH_TOKEN=oauth:your_twitch_token
   TWITCH_CHANNEL=target_twitch_channel
   DEEPSEEK_API_KEY=your_deepseek_api_key
   ```

3. Obtain a YouTube OAuth 2.0 client ID and place the `client_secrets.json` file in the project root directory.

## File Descriptions

- **stream_chat_recorder.py**: Main script that records Twitch streams and logs chat messages in real-time. Creates a video file and a CSV of chat messages.
  
- **chat_log_test.py**: Testing script for connecting to Twitch chat without recording the stream. Useful for debugging chat connection issues.
  
- **clip_maker.py**: Analyzes the recorded chat log to find interesting moments, creates highlight clips, adds titles and descriptions using AI, and prepares videos for YouTube.
  
- **upload_video.py**: Handles authentication and uploading of videos to YouTube using the YouTube Data API v3.

- **.env_template.txt**: Template for the environment variables required by the application.

## Usage

1. **Record a stream and capture chat**:
   ```bash
   python stream_chat_recorder.py
   ```
   This will start recording the Twitch channel specified in your `.env` file and log all chat messages. The recording continues until the stream ends or you manually stop it (Ctrl+C).

2. **Process highlights and generate clips** (automatically runs after recording):
   ```bash
   python clip_maker.py
   ```
   This analyzes the chat log, identifies peak moments, and creates short vertical format videos suitable for YouTube Shorts. It saves these clips in the `highlights/{channel_name}/freq` and `highlights/{channel_name}/sent` directories.

3. **Upload a clip to YouTube**:
   ```bash
   python upload_video.py --file="highlight_1_vertical.mp4" --title="Exciting Moment #shorts" --description="Check out this amazing moment!" --keywords="twitch,shorts,gaming" --category="23" --privacyStatus="public"
   ```
   This can be done manually, or the clip_maker.py will attempt to do this automatically for each clip it creates.

## How It Works

1. **Stream Recording**: The system connects to the specified Twitch channel using Streamlink and begins recording the stream to a local MP4 file.

2. **Chat Logging**: Simultaneously, it connects to the Twitch IRC server to capture chat messages, which are timestamped and saved to a CSV file.

3. **Chat Analysis**: After recording, the system analyzes the chat log to identify:
   - **Frequency Peaks**: Moments when chat activity suddenly increases
   - **Sentiment Peaks**: Moments when chat sentiment shows strong emotional reactions

4. **Clip Generation**: For each identified peak, the system:
   - Extracts a ~50-second clip centered on the peak moment
   - Transcribes the audio using OpenAI's Whisper
   - Generates a title and description using DeepSeek AI based on the chat messages and transcription
   - Creates a vertical format video suitable for YouTube Shorts with the stream footage overlaid on a blurred background
   - Adds the AI-generated title to the video

5. **YouTube Upload**: Finally, it uploads each clip to YouTube with appropriate metadata.

## Technologies Used

- **Python**: Core programming language
- **Streamlink**: For capturing Twitch streams
- **Socket/IRC**: For connecting to Twitch chat
- **Pandas/NumPy/SciPy**: For data analysis and peak detection
- **NLTK**: For sentiment analysis of chat messages
- **OpenAI Whisper**: For speech-to-text transcription
- **DeepSeek AI**: For generating titles and descriptions
- **ffmpeg**: For video processing and formatting
- **YouTube Data API v3**: For uploading videos to YouTube
- **OAuth2**: For YouTube authentication

## Troubleshooting

- **Chat Connection Issues**: If you're having trouble connecting to Twitch chat, verify your OAuth token is valid and has the proper scopes.

- **Missing Highlights**: If no highlights are being detected, try adjusting the `FREQUENCY_STD_COEF` and `SENTIMENT_STD_COEF` values in `clip_maker.py` to make the peak detection more or less sensitive.

- **YouTube Upload Errors**: Make sure your `client_secrets.json` file is correctly configured and that you have the YouTube Data API enabled in your Google Cloud Console.

- **Video Processing Errors**: Ensure ffmpeg is properly installed and available in your system PATH.

- **API Rate Limits**: If you're processing many clips, be aware of rate limits for the DeepSeek API and YouTube API.
