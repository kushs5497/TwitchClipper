# TwitchClipper

TwitchClipper is a sophisticated automated content creation pipeline that leverages advanced AI models, data analysis techniques, and video processing algorithms to transform Twitch streams into engaging YouTube Shorts. This project demonstrates complex integration of multiple technologies to create an end-to-end solution for automated content repurposing—a growing need in the creator economy.

## Project Overview

This system implements a multi-stage pipeline architecture that handles:
- Real-time data acquisition from Twitch streams and IRC chat
- Advanced time-series analysis for moment detection
- Natural language processing for sentiment analysis
- AI-driven content generation using state-of-the-art language models
- Automated video editing and optimization for short-form content
- Programmatic publishing via the YouTube API

The result is a fully autonomous system that can identify engaging moments from hours of content, transform them into optimized short-form videos, and publish them to YouTube—all without human intervention.

## Architectural Decisions

### Multi-threaded Processing Architecture
The system employs a multi-threaded architecture to handle concurrent tasks:
- A dedicated thread for stream recording using Streamlink's CLI interface
- A separate thread for IRC chat connection and message processing
- Background threads for data logging and synchronization
- Main thread orchestration with proper event handling for graceful termination

This approach allows the system to simultaneously record high-quality video while capturing and processing chat data in real-time.

### Time-series Analysis for Content Identification
The moment detection algorithm implements:
- Gaussian smoothing of chat frequency and sentiment data
- Peak detection using signal processing techniques from SciPy
- Dynamic thresholding based on statistical properties of the chat distribution
- Multi-factor scoring combining both frequency and sentiment indicators

```python
# Example of the sophisticated peak detection algorithm
def compute_freq_peaks(chat_data):
    chat_data["count"] = 1
    chat_counts = chat_data.resample(FREQUENCY_RESAMPLE_RATE, on="timestamp").sum()["count"].fillna(0)
    smoothed_counts = gaussian_filter1d(chat_counts.values, sigma=FREQUENCY_GAUSSIAN_SIGMA)
    threshold_counts = chat_counts.mean() + FREQUENCY_STD_COEF * chat_counts.std()
    peaks_counts, _ = find_peaks(smoothed_counts, height=threshold_counts, distance=30)
    peak_times_freq = chat_counts.index[peaks_counts]
    return chat_counts, smoothed_counts, peaks_counts, peak_times_freq
```

### AI Integration Layer
The system seamlessly integrates multiple AI services:
- DeepSeek's language model for contextual content generation
- OpenAI's Whisper for accurate speech-to-text processing
- NLTK's sentiment analysis for emotional response detection
- Custom prompt engineering for genre-appropriate content creation

### Modular Video Processing Pipeline
The video processing pipeline is designed with high modularity:
- Separation of concerns between detection, extraction, and enhancement
- Programmatic ffmpeg command generation for complex video effects
- Parallelizable processing for multiple highlight clips
- Error handling and retry logic for resilient media processing

## Features

### Automated Stream Recording
- Leverages Streamlink for reliable, high-quality stream capture
- Handles network interruptions and reconnection scenarios
- Configurable quality settings to balance file size and visual fidelity
- Proper timestamp synchronization for accurate moment identification

### Real-time Chat Analysis
- Direct IRC connection to Twitch chat servers
- Message filtering and sanitization for analysis preparation
- Continuous logging with proper timestamp correlation to video content
- Low-latency processing for immediate data availability

### Advanced Highlight Detection
- Dual-metric analysis combining message frequency and sentiment scores
- Adaptive thresholding to accommodate different stream styles and audience sizes
- Tunable sensitivity parameters for different content types
- Smoothing algorithms to reduce false positives from momentary spikes

```python
# Example of the sentiment analysis implementation
def compute_sent_peaks(chat_data):
    chat_data["sentiment"] = chat_data["message"].apply(lambda x: sia.polarity_scores(x)["compound"]).abs()
    chat_sentiment = chat_data.resample(SENTIMENT_RESAMPLE_RATE, on="timestamp").mean(numeric_only=True)["sentiment"].fillna(0)
    smoothed_sentiment = gaussian_filter1d(chat_sentiment.values, sigma=SENTIMENT_GAUSSIAN_SIGMA)
    threshold_sent = chat_sentiment.mean() + SENTIMENT_STD_COEF * chat_sentiment.std()
    peaks_sent, _ = find_peaks(smoothed_sentiment, height=threshold_sent, distance=30)
    peak_times_sent = chat_sentiment.index[peaks_sent]
    return chat_sentiment, smoothed_sentiment, peaks_sent, peak_times_sent
```

### AI-Powered Content Generation
- Context-aware titling system that understands gaming terminology and stream culture
- Multi-modal input processing combining chat data and audio transcription
- Custom prompt engineering for genre-appropriate, platform-optimized content
- Automatic content sanitization to ensure platform compliance

### Professional Video Production
- Dynamic compositing with blurred background for visual enhancement
- Automatic aspect ratio conversion optimized for mobile viewing
- Custom typography with configurable styling and animation
- Intelligent clip extraction with proper lead-in and lead-out timing

### Automated YouTube Upload
- OAuth2 implementation for secure API access
- Metadata optimization for YouTube algorithm performance
- Proper categorization and tagging for discoverability
- Exponential backoff retry strategy for API resilience

## Technical Implementation

### System Architecture

The application is structured around three main components:

1. **Data Acquisition Layer**: 
   - `stream_chat_recorder.py` - Manages concurrent recording and chat capture
   - Socket-based IRC client implementation for real-time data collection

2. **Analysis Engine**:
   - Time-series processing for moment identification
   - NLP components for sentiment scoring and content understanding

3. **Production Pipeline**:
   - Video processing modules with ffmpeg integration
   - AI-driven content generation with DeepSeek and Whisper
   - YouTube API integration for publishing

### Advanced Techniques Utilized

- **Signal Processing**: Employs Gaussian filtering and peak detection algorithms from SciPy
- **Natural Language Processing**: Utilizes NLTK's VADER sentiment analysis model
- **Multi-threading**: Implements thread synchronization with event flags and queues
- **API Integration**: Incorporates multiple external services with proper authentication
- **Error Handling**: Robust exception management with graceful degradation
- **Video Processing**: Complex ffmpeg parameter generation for precise media manipulation

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

## Future Improvements

### Enhanced Moment Detection
- Implementation of multivariate analysis incorporating audio levels and visual scene detection
- Machine learning model training on historical chat data to improve peak relevance
- Streamer-specific calibration to account for different audience interaction patterns

### Content Optimization
- A/B testing framework for title and description generation
- Thumbnail generation using frame analysis and object detection
- Custom intro/outro generation based on channel branding

### Infrastructure Enhancements
- Containerization with Docker for consistent deployment
- Cloud-based processing for increased scalability
- Real-time analytics dashboard for system monitoring

### Additional Platform Support
- Extension to other streaming platforms (Facebook Gaming, YouTube Live)
- Support for additional short-form video platforms (TikTok, Instagram Reels)
- Integration with content management systems for creators

## Troubleshooting

- **Chat Connection Issues**: If you're having trouble connecting to Twitch chat, verify your OAuth token is valid and has the proper scopes.

- **Missing Highlights**: If no highlights are being detected, try adjusting the `FREQUENCY_STD_COEF` and `SENTIMENT_STD_COEF` values in `clip_maker.py` to make the peak detection more or less sensitive.

- **YouTube Upload Errors**: Make sure your `client_secrets.json` file is correctly configured and that you have the YouTube Data API enabled in your Google Cloud Console.

- **Video Processing Errors**: Ensure ffmpeg is properly installed and available in your system PATH.

- **API Rate Limits**: If you're processing many clips, be aware of rate limits for the DeepSeek API and YouTube API.

## Technical Challenges Overcome

The development of TwitchClipper required solving several complex technical challenges:

1. **Accurate Timestamp Synchronization**: Ensuring perfect alignment between video content and chat messages required custom offset calculation and continuous adjustment.

2. **Efficient Video Processing**: Processing large video files efficiently while minimizing memory usage was achieved through careful stream handling and chunk processing.

3. **Contextual Content Generation**: Creating relevant, engaging titles required sophisticated prompt engineering and context-aware language model utilization.

4. **API Reliability**: Implementing exponential backoff strategies and proper error handling for external API dependencies ensured system resilience.

5. **Performance Optimization**: Balancing computational requirements with processing speed through selective application of resource-intensive algorithms.
