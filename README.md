# How to Chat with your Youtube Video

## Prerequisites:
Install Python and pip.

## Step 1: Create a virtual environment and Install the required packages:

```bash
$ python3 -m venv .venv
$ source .venv/bin/activate
$ pip install -r requirements.txt
```

## Step 2: Set Up Environment Variables
  ### 2.1: Create OpenAI and Pinecone accounts and get the API Keys, Endpoints.

  1. **OpenAI:** https://platform.openai.com/api-keys
  2. **Pinecone:** https://app.pinecone.io/

  ### 2.2: Create a .env file in the root directory of your project and fill in the following details:
```plaintext
SECRET_KEY=Enter your Flask Application Secret Key here

EMBEDDINGS_MODEL=Enter your Embedding Model here
OPENAI_API_KEY=Enter your OpenAI API key here

ANTHROPIC_API_KEY=Enter your Anthropic API key here

PINECONE_INDEX_NAME=Enter your Pinecone Index Name here
PINECONE_API_ENV=Enter your Pinecone Environment endpoint here
PINECONE_API_KEY=Enter your Pinecone API key here
```

### 2.3: Speech-to-Text (STT) Fallback (Optional)

This application includes a feature to use OpenAI's Whisper API as a fallback if YouTube transcripts are unavailable (e.g., disabled by the uploader or not found automatically).
When enabled, if the system cannot retrieve a transcript directly from YouTube, it will attempt to:
1. Download the audio from the YouTube video using `yt-dlp`.
2. Transcribe the downloaded audio using OpenAI's Whisper API (`whisper-1` model).

This significantly enhances the chances of getting a transcript, especially for videos without pre-existing captions.

**Configuration:**

*   This feature relies on your `OPENAI_API_KEY` being correctly configured in the `.env` file.
*   **`ENABLE_STT_FALLBACK`**: This environment variable controls the STT fallback mechanism.
    *   Set to `false` to disable the STT fallback.
    *   If not set, it defaults to `true` (enabled).
    *   Example: `ENABLE_STT_FALLBACK=false`

**Dependencies:**
*   The STT fallback feature uses `yt-dlp` for downloading audio and the `openai` library for transcription. These should be included in your `requirements.txt` for production setups. The application attempts to handle these, but a robust setup should ensure they are listed as dependencies.

## Step 3: Run the Application
Execute the application.py file using Python. This will start the Flask application:
```bash
$ python application.py
```

## Step 4: Enjoy your experience with the application!
Load your video and ask your questions.

