# MedASR Transcription Server

OpenAI-compatible speech-to-text transcription server using [google/medasr](https://huggingface.co/google/medasr) model.

## Features

- OpenAI API compatible endpoint (`POST /v1/audio/transcriptions`)
- Supports common audio formats: mp3, wav, flac, m4a, ogg, webm, mp4
- Model loaded once at startup and cached in memory (no reloading per request)
- GPU acceleration support (auto-detected)
- YAML configuration
- Temporary audio files cleaned up after each request

## Installation

```bash
# Create virtual environment (recommended)
python -m venv venv
./venv/scripts/activate

# Install dependencies
pip install -r requirements.txt
```

## Configuration

Edit `config.yaml` to customize:

```yaml
server:
  host: "0.0.0.0"
  port: 8000

model:
  name: "google/medasr"
  chunk_length_s: 20
  stride_length_s: 2
  device: "auto"  # "auto", "cuda", or "cpu"

huggingface:
  token: null  # Optional: HF token for private models or rate limit increase
```

## Usage

```bash
# Start the server
python main.py
```

The server will start at `http://localhost:8000`.

## API Usage

### Transcribe Audio

```bash
curl -X POST http://localhost:8000/v1/audio/transcriptions \
  -H "Content-Type: multipart/form-data" \
  -F "file=@audio.wav"
```

Response:

```json
{
  "text": "transcribed text here"
}
```

### Health Check

```bash
curl http://localhost:8000/health
```

Response:

```json
{
  "status": "healthy",
  "model": "google/medasr"
}
```

## OpenAI Client Compatible Usage

You can use this server with OpenAI client libraries by setting the base URL:

```python
from openai import OpenAI

client = OpenAI(
    api_key="dummy",  # No actual API key needed
    base_url="http://localhost:8000/v1"
)

with open("audio.wav", "rb") as audio_file:
    transcript = client.audio.transcriptions.create(
        file=audio_file,
        model="google/medasr"
    )

print(transcript.text)
```

## Requirements

- Python 3.10+
- transformers >= 5.0.0
- torch >= 2.0.0
- CUDA-compatible GPU (optional, for faster inference)
