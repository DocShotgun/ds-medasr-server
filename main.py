"""
OpenAI-compatible MedASR Transcription Server

This server provides an OpenAI API compatible endpoint for speech-to-text
transcription using the google/medasr model from Hugging Face.

Usage:
    python main.py

The server will start at http://localhost:8000 by default.

Note: The model is loaded once at startup and cached in memory for all
subsequent requests. Temporary audio files are cleaned up after each request.
"""

import os
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path

import librosa
import torch
import yaml
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from transformers import AutoModelForCTC, AutoProcessor

# Load configuration
CONFIG_PATH = Path(__file__).parent / "config.yaml"


def load_config(config_path: Path = CONFIG_PATH) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


config = load_config()

# Global model and processor instances (loaded lazily)
_model = None
_processor = None


def get_device() -> str:
    """Determine the device to use for inference."""
    device_setting = config.get("model", {}).get("device", "auto")
    if device_setting == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device_setting


def load_model():
    """Load the MedASR model and processor."""
    global _model, _processor

    if _model is not None and _processor is not None:
        return _model, _processor

    model_name = config.get("model", {}).get("name", "google/medasr")
    device = get_device()

    print(f"Loading model {model_name} on device: {device}")

    # Set HuggingFace token if provided
    hf_token = config.get("huggingface", {}).get("token")
    if hf_token:
        os.environ["HF_TOKEN"] = hf_token

    _processor = AutoProcessor.from_pretrained(model_name)
    _model = AutoModelForCTC.from_pretrained(model_name).to(device)

    print("Model loaded successfully!")
    return _model, _processor


def transcribe_audio(audio_path: str) -> str:
    """
    Transcribe an audio file using the MedASR model.

    Args:
        audio_path: Path to the audio file

    Returns:
        Transcribed text
    """
    global _model, _processor

    if _model is None or _processor is None:
        load_model()

    device = get_device()

    # Load and resample audio to 16kHz
    speech, sample_rate = librosa.load(audio_path, sr=16000)

    # Process audio with processor
    inputs = _processor(
        speech,
        sampling_rate=sample_rate,
        return_tensors="pt",
        padding=True
    )

    # Move inputs to the same device as the model
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Generate predictions
    with torch.no_grad():
        outputs = _model.generate(**inputs)

    # Decode the predictions
    transcription = _processor.batch_decode(outputs, skip_special_tokens=True)[0]

    return transcription


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown events."""
    # Startup: Load model
    print("Starting MedASR server...")
    try:
        load_model()
        print("Server ready!")
    except Exception as e:
        print(f"Warning: Could not load model at startup: {e}")
        print("Model will be loaded on first request.")
    yield
    # Shutdown
    print("Shutting down MedASR server...")


app = FastAPI(
    title="MedASR Transcription API",
    description="OpenAI-compatible speech-to-text transcription using google/medasr",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "model": config.get("model", {}).get("name", "google/medasr")}


@app.post("/v1/audio/transcriptions")
async def create_transcription(
    file: UploadFile = File(...),
    model: str = Form(default="google/medasr"),
):
    """
    OpenAI-compatible audio transcription endpoint.

    Request body (multipart/form-data):
        - file: Audio file (mp3, wav, flac, m4a, ogg, etc.)
        - model: Model name (optional, ignored - always uses google/medasr)

    Returns:
        JSON with 'text' field containing the transcription.
    """
    # Validate file type
    allowed_extensions = {".mp3", ".wav", ".flac", ".m4a", ".ogg", ".webm", ".mp4"}
    file_ext = Path(file.filename).suffix.lower()

    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported audio format: {file_ext}. Allowed: {', '.join(allowed_extensions)}",
        )

    # Save uploaded file to temporary location
    with tempfile.NamedTemporaryFile(suffix=file_ext, delete=False) as temp_file:
        try:
            # Read file content in chunks to avoid memory issues with large files
            while True:
                chunk = await file.read(1024 * 1024)  # 1MB chunks
                if not chunk:
                    break
                temp_file.write(chunk)
            temp_file_path = temp_file.name

            # Transcribe
            try:
                transcription = transcribe_audio(temp_file_path)
            except Exception as e:
                raise HTTPException(
                    status_code=500,
                    detail=f"Transcription failed: {str(e)}",
                )

            # Return OpenAI-compatible response
            return JSONResponse(content={"text": transcription})

        finally:
            # Clean up temporary file
            try:
                os.unlink(temp_file_path)
            except OSError:
                pass


def main():
    """Run the server using uvicorn."""
    import uvicorn

    host = config.get("server", {}).get("host", "0.0.0.0")
    port = config.get("server", {}).get("port", 8000)

    print(f"Starting MedASR server on {host}:{port}")
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
