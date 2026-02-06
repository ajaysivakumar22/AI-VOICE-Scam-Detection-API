from fastapi import FastAPI, Depends, HTTPException, UploadFile, File, Form, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse
from fastapi.openapi.docs import get_swagger_ui_html, get_redoc_html
from pathlib import Path
from app.schemas import VoiceRequest, VoiceResponse
from app.auth import validate_api_key
from app.config import SUPPORTED_LANGUAGES, API_KEY
from app.audio.decoder import decode_base64_audio, decode_audio_bytes
from app.audio.features import extract_features
from app.ml.predictor import predict
from app.ml.explanation import generate_explanation
from app.ml.model import load_model_and_baselines
import librosa
import numpy as np

# Custom API description with markdown
API_DESCRIPTION = """
## 🎙️ AI Voice Detection API

Detect whether an audio sample contains **human voice** or **AI-generated speech** using advanced audio analysis.

### 🔑 Authentication
All API endpoints require an `x-api-key` header for authentication.

### 🌍 Supported Languages
- 🇬🇧 English
- 🇮🇳 Tamil  
- 🇮🇳 Hindi
- 🇮🇳 Malayalam
- 🇮🇳 Telugu

### 📊 Analysis Features
The API analyzes **18 audio features** including:
- **MFCCs** (Mel-Frequency Cepstral Coefficients) - Voice texture analysis
- **Pitch Variation** - Natural speech has more pitch variety
- **Energy Dynamics** - Human speech has organic breathing patterns
- **Spectral Centroid** - Frequency distribution analysis

### 🎯 Response
Returns classification (`HUMAN` or `AI_GENERATED`), confidence score (0-1), and detailed explanation.
"""

app = FastAPI(
    title="🎙️ AI Voice Detector",
    version="1.0.0",
    description=API_DESCRIPTION,
    docs_url=None,  # Disable default docs
    redoc_url=None,  # Disable default redoc
    openapi_tags=[
        {
            "name": "Voice Detection",
            "description": "Endpoints for analyzing audio and detecting AI-generated voices"
        },
        {
            "name": "Health",
            "description": "API health check endpoints"
        }
    ]
)

# Load model on startup to avoid delay on first request
@app.on_event("startup")
async def startup_event():
    try:
        load_model_and_baselines()
        print("Model and baselines loaded successfully on startup.")
    except Exception as e:
        print(f"Error loading model on startup: {e}")

# Redirect the root to the API documentation
@app.get("/", include_in_schema=False)
async def root_redirect():
    return RedirectResponse(url="/docs")


# FIX 4: Global Exception Handler for consistent error format
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Return all errors in standard format."""
    return JSONResponse(
        status_code=400,
        content={
            "status": "error",
            "message": str(exc)
        }
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions in standard format."""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "status": "error",
            "message": exc.detail
        }
    )


@app.get("/docs", include_in_schema=False)
async def custom_swagger_ui():
    """Custom Swagger UI with enhanced styling."""
    return get_swagger_ui_html(
        openapi_url="/openapi.json",
        title="🎙️ AI Voice Detector - API Documentation",
        swagger_css_url="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui.css",
        swagger_ui_parameters={
            "syntaxHighlight.theme": "monokai",
            "docExpansion": "list",
            "filter": True,
            "tryItOutEnabled": True,
            "persistAuthorization": True,
            "displayRequestDuration": True
        }
    )


@app.get("/redoc", include_in_schema=False)
async def custom_redoc():
    """Custom ReDoc documentation."""
    return get_redoc_html(
        openapi_url="/openapi.json",
        title="🎙️ AI Voice Detector - API Reference"
    )


@app.get("/health", tags=["Health"])
async def health_check():
    """Check if the API is running."""
    return {
        "status": "healthy",
        "version": "1.0.0",
        "service": "AI Voice Detection API"
    }


@app.post("/api/voice-detection", response_model=VoiceResponse, tags=["Voice Detection"])
def detect_voice(
    request: VoiceRequest,
    api_key: str = Depends(validate_api_key)
):
    """
    🎙️ Detect AI-Generated Voice
    
    Analyzes a Base64-encoded MP3 audio sample to determine whether the voice is 
    **AI-generated** or **Human**.
    
    ### Supported Languages
    - Tamil, English, Hindi, Malayalam, Telugu
    
    ### Request Format
    - `language`: One of the 5 supported languages
    - `audioFormat`: Must be "mp3"
    - `audioBase64`: Base64-encoded MP3 audio data
    
    ### Response
    Returns classification (`AI_GENERATED` or `HUMAN`), confidence score (0.0-1.0), 
    and explanation.
    """
    # Step 1: Decode audio
    waveform, sample_rate = decode_base64_audio(request.audioBase64, "mp3")

    # Step 2: Duration check & Truncation (Safety for Render timeouts)
    duration = len(waveform) / sample_rate
    MAX_DURATION = 30.0  # Limit to 30s to prevent memory issues
    if duration > MAX_DURATION:
        waveform = waveform[:int(MAX_DURATION * sample_rate)]

    # Step 3: Silence trimming (VAD)
    intervals = librosa.effects.split(waveform, top_db=25)
    if intervals.any():
        waveform = np.concatenate([waveform[s:e] for s, e in intervals])

    # Step 4: Extract features
    features = extract_features(waveform, sample_rate)

    # Step 5: ML prediction
    classification, confidence = predict(features)
    
    # Step 6: Generate explanation
    explanation = generate_explanation(features, classification, confidence)

    return VoiceResponse(
        status="success",
        language=request.language,
        classification=classification,
        confidenceScore=confidence,
        explanation=explanation
    )


@app.post("/api/voice-detection/upload", response_model=VoiceResponse, include_in_schema=False)
async def detect_voice_upload(
    audio_file: UploadFile = File(..., description="🎵 Audio file to analyze (MP3 only)"),
    language: str = Form(..., description="🌍 Language of the audio: Tamil, English, Hindi, Malayalam, Telugu"),
    api_key: str = Depends(validate_api_key)
):
    """
    🎤 Upload Audio File for Voice Detection
    
    Analyze an uploaded audio file to determine if it contains **human voice** or **AI-generated speech**.
    
    ### Supported Formats
    - MP3, WAV, M4A, MPEG, OGG
    
    ### Minimum Requirements
    - Audio can be of any duration
    - Maximum file size: **50MB**
    
    ### Response
    Returns classification result with confidence score and detailed analysis.
    """
    # Step 1: Read file contents
    audio_bytes = await audio_file.read()
    if len(audio_bytes) == 0:
        raise HTTPException(status_code=400, detail="Empty audio file")

    # Step 2: Decode audio (Detect format)
    filename = audio_file.filename or "audio.mp3"
    file_ext = filename.split('.')[-1].lower() if '.' in filename else "mp3"
    waveform, sample_rate = decode_audio_bytes(audio_bytes, file_ext)

    # Step 3: Duration check & Truncation (Safety for Render timeouts)
    duration = len(waveform) / sample_rate
    MAX_DURATION = 30.0  # Limit to 30s to prevent memory issues
    if duration > MAX_DURATION:
        waveform = waveform[:int(MAX_DURATION * sample_rate)]

    # Step 4: Silence trimming (VAD)
    intervals = librosa.effects.split(waveform, top_db=25)
    if intervals.any():
        waveform = np.concatenate([waveform[s:e] for s, e in intervals])

    # Step 5: Extract features
    features = extract_features(waveform, sample_rate)

    # Step 6: ML prediction
    classification, confidence = predict(features)
    
    # Step 7: Generate explanation
    explanation = generate_explanation(features, classification, confidence)

    return VoiceResponse(
        status="success",
        language=language,
        classification=classification,
        confidenceScore=confidence,
        explanation=explanation
    )
