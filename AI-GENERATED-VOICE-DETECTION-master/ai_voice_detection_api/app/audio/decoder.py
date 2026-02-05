import base64
import io
import os
import subprocess
import binascii
import tempfile
import numpy as np
import librosa
import shutil
from fastapi import HTTPException

# Configure FFmpeg path
# On Render/Linux, ffmpeg is usually in /usr/bin which is in PATH
FFMPEG_EXE = "ffmpeg"

# Specific Windows fallback for local development
# This path is machine-specific but wrapped in a check so it doesn't break deployment
LOCAL_FFMPEG_PATH = r"C:\Users\ajays\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-8.0.1-full_build\bin"

if os.name == 'nt' and os.path.exists(LOCAL_FFMPEG_PATH):
    # Add to PATH so subprocess and pydub can find it
    # We do this dynamically so it only affects this run
    if LOCAL_FFMPEG_PATH not in os.environ["PATH"]:
        os.environ["PATH"] = LOCAL_FFMPEG_PATH + os.pathsep + os.environ["PATH"]

# Validates if ffmpeg is available
if not shutil.which(FFMPEG_EXE):
    # This warning helps debug deployment issues
    print("Warning: ffmpeg not found in PATH.")

# Try to configure pydub
try:
    from pydub import AudioSegment
    # pydub auto-detects from PATH, but we can be explicit if needed
    # AudioSegment.converter = FFMPEG_EXE 
    PYDUB_AVAILABLE = True
except ImportError:
    PYDUB_AVAILABLE = False


def decode_base64_audio(audio_base64: str, audio_format: str = "mp3"):
    """
    Decode base64 audio to waveform with comprehensive validation.
    
    Args:
        audio_base64: Base64 encoded audio string
        audio_format: Audio format (mp3, wav, etc.)
    
    Returns:
        tuple: (waveform as np.ndarray, sample_rate as int)
    """
    try:
        # Remove data URI prefix if present (e.g., "data:audio/mp3;base64,")
        if "base64," in audio_base64:
            audio_base64 = audio_base64.split("base64,")[1]
        
        # Remove whitespace and newlines
        audio_base64 = audio_base64.strip().replace("\n", "").replace("\r", "")
        
        # Validate base64 string is not empty
        if not audio_base64:
            raise HTTPException(
                status_code=400,
                detail="Empty base64 audio data"
            )
        
        # Decode base64 to bytes
        audio_bytes = base64.b64decode(audio_base64)
        
        # Validate audio bytes are not empty
        if len(audio_bytes) == 0:
            raise HTTPException(
                status_code=400,
                detail="Decoded audio data is empty"
            )
        
        return decode_audio_bytes(audio_bytes, audio_format)

    except binascii.Error as e:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid Base64 encoding: {str(e)}"
        )


def decode_audio_bytes(audio_bytes: bytes, audio_format: str = "mp3"):
    """
    Decode audio bytes to waveform.
    
    Args:
        audio_bytes: Raw audio bytes
        audio_format: Audio format (mp3, wav, etc.)
    
    Returns:
        tuple: (waveform as np.ndarray, sample_rate as int)
    """
    # Method 1: Try direct librosa loading for WAV files
    if audio_format.lower() == "wav":
        try:
            # Force sr=22050 to match training data
            waveform, sample_rate = librosa.load(
                io.BytesIO(audio_bytes),
                sr=22050,
                mono=True
            )
            return waveform, sample_rate
        except Exception:
            pass

    # Method 2: Try using FFmpeg directly via subprocess
    if os.path.exists(FFMPEG_EXE):
        try:
            with tempfile.NamedTemporaryFile(suffix=f'.{audio_format}', delete=False) as temp_in:
                temp_in.write(audio_bytes)
                temp_in_path = temp_in.name
            
            temp_out_path = temp_in_path.replace(f'.{audio_format}', '.wav')
            
            # Run FFmpeg to convert to WAV
            # CRITICAL: Force 22050Hz sample rate for consistency with training data
            result = subprocess.run(
                [FFMPEG_EXE, '-i', temp_in_path, '-f', 'wav', '-acodec', 'pcm_s16le', '-ar', '22050', '-ac', '1', '-y', temp_out_path],
                capture_output=True,
                timeout=30
            )
            
            if result.returncode == 0 and os.path.exists(temp_out_path):
                # Load with sr=None because we already forced it to 22050 in ffmpeg
                # This ensures we get exactly what ffmpeg produced
                waveform, sample_rate = librosa.load(temp_out_path, sr=None, mono=True)
                # Cleanup
                os.unlink(temp_in_path)
                os.unlink(temp_out_path)
                return waveform, sample_rate
            else:
                # Cleanup on failure
                if os.path.exists(temp_in_path):
                    os.unlink(temp_in_path)
                if os.path.exists(temp_out_path):
                    os.unlink(temp_out_path)
        except Exception as e:
            pass

    # Method 3: Try pydub as fallback
    if PYDUB_AVAILABLE:
        try:
            audio_segment = AudioSegment.from_file(
                io.BytesIO(audio_bytes),
                format=audio_format
            )

            wav_io = io.BytesIO()
            audio_segment.export(wav_io, format="wav")
            wav_io.seek(0)

            # Force sr=22050 to match training data
            waveform, sample_rate = librosa.load(
                wav_io,
                sr=22050,
                mono=True
            )
            return waveform, sample_rate
        except Exception as e:
            print("Warning: pydub import failed or FFmpeg not found.", e)

    # Fallback to librosa dynamic load (might be slow / wrong SR)
    # But better than crashing
    try:
        # Force sr=22050
        waveform, sample_rate = librosa.load(
            io.BytesIO(audio_bytes),
            sr=22050, 
            mono=True
        )
        return waveform, sample_rate
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Failed to decode audio. Please ensure valid MP3/WAV format. Error: {str(e)}"
        )

    raise HTTPException(
        status_code=400,
        detail="Audio decoding failed: No suitable decoder available"
    )
