"""
Create a test audio file and test the Voice Detection API
"""

import base64
import requests
import numpy as np
from pydub import AudioSegment
from pydub.generators import Sine
import io

API_URL = "http://127.0.0.1:8000/api/voice-detection"
API_KEY = "YOUR_SECRET_API_KEY"

def create_test_audio():
    """Create a simple 2-second sine wave audio"""
    print("Creating test audio file...")
    
    # Generate a 2-second 440Hz sine wave (A note)
    tone = Sine(440).to_audio_segment(duration=2000)  # 2 seconds
    
    # Export to MP3 in memory
    mp3_buffer = io.BytesIO()
    tone.export(mp3_buffer, format="mp3")
    mp3_bytes = mp3_buffer.getvalue()
    
    # Also save to file for reference
    tone.export("d:/GuviHCL_Hackathon/test_audio.mp3", format="mp3")
    print("Saved test_audio.mp3 to d:/GuviHCL_Hackathon/")
    
    return mp3_bytes

def test_api(audio_bytes):
    """Send audio to API and print response"""
    audio_base64 = base64.b64encode(audio_bytes).decode("utf-8")
    
    print(f"Audio size: {len(audio_bytes)} bytes")
    print(f"Base64 length: {len(audio_base64)} characters")
    print()
    
    payload = {
        "language": "English",
        "audioFormat": "mp3",
        "audioBase64": audio_base64
    }
    
    headers = {
        "x-api-key": API_KEY,
        "Content-Type": "application/json"
    }
    
    print("Sending request to API...")
    response = requests.post(API_URL, json=payload, headers=headers)
    
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")

if __name__ == "__main__":
    audio_bytes = create_test_audio()
    test_api(audio_bytes)
