"""
Test script for the Voice Detection API
Usage: python test_api.py path/to/your/audio.mp3
"""

import base64
import requests
import sys

API_URL = "http://127.0.0.1:8000/api/voice-detection"
API_KEY = "YOUR_SECRET_API_KEY"

def test_api(audio_file_path: str, audio_format: str = "mp3"):
    # Read and encode the audio file to base64
    with open(audio_file_path, "rb") as f:
        audio_bytes = f.read()
    
    audio_base64 = base64.b64encode(audio_bytes).decode("utf-8")
    
    print(f"📁 File: {audio_file_path}")
    print(f"🎵 Format: {audio_format}")
    print(f"📊 Size: {len(audio_bytes)} bytes")
    print(f"🔤 Base64 length: {len(audio_base64)} characters")
    print()
    
    # Prepare the request
    payload = {
        "language": "English",
        "audioFormat": audio_format,
        "audioBase64": audio_base64
    }
    
    headers = {
        "x-api-key": API_KEY,
        "Content-Type": "application/json"
    }
    
    print("📤 Sending request to API...")
    response = requests.post(API_URL, json=payload, headers=headers)
    
    print(f"📥 Status: {response.status_code}")
    print(f"📋 Response:")
    print(response.json())

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_api.py <path_to_audio_file> [format]")
        print("Example: python test_api.py sample.mp3")
        print("Example: python test_api.py audio.mpeg mpeg")
    else:
        file_path = sys.argv[1]
        # Auto-detect format from extension
        ext = file_path.split(".")[-1].lower()
        audio_format = sys.argv[2] if len(sys.argv) > 2 else ext
        test_api(file_path, audio_format)
