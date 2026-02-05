"""
Test script for the AI Voice Detection API.

Run this script while the server is running to test the endpoints.
"""

import requests
import json

API_URL = "http://127.0.0.1:8000/api/voice-detection/upload"
API_KEY = "YOUR_SECRET_API_KEY"

def test_voice_detection(audio_path: str, language: str = "English"):
    """Test the voice detection API with an audio file."""
    headers = {
        "x-api-key": API_KEY
    }
    
    files = {
        "audio_file": open(audio_path, "rb")
    }
    
    data = {
        "language": language
    }
    
    try:
        response = requests.post(API_URL, headers=headers, files=files, data=data)
        print(f"Status Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        return response.json()
    except Exception as e:
        print(f"Error: {e}")
        return None
    finally:
        files["audio_file"].close()


if __name__ == "__main__":
    import sys
    import os
    
    if len(sys.argv) > 1:
        audio_file = sys.argv[1]
    else:
        # Default test file (updated to be path-agnostic)
        print("Usage: python test_api.py <path_to_audio_file>")
        print("No audio file provided. Creating a dummy test file if none exists...")
        audio_file = "test_audio_sample.mp3"
        
        # Only create if it doesn't exist (and if we can - requires pydub or similar, 
        # but here we just want to avoid the crash if the user runs it blindly)
        if not os.path.exists(audio_file):
             print(f"Warning: Default file '{audio_file}' not found.")
             print("Please provide a path: python test_api.py <file.wav>")
             sys.exit(1)
    
    print(f"Testing with: {audio_file}")
    print("-" * 50)
    test_voice_detection(audio_file)
