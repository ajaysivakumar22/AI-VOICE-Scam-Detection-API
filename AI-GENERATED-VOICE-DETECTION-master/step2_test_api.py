"""
Step 2: Test the API with the audio file (run after step1)
"""
import base64
import requests

API_URL = "http://127.0.0.1:8000/api/voice-detection"
API_KEY = "YOUR_SECRET_API_KEY"

# Read the WAV file
wav_path = "d:/GuviHCL_Hackathon/test_audio.wav"
with open(wav_path, "rb") as f:
    wav_bytes = f.read()

# Encode to base64
audio_base64 = base64.b64encode(wav_bytes).decode("utf-8")
print(f"File: {wav_path}")
print(f"Size: {len(wav_bytes)} bytes")
print(f"Base64 length: {len(audio_base64)} chars")

# Send to API
payload = {
    "language": "English",
    "audioFormat": "wav",
    "audioBase64": audio_base64
}

headers = {
    "x-api-key": API_KEY,
    "Content-Type": "application/json"
}

print("\nSending to API...")
response = requests.post(API_URL, json=payload, headers=headers)
print(f"Status: {response.status_code}")
print(f"Response: {response.json()}")
