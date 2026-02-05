"""
Step 1: Create a test WAV audio file (run this first)
"""
import numpy as np
import soundfile as sf

# Create a 2-second 440Hz sine wave
sample_rate = 44100
duration = 2  # seconds
t = np.linspace(0, duration, int(sample_rate * duration), False)
audio_data = 0.5 * np.sin(2 * np.pi * 440 * t).astype(np.float32)

# Save to WAV file
wav_path = "d:/GuviHCL_Hackathon/test_audio.wav"
sf.write(wav_path, audio_data, sample_rate)
print(f"Created: {wav_path}")
print(f"Duration: {duration} seconds")
print(f"Sample rate: {sample_rate} Hz")
