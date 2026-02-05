# 🎓 Model Training Guide

## Quick Start

### Step 1: Collect Training Data

You need audio samples of both **human voices** and **AI-generated voices**.

#### Where to get Human Voice samples:
- 🎙️ Record yourself speaking
- 🎧 Download podcast clips (human speakers)
- 📺 Use YouTube audio (interviews, speeches)
- 🗣️ LibriSpeech dataset: https://www.openslr.org/12
- 🎵 VoxCeleb dataset: https://www.robots.ox.ac.uk/~vgg/data/voxceleb/

#### Where to get AI Voice samples:
- 🤖 ElevenLabs: https://elevenlabs.io/
- 🔊 Google Cloud TTS: https://cloud.google.com/text-to-speech
- 🎤 Amazon Polly: https://aws.amazon.com/polly/
- 💬 ChatGPT voice mode recordings
- 🎯 Resemble.ai: https://www.resemble.ai/
- 📢 Microsoft Azure TTS

### Step 2: Organize Files

Place your audio files in these folders:

```
training/
└── data/
    ├── human/          ← Put human voice files here
    │   ├── human_1.mp3
    │   ├── human_2.wav
    │   └── ...
    └── ai/             ← Put AI-generated files here
        ├── ai_1.mp3
        ├── ai_2.wav
        └── ...
```

**Recommendations:**
- Minimum: 20 samples each (40 total)
- Ideal: 50+ samples each (100+ total)
- Formats: MP3, WAV, M4A, OGG, FLAC
- Duration: Any length supported (previously 1-30 seconds)

### Step 3: Train the Model

```bash
cd D:\GuviHCL_Hackathon\ai_voice_detection_api
py training/train_model.py
```

### Step 4: Restart the API

The API will automatically load the trained model on restart:

```bash
py -m uvicorn app.main:app --reload
```

---

## 📊 Expected Output

```
============================================================
🎙️  AI VOICE DETECTION - MODEL TRAINING
============================================================

📁 Data directories:
   Human voices: training\data\human
   AI voices:    training\data\ai

============================================================
📥 LOADING TRAINING DATA
============================================================
📂 Found 50 audio files in human/
  ✅ [1/50] human_01.mp3 (5.2s)
  ...

📂 Found 50 audio files in ai/
  ✅ [1/50] ai_generated_01.mp3 (4.8s)
  ...

📊 Dataset Summary:
   Human samples: 50
   AI samples:    50
   Total:         100

============================================================
🤖 TRAINING MODEL
============================================================
Training Random Forest classifier...

📈 Cross-validation scores: [0.92 0.88 0.94 0.90 0.86]
   Mean accuracy: 90.00% (+/- 5.66%)

============================================================
📊 MODEL EVALUATION
============================================================

Test Accuracy: 92.00%

Classification Report:
              precision    recall  f1-score   support
       HUMAN       0.91      0.93      0.92        10
AI_GENERATED       0.93      0.91      0.92        10

✅ Model saved to: models/voice_classifier.pkl
```

---

## 💡 Tips for Better Accuracy

1. **Balanced dataset**: Equal number of human and AI samples
2. **Diverse voices**: Different ages, genders, accents
3. **Diverse AI sources**: Use multiple TTS providers
4. **Clean audio**: Avoid noisy recordings
5. **Similar conditions**: Match recording quality between classes
6. **More data**: More samples = better accuracy
