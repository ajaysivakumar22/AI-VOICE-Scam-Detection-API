# 🎙️ AI Voice Scam Detection API

A secure REST API that detects whether a given voice sample is **AI-generated** or **Human-spoken**.  
Built for the **HCL × GUVI Hackathon** with a focus on robustness, fairness, and explainability.

---

## 🚀 Features
- Detects **AI_GENERATED** vs **HUMAN** voice
- Supports **Tamil, English, Hindi, Malayalam, Telugu**
- Accepts **Base64-encoded MP3 audio**
- Returns prediction with **confidence score**
- Secured using **API Key authentication**
- Fully documented via **Swagger (`/docs`)**

---

## 🛠️ Tech Stack
- **FastAPI** – REST API
- **Python** – Core logic
- **Librosa / Pydub** – Audio processing
- **Scikit-learn** – ML classification
- **Render** – Deployment

---

## 📌 API Endpoint

### POST `/api/voice-detection`

#### Headers
