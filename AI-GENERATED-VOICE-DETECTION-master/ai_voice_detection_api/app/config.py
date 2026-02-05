import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env from project root
env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

API_KEY = os.getenv("API_KEY")

SUPPORTED_LANGUAGES = [
    "Tamil", "English", "Hindi", "Malayalam", "Telugu"
]
