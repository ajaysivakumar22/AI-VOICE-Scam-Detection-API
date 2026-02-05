import pickle
import json
from pathlib import Path

# Define paths relative to the project root for robustness
BASE_DIR = Path(__file__).resolve().parent.parent.parent
MODEL_PATH = BASE_DIR / 'models' / 'voice_classifier.pkl'
BASELINE_PATH = BASE_DIR / 'training' / 'feature_baselines.json'

_model = None
_baselines = None

def load_model_and_baselines():
    global _model, _baselines

    if _model is None:
        if MODEL_PATH.exists():
            with open(MODEL_PATH, 'rb') as f:
                _model = pickle.load(f)
        else:
            raise FileNotFoundError(f'Model file not found at {MODEL_PATH}')

    if _baselines is None:
        if BASELINE_PATH.exists():
            with open(BASELINE_PATH, 'r') as f:
                _baselines = json.load(f)
        else:
            # Fallback for deployment scenarios where training might not exist
            fallback = BASE_DIR / 'models' / 'feature_baselines.json'
            if fallback.exists():
                with open(fallback, 'r') as f:
                    _baselines = json.load(f)
            else:
                print(f'Warning: Baselines not found at {BASELINE_PATH}')
                _baselines = {}

    return _model, _baselines
