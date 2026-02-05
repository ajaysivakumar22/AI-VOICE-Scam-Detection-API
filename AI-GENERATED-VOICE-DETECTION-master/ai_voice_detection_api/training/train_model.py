import os
import json
import base64
import pickle
import numpy as np
import librosa
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from collections import defaultdict
import sys
from pathlib import Path
import random

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.audio.decoder import decode_base64_audio as decode_base64_mp3
from app.audio.features import extract_features

# -----------------------------
# Silence Removal (VAD)
# -----------------------------
def remove_silence(waveform, sr):
    try:
        intervals = librosa.effects.split(waveform, top_db=25)
        if len(intervals) == 0:
            return waveform
        return np.concatenate([waveform[s:e] for s, e in intervals])
    except:
        return waveform

# -----------------------------
# Data Augmentation
# -----------------------------
def augment_audio(waveform, sr):
    augmented = []

    # Original
    augmented.append(waveform)

    # Time stretch
    try:
        augmented.append(librosa.effects.time_stretch(waveform, rate=0.95))
        augmented.append(librosa.effects.time_stretch(waveform, rate=1.05))
    except:
        pass

    # Volume scaling
    augmented.append(waveform * 0.9)
    augmented.append(waveform * 1.1)

    return augmented

# -----------------------------
# Load Dataset
# -----------------------------
def load_dataset(folder, label, max_files=None):
    X, y, raw_features = [], [], []

    if not os.path.exists(folder):
        print(f'Directory not found: {folder}')
        return np.array(X), np.array(y), raw_features

    files = [f for f in os.listdir(folder) if f.lower().endswith(('.mp3', '.mpeg', '.wav', '.ogg', '.flac'))]
    
    if max_files:
        if len(files) > max_files:
            print(f"Downsampling {folder} from {len(files)} to {max_files} files...")
            random.shuffle(files)
            files = files[:max_files]
        else:
            print(f"Loading all {len(files)} files from {folder}...")

    for i, file in enumerate(files):
        path = os.path.join(folder, file)
        try:
            with open(path, 'rb') as f:
                # Use base64 logic as requested, though direct load is possible
                # This ensures consistent pipeline with API
                audio_base64 = base64.b64encode(f.read()).decode()

            # Determine format from extension
            ext = file.split('.')[-1].lower()
            if ext == 'mpeg': ext = 'mp3' # Treat mpeg as mp3 for decoder simple match
            
            waveform, sr = decode_base64_mp3(audio_base64, audio_format=ext)
            # Basic validation
            if len(waveform) < 100: continue
            
            waveform = remove_silence(waveform, sr)

            if len(waveform) < 1000: # Skip extremely short clips after silence removal
                 continue

            for aug in augment_audio(waveform, sr):
                features = extract_features(aug, sr)
                X.append(features)
                y.append(label)
                raw_features.append(features)
                
            if i % 10 == 0:
                print(f'Processed {i+1} files in {folder}')
        except Exception as e:
            print(f'Error processing {file}: {e}')
    return np.array(X), np.array(y), raw_features

# -----------------------------
# Load Data
# -----------------------------
ai_folder = str(Path(__file__).parent / 'data' / 'ai')
human_folder = str(Path(__file__).parent / 'data' / 'human')

# Count AI files first to balance the dataset
ai_files_count = len([f for f in os.listdir(ai_folder) if f.lower().endswith(('.mp3', '.mpeg', '.wav', '.ogg', '.flac'))])
print(f"Found {ai_files_count} AI samples availability.")

print('Loading AI data...')
# Load all AI data
X_a, y_a, feats_a = load_dataset(ai_folder, 1)
print(f'Loaded {len(X_a)} AI samples (augmented).')

print(f'Loading Human data (matching AI count: {ai_files_count})...')
# Downsample human data to match AI count for perfect balance
X_h, y_h, feats_h = load_dataset(human_folder, 0, max_files=ai_files_count)
print(f'Loaded {len(X_h)} human samples (augmented).')

if len(X_h) == 0 or len(X_a) == 0:
    print('Error: insufficient data. Exiting.')
    sys.exit(1)

X = np.vstack([X_h, X_a])
y = np.concatenate([y_h, y_a])

# -----------------------------
# Train / Test Split
# -----------------------------
print('Splitting data...')
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -----------------------------
# Model + Auto Tuning
# -----------------------------
print('Tuning hyperparameters with GridSearchCV...')
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 8, 16],
    'min_samples_split': [2, 5]
}

grid = GridSearchCV(
    RandomForestClassifier(random_state=42, class_weight='balanced'),
    param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)

grid.fit(X_train, y_train)
model = grid.best_estimator_

print('Best Params:', grid.best_params_)
print('Train Accuracy:', model.score(X_train, y_train))
print('Test Accuracy:', model.score(X_test, y_test))

# -----------------------------
# Save Model
# -----------------------------
models_dir = Path(__file__).resolve().parent.parent / 'models'
models_dir.mkdir(exist_ok=True)
model_path = models_dir / 'voice_classifier.pkl'

with open(model_path, 'wb') as f:
    pickle.dump(model, f)
print(f'Model saved to {model_path}')

# -----------------------------
# Feature Baselines (NO MAGIC NUMBERS)
# -----------------------------
print('Calculating feature baselines...')
def compute_stats(features):
    return {
        'mean': float(np.mean(features)),
        'std': float(np.std(features)),
        'p5': float(np.percentile(features, 5)),
        'p95': float(np.percentile(features, 95))
    }

baselines = defaultdict(dict)

feats_h = np.array(feats_h)
feats_a = np.array(feats_a)

# Ensure consistent shape
min_len = min(len(feats_h), len(feats_a))
if min_len > 0:
    num_features = feats_h.shape[1]
    
    for idx in range(num_features):
        baselines[f'feature_{idx}']['human'] = compute_stats(feats_h[:, idx])
        baselines[f'feature_{idx}']['ai'] = compute_stats(feats_a[:, idx])

    baseline_path = Path(__file__).parent / 'feature_baselines.json'
    with open(baseline_path, 'w') as f:
        json.dump(baselines, f, indent=2)
    print(f'Baselines saved to {baseline_path}')

print(' Training complete. Model & baselines saved.')
