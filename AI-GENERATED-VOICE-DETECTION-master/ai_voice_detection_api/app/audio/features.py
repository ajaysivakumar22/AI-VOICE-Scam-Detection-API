import numpy as np
import librosa
from typing import Union
import scipy.stats


def extract_features(waveform: np.ndarray, sample_rate: Union[int, float]) -> np.ndarray:
    """
    Extract exact feature set for hackathon compliance (No hard-coding).
    
    Features (Total: 46):
    - MFCC (13) mean
    - MFCC Δ (13) mean
    - MFCC ΔΔ (13) mean
    - Pitch mean, std
    - RMS mean, std
    - Spectral centroid mean
    - Jitter mean
    - Shimmer mean
    """
    features = []

    # Ensure minimum length (0.5s)
    if len(waveform) < 2048:
        waveform = np.pad(waveform, (0, 2048 - len(waveform)))

    # 1. MFCCs (13) mean
    mfcc = librosa.feature.mfcc(y=waveform, sr=sample_rate, n_mfcc=13)
    features.extend(np.mean(mfcc, axis=1))

    # 2. MFCC Delta (13) mean
    mfcc_delta = librosa.feature.delta(mfcc)
    features.extend(np.mean(mfcc_delta, axis=1))

    # 3. MFCC Delta-Delta (13) mean
    mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
    features.extend(np.mean(mfcc_delta2, axis=1))

    # Pitch/F0 Extraction
    try:
        pitch = librosa.yin(waveform, fmin=50, fmax=400, sr=sample_rate)
        pitch_valid = pitch[pitch > 0]
    except:
        pitch_valid = np.array([])
        
    # 4. Pitch mean & std
    if len(pitch_valid) > 10:
        features.append(np.mean(pitch_valid))
        features.append(np.std(pitch_valid))
    else:
        features.extend([0.0, 0.0])

    # RMS Energy
    rms = librosa.feature.rms(y=waveform)[0]

    # 5. RMS mean & std
    features.append(np.mean(rms))
    features.append(np.std(rms))

    # 6. Spectral Centroid mean
    centroid = librosa.feature.spectral_centroid(y=waveform, sr=sample_rate)[0]
    features.append(np.mean(centroid))

    # 7. Jitter (mean)
    try:
        if len(pitch_valid) > 1:
            periods = 1.0 / pitch_valid
            jitter = np.mean(np.abs(np.diff(periods)))
            features.append(jitter)
        else:
            features.append(0.0)
    except:
        features.append(0.0)

    # 8. Shimmer (mean)
    try:
        if len(rms) > 1:
            shimmer = np.mean(np.abs(np.diff(rms)))
            features.append(shimmer)
        else:
            features.append(0.0)
    except:
        features.append(0.0)

    # Sanitize
    return np.nan_to_num(np.array(features, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)