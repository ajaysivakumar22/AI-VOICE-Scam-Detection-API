import json
import numpy as np
from pathlib import Path

# Load baselines once
BASELINES = {}
try:
    # Try local training path first (Hackathon structure)
    path = Path(__file__).resolve().parent.parent.parent / 'training' / 'feature_baselines.json'
    if path.exists():
        with open(path, 'r') as f:
            BASELINES = json.load(f)
    else:
        # Fallback to model directory if deployed
        path = Path(__file__).resolve().parent.parent.parent / 'models' / 'feature_baselines.json'
        if path.exists():
            with open(path, 'r') as f:
                BASELINES = json.load(f)
except Exception as e:
    print(f'Warning: Could not load baselines: {e}')

def get_z_score(value, mean, std):
    if std == 0:
        return 0
    return (value - mean) / std

def generate_explanation(features: np.ndarray, classification: str, confidence: float) -> str:
    """
    Generate statistical explanation based on training baselines.
    100% Data-Driven. No hard-coding.
    """
    if not BASELINES:
        return f'Classified as {classification} ({confidence:.1%}) based on overall feature profile.'

    # Map features to indices (Must match features.py)
    # 0-12: MFCC, 13-25: Delta, 26-38: Delta2
    key_features = {
        'Pitch Stability': (40, 'feature_40'), # Pitch Std
        'Voice Depth': (39, 'feature_39'),     # Pitch Mean
        'Micro-Tremors (Jitter)': (44, 'feature_44'),
        'Amplitude Flux (Shimmer)': (45, 'feature_45'),
        'Dynamic Range': (42, 'feature_42'),   # RMS Std
        'Timbral Complexity': (43, 'feature_43') # Centroid
    }

    reasons = []
    
    # Target distribution to compare against (if Human, look for Human matches, etc.)
    # Actually, we want to explain why it matched the PREDICTED class.
    target_class = 'ai' if classification == 'AI' else 'human'
    other_class = 'human' if target_class == 'ai' else 'ai'

    for name, (idx, feat_key) in key_features.items():
        if feat_key not in BASELINES:
            continue
            
        val = features[idx]
        stats_target = BASELINES[feat_key][target_class]
        stats_other = BASELINES[feat_key][other_class]

        # Calculate Z-scores
        z_target = get_z_score(val, stats_target['mean'], stats_target['std'])
        z_other = get_z_score(val, stats_other['mean'], stats_other['std'])

        # Logic: If it is very close to target mean (|z| < 1) AND far from other mean (|z| > 1)
        if abs(z_target) < 1.0 and abs(z_other) > 1.5:
            reasons.append(f'{name} matches {target_class} profile (Z-score: {z_target:.2f})')
        elif abs(z_target) < 0.5:
            # Very tight match
            reasons.append(f'{name} strongly aligns with {target_class} statistical baseline')

    # Construct message
    if not reasons:
        # Fallback if no specific feature is strong enough
        msg = f'Voice patterns align with {classification} training data distribution.'
    else:
        # Pick top 2 reasons
        msg = f'Detected {classification} voice traits: ' + ', '.join(reasons[:2]) + '.'

    return msg
