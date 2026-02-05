"""
Create a voice classifier model for AI vs Human detection.

This script creates a simple classifier based on voice feature patterns.
For a hackathon, this uses a heuristic-based approach that mimics ML behavior.

Run this script once to generate the model file.
"""

import pickle
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from pathlib import Path


class VoiceClassifier(BaseEstimator, ClassifierMixin):
    """
    Voice classifier for AI-generated vs Human voice detection.
    
    Uses feature-based heuristics that analyze:
    - Pitch consistency (AI voices are unnaturally consistent)
    - Energy patterns (AI voices have robotic energy)
    - MFCC variance (AI voices have synthetic texture)
    - Spectral characteristics
    
    Classes:
        0 = HUMAN
        1 = AI_GENERATED
    """
    
    def __init__(self):
        self.classes_ = np.array([0, 1])  # 0=HUMAN, 1=AI_GENERATED
        self.is_fitted_ = True
    
    def fit(self, X, y=None):
        """Model doesn't need training - uses feature analysis."""
        return self
    
    def predict(self, X):
        """Predict class labels."""
        proba = self.predict_proba(X)
        return (proba[:, 1] >= 0.6).astype(int)
    
    def predict_proba(self, X):
        """
        Predict class probabilities based on feature analysis.
        
        Features expected (18 total):
        - 0-12: MFCCs (13 coefficients)
        - 13: Pitch mean
        - 14: Pitch std (variation)
        - 15: RMS mean (energy)
        - 16: RMS std (energy variation)  
        - 17: Spectral centroid
        """
        X = np.atleast_2d(X)
        probas = []
        
        for features in X:
            ai_score = 0.0
            total_weight = 0.0
            
            # Feature 14: Pitch standard deviation
            # Low pitch variation = more likely AI
            if len(features) > 14:
                pitch_std = features[14]
                if pitch_std < 10:
                    ai_score += 0.35  # Very consistent pitch = AI
                elif pitch_std < 20:
                    ai_score += 0.15  # Somewhat consistent
                else:
                    ai_score += 0.0   # Natural variation = Human
                total_weight += 0.35
            
            # Feature 16: RMS standard deviation (energy variation)
            # Low energy variation = more likely AI
            if len(features) > 16:
                rms_std = features[16]
                if rms_std < 0.01:
                    ai_score += 0.25  # Very consistent energy = AI
                elif rms_std < 0.03:
                    ai_score += 0.10  # Somewhat consistent
                else:
                    ai_score += 0.0   # Natural variation = Human
                total_weight += 0.25
            
            # MFCC variance (features 0-12)
            # Low variance in MFCCs = synthetic texture
            if len(features) >= 13:
                mfcc_std = np.std(features[0:13])
                if mfcc_std < 25:
                    ai_score += 0.20  # Synthetic texture
                elif mfcc_std < 40:
                    ai_score += 0.08
                else:
                    ai_score += 0.0   # Natural texture
                total_weight += 0.20
            
            # Feature 17: Spectral centroid
            # Unusual spectral centroid can indicate AI
            if len(features) > 17:
                spectral = features[17]
                if spectral > 3500 or spectral < 1000:
                    ai_score += 0.10  # Unusual frequency
                else:
                    ai_score += 0.0   # Normal frequency
                total_weight += 0.10
            
            # Feature 13: Pitch mean
            # Unusual pitch can indicate AI
            if len(features) > 13:
                pitch_mean = features[13]
                if pitch_mean < 80 or pitch_mean > 280:
                    ai_score += 0.10  # Unusual pitch
                total_weight += 0.10
            
            # Normalize to probability
            if total_weight > 0:
                prob_ai = min(ai_score / total_weight, 1.0)
            else:
                prob_ai = 0.5
            
            # Add some reasonable bounds
            prob_ai = max(0.05, min(0.95, prob_ai))
            
            probas.append([1 - prob_ai, prob_ai])
        
        return np.array(probas)


def create_model():
    """Create and save the voice classifier model."""
    # Create model
    model = VoiceClassifier()
    
    # Save to models directory
    model_dir = Path(__file__).resolve().parent / "ai_voice_detection_api" / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    
    model_path = model_dir / "voice_classifier.pkl"
    
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    
    print(f"Model saved to: {model_path}")
    
    # Test the model
    print("\nTesting model with sample features...")
    
    # Simulate AI voice features (low variation)
    ai_features = np.array([
        -200, 80, 15, -5, 10, 5, -3, 8, -2, 6, 4, -1, 3,  # MFCCs
        150,   # pitch mean
        8,     # pitch std (low = AI)
        0.05,  # rms mean
        0.008, # rms std (low = AI)
        2800   # spectral centroid
    ])
    
    # Simulate Human voice features (high variation)
    human_features = np.array([
        -180, 95, 20, -8, 15, 8, -5, 12, -4, 9, 6, -2, 5,  # MFCCs
        140,   # pitch mean
        35,    # pitch std (high = Human)
        0.08,  # rms mean
        0.045, # rms std (high = Human)
        2200   # spectral centroid
    ])
    
    ai_proba = model.predict_proba(ai_features.reshape(1, -1))[0]
    human_proba = model.predict_proba(human_features.reshape(1, -1))[0]
    
    print(f"AI voice test:    P(Human)={ai_proba[0]:.2f}, P(AI)={ai_proba[1]:.2f}")
    print(f"Human voice test: P(Human)={human_proba[0]:.2f}, P(AI)={human_proba[1]:.2f}")
    
    return model


if __name__ == "__main__":
    create_model()
