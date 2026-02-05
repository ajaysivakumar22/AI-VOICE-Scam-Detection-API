import numpy as np
from app.ml.model import load_model_and_baselines

def predict(features: np.ndarray):
    model, _ = load_model_and_baselines()

    if model is None:
        # Fallback if model failed to load (should not happen if file exists)
        return 'HUMAN', 0.0

    # Reshape for single sample prediction
    features = features.reshape(1, -1)
    
    # Predict probability of being AI
    prob_ai = model.predict_proba(features)[0][1]

    # Threshold for classification (0.5 is standard, 0.6 is conservative for AI)
    THRESHOLD = 0.5
    
    classification = 'AI_GENERATED' if prob_ai >= THRESHOLD else 'HUMAN'
    
    # Calculate confidence: how far is the probability from 0.5?
    # If prob_ai is 0.9, confidence is 0.9 (Very confident AI)
    # If prob_ai is 0.1, confidence is 0.9 (Very confident Human)
    # If prob_ai is 0.5, confidence is 0.5 (Uncertain)
    
    if classification == 'AI_GENERATED':
        confidence = prob_ai
    else:
        confidence = 1.0 - prob_ai

    return classification, round(float(confidence), 2)
