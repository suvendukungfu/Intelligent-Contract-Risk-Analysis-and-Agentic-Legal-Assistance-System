"""
models/explainability.py
-------------------------
XAI (Explainable AI) Engine for Milestone 4.
Extracts mathematical feature importance from the Logistic Regression matrix
to prove exactly which words drove the risk classification.
"""

import numpy as np
import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

def generate_xai_importance(clause: str, model_tuple: Any) -> Dict[str, float]:
    """
    Extracts high-weight tokens from a specific clause using the trained model's coefficients.
    This acts as a high-performance alternative to LIME/SHAP for linear models.

    Args:
        clause: The text string to analyze.
        model_tuple: The (model, vectorizer) loaded from inference.

    Returns:
        Dict mapping critical tokens to their isolated risk contribution weight.
    """
    if not model_tuple or not clause:
        return {}

    try:
        model, vectorizer = model_tuple
        
        # 1. Transform the single clause
        vec = vectorizer.transform([clause])
        
        # 2. Get non-zero features and their indexes
        feature_indices = vec.nonzero()[1]
        feature_names = vectorizer.get_feature_names_out()
        
        # 3. Get the Logistic Regression coefficients (assuming High Risk is index 1 or simply positive)
        # Check if it's binary or multi-class
        if len(model.classes_) == 2:
            coeffs = model.coef_[0]
        else:
            # Multi-class naive assumption - taking max absolute weight across classes
            coeffs = np.max(np.abs(model.coef_), axis=0)

        # 4. Map the words in this specific clause to their global mathematical risk weight
        importance = {}
        for idx in feature_indices:
            word = feature_names[idx]
            weight = coeffs[idx] * vec[0, idx] # TF-IDF term frequency * global model weight
            
            # We only care about words that mathematically PUSH the risk artificially high
            if weight > 0.05:  
                importance[word] = round(float(weight), 3)

        # Sort by impact
        sorted_importance = {k: v for k, v in sorted(importance.items(), key=lambda item: item[1], reverse=True)[:5]}
        return sorted_importance

    except Exception as e:
        logger.error(f"[XAI] Failed to compute feature importance: {e}")
        return {}
