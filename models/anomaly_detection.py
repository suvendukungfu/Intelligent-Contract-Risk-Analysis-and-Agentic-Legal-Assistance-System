"""
models/anomaly_detection.py
----------------------------
Unsupervised Semantic Anomaly Detection mapped to Milestone 4.
Detects mathematically abnormal clauses using IsolationForest on Sentence Embeddings.
"""

import numpy as np
import logging
from typing import List, Dict, Any, Tuple
from sklearn.ensemble import IsolationForest

try:
    from sentence_transformers import SentenceTransformer
    # Reusing the existing local model caching
    _semantic_model = SentenceTransformer("all-MiniLM-L6-v2", cache_folder="/tmp/hf_cache")
    USE_ANOMALY = True
except ImportError:
    USE_ANOMALY = False

logger = logging.getLogger(__name__)

def detect_semantic_anomalies(clauses: List[str]) -> List[Dict[str, Any]]:
    """
    Computes density-based anomalies using IsolationForest.
    Useful for catching "Zero-Day" predatory legal clauses that ML models weren't trained on.

    Returns:
        List of dicts containing 'is_anomaly' boolean and semantic 'anomaly_score'.
    """
    if not USE_ANOMALY or not clauses:
        logger.warning("[Anomaly] sentence-transformers missing or no clauses. Skipping.")
        return [{"is_anomaly": False, "anomaly_score": 0.0} for _ in clauses]
    
    if len(clauses) < 3:
        # Too few clauses to cluster meaningfully
        return [{"is_anomaly": False, "anomaly_score": 0.0} for _ in clauses]

    try:
        # Convert clauses to semantic tensors (N, 384)
        embeddings = _semantic_model.encode(clauses, convert_to_numpy=True)
        
        # Fit Isolation Forest (Contamination sets the rough expected anomaly rate)
        # We expect a standard contract to have maybe 10-15% novel weird clauses.
        clf = IsolationForest(n_estimators=100, contamination=0.15, random_state=42)
        
        preds = clf.fit_predict(embeddings)
        scores = clf.score_samples(embeddings) # Lower score indicates mathematically unusual
        
        # Normalize scores to 0-100 range for readability (higher = more anomalous)
        # score_samples are usually negative. E.g., -0.6 (normal) to -1.0 (anomalous)
        normalized_scores = 100 * (scores.min() - scores) / (scores.min() - scores.max())
        
        results = []
        for i, pred in enumerate(preds):
            # pred == -1 means anomaly
            results.append({
                "is_anomaly": bool(pred == -1),
                "anomaly_score": round(float(normalized_scores[i]), 1)
            })
            
        return results

    except Exception as e:
        logger.error(f"[Anomaly] Failed to process embeddings: {e}")
        return [{"is_anomaly": False, "anomaly_score": 0.0} for _ in clauses]
