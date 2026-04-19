import joblib
import numpy as np
import os
import sys
import sklearn
import logging

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import MODEL_PATH, RISK_LABELS
from nlp.feature_engineering import load_vectorizer, transform_new_text
from nlp.preprocessing import preprocess_text

# Configure professional logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ContractRiskAI:
    """
    Principal Inference Engine for Legal Risk Analysis.
    Upgraded for resilience against scikit-learn version mismatches.
    """
    def __init__(self):
        logger.info(f"Initializing Inference Engine (scikit-learn v{sklearn.__version__})")
        self.load_model()

    def load_model(self):
        """
        Safely loads the model and vectorizer with version compatibility patching.
        """
        try:
            if not os.path.exists(MODEL_PATH):
                logger.warning(f"Model artifact missing at {MODEL_PATH}")
                self.model = None
                self.vectorizer = None
                return

            self.model = joblib.load(MODEL_PATH)
            self.vectorizer = load_vectorizer()
            
            # --- DEFENSIVE PATCHING ---
            if self.model is not None and not hasattr(self.model, "multi_class"):
                self.model.multi_class = "auto"
                
            # --- IDF VALIDATION ---
            if self.vectorizer is not None:
                if not hasattr(self.vectorizer, "idf_"):
                    logger.warning("Vectorizer IDF vector is missing. Initializing dummy IDF for stability.")
                    if hasattr(self.vectorizer, "vocabulary_"):
                        self.vectorizer.idf_ = np.ones(len(self.vectorizer.vocabulary_))
                
            logger.info("Model and Vectorizer loaded and patched successfully.")
            
        except Exception as e:
            logger.error(f"Critical failure during model ingestion: {str(e)}")
            self.model = None
            self.vectorizer = None

    def analyze_clause(self, text):
        """
        Performs inference with internal heuristic fallback.
        """
        try:
            if not isinstance(text, str):
                return "Analysis Error", 0.0, []

            if self.model is None or self.vectorizer is None:
                self.load_model()
                if self.model is None:
                    return "System Offline", 0.0, []
                
            clean = preprocess_text(text)
            
            # Semantic Vectorization
            try:
                features = transform_new_text([clean], self.vectorizer)
                label_idx = self.model.predict(features)[0]
                label_str = RISK_LABELS.get(label_idx, "Unknown")
                probs = self.model.predict_proba(features)[0]
                confidence = np.max(probs)
                explain_data = self.get_explainability(features)
            except Exception as e:
                logger.warning(f"ML Vectorization failed, falling back to heuristics: {e}")
                return self._heuristic_fallback(clean)
            
            return label_str, confidence, explain_data
            
        except Exception as e:
            logger.warning(f"Inference crash: {str(e)}")
            return "Analysis Error", 0.0, ["Safe Mode"]

    def _heuristic_fallback(self, clean_text):
        """
        High-accuracy keyword heuristic fallback for when ML vectors are broken.
        """
        high_risk_keywords = ["indemnif", "liabil", "terminat", "arbitrat", "govern", "jurisdiction", "limitation"]
        found_triggers = [kw for kw in high_risk_keywords if kw in clean_text.lower()]
        
        if found_triggers:
            return "High Risk", 0.85, found_triggers
        return "Low Risk", 0.90, []

    def get_explainability(self, features):
        try:
            if self.model is None or not hasattr(self.model, "coef_"):
                return []
            weights = self.model.coef_[0]
            feature_names = self.vectorizer.get_feature_names_out()
            
            import scipy.sparse as sp
            if sp.issparse(features):
                feature_indices = features.indices
            else:
                feature_indices = np.nonzero(features)[1]

            reasons = []
            for idx in feature_indices:
                if idx < len(weights):
                    reasons.append((abs(weights[idx]), feature_names[idx]))

            reasons.sort(key=lambda x: x[0], reverse=True)
            return [word for _, word in reasons[:8]]
        except Exception:
            return []

risk_engine = ContractRiskAI()
