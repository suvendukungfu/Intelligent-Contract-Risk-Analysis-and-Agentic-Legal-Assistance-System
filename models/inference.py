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
    
    SENIOR ENG NOTE: This class implements defensive patching for scikit-learn 
    LogisticRegression objects. Traditional unpickling (joblib/pickle) can fail 
    when the environment version differs from the training version, often resulting 
    in missing attributes like 'multi_class'.
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
            # LogisticRegression changed internal attribute names across versions.
            # We ensure 'multi_class' exists to prevent scikit-learn's internal 
            # predict() from crashing on older/newer pickled models.
            if self.model is not None and not hasattr(self.model, "multi_class"):
                logger.info("Patching missing 'multi_class' attribute for compatibility.")
                self.model.multi_class = "auto"
                
            logger.info("Model and Vectorizer loaded successfully.")
            
        except Exception as e:
            logger.error(f"Critical failure during model ingestion: {str(e)}")
            self.model = None
            self.vectorizer = None

    def analyze_clause(self, text):
        """
        Performs inference with fallback handling. 
        Returns (label, confidence, reasoning).
        """
        try:
            if self.model is None or self.vectorizer is None:
                self.load_model()
                if self.model is None:
                    return "System Offline", 0.0, []
                
            # 1. Linguistic Preprocessing
            clean = preprocess_text(text)
            
            # 2. Semantic Vectorization
            features = transform_new_text([clean], self.vectorizer)
            
            # 3. Model Inference
            # SENIOR ENG NOTE: We use predict() and predict_proba() only.
            # Attributes like multi_class are used internally by scikit-learn; 
            # we should never access them directly in production logic.
            label_idx = self.model.predict(features)[0]
            label_str = RISK_LABELS.get(label_idx, "Unknown")
            
            # 4. Confidence Score (Probability)
            probs = self.model.predict_proba(features)[0]
            confidence = np.max(probs)
            
            # 5. Explainability Synthesis
            explain_data = self.get_explainability(features)
            
            return label_str, confidence, explain_data
            
        except Exception as e:
            logger.warning(f"Inference interrupted for clause segment: {str(e)}")
            # Return neutral fallback to maintain UI stability
            return "Analysis Error", 0.0, ["Safe Mode Engaged"]

    def get_explainability(self, features):
        """
        Extracts feature importance for local interpretability.
        Works for both sparse (TF-IDF) and dense feature matrices.
        """
        try:
            if self.model is None or not hasattr(self.model, "coef_"):
                return []

            weights      = self.model.coef_[0]
            feature_names = self.vectorizer.get_feature_names_out()

            # Support both sparse matrices (hasattr .indices) and dense arrays
            import scipy.sparse as sp
            if sp.issparse(features):
                feature_indices = features.indices
            else:
                feature_indices = np.nonzero(features)[1]

            reasons = []
            for idx in feature_indices:
                if idx < len(weights):
                    weight = weights[idx]
                    if weight > 0:
                        reasons.append((weight, feature_names[idx]))

            reasons.sort(key=lambda x: x[0], reverse=True)
            return [word for _, word in reasons[:5]]
        except Exception:
            return []

# Singleton instance for platform-wide access
risk_engine = ContractRiskAI()
