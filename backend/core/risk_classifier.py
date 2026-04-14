"""Risk classification for contract clauses."""

import logging
from pathlib import Path
from typing import List, Optional, Dict, Any
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
import joblib

from api.models import RiskPrediction

logger = logging.getLogger(__name__)


class RiskClassifier:
    def __init__(self, classifier_type: str = 'random_forest', model_version: str = 'v1', random_state: int = 42):
        self.classifier_type = classifier_type
        self.model_version = model_version
        self.random_state = random_state
        
        if classifier_type == 'logistic_regression':
            self.classifier = LogisticRegression(max_iter=1000, random_state=random_state, class_weight='balanced', n_jobs=-1)
        elif classifier_type == 'random_forest':
            self.classifier = RandomForestClassifier(n_estimators=100, max_depth=20, min_samples_split=5, 
                                                    min_samples_leaf=2, random_state=random_state, 
                                                    class_weight='balanced', n_jobs=-1)
        else:
            raise ValueError(f"Unknown classifier type: {classifier_type}")
        
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(['high_risk', 'medium_risk', 'low_risk', 'no_risk'])
        self.is_trained = False
        self.training_metadata: Dict[str, Any] = {}
        
        logger.info(f"Initialized {classifier_type} classifier")
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray, 
              X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None) -> Dict[str, Any]:
        if X_train.shape[0] == 0:
            raise ValueError("Training data is empty")
        if X_train.shape[0] != len(y_train):
            raise ValueError(f"Feature matrix and labels have different lengths")
        
        logger.info(f"Training {self.classifier_type} on {X_train.shape[0]} examples")
        
        y_train_encoded = self.label_encoder.transform(y_train)
        self.classifier.fit(X_train, y_train_encoded)
        
        train_predictions = self.classifier.predict(X_train)
        train_accuracy = np.mean(train_predictions == y_train_encoded)
        
        unique, counts = np.unique(y_train, return_counts=True)
        label_dist = dict(zip(unique, counts))
        
        self.training_metadata = {
            'n_train_samples': X_train.shape[0],
            'n_features': X_train.shape[1],
            'train_accuracy': float(train_accuracy),
            'label_distribution': label_dist,
            'classifier_type': self.classifier_type,
            'model_version': self.model_version
        }
        
        if X_val is not None and y_val is not None:
            y_val_encoded = self.label_encoder.transform(y_val)
            val_predictions = self.classifier.predict(X_val)
            val_accuracy = np.mean(val_predictions == y_val_encoded)
            self.training_metadata['n_val_samples'] = X_val.shape[0]
            self.training_metadata['val_accuracy'] = float(val_accuracy)
            logger.info(f"Validation accuracy: {val_accuracy:.4f}")
        
        self.is_trained = True
        logger.info(f"Training complete: accuracy {train_accuracy:.4f}")
        return self.training_metadata
    
    def predict(self, features: np.ndarray) -> List[RiskPrediction]:
        if not self.is_trained:
            raise RuntimeError("Classifier must be trained before making predictions")
        if features.shape[0] == 0:
            return []
        
        predictions_encoded = self.classifier.predict(features)
        probabilities = self.classifier.predict_proba(features)
        predictions = self.label_encoder.inverse_transform(predictions_encoded)
        
        results = []
        for i, (pred, probs) in enumerate(zip(predictions, probabilities)):
            confidence = float(probs[predictions_encoded[i]])
            results.append(RiskPrediction(
                clause_id=f"clause-{i}",
                risk_label=pred,
                confidence=confidence,
                model_version=self.model_version
            ))
        
        return results
    
    def predict_single(self, features: np.ndarray) -> RiskPrediction:
        if features.ndim == 1:
            features = features.reshape(1, -1)
        return self.predict(features)[0]
    
    def save(self, model_path: str) -> None:
        if not self.is_trained:
            raise RuntimeError("Cannot save untrained classifier")
        
        model_path = Path(model_path)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        
        model_data = {
            'classifier': self.classifier,
            'label_encoder': self.label_encoder,
            'classifier_type': self.classifier_type,
            'model_version': self.model_version,
            'training_metadata': self.training_metadata,
            'is_trained': self.is_trained
        }
        
        joblib.dump(model_data, model_path)
        logger.info(f"Model saved to {model_path}")
    
    @classmethod
    def load(cls, model_path: str) -> 'RiskClassifier':
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        logger.info(f"Loading model from {model_path}")
        model_data = joblib.load(model_path)
        
        instance = cls(
            classifier_type=model_data['classifier_type'],
            model_version=model_data['model_version']
        )
        
        instance.classifier = model_data['classifier']
        instance.label_encoder = model_data['label_encoder']
        instance.training_metadata = model_data['training_metadata']
        instance.is_trained = model_data['is_trained']
        
        logger.info(f"Loaded {instance.classifier_type} model")
        return instance


_risk_classifier_instance: Optional[RiskClassifier] = None

def get_risk_classifier(model_path: Optional[str] = None, classifier_type: str = 'random_forest') -> RiskClassifier:
    global _risk_classifier_instance
    if _risk_classifier_instance is None:
        if model_path and Path(model_path).exists():
            _risk_classifier_instance = RiskClassifier.load(model_path)
        else:
            _risk_classifier_instance = RiskClassifier(classifier_type=classifier_type)
    return _risk_classifier_instance
