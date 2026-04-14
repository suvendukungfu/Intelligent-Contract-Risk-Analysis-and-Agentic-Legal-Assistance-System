"""Model evaluation for risk classifier."""

import json
import logging
from pathlib import Path
from typing import Dict, Optional
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)


class ModelEvaluator:
    def __init__(self, target_f1_score: float = 0.65):
        self.target_f1_score = target_f1_score
        self.labels = ['high_risk', 'medium_risk', 'low_risk', 'no_risk']
    
    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray, y_proba: Optional[np.ndarray] = None) -> Dict:
        if len(y_true) == 0:
            raise ValueError("Cannot evaluate on empty dataset")
        
        logger.info(f"Evaluating model on {len(y_true)} samples")
        
        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, labels=self.labels, average=None, zero_division=0
        )
        
        macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
            y_true, y_pred, labels=self.labels, average='macro', zero_division=0
        )
        
        weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
            y_true, y_pred, labels=self.labels, average='weighted', zero_division=0
        )
        
        per_class_metrics = {}
        for i, label in enumerate(self.labels):
            per_class_metrics[label] = {
                'precision': float(precision[i]),
                'recall': float(recall[i]),
                'f1_score': float(f1[i]),
                'support': int(support[i])
            }
        
        cm = confusion_matrix(y_true, y_pred, labels=self.labels)
        meets_target = macro_f1 >= self.target_f1_score
        
        logger.info(f"Macro F1: {macro_f1:.4f} ({'PASS' if meets_target else 'FAIL'})")
        
        return {
            'overall_metrics': {
                'accuracy': float(accuracy),
                'macro_precision': float(macro_precision),
                'macro_recall': float(macro_recall),
                'macro_f1_score': float(macro_f1),
                'weighted_precision': float(weighted_precision),
                'weighted_recall': float(weighted_recall),
                'weighted_f1_score': float(weighted_f1)
            },
            'per_class_metrics': per_class_metrics,
            'confusion_matrix': cm.tolist(),
            'labels': self.labels,
            'n_samples': len(y_true),
            'meets_target_f1': meets_target,
            'target_f1_score': self.target_f1_score
        }
    
    def generate_classification_report(self, y_true: np.ndarray, y_pred: np.ndarray) -> str:
        return classification_report(y_true, y_pred, labels=self.labels, target_names=self.labels, zero_division=0)
    
    def plot_confusion_matrix(self, confusion_matrix: np.ndarray, output_path: Optional[str] = None, normalize: bool = False) -> None:
        cm = np.array(confusion_matrix)
        
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            fmt, title = '.2f', 'Normalized Confusion Matrix'
        else:
            fmt, title = 'd', 'Confusion Matrix'
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues', xticklabels=self.labels, yticklabels=self.labels)
        plt.title(title, fontsize=16, pad=20)
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.tight_layout()
        
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"Confusion matrix saved to {output_path}")
        
        plt.close()
    
    def save_metrics(self, metrics: Dict, output_path: str) -> None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"Metrics saved to {output_path}")


def evaluate_model(classifier, X_test: np.ndarray, y_test: np.ndarray, 
                   output_dir: str = "backend/ml/evaluation", save_plots: bool = True) -> Dict:
    logger.info("Starting model evaluation")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    predictions = classifier.predict(X_test)
    y_pred = np.array([pred.risk_label for pred in predictions])
    
    evaluator = ModelEvaluator(target_f1_score=0.65)
    metrics = evaluator.evaluate(y_test, y_pred)
    
    report = evaluator.generate_classification_report(y_test, y_pred)
    with open(output_dir / "classification_report.txt", 'w') as f:
        f.write(report)
    
    if save_plots:
        evaluator.plot_confusion_matrix(metrics['confusion_matrix'], output_path=output_dir / "confusion_matrix.png")
        evaluator.plot_confusion_matrix(metrics['confusion_matrix'], output_path=output_dir / "confusion_matrix_normalized.png", normalize=True)
    
    evaluator.save_metrics(metrics, output_dir / "metrics.json")
    
    logger.info(f"Evaluation complete. Accuracy: {metrics['overall_metrics']['accuracy']:.4f}, F1: {metrics['overall_metrics']['macro_f1_score']:.4f}")
    return metrics
