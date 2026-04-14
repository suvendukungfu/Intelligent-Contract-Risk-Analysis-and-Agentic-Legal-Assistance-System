"""Training pipeline for risk classifier."""

import argparse
import logging
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.feature_extractor import FeatureExtractor
from core.risk_classifier import RiskClassifier
from ml.train import DataPreprocessor
from ml.evaluate import evaluate_model

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def train_risk_classifier(data_dir: str = "backend/data/training_data",
                          model_output_path: str = "backend/ml/models/risk_classifier.pkl",
                          classifier_type: str = "random_forest",
                          evaluation_output_dir: str = "backend/ml/evaluation") -> None:
    
    logger.info("Starting training pipeline")
    
    preprocessor = DataPreprocessor(data_dir=data_dir)
    train_examples = preprocessor.load_split_data('train')
    val_examples = preprocessor.load_split_data('val')
    test_examples = preprocessor.load_split_data('test')
    
    if not train_examples:
        logger.error("No training data found. Run data preparation first.")
        return
    
    logger.info(f"Loaded {len(train_examples)} train, {len(val_examples)} val, {len(test_examples)} test examples")
    
    feature_extractor = FeatureExtractor()
    
    train_texts = [ex.clause_text for ex in train_examples]
    train_labels = np.array([ex.risk_label for ex in train_examples])
    X_train = feature_extractor.batch_extract(train_texts, batch_size=32)
    
    X_val, val_labels = None, None
    if val_examples:
        val_texts = [ex.clause_text for ex in val_examples]
        val_labels = np.array([ex.risk_label for ex in val_examples])
        X_val = feature_extractor.batch_extract(val_texts, batch_size=32)
    
    X_test, test_labels = None, None
    if test_examples:
        test_texts = [ex.clause_text for ex in test_examples]
        test_labels = np.array([ex.risk_label for ex in test_examples])
        X_test = feature_extractor.batch_extract(test_texts, batch_size=32)
    
    logger.info(f"Training {classifier_type} classifier")
    classifier = RiskClassifier(classifier_type=classifier_type, model_version='v1')
    classifier.train(X_train, train_labels, X_val, val_labels)
    
    if X_test is not None and test_labels is not None:
        logger.info("Evaluating on test set")
        metrics = evaluate_model(classifier, X_test, test_labels, output_dir=evaluation_output_dir, save_plots=True)
        
        if metrics['meets_target_f1']:
            logger.info("✓ Model meets F1 score requirement (≥ 0.65)")
        else:
            logger.warning(f"✗ Model F1 score {metrics['overall_metrics']['macro_f1_score']:.4f} below target 0.65")
    
    logger.info(f"Saving model to {model_output_path}")
    classifier.save(model_output_path)
    logger.info("Training pipeline complete")


def main():
    parser = argparse.ArgumentParser(description="Train risk classifier")
    parser.add_argument('--data-dir', type=str, default='backend/data/training_data')
    parser.add_argument('--model-output', type=str, default='backend/ml/models/risk_classifier.pkl')
    parser.add_argument('--classifier-type', type=str, choices=['logistic_regression', 'random_forest'], default='random_forest')
    parser.add_argument('--eval-output', type=str, default='backend/ml/evaluation')
    args = parser.parse_args()
    
    train_risk_classifier(
        data_dir=args.data_dir,
        model_output_path=args.model_output,
        classifier_type=args.classifier_type,
        evaluation_output_dir=args.eval_output
    )


if __name__ == "__main__":
    main()
