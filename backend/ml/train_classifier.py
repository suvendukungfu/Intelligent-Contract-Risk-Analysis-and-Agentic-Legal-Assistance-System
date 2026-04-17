"""
Complete training pipeline for the Risk Classifier.

This script:
1. Generates or loads training data
2. Extracts features using sentence embeddings
3. Trains the risk classifier
4. Evaluates performance
5. Saves the trained model

Usage:
    python -m backend.ml.train_classifier --generate-data --num-examples 200
"""

import sys
import logging
import argparse
from pathlib import Path
import numpy as np

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ml.generate_synthetic_data import generate_synthetic_dataset
from ml.train import DataPreprocessor, prepare_training_data
from core.feature_extractor import FeatureExtractor
from core.risk_classifier import RiskClassifier

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Main training pipeline."""
    parser = argparse.ArgumentParser(description="Train the risk classification model")
    parser.add_argument(
        "--generate-data",
        action="store_true",
        help="Generate synthetic training data"
    )
    parser.add_argument(
        "--num-examples",
        type=int,
        default=200,
        help="Number of examples per class to generate (default: 200)"
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="backend/data/raw/synthetic_contracts.json",
        help="Path to training data file"
    )
    parser.add_argument(
        "--model-output",
        type=str,
        default="backend/ml/models/risk_classifier_v1.pkl",
        help="Path to save trained model"
    )
    parser.add_argument(
        "--classifier-type",
        type=str,
        default="random_forest",
        choices=["random_forest", "logistic_regression"],
        help="Type of classifier to train"
    )
    
    args = parser.parse_args()
    
    logger.info("=" * 60)
    logger.info("RISK CLASSIFIER TRAINING PIPELINE")
    logger.info("=" * 60)
    
    # Step 1: Generate or load training data
    if args.generate_data:
        logger.info("\n[Step 1/5] Generating synthetic training data...")
        generate_synthetic_dataset(
            num_examples_per_class=args.num_examples,
            output_path=args.data_path
        )
    else:
        logger.info(f"\n[Step 1/5] Using existing data from {args.data_path}")
    
    # Step 2: Load and preprocess data
    logger.info("\n[Step 2/5] Loading and preprocessing data...")
    preprocessor = DataPreprocessor()
    
    # Load data
    examples = preprocessor.load_kaggle_dataset(args.data_path)
    
    if not examples:
        logger.error("No training data found! Use --generate-data to create synthetic data.")
        return
    
    logger.info(f"Loaded {len(examples)} examples")
    
    # Clean data
    examples = preprocessor.clean_data(examples)
    
    # Split data (70% train, 15% val, 15% test)
    train_examples, val_examples, test_examples = preprocessor.split_data(examples)
    
    # Save splits for future use
    preprocessor.save_split_data(train_examples, val_examples, test_examples)
    
    # Step 3: Extract features
    logger.info("\n[Step 3/5] Extracting features using sentence embeddings...")
    feature_extractor = FeatureExtractor(model_name='all-MiniLM-L6-v2')
    
    # Extract training features
    logger.info("Extracting training features...")
    train_texts = [ex.clause_text for ex in train_examples]
    train_labels = [ex.risk_label for ex in train_examples]
    X_train = feature_extractor.batch_extract(train_texts, batch_size=32)
    y_train = np.array(train_labels)
    
    logger.info(f"Training features shape: {X_train.shape}")
    
    # Extract validation features
    logger.info("Extracting validation features...")
    val_texts = [ex.clause_text for ex in val_examples]
    val_labels = [ex.risk_label for ex in val_examples]
    X_val = feature_extractor.batch_extract(val_texts, batch_size=32)
    y_val = np.array(val_labels)
    
    logger.info(f"Validation features shape: {X_val.shape}")
    
    # Extract test features
    logger.info("Extracting test features...")
    test_texts = [ex.clause_text for ex in test_examples]
    test_labels = [ex.risk_label for ex in test_examples]
    X_test = feature_extractor.batch_extract(test_texts, batch_size=32)
    y_test = np.array(test_labels)
    
    logger.info(f"Test features shape: {X_test.shape}")
    
    # Log cache statistics
    cache_stats = feature_extractor.get_cache_stats()
    logger.info(f"Feature extraction cache stats: {cache_stats}")
    
    # Step 4: Train classifier
    logger.info(f"\n[Step 4/5] Training {args.classifier_type} classifier...")
    classifier = RiskClassifier(
        classifier_type=args.classifier_type,
        model_version='v1'
    )
    
    # Train with validation data
    training_metadata = classifier.train(X_train, y_train, X_val, y_val)
    
    logger.info("Training complete!")
    logger.info(f"Training metadata: {training_metadata}")
    
    # Step 5: Evaluate on test set
    logger.info("\n[Step 5/5] Evaluating on test set...")
    
    # Make predictions
    test_predictions = classifier.predict(X_test)
    predicted_labels = [pred.risk_label for pred in test_predictions]
    
    # Calculate accuracy
    correct = sum(1 for pred, true in zip(predicted_labels, y_test) if pred == true)
    test_accuracy = correct / len(y_test)
    
    logger.info(f"Test Accuracy: {test_accuracy:.4f}")
    
    # Calculate per-class accuracy
    from collections import defaultdict
    class_correct = defaultdict(int)
    class_total = defaultdict(int)
    
    for pred, true in zip(predicted_labels, y_test):
        class_total[true] += 1
        if pred == true:
            class_correct[true] += 1
    
    logger.info("\nPer-class accuracy:")
    for label in sorted(class_total.keys()):
        acc = class_correct[label] / class_total[label] if class_total[label] > 0 else 0
        logger.info(f"  {label}: {acc:.4f} ({class_correct[label]}/{class_total[label]})")
    
    # Calculate confusion matrix
    from sklearn.metrics import confusion_matrix, classification_report
    
    labels = ['high_risk', 'medium_risk', 'low_risk', 'no_risk']
    cm = confusion_matrix(y_test, predicted_labels, labels=labels)
    
    logger.info("\nConfusion Matrix:")
    logger.info(f"{'':15} " + " ".join(f"{label:12}" for label in labels))
    for i, label in enumerate(labels):
        logger.info(f"{label:15} " + " ".join(f"{cm[i][j]:12}" for j in range(len(labels))))
    
    # Classification report
    logger.info("\nClassification Report:")
    report = classification_report(y_test, predicted_labels, labels=labels, target_names=labels)
    logger.info(f"\n{report}")
    
    # Save model
    logger.info(f"\nSaving model to {args.model_output}...")
    classifier.save(args.model_output)
    
    logger.info("\n" + "=" * 60)
    logger.info("TRAINING COMPLETE!")
    logger.info("=" * 60)
    logger.info(f"\nModel saved to: {args.model_output}")
    logger.info(f"Test Accuracy: {test_accuracy:.4f}")
    logger.info(f"Training examples: {len(train_examples)}")
    logger.info(f"Validation examples: {len(val_examples)}")
    logger.info(f"Test examples: {len(test_examples)}")
    
    # Check if model meets F1 score requirement (≥ 0.65)
    from sklearn.metrics import f1_score
    f1 = f1_score(y_test, predicted_labels, labels=labels, average='macro')
    logger.info(f"Macro F1 Score: {f1:.4f}")
    
    if f1 >= 0.65:
        logger.info("✓ Model meets F1 score requirement (≥ 0.65)")
    else:
        logger.warning(f"⚠ Model F1 score ({f1:.4f}) is below requirement (0.65)")
        logger.warning("  Consider: increasing training data, tuning hyperparameters, or trying a different classifier")


if __name__ == "__main__":
    main()
