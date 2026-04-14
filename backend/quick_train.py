"""Quick training script for testing purposes."""

import json
import logging
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split

from core.feature_extractor import FeatureExtractor
from core.risk_classifier import RiskClassifier

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load sample data
data_file = Path("data/raw/sample_training_data.jsonl")
texts = []
labels = []

with open(data_file, 'r') as f:
    for line in f:
        item = json.loads(line)
        texts.append(item['clause_text'])
        labels.append(item['risk_label'])

logger.info(f"Loaded {len(texts)} training examples")

# Split data
X_train_texts, X_test_texts, y_train, y_test = train_test_split(
    texts, labels, test_size=0.2, random_state=42, stratify=labels
)

logger.info(f"Train: {len(X_train_texts)}, Test: {len(X_test_texts)}")

# Extract features
logger.info("Extracting features...")
feature_extractor = FeatureExtractor()
X_train = feature_extractor.batch_extract(X_train_texts, batch_size=8)
X_test = feature_extractor.batch_extract(X_test_texts, batch_size=8)

logger.info(f"Feature shape: {X_train.shape}")

# Train classifier
logger.info("Training classifier...")
classifier = RiskClassifier(classifier_type='logistic_regression', model_version='v1')
y_train_array = np.array(y_train)
y_test_array = np.array(y_test)

classifier.train(X_train, y_train_array, X_test, y_test_array)

# Save model
model_dir = Path("ml/models")
model_dir.mkdir(parents=True, exist_ok=True)
model_path = model_dir / "risk_classifier.pkl"

logger.info(f"Saving model to {model_path}")
classifier.save(str(model_path))

logger.info("Training complete!")
