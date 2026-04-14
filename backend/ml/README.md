# ML Module

Machine learning components for contract risk classification.

## Quick Start

### 1. Prepare Data
```bash
python3 -m backend.ml.train \
    --kaggle backend/data/raw/sample_training_data.jsonl \
    --output backend/data/training_data
```

### 2. Train Classifier
```bash
python3 -m backend.ml.train_classifier \
    --data-dir backend/data/training_data \
    --model-output backend/ml/models/risk_classifier.pkl
```

### 3. Use Model
```python
from core.risk_classifier import RiskClassifier
from core.feature_extractor import FeatureExtractor

classifier = RiskClassifier.load('backend/ml/models/risk_classifier.pkl')
extractor = FeatureExtractor()

features = extractor.extract_single("Contract clause text")
prediction = classifier.predict_single(features)
print(f"{prediction.risk_label} ({prediction.confidence:.2f})")
```

## Files

- `train.py` - Data preprocessing
- `train_classifier.py` - Training pipeline
- `evaluate.py` - Model evaluation
- `models/` - Saved models
- `evaluation/` - Metrics and plots

## Requirements

- F1 score ≥ 0.65
- Risk labels: {high_risk, medium_risk, low_risk, no_risk}
- Confidence scores: [0.0, 1.0]
