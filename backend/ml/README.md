# ML Training Pipeline

This directory contains the machine learning training pipeline for the Contract Risk Analysis System.

## 📁 Files

- **`train_classifier.py`** - Complete training pipeline (main script)
- **`train.py`** - Data preprocessing utilities
- **`generate_synthetic_data.py`** - Synthetic data generation
- **`evaluate.py`** - Model evaluation utilities
- **`models/`** - Directory for saved models

## 🚀 Quick Start

### Step 1: Install Dependencies

```bash
cd backend
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### Step 2: Generate Training Data & Train Model

```bash
# Generate 200 examples per class and train
python -m ml.train_classifier --generate-data --num-examples 200

# Or use more data for better accuracy
python -m ml.train_classifier --generate-data --num-examples 500
```

This will:
1. ✅ Generate synthetic training data (800-2000 examples)
2. ✅ Split into train/val/test (70/15/15)
3. ✅ Extract features using sentence embeddings
4. ✅ Train a Random Forest classifier
5. ✅ Evaluate on test set
6. ✅ Save model to `ml/models/risk_classifier_v1.pkl`

### Step 3: Check Results

The script will output:
- Training accuracy
- Validation accuracy
- Test accuracy
- Per-class accuracy
- Confusion matrix
- F1 score (must be ≥ 0.65)

## 📊 Understanding the Output

### Example Output:
```
[Step 1/5] Generating synthetic training data...
Generated 800 examples

[Step 2/5] Loading and preprocessing data...
Loaded 800 examples
Cleaned dataset contains 800 examples
Label distribution: {'high_risk': 200, 'medium_risk': 200, 'low_risk': 200, 'no_risk': 200}

Data split complete:
  Train: 560 examples (70.0%)
  Val:   120 examples (15.0%)
  Test:  120 examples (15.0%)

[Step 3/5] Extracting features using sentence embeddings...
Training features shape: (560, 384)
Validation features shape: (120, 384)
Test features shape: (120, 384)

[Step 4/5] Training random_forest classifier...
Training complete!
Training accuracy: 0.9821
Validation accuracy: 0.8750

[Step 5/5] Evaluating on test set...
Test Accuracy: 0.8667

Per-class accuracy:
  high_risk: 0.9333 (28/30)
  medium_risk: 0.8667 (26/30)
  low_risk: 0.8333 (25/30)
  no_risk: 0.8333 (25/30)

Macro F1 Score: 0.8650
✓ Model meets F1 score requirement (≥ 0.65)
```

## 🎯 What Each Metric Means

### **Accuracy**
- **What it is**: Percentage of correct predictions
- **Example**: 0.8667 = 86.67% correct
- **Good value**: > 0.80 (80%)

### **F1 Score**
- **What it is**: Balance between precision and recall
- **Example**: 0.8650 = Good balance
- **Requirement**: ≥ 0.65 (65%)
- **Why it matters**: Better than accuracy for imbalanced data

### **Confusion Matrix**
Shows where the model makes mistakes:
```
                high_risk    medium_risk  low_risk     no_risk
high_risk       28           2            0            0
medium_risk     1            26           3            0
low_risk        0            2            25           3
no_risk         0            0            2            28
```

**Reading it:**
- Diagonal = Correct predictions
- Off-diagonal = Mistakes
- Example: 2 high_risk clauses were misclassified as medium_risk

## 🔧 Advanced Usage

### Use Different Classifier

```bash
# Logistic Regression (faster, simpler)
python -m ml.train_classifier --generate-data --classifier-type logistic_regression

# Random Forest (slower, more accurate)
python -m ml.train_classifier --generate-data --classifier-type random_forest
```

### Use Your Own Data

```bash
# Prepare your data in JSON format:
# [
#   {"clause_text": "...", "risk_label": "high_risk"},
#   {"clause_text": "...", "risk_label": "low_risk"},
#   ...
# ]

python -m ml.train_classifier --data-path path/to/your/data.json
```

### Custom Model Path

```bash
python -m ml.train_classifier \
  --generate-data \
  --model-output ml/models/my_custom_model.pkl
```

## 📚 How It Works

### 1. **Data Generation**
Creates realistic contract clauses with risk labels:
- **High Risk**: "unlimited liability", "waive all rights"
- **Medium Risk**: "90-day notice", "non-compete 1 year"
- **Low Risk**: "2 weeks notice", "comply with policies"
- **No Risk**: "governed by laws", "entire agreement"

### 2. **Feature Extraction**
Converts text to numbers using **sentence-transformers**:
```
"unlimited liability" → [0.23, -0.45, 0.67, ..., 0.12]  (384 numbers)
```

The model understands semantic meaning:
- Similar clauses → Similar numbers
- Different clauses → Different numbers

### 3. **Training**
The classifier learns patterns:
```
IF embedding contains [high values in dimensions 10, 45, 200]
THEN risk_label = "high_risk"
```

### 4. **Evaluation**
Tests on unseen data to measure real-world performance.

## 🐛 Troubleshooting

### Error: "Model 'en_core_web_sm' not found"
```bash
python -m spacy download en_core_web_sm
```

### Error: "sentence-transformers not installed"
```bash
pip install sentence-transformers
```

### Low F1 Score (< 0.65)
Try:
1. Generate more training data: `--num-examples 500`
2. Use Random Forest: `--classifier-type random_forest`
3. Check data quality in `backend/data/raw/`

### Model file not found in API
Make sure the model is saved to the correct path:
```bash
ls backend/ml/models/risk_classifier_v1.pkl
```

## 📈 Next Steps

After training:
1. ✅ Model is saved to `ml/models/risk_classifier_v1.pkl`
2. ✅ API will automatically load it when you start the server
3. ✅ Test it: `POST /api/v1/analyze/milestone1` with a contract file

## 🎓 Learning Resources

- **Sentence Transformers**: https://www.sbert.net/
- **Random Forest**: https://scikit-learn.org/stable/modules/ensemble.html#forest
- **F1 Score**: https://en.wikipedia.org/wiki/F-score
- **Confusion Matrix**: https://en.wikipedia.org/wiki/Confusion_matrix
