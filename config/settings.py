import os

# --- PATHS ---
# This helps the code know where it is located on your computer.
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Folder for our datasets
DATA_DIR = os.path.join(BASE_DIR, "data")

# Folder for saved model files (like the brain of our AI)
ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")

# --- DATASET ---
# The specific file we are using for training.
DATASET_PATH = os.path.join(DATA_DIR, "legal_docs_modified.csv")

# --- MODEL PATHS ---
# Where we save the "rules" the AI learned.
VECTORIZER_PATH = os.path.join(ARTIFACTS_DIR, "vectorizer.pkl")
MODEL_PATH = os.path.join(ARTIFACTS_DIR, "classifier.pkl")

# --- LABELS ---
# Human-friendly names for our risk levels.
# In our dataset, 0 means Low and 1 means High.
RISK_LABELS = {
    0: "Low Risk",
    1: "High Risk"
}

# Colors for our Streamlit dashboard
RISK_COLORS = {
    "Low Risk": "green",
    "High Risk": "red",
    "Medium Risk": "orange" 
}
