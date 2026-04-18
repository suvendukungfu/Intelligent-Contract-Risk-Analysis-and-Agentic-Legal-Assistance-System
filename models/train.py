import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
from nlp.preprocessing import preprocess_text, batch_preprocess_texts
from nlp.feature_engineering import create_tfidf_features
from config.settings import DATASET_PATH, MODEL_PATH, ARTIFACTS_DIR

def train_system():
    """
    This is the AI Training Lab. 
    Here, our models 'study' thousands of legal examples to learn the 
    patterns of High-Risk language.
    """
    if not os.path.exists(DATASET_PATH):
        print(f"❌ Error: Dataset not found at {DATASET_PATH}")
        return

    # 📚 Phase 1: Ingestion & Data Health
    print("--- 📚 Phase 1: Loading & Cleaning Data ---")
    df = pd.read_csv(DATASET_PATH)
    
    # ML Logic: Drop missing values early to prevent arithmetic errors later.
    df = df.dropna(subset=['clause_text', 'clause_status'])
    
    print(f"AI is cleaning {len(df)} clauses using the spaCy pipeline (batch mode)...")
    df['clean_text'] = batch_preprocess_texts(df['clause_text'])
    
    # 🔢 Phase 2: Feature Engineering (TF-IDF)
    print("--- 🔢 Phase 2: Turning Words into Numbers ---")
    # TF-IDF captures word importance relative to the whole corpus.
    X, vectorizer = create_tfidf_features(df['clean_text'])
    y = df['clause_status']
    
    # Split: 80% for learning (Study), 20% for testing (Exam)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 🤖 Phase 3: Competitive Model Training
    print("--- 🤖 Phase 3: Training Competitive Models ---")
    
    # 1. Logistic Regression: The Reliable Veteran
    # class_weight='balanced' automatically adjusts for if one class (Low Risk) 
    # has more data than another (High Risk).
    lr_model = LogisticRegression(class_weight='balanced', max_iter=1000)
    lr_model.fit(X_train, y_train)
    
    # 2. Decision Tree: The Logical Thinker
    # Follows a flow-chart like decision path (If word 'Liability' > 0.4 -> High Risk)
    dt_model = DecisionTreeClassifier(class_weight='balanced', max_depth=10)
    dt_model.fit(X_train, y_train)
    
    # Save the winners!
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    joblib.dump(lr_model, MODEL_PATH)
    print(f"✅ Primary Brain (Logistic Regression) saved to {MODEL_PATH}")
    
    # 📝 Phase 4: Professional Evaluation
    print("\n--- 📝 Phase 4: Model Benchmarking (The Final Exam) ---")
    
    results = []
    models = [("Logistic Regression", lr_model)] # Focus on primary brain
    
    for name, model in models:
        y_pred = model.predict(X_test)
        
        # Calculate professional AI metrics
        p, r, f, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted', zero_division=0)
        
        results.append({
            "Model": name,
            "Precision (Accuracy)": f"{p:.2%}",
            "Recall (Risk Detection)": f"{r:.2%}",
            "F1-Score": f"{f:.2%}"
        })
        
        # 🧪 Cross-validation to ensure consistency
        from sklearn.model_selection import cross_val_score
        cv_scores = cross_val_score(model, X, y, cv=5)
        print(f"📊 {name} CV Mean Accuracy: {cv_scores.mean():.2%} (+/- {cv_scores.std()*2:.2%})")

        # 📊 NEW: Matrix Heatmap generation
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
        plt.title(f'Confusion Matrix: {name}')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        
        # Save visualization for the portfolio!
        chart_path = os.path.join(ARTIFACTS_DIR, f"{name.lower().replace(' ', '_')}_matrix.png")
        plt.savefig(chart_path)
        plt.close()
        
        # Log metrics to a JSON for the UI to read
        import json
        metrics_log = {
            "precision": p,
            "recall": r,
            "f1": f,
            "cv_mean": cv_scores.mean(),
            "cv_std": cv_scores.std()
        }
        with open(os.path.join(ARTIFACTS_DIR, "metrics.json"), "w") as f_json:
            json.dump(metrics_log, f_json)
        
        print(f"📉 Evaluation Matrix saved for {name}")

    # Display clean table
    eval_df = pd.DataFrame(results)
    print("\n" + eval_df.to_string(index=False))
    
    print("\n💡 Logic Insights:")
    print("- Precision: When the AI flags a risk, how often is it actually a risk?")
    print("- Recall: Out of ALL the real risks, how many did the AI successfully catch?")
    print("- F1-Score: The harmonic mean; our 'final grade' for model balance.")

if __name__ == "__main__":
    train_system()
