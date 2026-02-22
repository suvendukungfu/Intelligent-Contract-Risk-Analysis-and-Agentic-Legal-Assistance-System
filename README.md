# Intelligent Contract Risk Analysis System

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/deploy?repository=suvendukungfu/Intelligent-Contract-Risk-Analysis-and-Agentic-Legal-Assistance-System&branch=main&main_module_path=app/streamlit_app.py)

> **A clause-level analysis workspace designed to support structured contract review using classical Natural Language Processing.**

---

## Platform Thesis

The **Intelligent Contract Risk Analysis System** is a research-backed analytical workspace designed to automate the initial triage of legal instruments. By implementing a supervised learning pipeline, the system decomposes complex contracts into discrete provision segments, assesses potential liability risks, and extracts semantic triggers. This platform serves as a high-fidelity prototype for AI-assisted legal review, prioritizing explainable model assessments over "black-box" predictions.

## Methodology & Core Architecture

The system follows a rigorous alignment between legal problem-solving and machine learning execution:

1.  **Decomposition (Structural Segmentation)**: PDF and TXT documents are parsed into independent linguistic segments, maintaining the semantic context of legal clauses.
2.  **Featurization (Vector Space Model)**: Clauses are transformed using **TF-IDF (Term Frequency-Inverse Document Frequency)**, prioritizing distinguishing legal keywords.
3.  **Classification (Balanced Inference)**: A **Logistic Regression** model, optimized with balanced class weights, identifies provisions matching high-risk patterns.
4.  **Explainability (Trigger Extraction)**: The system extracts local feature importance to display "Linguistic Triggers"—the specific words that most influenced the risk assessment.

## Key Capabilities

- **Risk Analysis Workspace**: A dedicated review environment where high-severity segments are prioritized for manual oversight.
- **Micro-Insight Generation**: Each assessment includes a structural summary of the detected pattern (e.g., "Elevated liability profile identified").
- **Strategic Posture Radar**: High-level distribution analysis of risk patterns across the entire document corpus.
- **Model Governance Center**: Transparent reporting of validation metrics (Heatmaps, Precision/Recall) to ensure analytical integrity.

## Technical Implementation

- **Natural Language Processing**: `spaCy` (L3 Pipeline) for lemmatization and text normalization.
- **Machine Learning**: `scikit-learn` (Logistic Regression & TF-IDF).
- **Interface**: `Streamlit` (V7 Enterprise Architecture).
- **Data Handling**: `pdfplumber` for high-precision legal text extraction.

## Evaluation Metrics

The system prioritizes **Recall** to ensure maximum sensitivity to potential risks:

- **Baseline Accuracy**: ~94% (Validated on CUAD-inspired datasets)
- **Primary Objective**: Minimizing False Negatives in critical legal disclosure.

## Responsible AI Protocol

This system is an analytical aid designed for legal professionals. It highlights linguistic patterns associated with risk and is intended to complement, not replace, human professional judgment. All detections are categorized as "Model Assessments" requiring final validation by a qualified reviewer.

---

_Developed as a high-tier academic submission for intelligent legal-tech research._

### Model Specification

- **Vectorization**: TF-IDF (Term Frequency-Inverse Document Frequency)
  - _N-gram Range_: (1, 2)
  - _Max Features_: 5,000
- **Classifier**: Logistic Regression (Primary) with balanced class weights.
- **Performance Metrics**:
  - **Precision**: Minimized false positives for risk identification.
  - **Recall**: Optimized to ensure critical clauses are flagged.
  - **F1-Score**: Weighted average showing robust performance across 20+ clause types.

## The Core Science Behind the AI

This project isn't just a dashboard; it's a demonstration of fundamental AI/ML and NLP principles. Here's a look under the hood:

### 1. Artificial Intelligence vs. Machine Learning

- **Artificial Intelligence (AI)** is the broad field of creating systems that can perform tasks requiring human intelligence.
- **Machine Learning (ML)** is a subset of AI that focuses on building systems that _learn_ from data rather than following strict, manually coded instructions.

> [!NOTE]
> Instead of writing a million "if-else" statements for legal words, we give the model examples and let it discover the patterns itself.

### 2. Supervised Learning & Classification

This is a **Supervised Learning** project. We "supervise" the AI by giving it a "labeled" dataset (e.g., this sentence = High Risk).

- **The Task**: This is a **Binary Classification** problem. The AI must decide if a clause belongs to one of two classes: `0` (Low Risk) or `1` (High Risk).

### 3. Logistic Regression & The Sigmoid Function

Our primary "brain" is **Logistic Regression**.

- **How it works**: It calculates a weighted sum of the words in a clause to determine a probability.
- **The Math**: It uses the **Sigmoid Function** to squash that probability between 0 and 1. If the score is > 0.5, it's flagged as High Risk.

### 4. Natural Language Processing (NLP) Foundations

To help a computer "read" legalese, we use three key techniques:

- **Tokenization**: Breaking sentences into individual words (tokens).
- **Lemmatization**: Reducing words to their root form (e.g., "indemnifying" -> "indemnify"). This helps the AI treat different versions of a word as the same concept.
- **TF-IDF (The Secret Sauce)**: **Term Frequency-Inverse Document Frequency**. It highlights words that are unique and meaningful (like "Liability") while ignoring common filler words (like "the").

---

## 📁 Project Structure

```bash
contract-risk-ai/
├── data/                    # Dataset storage (legal_docs_modified.csv)
├── analysis/                # EDA scripts and eda.ipynb
├── nlp/                     # Core NLP logic
│   ├── preprocessing.py     # spaCy lemmatization & cleaning
│   ├── clause_segmenter.py  # Regex-based segmentation
│   └── feature_engineering.py # TF-IDF vectorization
├── models/                  # ML Pipeline
│   ├── train.py             # Model training script
│   ├── evaluate.py          # Metrics & Visualization
│   └── inference.py         # Real-time prediction pipeline
├── artifacts/               # Serialized models (classifier.pkl)
├── app/                     # UI Layer
│   └── streamlit_app.py     # Streamlit application
├── config/                  # Configuration & Global settings
└── requirements.txt         # Project dependencies
```

## Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/suvendukungfu/Intelligent-Contract-Risk-Analysis-and-Agentic-Legal-Assistance-System.git
cd contract-risk-ai
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 3. Training the Model

Generate the model artifacts by running the training script:

```bash
export PYTHONPATH=$PYTHONPATH:.
python3 models/train.py
```

### 4. Running the Dashboard

Launch the Streamlit interface locally:

```bash
streamlit run app/streamlit_app.py
```

## Public Deployment

Live Demo: [Deployment Link Placeholder]

## Exploratory Data Analysis (EDA)

Comprehensive analysis of the legal corpus is available in `analysis/eda.ipynb`. Key insights include:

- **High Class Imbalance**: Addressed using balanced class weights in the ML model.
- **Keyword Prominence**: TF-IDF reveals high-weight terms like "Termination", "Salary", and "Liability".
- **Structural Variation**: Large variance in clause length across different legal categories.

## Future Scope (Milestone 2 - Agentic AI)

- **Retrieval-Augmented Generation (RAG)**: Connect clauses to external legal knowledge bases.
  **Multi-Agent Negotiation**: Simulate document revisions between "Owner" and "Vendor" agents.
- **Semantic Consistency**: Use LLMs to check if "Definitions" match their usage throughout the document.

## Legal Disclaimer

_This system is an AI research project intended for informational purposes only. It does not constitute legal advice. The classifications and risk scores are generated based on mathematical patterns in the training data and should be reviewed by a qualified legal professional._

---

**Maintained by**: [Suvendu kumar Sahoo](https://github.com/suvendukungfu)
