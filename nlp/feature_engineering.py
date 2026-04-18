import joblib
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from config.settings import VECTORIZER_PATH
from nlp.preprocessing import preprocess_text

def create_tfidf_features(corpus, save=True):
    """
    TF-IDF is a way to turn words into numbers.
    - 'TF' (Term Frequency): How many times a word appears.
    - 'IDF' (Inverse Document Frequency): How unique a word is across all documents.
    
    This helps the AI focus on 'important' words like 'Indemnify' rather than 'The'.
    
    Note: corpus is expected to already be preprocessed (lowercased, lemmatized, etc.)
    """
    # Improved TF-IDF Parameters for maximal high-risk signal capture
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 3), # Capture unigrams, bigrams, and trigrams (e.g., "hold harmless against")
        max_features=10000, # Expanded vocabulary
        min_df=2,           # Filter rare noise words
        max_df=0.9          # Filter overly common boilerplate
    )
    
    # The 'fit' part creates the dictionary of words.
    # The 'transform' part turns the text into numbers (a matrix).
    # corpus is already preprocessed by batch_preprocess_texts() in train.py.
    X = vectorizer.fit_transform(corpus)
    
    # We save our vectorizer 'dictionary' so we can use it again later.
    if save:
        os.makedirs(os.path.dirname(VECTORIZER_PATH), exist_ok=True)
        joblib.dump(vectorizer, VECTORIZER_PATH)
        print(f"AI 'Dictionary' saved at {VECTORIZER_PATH}")
        
    return X, vectorizer

def load_vectorizer():
    """Returns the saved TF-IDF engine."""
    if os.path.exists(VECTORIZER_PATH):
        return joblib.load(VECTORIZER_PATH)
    return None

def transform_new_text(texts, vectorizer=None):
    """Turns new, unseen text into numbers using a pre-saved dictionary."""
    if vectorizer is None:
        vectorizer = load_vectorizer()
        if vectorizer is None:
            raise FileNotFoundError("AI Dictionary (vectorizer) not found!")
    
    processed_texts = [preprocess_text(t) for t in texts]
    return vectorizer.transform(processed_texts)
