import unittest
import os
import sys
import pandas as pd
import numpy as np

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nlp.preprocessing import preprocess_text, batch_preprocess_texts
from nlp.clause_segmenter import segment_clauses
from models.inference import risk_engine
from config.settings import MODEL_PATH, DATASET_PATH

class TestContractRiskSystem(unittest.TestCase):
    
    # --- NLP Component Tests ---
    
    def test_preprocessing_single(self):
        """Test if single text preprocessing works and removes stopwords/punctuation."""
        test_text = "The parties are currently agreeing to the terms!"
        processed = preprocess_text(test_text)
        self.assertIsInstance(processed, str)
        # 'The' and 'the' are stopwords, '!' is punctuation
        self.assertNotIn("The", processed)
        self.assertNotIn("!", processed)
        # Check if lemmatization or at least cleaning happened
        self.assertTrue(len(processed) > 0)

    def test_preprocessing_batch(self):
        """Test if batch preprocessing yields correct list length."""
        texts = ["Agreement one.", "Clause two with details."]
        processed = batch_preprocess_texts(texts)
        self.assertEqual(len(processed), 2)
        self.assertIsInstance(processed[0], str)

    def test_clause_segmentation(self):
        """Test if regex-based segmentation splits text correctly."""
        text = "1. First section text.\n\n2. Second section text.\nArticle III: Third section."
        clauses = segment_clauses(text)
        # Should find at least 3 clauses
        self.assertGreaterEqual(len(clauses), 3)
        self.assertIn("First section", clauses[0])
        self.assertIn("Second section", clauses[1])
        self.assertIn("Article III", clauses[2])

    def test_segmentation_empty(self):
        """Test handling of empty or tiny strings."""
        self.assertEqual(segment_clauses(""), [])
        self.assertEqual(segment_clauses("Short"), []) # Threshold is 10 chars

    # --- Inference Engine Tests ---

    def test_inference_engine_load(self):
        """Check if model and vectorizer are loaded correctly."""
        # Ensure model exists before testing load
        if os.path.exists(MODEL_PATH):
            risk_engine.load_model()
            self.assertIsNotNone(risk_engine.model)
            self.assertIsNotNone(risk_engine.vectorizer)

    def test_analyze_clause(self):
        """Test the end-to-end analysis of a single clause."""
        if os.path.exists(MODEL_PATH):
            test_clause = "The Consultant shall indemnify the Client against all liabilities and claims."
            label, confidence, keywords = risk_engine.analyze_clause(test_clause)
            
            self.assertIn(label, ["High Risk", "Low Risk", "Unknown"])
            self.assertGreaterEqual(confidence, 0.0)
            self.assertLessEqual(confidence, 1.0)
            self.assertIsInstance(keywords, list)

    # --- Data Integrity Tests ---

    def test_dataset_exists(self):
        """Ensure the training dataset is present."""
        self.assertTrue(os.path.exists(DATASET_PATH), f"Dataset missing at {DATASET_PATH}")

    def test_dataset_columns(self):
        """Check if dataset has required columns."""
        if os.path.exists(DATASET_PATH):
            df = pd.read_csv(DATASET_PATH, nrows=5)
            self.assertIn('clause_text', df.columns)
            self.assertIn('clause_status', df.columns)

if __name__ == "__main__":
    unittest.main()
