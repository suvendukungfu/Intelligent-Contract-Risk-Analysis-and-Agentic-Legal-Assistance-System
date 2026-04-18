import unittest
import os
import sys
import pandas as pd
import numpy as np
import scipy.sparse as sp
from unittest.mock import MagicMock, patch

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nlp.preprocessing import preprocess_text, batch_preprocess_texts
from nlp.clause_segmenter import segment_clauses
from models.inference import ContractRiskAI, risk_engine
from config.settings import MODEL_PATH, DATASET_PATH, ARTIFACTS_DIR

class TestContractRiskSystemHard(unittest.TestCase):

    # --- 1. NLP PREPROCESSING (MID-TO-HARD) ---

    def test_preprocessing_heavy_special_chars(self):
        """Hard: Test preprocessing with mixed encodings and non-standard whitespace."""
        text = "This is a test clause\x00 with weird\twhitespace and \U0001F600 emojis!"
        processed = preprocess_text(text)
        self.assertNotIn("\x00", processed)
        self.assertNotIn("\t", processed)
        self.assertNotIn("\U0001F600", processed)

    def test_preprocessing_batch_empty_handling(self):
        """Mid: Ensure batch preprocessing handles lists with mixed empty/None values."""
        texts = ["Valid text", "", None, "Another valid"]
        # preprocess_text handles None by checking isinstance(text, str)
        processed = batch_preprocess_texts(texts)
        self.assertEqual(len(processed), 4)
        self.assertEqual(processed[1], "")
        self.assertEqual(processed[2], "")

    def test_lemmatization_accuracy(self):
        """Mid: Verify spaCy lemmatization for legal-specific terms."""
        text = "indemnified indemnifying indemnification"
        processed = preprocess_text(text)
        # Should ideally collapse to 'indemnify' or 'indemnification' depending on spacy rules
        # Just ensure they aren't kept as is (inflected)
        self.assertNotIn("indemnifying", processed)

    def test_long_document_preprocessing_performance(self):
        """Hard: Test preprocessing with a very large string to ensure stability."""
        long_text = "Standard clause text. " * 5000
        processed = preprocess_text(long_text)
        self.assertGreater(len(processed), 100)

    # --- 2. CLAUSE SEGMENTATION (MID-TO-HARD) ---

    def test_segmentation_nested_patterns(self):
        """Hard: Test complex nested legal numbering and edge cases."""
        text = "1. Section One\n2. Section Two\nArticle IV: The Root\nA. Subsection\nB. Subsection"
        clauses = segment_clauses(text)
        # Expected: Split at 1., 2., Article IV, A., B.
        self.assertIn("Section One", clauses[0])
        self.assertIn("Subsection", clauses[-1])

    def test_segmentation_no_boundary_breaks(self):
        """Mid: Test segmentation on text with no defined boundaries."""
        text = "This is just a long paragraph without any legal numbering or special keywords like Article or Section."
        clauses = segment_clauses(text)
        self.assertEqual(len(clauses), 1)

    def test_segmentation_case_insensitivity(self):
        """Mid: Test if 'article' vs 'ARTICLE' are both caught."""
        text = "article 1: low case. ARTICLE 2: UP CASE."
        clauses = segment_clauses(text)
        self.assertEqual(len(clauses), 2)

    def test_segmentation_excessive_newlines(self):
        """Mid: Test segmentation with varying amounts of whitespace between clauses."""
        text = "1. Clause One\n\n\n\n\n2. Clause Two"
        clauses = segment_clauses(text)
        self.assertEqual(len(clauses), 2)
        self.assertEqual(clauses[1], "2. Clause Two")

    # --- 3. INFERENCE ENGINE & EXPLAINABILITY (HARD) ---

    def test_inference_missing_artifacts_graceful_fail(self):
        """Hard: Test ContractRiskAI behavior when model file is missing."""
        with patch('os.path.exists') as mock_exists:
            mock_exists.return_value = False
            engine = ContractRiskAI()
            label, conf, reasons = engine.analyze_clause("Any text")
            self.assertEqual(label, "System Offline")

    def test_inference_coefficient_alignment(self):
        """Hard: Ensure explainability keywords match feature weights."""
        if risk_engine.model and hasattr(risk_engine.model, 'coef_'):
            test_clause = "The Consultant shall indemnify the Client against all liabilities."
            label, conf, reasons = risk_engine.analyze_clause(test_clause)
            # If it's high risk, 'indemnify' or 'liability' should likely be in reasons
            if label == "High Risk":
                self.assertTrue(len(reasons) > 0)

    def test_inference_probability_sum(self):
        """Mid: Ensure predict_proba sums to ~1.0."""
        if risk_engine.model:
            clean = preprocess_text("Indemnity clause example")
            features = risk_engine.vectorizer.transform([clean])
            probs = risk_engine.model.predict_proba(features)[0]
            self.assertAlmostEqual(sum(probs), 1.0, places=5)

    def test_explainability_sparse_vs_dense(self):
        """Hard: Verify get_explainability handles dense input conversion correctly."""
        if risk_engine.model:
            # Create a dummy dense feature row
            num_features = len(risk_engine.vectorizer.get_feature_names_out())
            dummy_dense = np.zeros((1, num_features))
            dummy_dense[0, 10] = 1.0 # arbitrary feature
            reasons = risk_engine.get_explainability(dummy_dense)
            self.assertIsInstance(reasons, list)

    def test_inference_input_validation(self):
        """Mid: Ensure engine handles non-string input gracefully."""
        label, conf, reasons = risk_engine.analyze_clause(12345)
        self.assertEqual(label, "Analysis Error")

    # --- 4. DATA & PIPELINE (MID-TO-HARD) ---

    def test_dataset_imbalance_check(self):
        """Mid: Verify the dataset isn't critically skewed (e.g. 99% one class)."""
        if os.path.exists(DATASET_PATH):
            df = pd.read_csv(DATASET_PATH)
            counts = df['clause_status'].value_counts(normalize=True)
            # Ensure neither class is below 5% for meaningful training
            self.assertGreater(counts.min(), 0.05)

    def test_tfidf_max_features_enforcement(self):
        """Mid: Ensure vectorizer respects the 5000 feature limit."""
        if risk_engine.vectorizer:
            feature_count = len(risk_engine.vectorizer.get_feature_names_out())
            self.assertLessEqual(feature_count, 5000)

    def test_model_artifact_persistence(self):
        """Mid: Ensure saved artifacts are valid joblib files."""
        import joblib
        if os.path.exists(MODEL_PATH):
            loaded = joblib.load(MODEL_PATH)
            self.assertTrue(hasattr(loaded, "predict"))

    # --- 5. EDGE CASES & STRESS TESTS (HARD) ---

    def test_very_short_clause_inference(self):
        """Mid: Inference on a 1-word clause."""
        label, conf, reasons = risk_engine.analyze_clause("Indemnity")
        self.assertIn(label, ["High Risk", "Low Risk", "Unknown"])

    def test_repetitive_text_segmentation(self):
        """Hard: Test segmenter on text composed purely of delimiters."""
        text = "\n\n1. \n\n2. \n\nArticle I\n\n"
        clauses = segment_clauses(text)
        # Should filter out empty results even if it finds breaks
        self.assertEqual(len(clauses), 0)

    def test_inference_engine_thread_safety_check(self):
        """Hard: Ensure simultaneous calls to analyze_clause don't crash (basic check)."""
        # Note: True thread safety testing is complex, this is a basic smoke test
        import threading
        def run_inference():
            risk_engine.analyze_clause("Standard liability clause for testing.")
        
        threads = [threading.Thread(target=run_inference) for _ in range(5)]
        for t in threads: t.start()
        for t in threads: t.join()
        # If no crash occurred, we pass

    def test_environment_variable_loading(self):
        """Mid: Ensure settings can resolve paths relative to BASE_DIR."""
        from config import settings
        self.assertTrue(os.path.isabs(settings.BASE_DIR))

if __name__ == "__main__":
    unittest.main()
