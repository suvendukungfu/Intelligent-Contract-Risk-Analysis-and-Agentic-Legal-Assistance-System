import unittest
import os
import sys
import json
import tempfile
import pandas as pd
import numpy as np
from unittest.mock import patch

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nlp.clause_segmenter import segment_clauses
from nlp.preprocessing import preprocess_text
from nlp.feature_engineering import transform_new_text
from models.inference import ContractRiskAI, risk_engine
from config.settings import DATASET_PATH, MODEL_PATH, VECTORIZER_PATH, ARTIFACTS_DIR

SAMPLE_DOC_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "sample_docs", "sample_nda.txt")

class TestE2EHard(unittest.TestCase):

    def test_e2e_sample_doc_inference_flow(self):
        """Hard: Full pipeline on sample NDA -> segments -> inference -> keyword extraction."""
        with open(SAMPLE_DOC_PATH, "r", encoding="utf-8") as f:
            raw_text = f.read()
        clauses = segment_clauses(raw_text)
        self.assertGreaterEqual(len(clauses), 8)

        results = []
        for clause in clauses[:10]:
            label, conf, keywords = risk_engine.analyze_clause(clause)
            results.append((label, conf, keywords))

        self.assertTrue(all(r[0] in ["High Risk", "Low Risk", "Unknown", "System Offline", "Analysis Error"] for r in results))
        self.assertTrue(all(0.0 <= r[1] <= 1.0 for r in results))

    def test_e2e_dataset_to_vectorizer(self):
        """Hard: Ensure corpus from dataset can be vectorized end-to-end."""
        df = pd.read_csv(DATASET_PATH)
        sample_texts = df["clause_text"].dropna().head(20).tolist()
        cleaned = [preprocess_text(t) for t in sample_texts]

        # Check vectorizer usage end-to-end
        features = transform_new_text(cleaned)
        self.assertEqual(features.shape[0], len(cleaned))

    def test_e2e_pipeline_handles_blank_clauses(self):
        """Hard: Ensure system doesn't crash on empty or whitespace-only clauses."""
        clauses = ["", "   ", "\n\n"]
        for c in clauses:
            label, conf, keywords = risk_engine.analyze_clause(c)
            self.assertIn(label, ["High Risk", "Low Risk", "Unknown", "System Offline", "Analysis Error"])

    def test_e2e_model_artifacts_exist(self):
        """Hard: Ensure model artifacts exist for production inference."""
        self.assertTrue(os.path.exists(MODEL_PATH), "Missing model artifact")
        self.assertTrue(os.path.exists(VECTORIZER_PATH), "Missing vectorizer artifact")

    def test_e2e_fallback_when_model_missing(self):
        """Hard: Simulate missing model and ensure safe fallback."""
        with patch("os.path.exists", return_value=False):
            engine = ContractRiskAI()
            label, conf, keywords = engine.analyze_clause("Confidentiality clause example.")
            self.assertEqual(label, "System Offline")
            self.assertEqual(conf, 0.0)

    def test_e2e_confidence_consistency(self):
        """Hard: Confidence must always be within [0,1] for sample NDA."""
        with open(SAMPLE_DOC_PATH, "r", encoding="utf-8") as f:
            raw_text = f.read()
        clauses = segment_clauses(raw_text)
        for clause in clauses[:10]:
            label, conf, _ = risk_engine.analyze_clause(clause)
            self.assertTrue(0.0 <= conf <= 1.0)

    def test_e2e_keywords_topk_nonempty_for_risky(self):
        """Hard: Ensure high-risk predictions include non-empty keywords list."""
        clause = "The Receiving Party shall indemnify, defend, and hold harmless the Disclosing Party."
        label, conf, keywords = risk_engine.analyze_clause(clause)
        if label == "High Risk":
            self.assertTrue(len(keywords) > 0)

    def test_e2e_large_document_scalability(self):
        """Hard: Simulate a very large document with repeated clauses."""
        clause = "The Receiving Party shall indemnify the Disclosing Party."
        big_doc = (clause + "\n\n") * 200
        clauses = segment_clauses(big_doc)
        self.assertGreater(len(clauses), 100)

        # Run inference on a subset to confirm no crash
        for c in clauses[:20]:
            label, conf, _ = risk_engine.analyze_clause(c)
            self.assertIn(label, ["High Risk", "Low Risk", "Unknown", "System Offline", "Analysis Error"])

    def test_e2e_vectorizer_consistency(self):
        """Hard: Ensure vocabulary stays consistent between load and transform."""
        if os.path.exists(VECTORIZER_PATH):
            clean = preprocess_text("This agreement shall be governed by the laws of Delaware.")
            vec_out = transform_new_text([clean])
            self.assertEqual(vec_out.shape[1], len(risk_engine.vectorizer.get_feature_names_out()))

    def test_e2e_metrics_json_schema(self):
        """Hard: Ensure metrics.json follows expected schema if it exists."""
        metrics_path = os.path.join(ARTIFACTS_DIR, "metrics.json")
        if os.path.exists(metrics_path):
            with open(metrics_path, "r") as f:
                m = json.load(f)
            for key in ["precision", "recall", "f1", "cv_mean", "cv_std"]:
                self.assertIn(key, m)

if __name__ == "__main__":
    unittest.main()
