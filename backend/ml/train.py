"""
Training data preprocessing pipeline for the Contract Risk Analysis System.

This module handles:
- Loading training data from various sources (Kaggle, Indian Kanoon)
- Data cleaning and validation
- Train/validation/test splitting (70/15/15)
- Handling missing or malformed entries

Requirements: 15.1, 15.2, 15.3, 15.4, 15.5
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
from sklearn.model_selection import train_test_split
import sys

# Add backend to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from api.models import TrainingExample

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataPreprocessor:
    """Handles loading, cleaning, and splitting of training data."""
    
    def __init__(self, data_dir: str = "backend/data/training_data"):
        """
        Initialize the data preprocessor.
        
        Args:
            data_dir: Directory containing training data files
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Valid risk labels
        self.valid_labels = {"high_risk", "medium_risk", "low_risk", "no_risk"}
        
    def load_kaggle_dataset(self, file_path: str) -> List[TrainingExample]:
        """
        Load training data from Kaggle legal dataset.
        
        Expected format: JSON or JSONL with fields:
        - clause_text: str
        - risk_label: str
        - contract_type: Optional[str]
        - jurisdiction: Optional[str]
        
        Args:
            file_path: Path to Kaggle dataset file
            
        Returns:
            List of TrainingExample objects
            
        Requirements: 15.1
        """
        examples = []
        file_path = Path(file_path)
        
        if not file_path.exists():
            logger.warning(f"Kaggle dataset not found at {file_path}")
            return examples
        
        try:
            # Handle both JSON and JSONL formats
            if file_path.suffix == '.jsonl':
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line_num, line in enumerate(f, 1):
                        try:
                            data = json.loads(line.strip())
                            example = self._parse_training_example(data, line_num)
                            if example:
                                examples.append(example)
                        except json.JSONDecodeError as e:
                            logger.warning(f"Skipping malformed JSON at line {line_num}: {e}")
                        except Exception as e:
                            logger.warning(f"Error processing line {line_num}: {e}")
            else:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        for idx, item in enumerate(data):
                            example = self._parse_training_example(item, idx)
                            if example:
                                examples.append(example)
                    else:
                        logger.error(f"Expected list in JSON file, got {type(data)}")
            
            logger.info(f"Loaded {len(examples)} examples from Kaggle dataset")
            
        except Exception as e:
            logger.error(f"Error loading Kaggle dataset: {e}")
        
        return examples
    
    def load_indian_kanoon_dataset(self, file_path: str) -> List[TrainingExample]:
        """
        Load training data from Indian Kanoon legal documents.
        
        Expected format: Similar to Kaggle dataset
        
        Args:
            file_path: Path to Indian Kanoon dataset file
            
        Returns:
            List of TrainingExample objects
            
        Requirements: 15.2
        """
        # Use same loading logic as Kaggle dataset
        examples = self.load_kaggle_dataset(file_path)
        logger.info(f"Loaded {len(examples)} examples from Indian Kanoon dataset")
        return examples
    
    def _parse_training_example(
        self, 
        data: Dict, 
        index: int
    ) -> Optional[TrainingExample]:
        """
        Parse and validate a single training example.
        
        Handles missing or malformed entries gracefully.
        
        Args:
            data: Dictionary containing training data
            index: Index or line number for logging
            
        Returns:
            TrainingExample if valid, None otherwise
            
        Requirements: 15.5
        """
        try:
            # Extract required fields
            clause_text = data.get('clause_text', '').strip()
            risk_label = data.get('risk_label', '').strip().lower()
            
            # Validate required fields
            if not clause_text:
                logger.warning(f"Skipping entry {index}: empty clause_text")
                return None
            
            if len(clause_text) < 10:
                logger.warning(f"Skipping entry {index}: clause_text too short ({len(clause_text)} chars)")
                return None
            
            if risk_label not in self.valid_labels:
                logger.warning(
                    f"Skipping entry {index}: invalid risk_label '{risk_label}'. "
                    f"Must be one of {self.valid_labels}"
                )
                return None
            
            # Extract optional fields
            contract_type = data.get('contract_type')
            jurisdiction = data.get('jurisdiction')
            
            # Create TrainingExample
            return TrainingExample(
                clause_text=clause_text,
                risk_label=risk_label,
                contract_type=contract_type,
                jurisdiction=jurisdiction
            )
            
        except Exception as e:
            logger.warning(f"Error parsing entry {index}: {e}")
            return None
    
    def clean_data(self, examples: List[TrainingExample]) -> List[TrainingExample]:
        """
        Clean and validate training data.
        
        - Remove duplicates
        - Normalize text
        - Validate labels
        
        Args:
            examples: List of training examples
            
        Returns:
            Cleaned list of training examples
            
        Requirements: 15.3
        """
        logger.info(f"Cleaning {len(examples)} examples...")
        
        # Remove duplicates based on clause text
        seen_texts = set()
        cleaned = []
        
        for example in examples:
            # Normalize text for duplicate detection
            normalized_text = ' '.join(example.clause_text.split()).lower()
            
            if normalized_text not in seen_texts:
                seen_texts.add(normalized_text)
                cleaned.append(example)
            else:
                logger.debug(f"Removing duplicate: {example.clause_text[:50]}...")
        
        logger.info(f"Removed {len(examples) - len(cleaned)} duplicates")
        logger.info(f"Cleaned dataset contains {len(cleaned)} examples")
        
        # Log label distribution
        label_counts = {}
        for example in cleaned:
            label_counts[example.risk_label] = label_counts.get(example.risk_label, 0) + 1
        
        logger.info(f"Label distribution: {label_counts}")
        
        return cleaned
    
    def split_data(
        self, 
        examples: List[TrainingExample],
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        random_state: int = 42
    ) -> Tuple[List[TrainingExample], List[TrainingExample], List[TrainingExample]]:
        """
        Split data into train, validation, and test sets.
        
        Default split: 70% train, 15% validation, 15% test
        Uses stratified splitting to maintain label distribution.
        
        Args:
            examples: List of training examples
            train_ratio: Proportion for training set
            val_ratio: Proportion for validation set
            test_ratio: Proportion for test set
            random_state: Random seed for reproducibility
            
        Returns:
            Tuple of (train_examples, val_examples, test_examples)
            
        Requirements: 15.4
        """
        # Validate ratios
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
            "Ratios must sum to 1.0"
        
        if len(examples) == 0:
            logger.warning("No examples to split")
            return [], [], []
        
        # Extract texts and labels
        texts = [ex.clause_text for ex in examples]
        labels = [ex.risk_label for ex in examples]
        
        # First split: separate test set
        train_val_texts, test_texts, train_val_labels, test_labels, train_val_indices, test_indices = \
            train_test_split(
                texts, 
                labels, 
                range(len(examples)),
                test_size=test_ratio,
                random_state=random_state,
                stratify=labels
            )
        
        # Second split: separate train and validation
        val_size_adjusted = val_ratio / (train_ratio + val_ratio)
        train_texts, val_texts, train_labels, val_labels, train_indices, val_indices = \
            train_test_split(
                train_val_texts,
                train_val_labels,
                train_val_indices,
                test_size=val_size_adjusted,
                random_state=random_state,
                stratify=train_val_labels
            )
        
        # Reconstruct TrainingExample objects
        train_examples = [examples[i] for i in train_indices]
        val_examples = [examples[i] for i in val_indices]
        test_examples = [examples[i] for i in test_indices]
        
        # Log split statistics
        logger.info(f"Data split complete:")
        logger.info(f"  Train: {len(train_examples)} examples ({len(train_examples)/len(examples)*100:.1f}%)")
        logger.info(f"  Val:   {len(val_examples)} examples ({len(val_examples)/len(examples)*100:.1f}%)")
        logger.info(f"  Test:  {len(test_examples)} examples ({len(test_examples)/len(examples)*100:.1f}%)")
        
        # Verify no overlap (Property 13: Data Split Non-Overlap)
        train_set = set(train_indices)
        val_set = set(val_indices)
        test_set = set(test_indices)
        
        assert len(train_set & val_set) == 0, "Train and validation sets overlap"
        assert len(train_set & test_set) == 0, "Train and test sets overlap"
        assert len(val_set & test_set) == 0, "Validation and test sets overlap"
        assert len(train_set | val_set | test_set) == len(examples), \
            "Union of splits does not equal original dataset"
        
        logger.info("Verified: No overlap between splits, union equals original dataset")
        
        return train_examples, val_examples, test_examples
    
    def save_split_data(
        self,
        train_examples: List[TrainingExample],
        val_examples: List[TrainingExample],
        test_examples: List[TrainingExample],
        output_dir: Optional[str] = None
    ) -> None:
        """
        Save split datasets to JSON files.
        
        Args:
            train_examples: Training examples
            val_examples: Validation examples
            test_examples: Test examples
            output_dir: Output directory (defaults to self.data_dir)
        """
        output_dir = Path(output_dir) if output_dir else self.data_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        
        datasets = {
            'train': train_examples,
            'val': val_examples,
            'test': test_examples
        }
        
        for name, examples in datasets.items():
            output_path = output_dir / f"{name}.json"
            data = [ex.model_dump() for ex in examples]
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Saved {len(examples)} examples to {output_path}")
    
    def load_split_data(
        self,
        split: str = 'train',
        data_dir: Optional[str] = None
    ) -> List[TrainingExample]:
        """
        Load previously saved split data.
        
        Args:
            split: Which split to load ('train', 'val', or 'test')
            data_dir: Data directory (defaults to self.data_dir)
            
        Returns:
            List of TrainingExample objects
        """
        data_dir = Path(data_dir) if data_dir else self.data_dir
        file_path = data_dir / f"{split}.json"
        
        if not file_path.exists():
            logger.warning(f"Split file not found: {file_path}")
            return []
        
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        examples = [TrainingExample(**item) for item in data]
        logger.info(f"Loaded {len(examples)} examples from {file_path}")
        
        return examples


def prepare_training_data(
    kaggle_path: Optional[str] = None,
    indian_kanoon_path: Optional[str] = None,
    output_dir: str = "backend/data/training_data"
) -> Tuple[List[TrainingExample], List[TrainingExample], List[TrainingExample]]:
    """
    Main function to prepare training data from all sources.
    
    This function:
    1. Loads data from Kaggle and Indian Kanoon datasets
    2. Cleans and validates the data
    3. Splits into train/val/test sets (70/15/15)
    4. Saves the splits to disk
    
    Args:
        kaggle_path: Path to Kaggle dataset file
        indian_kanoon_path: Path to Indian Kanoon dataset file
        output_dir: Directory to save processed data
        
    Returns:
        Tuple of (train_examples, val_examples, test_examples)
        
    Example:
        >>> train, val, test = prepare_training_data(
        ...     kaggle_path="data/kaggle_contracts.jsonl",
        ...     indian_kanoon_path="data/indian_kanoon.json"
        ... )
    """
    preprocessor = DataPreprocessor(data_dir=output_dir)
    
    # Load data from all sources
    all_examples = []
    
    if kaggle_path:
        kaggle_examples = preprocessor.load_kaggle_dataset(kaggle_path)
        all_examples.extend(kaggle_examples)
    
    if indian_kanoon_path:
        indian_kanoon_examples = preprocessor.load_indian_kanoon_dataset(indian_kanoon_path)
        all_examples.extend(indian_kanoon_examples)
    
    if not all_examples:
        logger.error("No training data loaded. Please provide valid dataset paths.")
        return [], [], []
    
    logger.info(f"Total examples loaded: {len(all_examples)}")
    
    # Clean data
    cleaned_examples = preprocessor.clean_data(all_examples)
    
    if len(cleaned_examples) < 10:
        logger.error(f"Insufficient training data: only {len(cleaned_examples)} examples")
        return [], [], []
    
    # Split data
    train_examples, val_examples, test_examples = preprocessor.split_data(cleaned_examples)
    
    # Save splits
    preprocessor.save_split_data(train_examples, val_examples, test_examples)
    
    return train_examples, val_examples, test_examples


if __name__ == "__main__":
    """
    Example usage for preparing training data.
    
    To use this script:
    1. Download datasets and place them in backend/data/raw/
    2. Run: python -m backend.ml.train
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Prepare training data for risk classification")
    parser.add_argument(
        "--kaggle",
        type=str,
        help="Path to Kaggle legal dataset"
    )
    parser.add_argument(
        "--indian-kanoon",
        type=str,
        help="Path to Indian Kanoon dataset"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="backend/data/training_data",
        help="Output directory for processed data"
    )
    
    args = parser.parse_args()
    
    # Prepare training data
    train, val, test = prepare_training_data(
        kaggle_path=args.kaggle,
        indian_kanoon_path=args.indian_kanoon,
        output_dir=args.output
    )
    
    print(f"\nTraining data preparation complete!")
    print(f"Train: {len(train)} examples")
    print(f"Val: {len(val)} examples")
    print(f"Test: {len(test)} examples")
