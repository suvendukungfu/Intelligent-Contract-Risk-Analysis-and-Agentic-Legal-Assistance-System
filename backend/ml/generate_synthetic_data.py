"""
Generate synthetic training data for the Contract Risk Analysis System.

This script creates realistic contract clauses with risk labels for training
the ML classifier. Used when real datasets are not available.
"""

import json
import random
from pathlib import Path
from typing import List, Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Template clauses for each risk category
HIGH_RISK_TEMPLATES = [
    "The employee agrees to unlimited liability for any damages caused during employment.",
    "The contractor waives all rights to dispute resolution and agrees to binding arbitration without appeal.",
    "The party agrees to indemnify and hold harmless the company from any and all claims, including those arising from the company's own negligence.",
    "The employee agrees to a non-compete clause preventing work in the same industry for 10 years after termination.",
    "The contractor agrees to transfer all intellectual property rights, including pre-existing works, to the company.",
    "The party agrees to automatic renewal with no option to cancel or terminate the agreement.",
    "The employee waives all rights to overtime compensation regardless of hours worked.",
    "The contractor agrees to pay liquidated damages of $1,000,000 for any breach, regardless of actual harm.",
    "The party agrees that the company may unilaterally modify any terms of this agreement without notice.",
    "The employee agrees to work any hours required by the company without additional compensation.",
    "The contractor agrees to exclusive dealing and may not work with any other clients during the term.",
    "The party agrees to waive all statutory rights and protections under applicable law.",
    "The employee agrees to mandatory arbitration with the company selecting the arbitrator.",
    "The contractor agrees to bear all costs and expenses of the company in any dispute.",
    "The party agrees to unlimited personal guarantee for all company obligations.",
]

MEDIUM_RISK_TEMPLATES = [
    "The employee agrees to a 90-day notice period for resignation.",
    "The contractor shall maintain confidentiality of all company information for 5 years after termination.",
    "The party agrees to a non-solicitation clause preventing contact with company clients for 2 years.",
    "The employee agrees to assignment of inventions created during employment.",
    "The contractor agrees to indemnify the company for third-party claims arising from the contractor's work.",
    "The party agrees to automatic renewal unless written notice is provided 60 days in advance.",
    "The employee agrees to relocation at the company's discretion with 30 days notice.",
    "The contractor agrees to a penalty of 10% of contract value for late delivery.",
    "The party agrees to binding arbitration for dispute resolution.",
    "The employee agrees to a non-compete clause for 1 year within a 50-mile radius.",
    "The contractor agrees to maintain professional liability insurance of at least $1,000,000.",
    "The party agrees to pay attorney's fees and costs if the company prevails in any dispute.",
    "The employee agrees to return all company property within 7 days of termination.",
    "The contractor agrees to a right of first refusal for future projects.",
    "The party agrees to jurisdiction in the company's home state for all disputes.",
]

LOW_RISK_TEMPLATES = [
    "The employee agrees to provide 2 weeks notice before resignation.",
    "The contractor shall maintain confidentiality of proprietary information during the term of the agreement.",
    "The party agrees to use reasonable efforts to perform the services described herein.",
    "The employee agrees to comply with company policies and procedures.",
    "The contractor agrees to deliver work product by the agreed-upon deadlines.",
    "The party agrees to communicate any conflicts of interest to the company.",
    "The employee agrees to participate in required training programs.",
    "The contractor agrees to provide monthly progress reports.",
    "The party agrees to maintain accurate records of work performed.",
    "The employee agrees to use company equipment for business purposes only.",
    "The contractor agrees to obtain company approval before subcontracting work.",
    "The party agrees to comply with all applicable laws and regulations.",
    "The employee agrees to attend scheduled meetings and reviews.",
    "The contractor agrees to provide reasonable notice of any delays.",
    "The party agrees to cooperate in good faith to resolve any disputes.",
]

NO_RISK_TEMPLATES = [
    "This agreement shall be governed by the laws of the State of California.",
    "The parties agree that this document constitutes the entire agreement between them.",
    "This agreement may be executed in counterparts, each of which shall be deemed an original.",
    "The employee shall receive a salary of $X per year, payable bi-weekly.",
    "The contractor shall provide services as described in Exhibit A.",
    "The parties agree to communicate primarily via email and phone.",
    "This agreement shall commence on the date first written above.",
    "The company shall provide the employee with necessary equipment and resources.",
    "The contractor shall invoice the company monthly for services rendered.",
    "The parties acknowledge that they have read and understood this agreement.",
    "This agreement may be amended only by written agreement signed by both parties.",
    "The employee shall be entitled to standard company benefits as described in the employee handbook.",
    "The contractor shall perform services in a professional and workmanlike manner.",
    "The parties agree to maintain open and honest communication throughout the term.",
    "This agreement shall be binding upon and inure to the benefit of the parties and their successors.",
]


def generate_variations(template: str, num_variations: int = 3) -> List[str]:
    """
    Generate variations of a template clause by adding context and modifiers.
    
    Args:
        template: Base template clause
        num_variations: Number of variations to generate
        
    Returns:
        List of clause variations
    """
    variations = [template]
    
    # Prefixes to add context
    prefixes = [
        "Section 1. ",
        "Article 2. ",
        "Clause 3. ",
        "Paragraph 4. ",
        "1. ",
        "2.1 ",
        "",
    ]
    
    # Suffixes to add additional context
    suffixes = [
        " This provision shall survive termination of this agreement.",
        " The parties acknowledge the importance of this provision.",
        " This clause is material to the agreement.",
        "",
        "",
    ]
    
    for _ in range(num_variations - 1):
        prefix = random.choice(prefixes)
        suffix = random.choice(suffixes)
        variation = f"{prefix}{template}{suffix}"
        variations.append(variation)
    
    return variations


def generate_synthetic_dataset(
    num_examples_per_class: int = 100,
    output_path: str = "backend/data/raw/synthetic_contracts.json"
) -> List[Dict]:
    """
    Generate a synthetic dataset of contract clauses with risk labels.
    
    Args:
        num_examples_per_class: Number of examples to generate per risk class
        output_path: Path to save the generated dataset
        
    Returns:
        List of training examples
    """
    logger.info(f"Generating synthetic dataset with {num_examples_per_class} examples per class")
    
    dataset = []
    
    # Contract types for variety
    contract_types = [
        "employment",
        "service_agreement",
        "nda",
        "lease",
        "purchase_agreement",
        "consulting",
        "partnership",
        "license"
    ]
    
    # Jurisdictions for variety
    jurisdictions = [
        "California",
        "New York",
        "Texas",
        "India",
        "UK",
        "Canada"
    ]
    
    # Generate examples for each risk class
    risk_classes = [
        ("high_risk", HIGH_RISK_TEMPLATES),
        ("medium_risk", MEDIUM_RISK_TEMPLATES),
        ("low_risk", LOW_RISK_TEMPLATES),
        ("no_risk", NO_RISK_TEMPLATES)
    ]
    
    for risk_label, templates in risk_classes:
        logger.info(f"Generating {num_examples_per_class} examples for {risk_label}")
        
        examples_generated = 0
        while examples_generated < num_examples_per_class:
            # Select random template
            template = random.choice(templates)
            
            # Generate variations
            variations = generate_variations(template, num_variations=3)
            
            for variation in variations:
                if examples_generated >= num_examples_per_class:
                    break
                
                example = {
                    "clause_text": variation,
                    "risk_label": risk_label,
                    "contract_type": random.choice(contract_types),
                    "jurisdiction": random.choice(jurisdictions)
                }
                
                dataset.append(example)
                examples_generated += 1
    
    # Shuffle dataset
    random.shuffle(dataset)
    
    # Save to file
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Generated {len(dataset)} examples and saved to {output_path}")
    
    # Log statistics
    label_counts = {}
    for example in dataset:
        label = example['risk_label']
        label_counts[label] = label_counts.get(label, 0) + 1
    
    logger.info(f"Label distribution: {label_counts}")
    
    return dataset


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate synthetic training data")
    parser.add_argument(
        "--num-per-class",
        type=int,
        default=100,
        help="Number of examples to generate per risk class"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="backend/data/raw/synthetic_contracts.json",
        help="Output file path"
    )
    
    args = parser.parse_args()
    
    # Generate dataset
    dataset = generate_synthetic_dataset(
        num_examples_per_class=args.num_per_class,
        output_path=args.output
    )
    
    print(f"\n✓ Generated {len(dataset)} synthetic training examples")
    print(f"✓ Saved to {args.output}")
