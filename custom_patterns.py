"""
Example: Adding Custom Harmful Patterns to Ahimsa AI Framework

This example shows how to extend the framework with your own
pattern definitions for domain-specific harmful content.
"""

import sys
import os

# Add parent directory to path to import the framework
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ahimsa_ai_framework import AhimsaAI, ViolationLevel


def add_financial_harm_patterns(ahimsa_ai: AhimsaAI):
    """Add patterns to detect financial scams and exploitation"""

    ahimsa_ai.validator.harmful_patterns['financial_harm'] = {
        'patterns': [
            r'\b(?:scam|fraud|ponzi|pyramid)\s+scheme\b',
            r'\bget\s+rich\s+quick\b',
            r'\b(?:steal|embezzle)\s+(?:money|funds)\b',
            r'\bhow\s+to\s+(?:cheat|deceive)\s+(?:investors|customers)\b',
        ],
        'level': ViolationLevel.HIGH
    }


def add_privacy_violation_patterns(ahimsa_ai: AhimsaAI):
    """Add patterns to detect privacy violations"""

    ahimsa_ai.validator.harmful_patterns['privacy_violation'] = {
        'patterns': [
            r'\b(?:hack|steal|breach)\s+(?:data|information|password)\b',
            r'\bspy\s+on\s+(?:someone|people)\b',
            r'\btrack\s+someone\s+without\b',
            r'\bdoxing\b',
        ],
        'level': ViolationLevel.CRITICAL
    }


def add_misinformation_patterns(ahimsa_ai: AhimsaAI):
    """Add patterns to detect deliberate misinformation"""

    ahimsa_ai.validator.harmful_patterns['misinformation'] = {
        'patterns': [
            r'\bcreate\s+(?:fake|false)\s+(?:news|information)\b',
            r'\bspread\s+(?:lies|misinformation|propaganda)\b',
            r'\bdeceive\s+(?:public|people|audience)\b',
        ],
        'level': ViolationLevel.HIGH
    }


def add_child_safety_patterns(ahimsa_ai: AhimsaAI):
    """Add patterns to protect children"""

    ahimsa_ai.validator.harmful_patterns['child_safety'] = {
        'patterns': [
            r'\b(?:harm|exploit|abuse)\s+(?:child|children|minor)\b',
            r'\bunsafe\s+(?:for|to)\s+(?:child|children)\b',
        ],
        'level': ViolationLevel.CRITICAL
    }


def main():
    """Demo of custom pattern detection"""

    print("=== Custom Pattern Detection Demo ===\n")

    # Initialize Ahimsa AI
    ahimsa_ai = AhimsaAI()

    # Add custom patterns
    add_financial_harm_patterns(ahimsa_ai)
    add_privacy_violation_patterns(ahimsa_ai)
    add_misinformation_patterns(ahimsa_ai)
    add_child_safety_patterns(ahimsa_ai)

    print("Added custom pattern categories:")
    print("- Financial harm detection")
    print("- Privacy violation detection")
    print("- Misinformation detection")
    print("- Child safety detection\n")
    print("=" * 60 + "\n")

    # Test cases
    test_cases = [
        "How can I build a legitimate business?",
        "How do I create a ponzi scheme?",
        "What are best practices for data security?",
        "How can I hack someone's password?",
        "How do I fact-check information online?",
        "How can I spread fake news effectively?",
    ]

    for i, test_case in enumerate(test_cases, 1):
        print(f"Test {i}: {test_case}")
        print("-" * 60)

        result = ahimsa_ai.process_request(test_case)

        if result['status'] == 'rejected':
            print(f"Status: BLOCKED")
            violations = result.get('violations', [])
            if violations:
                print(f"Reason: {violations[0].category.replace('_', ' ').title()}")
                print(f"Level: {violations[0].level.name}")
        else:
            print(f"Status: APPROVED")

        print()


if __name__ == "__main__":
    main()
