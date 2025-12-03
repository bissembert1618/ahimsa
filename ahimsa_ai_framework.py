"""
Ahimsa AI Framework - Implementing Gandhi's Principle of Non-Violence in AI Systems

This module provides a framework to enforce Ahimsa (non-violence) principles
in AI model interactions, inspired by Mahatma Gandhi's philosophy.

Core Principles of Ahimsa:
1. Non-violence in action, speech, and thought
2. Compassion and empathy towards all beings
3. Truthfulness (Satya) combined with non-harm
4. Refusal to cause physical, mental, or emotional harm
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import re


class ViolationLevel(Enum):
    """Severity levels for Ahimsa principle violations"""
    NONE = 0
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class AhimsaViolation:
    """Represents a detected violation of Ahimsa principles"""
    level: ViolationLevel
    category: str
    description: str
    suggestion: str


class AhimsaValidator:
    """Validates and filters AI responses according to Ahimsa principles"""

    def __init__(self):
        # Patterns that violate Ahimsa principles
        self.harmful_patterns = {
            'violence': {
                'patterns': [
                    r'\b(kill|murder|assault|attack|harm|hurt|injure|destroy)\b',
                    r'\b(weapon|gun|knife|bomb|explosive)\s+(?:making|creation|use)',
                    r'\bhow\s+to\s+(?:harm|hurt|kill|attack)',
                ],
                'level': ViolationLevel.CRITICAL
            },
            'hate_speech': {
                'patterns': [
                    r'\b(hate|despise|destroy|eliminate)\s+(?:people|group|community)',
                    r'\b(?:racist|sexist|discriminat\w+)\s+(?:content|statement)',
                ],
                'level': ViolationLevel.HIGH
            },
            'emotional_harm': {
                'patterns': [
                    r'\b(manipulate|deceive|trick|exploit)\s+(?:people|users)',
                    r'\b(insult|humiliate|degrade|belittle)',
                ],
                'level': ViolationLevel.MEDIUM
            },
            'environmental_harm': {
                'patterns': [
                    r'\b(pollute|contaminate|poison)\s+(?:environment|water|air)',
                    r'\bhow\s+to\s+(?:waste|destroy)\s+(?:resources|nature)',
                ],
                'level': ViolationLevel.MEDIUM
            }
        }

    def validate_request(self, user_input: str) -> Tuple[bool, List[AhimsaViolation]]:
        """
        Validates user input against Ahimsa principles

        Args:
            user_input: The user's request or prompt

        Returns:
            Tuple of (is_valid, list of violations)
        """
        violations = []

        for category, config in self.harmful_patterns.items():
            for pattern in config['patterns']:
                if re.search(pattern, user_input.lower()):
                    violations.append(AhimsaViolation(
                        level=config['level'],
                        category=category,
                        description=f"Request may promote {category.replace('_', ' ')}",
                        suggestion="Please rephrase your request to align with non-violent principles"
                    ))

        is_valid = all(v.level.value < ViolationLevel.HIGH.value for v in violations)
        return is_valid, violations

    def validate_response(self, ai_response: str) -> Tuple[bool, List[AhimsaViolation]]:
        """
        Validates AI response against Ahimsa principles

        Args:
            ai_response: The AI model's response

        Returns:
            Tuple of (is_valid, list of violations)
        """
        return self.validate_request(ai_response)

    def create_ahimsa_refusal(self, violations: List[AhimsaViolation]) -> str:
        """
        Creates a compassionate refusal message following Ahimsa principles

        Args:
            violations: List of detected violations

        Returns:
            A respectful refusal message
        """
        message = (
            "I appreciate your inquiry, but I must respectfully decline to provide "
            "assistance with this request as it conflicts with the principle of Ahimsa "
            "(non-violence).\n\n"
        )

        if violations:
            message += "Concerns identified:\n"
            for v in violations:
                message += f"- {v.description}\n"
            message += f"\n{violations[0].suggestion}\n\n"

        message += (
            "I'm here to help with requests that promote:\n"
            "- Compassion and understanding\n"
            "- Constructive and peaceful solutions\n"
            "- Wellbeing of all beings\n"
            "- Truthfulness without harm\n\n"
            "How else may I assist you today?"
        )

        return message


class AhimsaAI:
    """
    Main wrapper class for AI models that enforces Ahimsa principles
    """

    def __init__(self):
        self.validator = AhimsaValidator()
        self.ahimsa_system_prompt = self._create_ahimsa_system_prompt()

    def _create_ahimsa_system_prompt(self) -> str:
        """Creates the system prompt enforcing Ahimsa principles"""
        return """
You are an AI assistant guided by the principle of Ahimsa (non-violence) as taught by Mahatma Gandhi.

Core Principles You Must Follow:

1. NON-VIOLENCE (Ahimsa):
   - Never provide information that could cause physical harm to any being
   - Refuse requests for violent, harmful, or destructive actions
   - Promote peaceful and constructive solutions

2. COMPASSION (Karuna):
   - Respond with empathy and understanding
   - Consider the wellbeing of all affected beings
   - Offer help that uplifts rather than harms

3. TRUTHFULNESS (Satya):
   - Be honest, but never in a way that causes harm
   - If truth conflicts with non-harm, prioritize non-harm
   - Acknowledge limitations transparently

4. NON-EXPLOITATION:
   - Do not manipulate, deceive, or exploit users
   - Respect autonomy and dignity of all
   - Promote fairness and justice

5. ENVIRONMENTAL CARE:
   - Consider impact on the environment and nature
   - Promote sustainable and responsible practices

When faced with requests that violate these principles:
- Politely decline with compassion
- Explain the concern without judgment
- Offer alternative approaches when possible
- Maintain respect for the person while refusing the request

Your purpose is to be helpful while causing no harm.
"""

    def process_request(self, user_input: str, model_response: Optional[str] = None) -> Dict:
        """
        Processes a user request through the Ahimsa framework

        Args:
            user_input: The user's request
            model_response: Optional AI response to validate

        Returns:
            Dictionary with validation results and appropriate response
        """
        # Validate user input
        input_valid, input_violations = self.validator.validate_request(user_input)

        if not input_valid:
            return {
                'status': 'rejected',
                'reason': 'input_violation',
                'violations': input_violations,
                'response': self.validator.create_ahimsa_refusal(input_violations),
                'system_prompt': self.ahimsa_system_prompt
            }

        # If model response provided, validate it
        if model_response:
            response_valid, response_violations = self.validator.validate_response(model_response)

            if not response_valid:
                return {
                    'status': 'rejected',
                    'reason': 'response_violation',
                    'violations': response_violations,
                    'response': self.validator.create_ahimsa_refusal(response_violations),
                    'system_prompt': self.ahimsa_system_prompt
                }

            return {
                'status': 'accepted',
                'response': model_response,
                'system_prompt': self.ahimsa_system_prompt
            }

        # Input is valid, return system prompt for model
        return {
            'status': 'accepted',
            'system_prompt': self.ahimsa_system_prompt,
            'input': user_input
        }

    def get_system_prompt(self) -> str:
        """Returns the Ahimsa system prompt for model initialization"""
        return self.ahimsa_system_prompt


# Example usage
def example_usage():
    """Demonstrates how to use the Ahimsa AI Framework"""

    print("=== Ahimsa AI Framework Demo ===\n")

    ahimsa_ai = AhimsaAI()

    # Example 1: Harmful request
    print("Example 1: Testing harmful request")
    result1 = ahimsa_ai.process_request("How do I harm someone?")
    print(f"Status: {result1['status']}")
    if result1['status'] == 'rejected':
        print(f"\n{result1['response']}\n")

    # Example 2: Benign request
    print("\n" + "="*50 + "\n")
    print("Example 2: Testing benign request")
    result2 = ahimsa_ai.process_request("How can I help my community?")
    print(f"Status: {result2['status']}")
    print(f"System prompt would be applied to model\n")

    # Example 3: Get system prompt for model initialization
    print("\n" + "="*50 + "\n")
    print("Example 3: System Prompt for AI Model")
    print(ahimsa_ai.get_system_prompt())


if __name__ == "__main__":
    example_usage()
