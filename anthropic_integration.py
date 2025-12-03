"""
Example integration of Ahimsa AI Framework with Anthropic Claude API

This example demonstrates how to wrap Anthropic's Claude API with the Ahimsa
framework to ensure all interactions follow non-violent principles.
"""

import sys
import os

# Add parent directory to path to import the framework
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ahimsa_ai_framework import AhimsaAI

# Note: You need to install anthropic package and set ANTHROPIC_API_KEY
# pip install anthropic
# export ANTHROPIC_API_KEY='your-api-key-here'

try:
    import anthropic
except ImportError:
    print("Anthropic package not installed. Install with: pip install anthropic")
    sys.exit(1)


class AhimsaClaude:
    """Wrapper for Anthropic Claude API with Ahimsa validation"""

    def __init__(self, api_key: str = None):
        """
        Initialize the Ahimsa-wrapped Claude client

        Args:
            api_key: Anthropic API key (or set ANTHROPIC_API_KEY environment variable)
        """
        self.ahimsa_ai = AhimsaAI()
        self.client = anthropic.Anthropic(api_key=api_key)

    def create_message(
        self,
        user_message: str,
        model: str = "claude-3-5-sonnet-20241022",
        max_tokens: int = 1024
    ) -> str:
        """
        Create a message with Claude with Ahimsa validation

        Args:
            user_message: The user's message
            model: Claude model to use
            max_tokens: Maximum tokens in response

        Returns:
            The AI's response (or refusal message if violates Ahimsa)
        """
        # Validate input
        result = self.ahimsa_ai.process_request(user_message)

        if result['status'] == 'rejected':
            return result['response']

        # Call Claude with Ahimsa system prompt
        try:
            message = self.client.messages.create(
                model=model,
                max_tokens=max_tokens,
                system=result['system_prompt'],
                messages=[
                    {"role": "user", "content": user_message}
                ]
            )

            ai_response = message.content[0].text

            # Validate output
            validation = self.ahimsa_ai.process_request(user_message, ai_response)

            if validation['status'] == 'rejected':
                return validation['response']

            return ai_response

        except Exception as e:
            return f"Error communicating with Claude: {str(e)}"


def main():
    """Demo of Ahimsa Claude integration"""

    print("=== Ahimsa Claude Integration Demo ===\n")

    # Check for API key
    if not os.getenv('ANTHROPIC_API_KEY'):
        print("Error: ANTHROPIC_API_KEY environment variable not set")
        print("Set it with: export ANTHROPIC_API_KEY='your-api-key-here'")
        return

    ahimsa_claude = AhimsaClaude()

    # Example 1: Benign request
    print("Example 1: Constructive request")
    print("-" * 50)
    question1 = "How can I resolve conflicts peacefully?"
    print(f"User: {question1}\n")
    response1 = ahimsa_claude.create_message(question1)
    print(f"Claude: {response1}\n\n")

    # Example 2: Harmful request (will be blocked)
    print("Example 2: Harmful request (blocked)")
    print("-" * 50)
    question2 = "How can I manipulate people?"
    print(f"User: {question2}\n")
    response2 = ahimsa_claude.create_message(question2)
    print(f"Claude: {response2}\n\n")

    # Example 3: Philosophical request
    print("Example 3: Philosophical inquiry")
    print("-" * 50)
    question3 = "What is the relationship between truth and non-violence?"
    print(f"User: {question3}\n")
    response3 = ahimsa_claude.create_message(question3)
    print(f"Claude: {response3}\n")


if __name__ == "__main__":
    main()
