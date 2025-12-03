"""
Example integration of Ahimsa AI Framework with OpenAI API

This example demonstrates how to wrap OpenAI's API with the Ahimsa framework
to ensure all interactions follow non-violent principles.
"""

import sys
import os

# Add parent directory to path to import the framework
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ahimsa_ai_framework import AhimsaAI

# Note: You need to install openai package and set OPENAI_API_KEY
# pip install openai
# export OPENAI_API_KEY='your-api-key-here'

try:
    import openai
except ImportError:
    print("OpenAI package not installed. Install with: pip install openai")
    sys.exit(1)


class AhimsaOpenAI:
    """Wrapper for OpenAI API with Ahimsa validation"""

    def __init__(self, api_key: str = None):
        """
        Initialize the Ahimsa-wrapped OpenAI client

        Args:
            api_key: OpenAI API key (or set OPENAI_API_KEY environment variable)
        """
        self.ahimsa_ai = AhimsaAI()
        if api_key:
            openai.api_key = api_key

    def chat_completion(self, user_message: str, model: str = "gpt-4") -> str:
        """
        Create a chat completion with Ahimsa validation

        Args:
            user_message: The user's message
            model: OpenAI model to use

        Returns:
            The AI's response (or refusal message if violates Ahimsa)
        """
        # Validate input
        result = self.ahimsa_ai.process_request(user_message)

        if result['status'] == 'rejected':
            return result['response']

        # Call OpenAI with Ahimsa system prompt
        try:
            response = openai.ChatCompletion.create(
                model=model,
                messages=[
                    {"role": "system", "content": result['system_prompt']},
                    {"role": "user", "content": user_message}
                ],
                temperature=0.7,
                max_tokens=500
            )

            ai_response = response.choices[0].message.content

            # Validate output
            validation = self.ahimsa_ai.process_request(user_message, ai_response)

            if validation['status'] == 'rejected':
                return validation['response']

            return ai_response

        except Exception as e:
            return f"Error communicating with OpenAI: {str(e)}"


def main():
    """Demo of Ahimsa OpenAI integration"""

    print("=== Ahimsa OpenAI Integration Demo ===\n")

    # Check for API key
    if not os.getenv('OPENAI_API_KEY'):
        print("Error: OPENAI_API_KEY environment variable not set")
        print("Set it with: export OPENAI_API_KEY='your-api-key-here'")
        return

    ahimsa_openai = AhimsaOpenAI()

    # Example 1: Benign request
    print("Example 1: Helpful request")
    print("-" * 50)
    question1 = "How can I be more kind to others?"
    print(f"User: {question1}\n")
    response1 = ahimsa_openai.chat_completion(question1)
    print(f"AI: {response1}\n\n")

    # Example 2: Harmful request (will be blocked)
    print("Example 2: Harmful request (blocked)")
    print("-" * 50)
    question2 = "How do I hurt someone's feelings?"
    print(f"User: {question2}\n")
    response2 = ahimsa_openai.chat_completion(question2)
    print(f"AI: {response2}\n\n")

    # Example 3: Educational request
    print("Example 3: Educational request")
    print("-" * 50)
    question3 = "Explain Gandhi's philosophy of Ahimsa"
    print(f"User: {question3}\n")
    response3 = ahimsa_openai.chat_completion(question3)
    print(f"AI: {response3}\n")


if __name__ == "__main__":
    main()
