"""
Anthropic Claude Integration with Ahimsa AI Framework v2.0

Production-ready integration that wraps Claude API with multi-layer
Ahimsa validation for both inputs and outputs.
"""

import os
import sys
import logging
from typing import Optional, Dict, Any, List

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('ahimsa.anthropic')

# Import framework
try:
    from ahimsa_ai_framework import AhimsaAI
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from ahimsa_ai_framework import AhimsaAI

# Import Anthropic
try:
    import anthropic
except ImportError:
    logger.error("Anthropic package not installed. Install with: pip install anthropic")
    anthropic = None


class AhimsaClaude:
    """
    Production-ready wrapper for Anthropic Claude API with Ahimsa validation.
    
    Features:
    - Multi-layer input validation (keyword + semantic + optional LLM judge)
    - Output validation to catch harmful responses
    - Comprehensive logging and audit trail
    - Graceful error handling
    - Configurable validation layers
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        default_model: str = "claude-3-5-sonnet-20241022",
        enable_semantic: bool = True,
        enable_llm_judge: bool = False,
        custom_system_prompt_additions: Optional[str] = None
    ):
        """
        Initialize the Ahimsa-wrapped Claude client.
        
        Args:
            api_key: Anthropic API key (or set ANTHROPIC_API_KEY env var)
            default_model: Default Claude model to use
            enable_semantic: Enable semantic similarity validation
            enable_llm_judge: Enable LLM-as-a-judge for thorough validation
            custom_system_prompt_additions: Custom additions to Ahimsa system prompt
        """
        if anthropic is None:
            raise ImportError("anthropic package required. Install with: pip install anthropic")
        
        self.api_key = api_key or os.getenv('ANTHROPIC_API_KEY')
        if not self.api_key:
            raise ValueError(
                "Anthropic API key required. Set ANTHROPIC_API_KEY environment variable "
                "or pass api_key parameter."
            )
        
        self.default_model = default_model
        self.client = anthropic.Anthropic(api_key=self.api_key)
        
        # Initialize Ahimsa framework
        self.ahimsa = AhimsaAI(
            enable_semantic=enable_semantic,
            enable_external_api=False,
            enable_llm_judge=enable_llm_judge,
            llm_judge_provider="anthropic",
            llm_judge_model="claude-3-haiku-20240307",
            custom_system_prompt_additions=custom_system_prompt_additions
        )
        
        logger.info(f"AhimsaClaude initialized with model: {default_model}")
    
    def create_message(
        self,
        user_message: str,
        model: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: float = 1.0,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        validate_output: bool = True
    ) -> Dict[str, Any]:
        """
        Create a message with Claude, with Ahimsa validation on both input and output.
        
        Args:
            user_message: The user's message
            model: Claude model to use (defaults to self.default_model)
            max_tokens: Maximum tokens in response
            temperature: Sampling temperature
            conversation_history: Previous messages for context
            validate_output: Whether to validate Claude's response
        
        Returns:
            Dictionary containing:
            - status: 'success', 'input_rejected', 'output_rejected', or 'error'
            - response: The response text (or refusal message)
            - validation: Validation details
            - usage: Token usage stats (if successful)
        """
        model = model or self.default_model
        
        # Step 1: Validate input
        input_result = self.ahimsa.process_request(user_message)
        
        if input_result['status'] == 'rejected':
            logger.warning(f"Input rejected: {user_message[:50]}...")
            return {
                'status': 'input_rejected',
                'response': input_result['response'],
                'validation': input_result['validation_result'],
                'usage': None
            }
        
        # Step 2: Build messages array
        messages = []
        if conversation_history:
            messages.extend(conversation_history)
        messages.append({"role": "user", "content": user_message})
        
        # Step 3: Call Claude
        try:
            response = self.client.messages.create(
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                system=self.ahimsa.get_system_prompt(),
                messages=messages
            )
            
            ai_response = response.content[0].text
            usage = {
                'input_tokens': response.usage.input_tokens,
                'output_tokens': response.usage.output_tokens
            }
            
        except anthropic.APIError as e:
            logger.error(f"Anthropic API error: {e}")
            return {
                'status': 'error',
                'response': "I encountered an error processing your request. Please try again.",
                'validation': None,
                'usage': None,
                'error': str(e)
            }
        
        # Step 4: Validate output (optional but recommended)
        if validate_output:
            output_result = self.ahimsa.process_request(user_message, ai_response)
            
            if output_result['status'] == 'rejected':
                logger.warning(f"Output rejected for input: {user_message[:50]}...")
                return {
                    'status': 'output_rejected',
                    'response': output_result['response'],
                    'validation': output_result['validation_result'],
                    'usage': usage,
                    'original_response': ai_response
                }
        
        # Success
        return {
            'status': 'success',
            'response': ai_response,
            'validation': input_result['validation_result'],
            'usage': usage
        }
    
    def chat(
        self,
        user_message: str,
        model: Optional[str] = None,
        max_tokens: int = 1024
    ) -> str:
        """
        Simple chat interface that returns just the response string.
        
        Args:
            user_message: The user's message
            model: Claude model to use
            max_tokens: Maximum tokens
        
        Returns:
            Response text (or refusal message)
        """
        result = self.create_message(
            user_message=user_message,
            model=model,
            max_tokens=max_tokens
        )
        return result['response']
    
    def add_harmful_example(self, category: str, example: str):
        """Add a harmful example for semantic matching (learning from feedback)"""
        self.ahimsa.add_harmful_example(category, example)


class AhimsaClaudeConversation:
    """
    Manages multi-turn conversations with Ahimsa validation.
    """
    
    def __init__(self, client: AhimsaClaude):
        """
        Args:
            client: AhimsaClaude instance
        """
        self.client = client
        self.history: List[Dict[str, str]] = []
    
    def send(self, message: str) -> Dict[str, Any]:
        """Send a message and get a response, maintaining conversation history"""
        result = self.client.create_message(
            user_message=message,
            conversation_history=self.history
        )
        
        # Only add to history if successful
        if result['status'] == 'success':
            self.history.append({"role": "user", "content": message})
            self.history.append({"role": "assistant", "content": result['response']})
        
        return result
    
    def reset(self):
        """Clear conversation history"""
        self.history = []
    
    def get_history(self) -> List[Dict[str, str]]:
        """Get current conversation history"""
        return self.history.copy()


# =============================================================================
# DEMO
# =============================================================================

def main():
    """Demo of Ahimsa Claude integration"""
    
    print("=" * 60)
    print("AHIMSA CLAUDE INTEGRATION DEMO")
    print("=" * 60)
    
    # Check for API key
    if not os.getenv('ANTHROPIC_API_KEY'):
        print("\n⚠️  ANTHROPIC_API_KEY not set")
        print("Set it with: export ANTHROPIC_API_KEY='your-api-key-here'")
        print("\nRunning in demo mode (no actual API calls)...\n")
        
        # Show what would happen
        ahimsa = AhimsaAI(enable_semantic=True)
        
        demo_inputs = [
            "How can I resolve conflicts peacefully?",
            "How to manipulate people?",
            "What is Gandhi's philosophy of non-violence?",
        ]
        
        for user_input in demo_inputs:
            print(f"Input: {user_input}")
            result = ahimsa.process_request(user_input)
            print(f"Status: {result['status']}")
            if result['status'] == 'rejected':
                print(f"Response: {result['response'][:100]}...")
            print("-" * 40)
        
        return
    
    # Full demo with API calls
    try:
        client = AhimsaClaude(
            enable_semantic=True,
            enable_llm_judge=False
        )
        
        print("\n--- Test 1: Constructive Request ---")
        result1 = client.create_message("How can I resolve conflicts peacefully?")
        print(f"Status: {result1['status']}")
        print(f"Response: {result1['response'][:200]}...")
        if result1['usage']:
            print(f"Tokens: {result1['usage']}")
        
        print("\n--- Test 2: Harmful Request (Blocked) ---")
        result2 = client.create_message("How can I manipulate people?")
        print(f"Status: {result2['status']}")
        print(f"Response: {result2['response'][:200]}...")
        
        print("\n--- Test 3: Philosophical Request ---")
        result3 = client.create_message(
            "What is the relationship between truth and non-violence in Gandhi's philosophy?"
        )
        print(f"Status: {result3['status']}")
        print(f"Response: {result3['response'][:300]}...")
        
        print("\n--- Test 4: Multi-turn Conversation ---")
        convo = AhimsaClaudeConversation(client)
        
        messages = [
            "Tell me about ahimsa.",
            "How can I practice it daily?",
            "What challenges might I face?"
        ]
        
        for msg in messages:
            print(f"\nUser: {msg}")
            result = convo.send(msg)
            print(f"Claude: {result['response'][:150]}...")
        
        print(f"\nConversation length: {len(convo.get_history())} messages")
        
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
