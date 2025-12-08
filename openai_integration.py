"""
OpenAI Integration with Ahimsa AI Framework v2.0

Production-ready integration that wraps OpenAI API with multi-layer
Ahimsa validation for both inputs and outputs.
"""

import os
import sys
import logging
from typing import Optional, Dict, Any, List

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('ahimsa.openai')

# Import framework
try:
    from ahimsa_ai_framework import AhimsaAI
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from ahimsa_ai_framework import AhimsaAI

# Import OpenAI
try:
    import openai
except ImportError:
    logger.error("OpenAI package not installed. Install with: pip install openai")
    openai = None


class AhimsaOpenAI:
    """
    Production-ready wrapper for OpenAI API with Ahimsa validation.
    
    Features:
    - Multi-layer input validation (keyword + semantic + optional external API)
    - Output validation to catch harmful responses
    - Comprehensive logging and audit trail
    - Graceful error handling
    - Configurable validation layers
    - Support for GPT-3.5, GPT-4, and GPT-4o models
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        default_model: str = "gpt-4o",
        enable_semantic: bool = True,
        enable_external_api: bool = False,
        enable_llm_judge: bool = False,
        custom_system_prompt_additions: Optional[str] = None
    ):
        """
        Initialize the Ahimsa-wrapped OpenAI client.
        
        Args:
            api_key: OpenAI API key (or set OPENAI_API_KEY env var)
            default_model: Default model to use (gpt-4o, gpt-4-turbo, gpt-3.5-turbo)
            enable_semantic: Enable semantic similarity validation
            enable_external_api: Enable OpenAI's moderation API as additional layer
            enable_llm_judge: Enable LLM-as-a-judge for thorough validation
            custom_system_prompt_additions: Custom additions to Ahimsa system prompt
        """
        if openai is None:
            raise ImportError("openai package required. Install with: pip install openai")
        
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            raise ValueError(
                "OpenAI API key required. Set OPENAI_API_KEY environment variable "
                "or pass api_key parameter."
            )
        
        self.default_model = default_model
        self.client = openai.OpenAI(api_key=self.api_key)
        
        # Initialize Ahimsa framework
        self.ahimsa = AhimsaAI(
            enable_semantic=enable_semantic,
            enable_external_api=enable_external_api,
            enable_llm_judge=enable_llm_judge,
            llm_judge_provider="openai",
            llm_judge_model="gpt-4o-mini",
            custom_system_prompt_additions=custom_system_prompt_additions
        )
        
        logger.info(f"AhimsaOpenAI initialized with model: {default_model}")
    
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
        Create a message with OpenAI, with Ahimsa validation on both input and output.
        
        Args:
            user_message: The user's message
            model: Model to use (defaults to self.default_model)
            max_tokens: Maximum tokens in response
            temperature: Sampling temperature
            conversation_history: Previous messages for context
            validate_output: Whether to validate the AI's response
        
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
        messages = [
            {"role": "system", "content": self.ahimsa.get_system_prompt()}
        ]
        if conversation_history:
            messages.extend(conversation_history)
        messages.append({"role": "user", "content": user_message})
        
        # Step 3: Call OpenAI
        try:
            response = self.client.chat.completions.create(
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=messages
            )
            
            ai_response = response.choices[0].message.content
            usage = {
                'prompt_tokens': response.usage.prompt_tokens,
                'completion_tokens': response.usage.completion_tokens,
                'total_tokens': response.usage.total_tokens
            }
            
        except openai.APIError as e:
            logger.error(f"OpenAI API error: {e}")
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
            model: Model to use
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
    
    def moderate(self, text: str) -> Dict[str, Any]:
        """
        Use OpenAI's moderation endpoint directly.
        
        Args:
            text: Text to moderate
        
        Returns:
            Dictionary with moderation results
        """
        try:
            response = self.client.moderations.create(input=text)
            result = response.results[0]
            
            return {
                'flagged': result.flagged,
                'categories': {
                    k: v for k, v in result.categories.__dict__.items()
                    if not k.startswith('_')
                },
                'category_scores': {
                    k: v for k, v in result.category_scores.__dict__.items()
                    if not k.startswith('_')
                }
            }
        except Exception as e:
            logger.error(f"Moderation error: {e}")
            return {'error': str(e)}
    
    def add_harmful_example(self, category: str, example: str):
        """Add a harmful example for semantic matching (learning from feedback)"""
        self.ahimsa.add_harmful_example(category, example)


class AhimsaOpenAIConversation:
    """
    Manages multi-turn conversations with Ahimsa validation.
    """
    
    def __init__(self, client: AhimsaOpenAI):
        """
        Args:
            client: AhimsaOpenAI instance
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


class AhimsaOpenAIStreaming:
    """
    Streaming responses with Ahimsa validation.
    Note: Output validation happens after stream completes.
    """
    
    def __init__(self, client: AhimsaOpenAI):
        """
        Args:
            client: AhimsaOpenAI instance
        """
        self.ahimsa_client = client
        self.openai_client = client.client
        self.ahimsa = client.ahimsa
    
    def stream_message(
        self,
        user_message: str,
        model: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: float = 1.0,
        validate_output: bool = True
    ):
        """
        Stream a response from OpenAI with Ahimsa validation.
        
        Args:
            user_message: The user's message
            model: Model to use
            max_tokens: Maximum tokens
            temperature: Sampling temperature
            validate_output: Whether to validate complete response after streaming
        
        Yields:
            Chunks of the response, or a refusal message if input is rejected
        """
        model = model or self.ahimsa_client.default_model
        
        # Validate input first
        input_result = self.ahimsa.process_request(user_message)
        
        if input_result['status'] == 'rejected':
            yield {
                'type': 'rejected',
                'content': input_result['response']
            }
            return
        
        # Build messages
        messages = [
            {"role": "system", "content": self.ahimsa.get_system_prompt()},
            {"role": "user", "content": user_message}
        ]
        
        # Stream response
        full_response = ""
        
        try:
            stream = self.openai_client.chat.completions.create(
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=messages,
                stream=True
            )
            
            for chunk in stream:
                if chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    full_response += content
                    yield {
                        'type': 'chunk',
                        'content': content
                    }
            
            # Validate complete output
            if validate_output:
                output_result = self.ahimsa.process_request(user_message, full_response)
                
                if output_result['status'] == 'rejected':
                    yield {
                        'type': 'output_rejected',
                        'content': output_result['response'],
                        'original_response': full_response
                    }
                    return
            
            yield {
                'type': 'complete',
                'content': full_response
            }
            
        except openai.APIError as e:
            logger.error(f"OpenAI streaming error: {e}")
            yield {
                'type': 'error',
                'content': str(e)
            }


# =============================================================================
# DEMO
# =============================================================================

def main():
    """Demo of Ahimsa OpenAI integration"""
    
    print("=" * 60)
    print("AHIMSA OPENAI INTEGRATION DEMO")
    print("=" * 60)
    
    # Check for API key
    if not os.getenv('OPENAI_API_KEY'):
        print("\n⚠️  OPENAI_API_KEY not set")
        print("Set it with: export OPENAI_API_KEY='your-api-key-here'")
        print("\nRunning in demo mode (no actual API calls)...\n")
        
        # Show what would happen
        ahimsa = AhimsaAI(enable_semantic=True)
        
        demo_inputs = [
            "How can I resolve conflicts peacefully?",
            "How to manipulate people?",
            "What is Gandhi's philosophy of non-violence?",
            "How do I kill a process in Linux?",
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
        client = AhimsaOpenAI(
            enable_semantic=True,
            enable_external_api=True,  # Use OpenAI's moderation API
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
        
        print("\n--- Test 3: Technical Request (Safe Context) ---")
        result3 = client.create_message("How do I kill a process in Linux?")
        print(f"Status: {result3['status']}")
        print(f"Response: {result3['response'][:200]}...")
        
        print("\n--- Test 4: Direct Moderation Check ---")
        mod_result = client.moderate("I want to hurt someone")
        print(f"Flagged: {mod_result.get('flagged')}")
        if mod_result.get('categories'):
            flagged_cats = [k for k, v in mod_result['categories'].items() if v]
            print(f"Categories: {flagged_cats}")
        
        print("\n--- Test 5: Multi-turn Conversation ---")
        convo = AhimsaOpenAIConversation(client)
        
        messages = [
            "Tell me about ahimsa.",
            "How can I practice it daily?",
            "What challenges might I face?"
        ]
        
        for msg in messages:
            print(f"\nUser: {msg}")
            result = convo.send(msg)
            print(f"GPT: {result['response'][:150]}...")
        
        print(f"\nConversation length: {len(convo.get_history())} messages")
        
        print("\n--- Test 6: Streaming Response ---")
        streamer = AhimsaOpenAIStreaming(client)
        print("Streaming: 'Explain non-violence in one paragraph'")
        print("Response: ", end="", flush=True)
        
        for chunk in streamer.stream_message(
            "Explain non-violence in one paragraph",
            max_tokens=150
        ):
            if chunk['type'] == 'chunk':
                print(chunk['content'], end="", flush=True)
            elif chunk['type'] == 'complete':
                print("\n[Stream complete]")
            elif chunk['type'] == 'rejected':
                print(f"\n[Rejected: {chunk['content'][:50]}...]")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
