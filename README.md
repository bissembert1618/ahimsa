# Ahimsa AI Framework v2.1

A production-ready Python framework that implements Mahatma Gandhi's principle of **Ahimsa (non-violence)** for AI systems. This framework provides multi-layered validation to ensure AI models operate with compassion, truthfulness, and respect for all beings.

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Version](https://img.shields.io/badge/version-2.1.0-orange.svg)

## 🎮 Live Demo

**[Try the interactive demo →](https://bissembert1618.github.io/ahimsa/demo.html)**

Type any text and watch it flow through all four validation layers in real time — no installation or API keys needed. The demo simulates the full pipeline with the framework's actual regex patterns, semantic similarity scoring, and compassionate refusal messages.

## What's New in v2.1

- **Interactive browser demo** — Self-contained HTML page that simulates the full 4-layer pipeline. Animated visualization, pre-loaded examples, and a JSON audit log. [Try it live](https://bissembert1618.github.io/ahimsa/demo.html) or open `demo.html` locally.

## What's New in v2.0

- **Multi-layer validation pipeline** - Defense in depth with multiple detection methods
- **Context-aware keyword detection** - Dramatically reduced false positives
- **Semantic similarity matching** - Catches paraphrased harmful content using ML
- **External moderation API integration** - Leverage OpenAI's moderation endpoint
- **LLM-as-a-judge** - Most thorough validation using AI evaluation
- **Separate input/output validation** - Different strategies for requests vs responses
- **Comprehensive logging & audit trail** - Production-ready observability
- **Full test suite** - pytest-based testing with edge cases

## Core Principles

The framework enforces five key principles from Gandhi's philosophy:

| Principle | Sanskrit | Description |
|-----------|----------|-------------|
| **Non-violence** | Ahimsa | No physical, mental, or emotional harm |
| **Compassion** | Karuna | Empathetic and understanding responses |
| **Truthfulness** | Satya | Honesty without causing harm |
| **Non-exploitation** | - | Respect for autonomy and dignity |
| **Environmental Care** | - | Sustainable and responsible practices |

## Architecture
```
┌─────────────────────────────────────────────────────────────┐
│                     User Request                            │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                  VALIDATION PIPELINE                        │
│  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐   │
│  │   Layer 1     │  │   Layer 2     │  │   Layer 3     │   │
│  │   Keyword     │─▶│   Semantic    │─▶│  External API │   │
│  │  (Fast/Free)  │  │    (ML)       │  │   (OpenAI)    │   │
│  └───────────────┘  └───────────────┘  └───────────────┘   │
│                              │                              │
│                              ▼                              │
│                    ┌───────────────┐                       │
│                    │   Layer 4     │                       │
│                    │  LLM Judge    │                       │
│                    │  (Thorough)   │                       │
│                    └───────────────┘                       │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
              ┌───────────────┴───────────────┐
              │                               │
        ┌─────▼─────┐                 ┌───────▼───────┐
        │  BLOCKED  │                 │   ACCEPTED    │
        │ Refusal   │                 │ + System      │
        │ Message   │                 │   Prompt      │
        └───────────┘                 └───────────────┘
                                              │
                                              ▼
                                      ┌───────────────┐
                                      │   AI Model    │
                                      │ (Claude/GPT)  │
                                      └───────────────┘
                                              │
                                              ▼
                                      ┌───────────────┐
                                      │    Output     │
                                      │  Validation   │
                                      └───────────────┘
                                              │
                                              ▼
                                      ┌───────────────┐
                                      │   Response    │
                                      └───────────────┘
```

## Installation

### Basic Installation (Keyword validation only)
```bash
git clone https://github.com/bissembert1618/ahimsa.git
cd ahimsa
# No dependencies needed for basic usage
python ahimsa_ai_framework.py
```

### Full Installation (All features)
```bash
git clone https://github.com/bissembert1618/ahimsa.git
cd ahimsa
pip install -r requirements.txt
```

### Minimal Installation (Choose what you need)
```bash
# For semantic validation
pip install sentence-transformers scikit-learn numpy torch

# For Anthropic Claude integration
pip install anthropic

# For OpenAI integration
pip install openai

# For testing
pip install pytest pytest-cov
```

## Quick Start

### Basic Usage
```python
from ahimsa_ai_framework import AhimsaAI

# Initialize with default settings (keyword + semantic validation)
ahimsa = AhimsaAI()

# Process a request
result = ahimsa.process_request("How can I help my community?")

if result['status'] == 'accepted':
    print("✅ Request approved")
    print(f"System prompt ready for AI model")
else:
    print("🚫 Request blocked")
    print(result['response'])  # Compassionate refusal message
```

### With Anthropic Claude
```python
from anthropic_integration import AhimsaClaude

client = AhimsaClaude(api_key="your-api-key")

# Simple chat
response = client.chat("How can I resolve conflicts peacefully?")
print(response)

# Full response with metadata
result = client.create_message("Tell me about Gandhi's philosophy")
print(f"Status: {result['status']}")
print(f"Response: {result['response']}")
print(f"Tokens used: {result['usage']}")
```

### With OpenAI
```python
from openai_integration import AhimsaOpenAI

client = AhimsaOpenAI(api_key="your-api-key")

# Simple chat
response = client.chat("What is non-violent communication?")
print(response)

# With streaming
from openai_integration import AhimsaOpenAIStreaming

streamer = AhimsaOpenAIStreaming(client)
for chunk in streamer.stream_message("Explain ahimsa"):
    if chunk['type'] == 'chunk':
        print(chunk['content'], end='', flush=True)
```

### Multi-turn Conversations
```python
from anthropic_integration import AhimsaClaude, AhimsaClaudeConversation

client = AhimsaClaude()
convo = AhimsaClaudeConversation(client)

# Conversation with memory
convo.send("Tell me about ahimsa")
convo.send("How can I practice it?")
convo.send("What challenges might I face?")

# Get conversation history
print(f"Messages: {len(convo.get_history())}")

# Reset when done
convo.reset()
```

## Configuration Options

### Validation Layers
```python
ahimsa = AhimsaAI(
    # Layer 1: Context-aware keyword detection (always enabled)
    
    # Layer 2: Semantic similarity using sentence transformers
    enable_semantic=True,  # Default: True
    
    # Layer 3: External moderation API (OpenAI)
    enable_external_api=False,  # Default: False (requires OPENAI_API_KEY)
    
    # Layer 4: LLM-as-a-judge (most thorough, most expensive)
    enable_llm_judge=False,  # Default: False
    llm_judge_provider="anthropic",  # or "openai"
    llm_judge_model="claude-3-haiku-20240307",
    
    # Custom additions to system prompt
    custom_system_prompt_additions="Always be encouraging."
)
```

### Environment Variables
```bash
# For OpenAI integration and external moderation API
export OPENAI_API_KEY='your-openai-key'

# For Anthropic integration
export ANTHROPIC_API_KEY='your-anthropic-key'
```

## Validation Layers Explained

### Layer 1: Context-Aware Keyword Detection
- **Speed**: ~1ms
- **Cost**: Free
- **Accuracy**: Good for obvious cases
- **False Positives**: Low (context-aware)

Catches explicit harmful patterns while understanding safe contexts:
```python
# Blocked
"How do I kill someone?"

# Allowed (safe contexts)
"How do I kill a process in Linux?"
"My back is killing me"
"I bombed that interview"
```

### Layer 2: Semantic Similarity
- **Speed**: ~50-100ms
- **Cost**: Free (runs locally)
- **Accuracy**: Very good
- **False Positives**: Very low

Uses sentence transformers to catch paraphrased harmful content:
```python
# Both blocked (semantically similar)
"How do I kill someone?"
"What's the best way to eliminate a person permanently?"
```

### Layer 3: External Moderation API
- **Speed**: ~200-500ms
- **Cost**: Pay per request (very cheap)
- **Accuracy**: Excellent
- **False Positives**: Very low

Leverages OpenAI's trained moderation models.

### Layer 4: LLM-as-a-Judge
- **Speed**: ~1-3 seconds
- **Cost**: Pay per request
- **Accuracy**: Best
- **False Positives**: Lowest

Uses an LLM to evaluate content with full context understanding.

## Adding Custom Harmful Patterns

### Keyword Patterns
```python
from ahimsa_ai_framework import AhimsaAI, KeywordValidator, ViolationLevel

ahimsa = AhimsaAI()

# Access the keyword validator
for validator in ahimsa.pipeline.input_validators:
    if isinstance(validator, KeywordValidator):
        # Add custom harmful patterns
        validator.harmful_patterns['custom_category'] = {
            'patterns': [
                r'\byour_pattern_here\b',
                r'\banother_pattern\b',
            ],
            'level': ViolationLevel.HIGH
        }
        
        # Add safe context patterns
        validator.safe_contexts['trigger_word'] = [
            r'safe_context_pattern',
        ]
```

### Semantic Examples
```python
ahimsa = AhimsaAI()

# Add examples for semantic matching (learning from feedback)
ahimsa.add_harmful_example('violence', 'New harmful phrase to detect')
ahimsa.add_harmful_example('manipulation', 'Another harmful example')
```

## Testing

### Run All Tests
```bash
# With pytest (recommended)
pytest test_ahimsa_framework.py -v

# With coverage report
pytest test_ahimsa_framework.py -v --cov=ahimsa_ai_framework --cov-report=html

# Basic tests without pytest
python test_ahimsa_framework.py
```

### Run Demo

#### Interactive Browser Demo (no dependencies)
```bash
# Open the HTML demo in your browser
open demo.html
# Or visit the live version: https://bissembert1618.github.io/ahimsa/demo.html
```

#### Python Demo
```bash
# Framework demo
python ahimsa_ai_framework.py

# Anthropic integration demo
export ANTHROPIC_API_KEY='your-key'
python anthropic_integration.py

# OpenAI integration demo
export OPENAI_API_KEY='your-key'
python openai_integration.py
```

## Use Cases

| Use Case | Recommended Configuration |
|----------|---------------------------|
| **Chatbot MVP** | `enable_semantic=True` only |
| **Production Chatbot** | All layers enabled |
| **Content Moderation** | `enable_external_api=True` |
| **High-Stakes Applications** | `enable_llm_judge=True` |
| **Cost-Sensitive** | `enable_semantic=False`, keyword only |
| **Educational Platform** | All layers + custom patterns |

## Performance

| Configuration | Latency | Cost per 1K requests |
|---------------|---------|----------------------|
| Keyword only | ~1ms | $0 |
| + Semantic | ~50-100ms | $0 |
| + External API | ~200-500ms | ~$0.01 |
| + LLM Judge | ~1-3s | ~$0.50-2.00 |

## Logging & Monitoring

The framework includes comprehensive logging:
```python
import logging

# Set log level
logging.getLogger('ahimsa').setLevel(logging.DEBUG)

# Each validation includes:
# - Request ID (for tracking)
# - Timestamp
# - Processing time
# - Layers checked
# - Violations found
```

Example log output:
```
2024-01-15 10:30:45 - ahimsa - INFO - Validation complete: {
    'is_valid': False,
    'violations': [{'level': 'CRITICAL', 'category': 'violence_request', ...}],
    'request_id': 'a1b2c3d4e5f6g7h8',
    'processing_time_ms': 52.3,
    'layers_checked': ['keyword', 'semantic']
}
```

## Philosophy

This framework is inspired by Mahatma Gandhi's principle of Ahimsa, which extends beyond physical non-violence to encompass:

- **Thought**: No harmful intentions or mental violence
- **Speech**: No hurtful, deceptive, or manipulative communication
- **Action**: No destructive or exploitative behavior

> *"Non-violence is the greatest force at the disposal of mankind. It is mightier than the mightiest weapon of destruction devised by the ingenuity of man."*
> — Mahatma Gandhi

## Contributing

Contributions are welcome! Please read [CONTRIBUTING.md](CONTRIBUTING.md) for details.

### Development Setup
```bash
git clone https://github.com/bissembert1618/ahimsa.git
cd ahimsa
pip install -r requirements.txt
pytest test_ahimsa_framework.py -v
```

### Areas for Contribution

- Additional language support
- New validation patterns
- Performance optimizations
- Documentation improvements
- Integration examples

## Roadmap

- [x] Interactive browser demo (v2.1)
- [ ] Fine-tuned classifier for better accuracy
- [ ] Adversarial testing suite
- [ ] Web API endpoint
- [ ] Configuration file support (YAML)
- [ ] Admin dashboard for monitoring
- [ ] Multi-language support
- [ ] Rate limiting and abuse prevention
- [ ] Human-in-the-loop escalation

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Documentation

- [Research Paper (PDF)](https://github.com/bissembert1618/ahimsa/blob/main/ahimsa_scientific_paper)
- [Project on OSF](https://osf.io/d7jsv/overview)

## Acknowledgments

- Mahatma Gandhi for the principle of Ahimsa
- The open-source AI safety community
- Anthropic and OpenAI for their APIs
- All contributors who help make AI safer

## Contact

- **Issues**: [GitHub Issues](https://github.com/bissembert1618/ahimsa/issues)
- **Email**: bissembert (at) gmail.com

## Disclaimer

This framework is a tool to promote ethical AI behavior but is not foolproof. It should be used as part of a comprehensive approach to AI safety, not as a sole solution. Regular updates and monitoring are recommended.

---

**Made with ❤️ and Ahimsa**
