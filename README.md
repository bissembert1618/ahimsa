# Ahimsa AI Framework

A Python framework that implements Mahatma Gandhi's principle of Ahimsa (non-violence) for AI systems, ensuring AI models operate with compassion, truthfulness, and respect for all beings.

## Overview

This framework provides a validation and filtering layer that can be integrated with any AI model to enforce non-violent, compassionate, and ethical behavior based on Gandhi's philosophy.

## Core Principles

The framework enforces five key principles:

1. **Non-violence (Ahimsa)** - No physical, mental, or emotional harm
2. **Compassion (Karuna)** - Empathetic and understanding responses
3. **Truthfulness (Satya)** - Honesty without causing harm
4. **Non-exploitation** - Respect for autonomy and dignity
5. **Environmental Care** - Sustainable and responsible practices

## Features

- **Input Validation**: Filters harmful requests before they reach the AI model
- **Response Validation**: Ensures AI responses align with Ahimsa principles
- **Compassionate Refusals**: Provides respectful explanations when declining requests
- **System Prompt Generation**: Creates comprehensive prompts for AI model initialization
- **Violation Detection**: Identifies and categorizes different types of harmful content
- **Extensible Pattern Matching**: Easy to add custom harmful pattern definitions

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/ahimsa-ai-framework.git
cd ahimsa-ai-framework

# No external dependencies required - uses only Python standard library
python3 ahimsa_ai_framework.py
```

## Quick Start

```python
from ahimsa_ai_framework import AhimsaAI

# Initialize the framework
ahimsa_ai = AhimsaAI()

# Process a user request
result = ahimsa_ai.process_request("How can I help my community?")

if result['status'] == 'accepted':
    print("Request approved")
    # Send to your AI model with the Ahimsa system prompt
    system_prompt = result['system_prompt']
else:
    print(result['response'])  # Compassionate refusal message
```

## Integration with AI Models

### OpenAI API Example

```python
from ahimsa_ai_framework import AhimsaAI
import openai

ahimsa_ai = AhimsaAI()

def safe_completion(user_message):
    # Validate input
    result = ahimsa_ai.process_request(user_message)

    if result['status'] == 'rejected':
        return result['response']

    # Call OpenAI with Ahimsa system prompt
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": result['system_prompt']},
            {"role": "user", "content": user_message}
        ]
    )

    ai_response = response.choices[0].message.content

    # Validate output
    validation = ahimsa_ai.process_request(user_message, ai_response)
    return validation['response']
```

### Anthropic Claude Example

```python
from ahimsa_ai_framework import AhimsaAI
import anthropic

ahimsa_ai = AhimsaAI()
client = anthropic.Anthropic()

def safe_completion(user_message):
    result = ahimsa_ai.process_request(user_message)

    if result['status'] == 'rejected':
        return result['response']

    message = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=1024,
        system=result['system_prompt'],
        messages=[
            {"role": "user", "content": user_message}
        ]
    )

    ai_response = message.content[0].text
    validation = ahimsa_ai.process_request(user_message, ai_response)
    return validation['response']
```

## Customization

### Adding Custom Harmful Patterns

```python
from ahimsa_ai_framework import AhimsaAI, ViolationLevel

ahimsa_ai = AhimsaAI()

# Add custom patterns
ahimsa_ai.validator.harmful_patterns['custom_category'] = {
    'patterns': [
        r'\byour_pattern_here\b',
    ],
    'level': ViolationLevel.HIGH
}
```

## Running the Demo

```bash
python3 ahimsa_ai_framework.py
```

This will demonstrate:
- Blocking harmful requests
- Approving benign requests
- The complete Ahimsa system prompt

## Use Cases

- **Educational AI Systems**: Ensure student-facing AI promotes non-violence
- **Mental Health Chatbots**: Prevent harmful advice or manipulation
- **Content Moderation**: Filter requests and responses for harmful content
- **Corporate AI Assistants**: Align with ethical corporate values
- **Research**: Study the impact of ethical frameworks on AI behavior

## Philosophy

This framework is inspired by Mahatma Gandhi's principle of Ahimsa, which extends beyond physical non-violence to encompass:

- **Thought**: No harmful intentions or mental violence
- **Speech**: No hurtful, deceptive, or manipulative communication
- **Action**: No destructive or exploitative behavior

By implementing these principles in AI systems, we aim to create technology that serves humanity with compassion and respect.

## Contributing

Contributions are welcome! Please read [CONTRIBUTING.md](CONTRIBUTING.md) for details on our code of conduct and the process for submitting pull requests.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Mahatma Gandhi for the principle of Ahimsa
- The open-source community for ethical AI development
- All contributors who help make AI safer and more compassionate

## Roadmap

- [ ] Add support for multiple languages
- [ ] Implement machine learning-based harm detection
- [ ] Create web API endpoint
- [ ] Add detailed logging and analytics
- [ ] Develop browser extension for real-time filtering
- [ ] Create comprehensive test suite
- [ ] Add configuration file support

## Contact

For questions, suggestions, or discussions about ethical AI:
- Open an issue on GitHub
- Contact: bissembert (a) gmail.com

## Disclaimer

This framework is a tool to promote ethical AI behavior but is not foolproof. It should be used as part of a comprehensive approach to AI safety and ethics, not as a sole solution. Regular updates and monitoring are recommended.

---

*"Non-violence is the greatest force at the disposal of mankind. It is mightier than the mightiest weapon of destruction devised by the ingenuity of man."* - Mahatma Gandhi
