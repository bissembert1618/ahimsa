# Project Structure

This document provides a detailed overview of the Ahimsa AI Framework codebase.

## Directory Layout
```
ahimsa/
│
├── 📄 ahimsa_ai_framework.py    # Core framework (main file)
├── 📄 anthropic_integration.py  # Anthropic Claude integration
├── 📄 openai_integration.py     # OpenAI GPT integration
├── 📄 custom_patterns.py        # User-defined patterns (optional)
│
├── 📄 test_ahimsa_framework.py  # Test suite
│
├── 📄 requirements.txt          # Python dependencies
├── 📄 setup.py                  # Package setup (for pip install)
│
├── 📄 README.md                 # Main documentation
├── 📄 CHANGELOG.md              # Version history
├── 📄 CONTRIBUTING.md           # Contribution guidelines
├── 📄 PROJECT_STRUCTURE.md      # This file
├── 📄 LICENSE                   # MIT License
│
└── 📄 .gitignore                # Git ignore rules
```

---

## Core Framework (`ahimsa_ai_framework.py`)

### Enums
```python
class ViolationLevel(Enum):
    """Severity levels for violations"""
    NONE = 0      # No violation
    LOW = 1       # Minor concern
    MEDIUM = 2    # Moderate concern
    HIGH = 3      # Serious - blocks request
    CRITICAL = 4  # Severe - always blocks
```
```python
class ValidationSource(Enum):
    """Which layer detected the violation"""
    KEYWORD = "keyword"           # Layer 1
    SEMANTIC = "semantic"         # Layer 2
    EXTERNAL_API = "external_api" # Layer 3
    LLM_JUDGE = "llm_judge"       # Layer 4
```

### Data Classes
```python
@dataclass
class AhimsaViolation:
    """Represents a single detected violation"""
    level: ViolationLevel         # Severity
    category: str                 # Type of violation
    description: str              # Human-readable description
    suggestion: str               # How to fix
    source: ValidationSource      # Which layer detected it
    confidence: float = 1.0       # 0.0 to 1.0
    matched_pattern: str = None   # Regex that matched (if keyword)
    metadata: Dict = {}           # Additional info
```
```python
@dataclass
class ValidationResult:
    """Complete result from validation pipeline"""
    is_valid: bool                    # Pass/fail
    violations: List[AhimsaViolation] # All violations found
    input_text: str                   # Original input
    timestamp: datetime               # When validated
    request_id: str                   # Unique ID for tracking
    processing_time_ms: float         # Performance metric
    layers_checked: List[str]         # Which validators ran
```

### Validators (Layers)

#### Layer 1: KeywordValidator
```python
class KeywordValidator(BaseValidator):
    """
    Fast, rule-based detection using regex patterns.
    
    Features:
    - Harmful pattern detection
    - Safe context recognition
    - ~1ms latency
    - No external dependencies
    """
    
    # Harmful patterns by category
    harmful_patterns = {
        'violence_request': {...},
        'weapon_creation': {...},
        'manipulation_request': {...},
        'hate_content': {...},
        'self_harm': {...},
    }
    
    # Safe contexts that override harmful patterns
    safe_contexts = {
        'kill': ['kill process', 'killing it', ...],
        'attack': ['panic attack', 'attack surface', ...],
        'bomb': ['bombed interview', 'photo bomb', ...],
        ...
    }
```

#### Layer 2: SemanticValidator
```python
class SemanticValidator(BaseValidator):
    """
    ML-based detection using sentence embeddings.
    
    Features:
    - Catches paraphrased harmful content
    - Uses sentence-transformers
    - ~50-100ms latency
    - Configurable threshold
    """
    
    # Harmful examples for similarity matching
    harmful_examples = {
        'violence': ["How do I kill someone", ...],
        'weapon_creation': ["How to make a bomb", ...],
        'manipulation': ["How to manipulate people", ...],
        ...
    }
```

#### Layer 3: ExternalModerationValidator
```python
class ExternalModerationValidator(BaseValidator):
    """
    External API-based moderation.
    
    Features:
    - Uses OpenAI moderation endpoint
    - Professional-grade detection
    - ~200-500ms latency
    - Pay per request
    """
```

#### Layer 4: LLMJudgeValidator
```python
class LLMJudgeValidator(BaseValidator):
    """
    LLM-based evaluation for nuanced cases.
    
    Features:
    - Full context understanding
    - Highest accuracy
    - ~1-3s latency
    - Most expensive
    """
```

#### Output Validator
```python
class OutputValidator(BaseValidator):
    """
    Validates AI responses (different from input validation).
    
    Detects:
    - Harmful instructions being provided
    - Encouragement of violence
    - Manipulation tactics
    """
```

### Pipeline
```python
class AhimsaValidationPipeline:
    """
    Orchestrates multiple validators.
    
    Features:
    - Configurable layers
    - Fail-fast option
    - Aggregates results
    - Performance tracking
    """
    
    def validate_input(self, text: str) -> ValidationResult:
        """Run input through all enabled validators"""
        
    def validate_output(self, response: str, original_input: str) -> ValidationResult:
        """Validate AI response"""
```

### Support Classes
```python
class RefusalGenerator:
    """Generates compassionate refusal messages"""
    
    def generate(self, result: ValidationResult) -> str:
        """Create refusal message based on violation type"""
```
```python
class SystemPromptGenerator:
    """Creates Ahimsa system prompts for AI models"""
    
    @staticmethod
    def generate(custom_additions: str = None) -> str:
        """Generate complete system prompt"""
```

### Main Class
```python
class AhimsaAI:
    """
    Main public interface for the framework.
    
    Usage:
        ahimsa = AhimsaAI()
        result = ahimsa.process_request("user input")
    """
    
    def __init__(
        self,
        enable_semantic: bool = True,
        enable_external_api: bool = False,
        enable_llm_judge: bool = False,
        ...
    ):
        """Configure which validation layers to use"""
    
    def process_request(
        self,
        user_input: str,
        model_response: str = None
    ) -> Dict[str, Any]:
        """
        Main entry point.
        
        Returns:
            {
                'status': 'accepted' | 'rejected',
                'reason': 'input_violation' | 'response_violation' | None,
                'response': str,  # AI response or refusal message
                'validation_result': {...},
                'system_prompt': str
            }
        """
    
    def get_system_prompt(self) -> str:
        """Get the Ahimsa system prompt"""
    
    def add_harmful_example(self, category: str, example: str):
        """Add new example for semantic matching"""
```

---

## Integrations

### Anthropic Integration (`anthropic_integration.py`)
```python
class AhimsaClaude:
    """
    Wrapper for Anthropic Claude API.
    
    Methods:
        create_message() - Full response with metadata
        chat() - Simple string response
        add_harmful_example() - Learning from feedback
    """

class AhimsaClaudeConversation:
    """
    Multi-turn conversation manager.
    
    Methods:
        send() - Send message, get response
        reset() - Clear history
        get_history() - Get conversation history
    """
```

### OpenAI Integration (`openai_integration.py`)
```python
class AhimsaOpenAI:
    """
    Wrapper for OpenAI API.
    
    Methods:
        create_message() - Full response with metadata
        chat() - Simple string response
        moderate() - Direct moderation API access
        add_harmful_example() - Learning from feedback
    """

class AhimsaOpenAIConversation:
    """Multi-turn conversation manager"""

class AhimsaOpenAIStreaming:
    """Streaming response support"""
```

---

## Data Flow

### Request Processing Flow
```
User Input
    │
    ▼
┌───────────────────────────────────────────┐
│           AhimsaAI.process_request()      │
└───────────────────────────────────────────┘
    │
    ▼
┌───────────────────────────────────────────┐
│      AhimsaValidationPipeline             │
│                                           │
│  ┌─────────────────────────────────────┐  │
│  │ Layer 1: KeywordValidator           │  │
│  │ - Check harmful patterns            │  │
│  │ - Check safe contexts               │  │
│  └─────────────────────────────────────┘  │
│                    │                      │
│                    ▼                      │
│  ┌─────────────────────────────────────┐  │
│  │ Layer 2: SemanticValidator          │  │
│  │ - Compute text embedding            │  │
│  │ - Compare to harmful examples       │  │
│  └─────────────────────────────────────┘  │
│                    │                      │
│                    ▼                      │
│  ┌─────────────────────────────────────┐  │
│  │ Layer 3: ExternalModerationValidator│  │
│  │ - Call OpenAI moderation API        │  │
│  └─────────────────────────────────────┘  │
│                    │                      │
│                    ▼                      │
│  ┌─────────────────────────────────────┐  │
│  │ Layer 4: LLMJudgeValidator          │  │
│  │ - Ask LLM to evaluate               │  │
│  └─────────────────────────────────────┘  │
│                                           │
└───────────────────────────────────────────┘
    │
    ▼
┌───────────────────────────────────────────┐
│           ValidationResult                │
│  - is_valid: bool                         │
│  - violations: List[AhimsaViolation]      │
│  - request_id, timestamp, etc.            │
└───────────────────────────────────────────┘
    │
    ├─── If INVALID ──► RefusalGenerator.generate()
    │                         │
    │                         ▼
    │                   Compassionate Refusal
    │
    └─── If VALID ────► Return with System Prompt
                              │
                              ▼
                        Ready for AI Model
```

### Integration Flow
```
User
  │
  ▼
┌─────────────────────┐
│   AhimsaClaude /    │
│   AhimsaOpenAI      │
└─────────────────────┘
  │
  ├── 1. Validate input (AhimsaAI)
  │         │
  │         ├── REJECTED → Return refusal
  │         │
  │         └── ACCEPTED → Continue
  │
  ├── 2. Call AI API with system prompt
  │
  ├── 3. Validate output (AhimsaAI)
  │         │
  │         ├── REJECTED → Return refusal
  │         │
  │         └── ACCEPTED → Continue
  │
  └── 4. Return response
          │
          ▼
        User
```

---

## Configuration Matrix

| Feature | Param | Default | Requires |
|---------|-------|---------|----------|
| Keyword validation | Always on | - | Nothing |
| Semantic validation | `enable_semantic` | `True` | sentence-transformers |
| External API | `enable_external_api` | `False` | openai, OPENAI_API_KEY |
| LLM Judge | `enable_llm_judge` | `False` | anthropic/openai, API key |
| Custom prompt | `custom_system_prompt_additions` | `None` | Nothing |

---

## File Dependencies
```
ahimsa_ai_framework.py
    └── (no internal dependencies)

anthropic_integration.py
    └── ahimsa_ai_framework.py

openai_integration.py
    └── ahimsa_ai_framework.py

test_ahimsa_framework.py
    └── ahimsa_ai_framework.py
```

---

## External Dependencies

### Required (for full features)

| Package | Version | Purpose |
|---------|---------|---------|
| sentence-transformers | >=2.2.0 | Semantic similarity |
| scikit-learn | >=1.0.0 | Cosine similarity |
| numpy | >=1.21.0 | Numerical operations |
| torch | >=2.0.0 | ML backend |

### Optional

| Package | Version | Purpose |
|---------|---------|---------|
| anthropic | >=0.18.0 | Claude integration |
| openai | >=1.0.0 | GPT integration |
| pytest | >=7.0.0 | Testing |

---

## Performance Characteristics

| Component | Latency | Memory | CPU |
|-----------|---------|--------|-----|
| KeywordValidator | ~1ms | Low | Low |
| SemanticValidator (init) | ~5s | ~500MB | High |
| SemanticValidator (inference) | ~50-100ms | - | Medium |
| ExternalModerationValidator | ~200-500ms | Low | Low |
| LLMJudgeValidator | ~1-3s | Low | Low |

---

## Extending the Framework

See [CONTRIBUTING.md](CONTRIBUTING.md) for:

- Adding new validation patterns
- Creating new validators
- Writing tests
- Submitting pull requests
