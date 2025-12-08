# Contributing to Ahimsa AI Framework

First off, thank you for considering contributing to the Ahimsa AI Framework! 🙏

This project aims to make AI systems safer and more ethical by implementing Gandhi's principle of non-violence. Every contribution helps make AI more compassionate.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [How Can I Contribute?](#how-can-i-contribute)
- [Development Setup](#development-setup)
- [Project Structure](#project-structure)
- [Coding Guidelines](#coding-guidelines)
- [Testing Guidelines](#testing-guidelines)
- [Pull Request Process](#pull-request-process)
- [Adding New Validation Patterns](#adding-new-validation-patterns)
- [Adding New Validators](#adding-new-validators)

---

## Code of Conduct

This project follows the principle of Ahimsa in all interactions. We expect all contributors to:

- **Be respectful** - Treat everyone with dignity and kindness
- **Be constructive** - Offer helpful feedback, not criticism
- **Be inclusive** - Welcome people of all backgrounds
- **Be patient** - Remember that everyone is learning
- **Be honest** - Communicate openly and transparently

Harassment, discrimination, or harmful behavior of any kind will not be tolerated.

---

## How Can I Contribute?

### 🐛 Reporting Bugs

Found a bug? Please open an issue with:

1. **Clear title** describing the problem
2. **Steps to reproduce** the issue
3. **Expected behavior** vs **actual behavior**
4. **Environment details** (Python version, OS, dependencies)
5. **Code sample** if applicable

Example:
```markdown
### Bug: False positive for "kill" in technical context

**Steps to reproduce:**
1. Initialize AhimsaAI with default settings
2. Call process_request("How to kill zombie processes")
3. Request is incorrectly rejected

**Expected:** Request should be accepted (technical context)
**Actual:** Request is rejected as violence

**Environment:** Python 3.10, Ubuntu 22.04, v2.0.0
```

### 💡 Suggesting Features

Have an idea? Open an issue with:

1. **Clear description** of the feature
2. **Use case** - Why is this needed?
3. **Proposed solution** (if you have one)
4. **Alternatives considered**

### 🔧 Contributing Code

We welcome:

- Bug fixes
- New validation patterns
- New validator layers
- Performance improvements
- Documentation improvements
- Test coverage improvements
- Integration examples

### 📚 Improving Documentation

Documentation improvements are always welcome:

- Fix typos or unclear explanations
- Add examples
- Improve README
- Add tutorials

---

## Development Setup

### Prerequisites

- Python 3.8 or higher
- Git
- (Optional) Virtual environment tool (venv, conda)

### Setup Steps
```bash
# 1. Fork the repository on GitHub

# 2. Clone your fork
git clone https://github.com/YOUR_USERNAME/ahimsa.git
cd ahimsa

# 3. Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 4. Install dependencies
pip install -r requirements.txt

# 5. Install development dependencies
pip install pytest pytest-cov black isort mypy

# 6. Run tests to verify setup
pytest test_ahimsa_framework.py -v

# 7. Create a branch for your changes
git checkout -b feature/your-feature-name
```

### Environment Variables (Optional)
```bash
# For testing integrations
export ANTHROPIC_API_KEY='your-key'
export OPENAI_API_KEY='your-key'
```

---

## Project Structure
```
ahimsa/
├── ahimsa_ai_framework.py    # Core framework
│   ├── ViolationLevel        # Enum for severity levels
│   ├── ValidationSource      # Enum for detection source
│   ├── AhimsaViolation       # Violation data class
│   ├── ValidationResult      # Result data class
│   ├── BaseValidator         # Abstract base class
│   ├── KeywordValidator      # Layer 1: Regex patterns
│   ├── SemanticValidator     # Layer 2: ML similarity
│   ├── ExternalModerationValidator  # Layer 3: API
│   ├── LLMJudgeValidator     # Layer 4: LLM evaluation
│   ├── OutputValidator       # Output validation
│   ├── AhimsaValidationPipeline    # Pipeline orchestrator
│   ├── RefusalGenerator      # Compassionate refusals
│   ├── SystemPromptGenerator # System prompt creation
│   └── AhimsaAI              # Main public interface
│
├── anthropic_integration.py  # Claude integration
│   ├── AhimsaClaude          # Main wrapper
│   └── AhimsaClaudeConversation  # Multi-turn support
│
├── openai_integration.py     # OpenAI integration
│   ├── AhimsaOpenAI          # Main wrapper
│   ├── AhimsaOpenAIConversation  # Multi-turn support
│   └── AhimsaOpenAIStreaming # Streaming support
│
├── test_ahimsa_framework.py  # Test suite
├── requirements.txt          # Dependencies
├── README.md                 # Documentation
├── CHANGELOG.md              # Version history
├── CONTRIBUTING.md           # This file
├── PROJECT_STRUCTURE.md      # Detailed structure
└── LICENSE                   # MIT License
```

---

## Coding Guidelines

### Style

We follow PEP 8 with these specifics:
```python
# Use type hints
def validate(self, text: str, context: Optional[Dict] = None) -> List[AhimsaViolation]:
    pass

# Use docstrings (Google style)
def process_request(self, user_input: str) -> Dict[str, Any]:
    """
    Process a user request through the Ahimsa framework.
    
    Args:
        user_input: The user's request text
    
    Returns:
        Dictionary with validation results and appropriate response
    
    Raises:
        ValueError: If user_input is None
    """
    pass

# Use descriptive variable names
harmful_patterns = {}  # Good
hp = {}  # Bad

# Use constants for magic values
SIMILARITY_THRESHOLD = 0.78  # Good
if similarity > 0.78:  # Bad - magic number
```

### Formatting
```bash
# Format code with black
black ahimsa_ai_framework.py

# Sort imports with isort
isort ahimsa_ai_framework.py

# Type check with mypy
mypy ahimsa_ai_framework.py
```

### Logging
```python
import logging

logger = logging.getLogger('ahimsa')

# Use appropriate log levels
logger.debug("Detailed debugging info")
logger.info("General information")
logger.warning("Something unexpected but not critical")
logger.error("Something failed")
```

---

## Testing Guidelines

### Running Tests
```bash
# Run all tests
pytest test_ahimsa_framework.py -v

# Run specific test class
pytest test_ahimsa_framework.py::TestKeywordValidator -v

# Run specific test
pytest test_ahimsa_framework.py::TestKeywordValidator::test_blocks_direct_violence -v

# Run with coverage
pytest test_ahimsa_framework.py -v --cov=ahimsa_ai_framework --cov-report=html

# Run without pytest (basic tests)
python test_ahimsa_framework.py
```

### Writing Tests
```python
class TestYourFeature:
    """Tests for your new feature"""
    
    @pytest.fixture
    def setup(self):
        """Setup for each test"""
        return YourClass()
    
    def test_feature_does_something(self, setup):
        """Clear description of what this tests"""
        # Arrange
        input_data = "test input"
        
        # Act
        result = setup.your_method(input_data)
        
        # Assert
        assert result.is_valid
        assert len(result.violations) == 0
    
    def test_feature_handles_edge_case(self, setup):
        """Test edge case: empty input"""
        result = setup.your_method("")
        assert result.is_valid
```

### Test Categories

1. **Unit tests** - Test individual functions/methods
2. **Integration tests** - Test components working together
3. **Edge case tests** - Test unusual inputs (empty, long, unicode)
4. **Regression tests** - Test previously fixed bugs

### What to Test

- ✅ All public methods
- ✅ Both positive and negative cases
- ✅ Edge cases (empty, null, very long)
- ✅ Error handling
- ❌ Private methods (test through public interface)
- ❌ External API calls (mock them)

---

## Pull Request Process

### Before Submitting

1. **Update from main**
```bash
   git fetch upstream
   git rebase upstream/main
```

2. **Run tests**
```bash
   pytest test_ahimsa_framework.py -v
```

3. **Format code**
```bash
   black *.py
   isort *.py
```

4. **Update documentation** if needed

5. **Update CHANGELOG.md** with your changes

### PR Template
```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Changes Made
- Change 1
- Change 2

## Testing
- [ ] All existing tests pass
- [ ] Added new tests for changes
- [ ] Tested manually

## Checklist
- [ ] Code follows project style
- [ ] Self-reviewed my code
- [ ] Added comments for complex parts
- [ ] Updated documentation
- [ ] No new warnings
```

### Review Process

1. Submit PR against `main` branch
2. Automated tests will run
3. Maintainer will review
4. Address any feedback
5. PR will be merged when approved

---

## Adding New Validation Patterns

### Adding Harmful Patterns
```python
# In KeywordValidator._init_harmful_patterns()

self.harmful_patterns['your_category'] = {
    'patterns': [
        r'\byour_regex_pattern\b',
        r'\banother_pattern\b',
    ],
    'level': ViolationLevel.HIGH  # or CRITICAL, MEDIUM, LOW
}
```

### Adding Safe Context Patterns
```python
# In KeywordValidator._init_safe_contexts()

self.safe_contexts['trigger_word'] = [
    r'safe_context_pattern',
    r'another_safe_pattern',
]
```

### Adding Semantic Examples
```python
# In SemanticValidator._get_harmful_examples()

'your_category': [
    "Example harmful phrase 1",
    "Example harmful phrase 2",
    "Example harmful phrase 3",  # Add 5-10 examples
],
```

### Testing New Patterns
```python
# Add tests in test_ahimsa_framework.py

def test_blocks_your_category(self, validator):
    """Your category should be blocked"""
    violations = validator.validate("Your harmful input")
    assert len(violations) > 0
    assert any(v.category == 'your_category' for v in violations)

def test_allows_safe_context(self, validator):
    """Safe context should be allowed"""
    violations = validator.validate("Your safe input")
    blocking = [v for v in violations if v.level.value >= ViolationLevel.HIGH.value]
    assert len(blocking) == 0
```

---

## Adding New Validators

To add a completely new validation layer:

### 1. Create the Validator Class
```python
from ahimsa_ai_framework import BaseValidator, AhimsaViolation, ViolationLevel, ValidationSource

class YourValidator(BaseValidator):
    """Description of your validator"""
    
    @property
    def name(self) -> str:
        return "your_validator"
    
    def __init__(self, your_param: str = "default"):
        self.your_param = your_param
        # Initialize your validator
    
    def validate(self, text: str, context: Optional[Dict] = None) -> List[AhimsaViolation]:
        """Validate text using your method"""
        violations = []
        
        # Your validation logic here
        if self._is_harmful(text):
            violations.append(AhimsaViolation(
                level=ViolationLevel.HIGH,
                category="your_category",
                description="Description of violation",
                suggestion="How to fix",
                source=ValidationSource.KEYWORD,  # or create new source
                confidence=0.9,
                metadata={'extra': 'info'}
            ))
        
        return violations
    
    def _is_harmful(self, text: str) -> bool:
        """Your detection logic"""
        return False
```

### 2. Integrate into Pipeline
```python
# In AhimsaValidationPipeline.__init__()

if enable_your_validator:
    self.input_validators.append(YourValidator())
```

### 3. Add Configuration Option
```python
# In AhimsaAI.__init__()

def __init__(
    self,
    # ... existing params ...
    enable_your_validator: bool = False,
):
    self.pipeline = AhimsaValidationPipeline(
        # ... existing params ...
        enable_your_validator=enable_your_validator,
    )
```

### 4. Add Tests
```python
class TestYourValidator:
    """Tests for YourValidator"""
    
    @pytest.fixture
    def validator(self):
        return YourValidator()
    
    def test_detects_harmful(self, validator):
        violations = validator.validate("harmful input")
        assert len(violations) > 0
    
    def test_allows_safe(self, validator):
        violations = validator.validate("safe input")
        assert len(violations) == 0
```

### 5. Document

Update README.md and CHANGELOG.md with your new validator.

---

## Questions?

- Open an issue for questions
- Email: bissembert (at) gmail.com

Thank you for contributing to ethical AI! 🙏
