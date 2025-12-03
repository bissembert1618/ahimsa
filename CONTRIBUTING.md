# Contributing to Ahimsa AI Framework

First off, thank you for considering contributing to the Ahimsa AI Framework! It's people like you who help make AI safer and more compassionate.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [How Can I Contribute?](#how-can-i-contribute)
- [Getting Started](#getting-started)
- [Development Workflow](#development-workflow)
- [Style Guidelines](#style-guidelines)
- [Commit Messages](#commit-messages)
- [Pull Request Process](#pull-request-process)

## Code of Conduct

This project is guided by the principle of Ahimsa (non-violence). We expect all contributors to:

- Be respectful and compassionate in all interactions
- Welcome and support newcomers
- Focus on what is best for the community
- Show empathy towards other community members
- Accept constructive criticism gracefully

## How Can I Contribute?

### Reporting Bugs

Before creating bug reports, please check existing issues to avoid duplicates. When you create a bug report, include:

- **Clear title and description**
- **Steps to reproduce** the behavior
- **Expected behavior**
- **Actual behavior**
- **Python version** and OS
- **Code samples** if applicable

### Suggesting Enhancements

Enhancement suggestions are tracked as GitHub issues. When creating an enhancement suggestion, include:

- **Clear title and description**
- **Use case** - why this would be useful
- **Proposed solution** - how it might work
- **Alternative solutions** you've considered

### Adding Harmful Pattern Detection

To improve the framework's ability to detect harmful content:

1. Identify patterns that violate Ahimsa principles
2. Create test cases demonstrating the pattern
3. Add regex patterns or detection logic
4. Ensure false positives are minimized
5. Document your additions

### Improving Documentation

- Fix typos or clarify confusing sections
- Add examples for different use cases
- Translate documentation to other languages
- Create tutorials or guides

## Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/your-username/ahimsa-ai-framework.git
   cd ahimsa-ai-framework
   ```
3. **Create a branch** for your changes:
   ```bash
   git checkout -b feature/your-feature-name
   ```

## Development Workflow

### Running Tests

```bash
# Run all tests
python3 -m pytest tests/

# Run with coverage
python3 -m pytest --cov=ahimsa_ai_framework tests/
```

### Code Quality

```bash
# Format code with black
black ahimsa_ai_framework.py

# Check style with flake8
flake8 ahimsa_ai_framework.py

# Type checking with mypy
mypy ahimsa_ai_framework.py
```

## Style Guidelines

### Python Style

- Follow [PEP 8](https://pep8.org/) style guide
- Use type hints for function parameters and return values
- Maximum line length: 100 characters
- Use descriptive variable names

### Code Organization

```python
# Good
def validate_request(self, user_input: str) -> Tuple[bool, List[AhimsaViolation]]:
    """
    Validates user input against Ahimsa principles

    Args:
        user_input: The user's request or prompt

    Returns:
        Tuple of (is_valid, list of violations)
    """
    pass

# Avoid
def check(self, x):
    pass
```

### Documentation

- Use docstrings for all public classes and methods
- Include examples in docstrings when helpful
- Keep comments concise and meaningful
- Update README.md for new features

## Commit Messages

Write clear, concise commit messages:

```
Add: New feature for multilingual support

- Implement language detection
- Add translation for error messages
- Update documentation

Resolves #123
```

Format:
- **Add:** New features
- **Fix:** Bug fixes
- **Update:** Changes to existing features
- **Docs:** Documentation changes
- **Test:** Adding or updating tests
- **Refactor:** Code changes that neither fix bugs nor add features

## Pull Request Process

1. **Update documentation** for any changed functionality
2. **Add tests** for new features
3. **Ensure all tests pass** before submitting
4. **Update README.md** if needed
5. **Describe your changes** clearly in the PR description

### PR Checklist

- [ ] Code follows the project's style guidelines
- [ ] Self-review of code completed
- [ ] Comments added for complex logic
- [ ] Documentation updated
- [ ] Tests added/updated and passing
- [ ] No breaking changes (or clearly documented)

### PR Description Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
Describe tests performed

## Related Issues
Fixes #123
```

## Review Process

1. Maintainers will review your PR
2. Address any requested changes
3. Once approved, a maintainer will merge your PR
4. Your contribution will be acknowledged in release notes

## Questions?

Feel free to:
- Open an issue with the "question" label
- Reach out to maintainers
- Join community discussions

## Recognition

Contributors will be:
- Listed in the project's contributors page
- Mentioned in release notes
- Acknowledged in the README

Thank you for helping make AI more compassionate and aligned with Ahimsa principles!

---

*"Be the change you wish to see in the world."* - Mahatma Gandhi
