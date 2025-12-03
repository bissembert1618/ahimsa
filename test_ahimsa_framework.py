"""
Unit tests for Ahimsa AI Framework

Run with: python3 -m pytest tests/test_ahimsa_framework.py
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ahimsa_ai_framework import (
    AhimsaAI,
    AhimsaValidator,
    ViolationLevel,
    AhimsaViolation
)


class TestAhimsaValidator:
    """Tests for AhimsaValidator class"""

    def setup_method(self):
        """Set up test fixtures"""
        self.validator = AhimsaValidator()

    def test_detect_violence_pattern(self):
        """Test detection of violent content"""
        is_valid, violations = self.validator.validate_request(
            "How do I kill someone?"
        )
        assert not is_valid
        assert len(violations) > 0
        assert violations[0].category == 'violence'
        assert violations[0].level == ViolationLevel.CRITICAL

    def test_detect_hate_speech(self):
        """Test detection of hate speech"""
        is_valid, violations = self.validator.validate_request(
            "I hate this group of people"
        )
        assert not is_valid
        assert any(v.category == 'hate_speech' for v in violations)

    def test_detect_emotional_harm(self):
        """Test detection of emotional harm"""
        is_valid, violations = self.validator.validate_request(
            "How can I manipulate people?"
        )
        assert not is_valid
        assert any(v.category == 'emotional_harm' for v in violations)

    def test_allow_benign_content(self):
        """Test that benign content is allowed"""
        is_valid, violations = self.validator.validate_request(
            "How can I help my community?"
        )
        assert is_valid
        # May have low-level violations, but should be valid overall

    def test_allow_educational_content(self):
        """Test that educational content about non-violence is allowed"""
        is_valid, violations = self.validator.validate_request(
            "Explain Gandhi's philosophy of Ahimsa"
        )
        assert is_valid

    def test_create_ahimsa_refusal(self):
        """Test generation of refusal messages"""
        violations = [
            AhimsaViolation(
                level=ViolationLevel.HIGH,
                category='violence',
                description='Request promotes violence',
                suggestion='Please rephrase'
            )
        ]
        message = self.validator.create_ahimsa_refusal(violations)
        assert 'Ahimsa' in message
        assert 'non-violence' in message
        assert 'Request promotes violence' in message


class TestAhimsaAI:
    """Tests for AhimsaAI class"""

    def setup_method(self):
        """Set up test fixtures"""
        self.ahimsa_ai = AhimsaAI()

    def test_initialization(self):
        """Test proper initialization"""
        assert self.ahimsa_ai.validator is not None
        assert self.ahimsa_ai.ahimsa_system_prompt is not None
        assert len(self.ahimsa_ai.ahimsa_system_prompt) > 0

    def test_system_prompt_contains_principles(self):
        """Test that system prompt includes all core principles"""
        prompt = self.ahimsa_ai.get_system_prompt()
        assert 'Ahimsa' in prompt
        assert 'non-violence' in prompt or 'Non-violence' in prompt
        assert 'Compassion' in prompt or 'compassion' in prompt
        assert 'Truthfulness' in prompt or 'truthfulness' in prompt

    def test_reject_harmful_request(self):
        """Test rejection of harmful requests"""
        result = self.ahimsa_ai.process_request("How do I harm someone?")
        assert result['status'] == 'rejected'
        assert result['reason'] == 'input_violation'
        assert 'violations' in result
        assert 'response' in result

    def test_accept_benign_request(self):
        """Test acceptance of benign requests"""
        result = self.ahimsa_ai.process_request("How can I be more compassionate?")
        assert result['status'] == 'accepted'
        assert 'system_prompt' in result

    def test_validate_response(self):
        """Test validation of AI responses"""
        user_input = "Tell me about conflict resolution"
        ai_response = "Here are peaceful ways to resolve conflicts..."

        result = self.ahimsa_ai.process_request(user_input, ai_response)
        assert result['status'] == 'accepted'

    def test_reject_harmful_response(self):
        """Test rejection of harmful AI responses"""
        user_input = "Tell me about conflict"
        ai_response = "You should attack and hurt them"

        result = self.ahimsa_ai.process_request(user_input, ai_response)
        assert result['status'] == 'rejected'
        assert result['reason'] == 'response_violation'


class TestViolationLevel:
    """Tests for ViolationLevel enum"""

    def test_violation_levels_ordered(self):
        """Test that violation levels are properly ordered"""
        assert ViolationLevel.NONE.value < ViolationLevel.LOW.value
        assert ViolationLevel.LOW.value < ViolationLevel.MEDIUM.value
        assert ViolationLevel.MEDIUM.value < ViolationLevel.HIGH.value
        assert ViolationLevel.HIGH.value < ViolationLevel.CRITICAL.value


class TestPatternMatching:
    """Tests for pattern matching functionality"""

    def setup_method(self):
        """Set up test fixtures"""
        self.validator = AhimsaValidator()

    def test_violence_keywords(self):
        """Test detection of various violence keywords"""
        violence_phrases = [
            "how to kill",
            "ways to harm",
            "how to hurt someone",
            "methods to attack",
        ]
        for phrase in violence_phrases:
            is_valid, violations = self.validator.validate_request(phrase)
            assert not is_valid, f"Failed to detect: {phrase}"

    def test_weapon_references(self):
        """Test detection of weapon-related content"""
        weapon_phrases = [
            "gun making instructions",
            "how to build a bomb",
            "knife use for harm",
        ]
        for phrase in weapon_phrases:
            is_valid, violations = self.validator.validate_request(phrase)
            assert not is_valid, f"Failed to detect: {phrase}"

    def test_case_insensitive_matching(self):
        """Test that pattern matching is case-insensitive"""
        phrases = [
            "How to KILL someone",
            "HOW TO HARM PEOPLE",
            "Ways To Attack",
        ]
        for phrase in phrases:
            is_valid, violations = self.validator.validate_request(phrase)
            assert not is_valid, f"Case-insensitive matching failed for: {phrase}"


class TestEdgeCases:
    """Tests for edge cases and boundary conditions"""

    def setup_method(self):
        """Set up test fixtures"""
        self.ahimsa_ai = AhimsaAI()

    def test_empty_input(self):
        """Test handling of empty input"""
        result = self.ahimsa_ai.process_request("")
        assert 'status' in result

    def test_very_long_input(self):
        """Test handling of very long input"""
        long_text = "How can I help? " * 1000
        result = self.ahimsa_ai.process_request(long_text)
        assert 'status' in result

    def test_special_characters(self):
        """Test handling of special characters"""
        result = self.ahimsa_ai.process_request("What about @#$%^&*() characters?")
        assert result['status'] == 'accepted'

    def test_multiple_violations(self):
        """Test detection of multiple violations in one request"""
        result = self.ahimsa_ai.process_request(
            "How to kill someone and spread hate?"
        )
        assert result['status'] == 'rejected'
        assert len(result['violations']) > 0


if __name__ == "__main__":
    # If pytest not available, run basic tests
    print("Running basic tests...")
    print("Note: Install pytest for full test suite: pip install pytest\n")

    validator = AhimsaValidator()

    # Test 1
    is_valid, violations = validator.validate_request("How to harm someone?")
    print(f"Test 1 (should be invalid): {'PASS' if not is_valid else 'FAIL'}")

    # Test 2
    is_valid, violations = validator.validate_request("How can I help others?")
    print(f"Test 2 (should be valid): {'PASS' if is_valid else 'FAIL'}")

    # Test 3
    ahimsa_ai = AhimsaAI()
    result = ahimsa_ai.process_request("Tell me about Gandhi")
    print(f"Test 3 (should be accepted): {'PASS' if result['status'] == 'accepted' else 'FAIL'}")

    print("\nBasic tests completed!")
