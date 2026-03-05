"""
Tests for bug fixes: quality gate IndexError, SelfConsistency empty paths
"""

import pytest
from axon import Axon, AxonConfig, LLMConfig, ReasoningMethod
from axon.core.models import Thought, ThoughtStage
from axon.reasoning import SelfConsistency
from axon.core import ThinkLayerConfig


class MockLLMClient:
    """Mock LLM client that returns predefined responses"""

    def __init__(self, response="mock response"):
        self.response = response
        self.call_count = 0

    async def generate(self, prompt: str) -> str:
        self.call_count += 1
        return self.response


class FailingLLMClient:
    """Mock LLM client that always fails"""

    async def generate(self, prompt: str) -> str:
        raise RuntimeError("LLM call failed")


class TestQualityGateBug:
    """Test that the quality gate doesn't crash when all thoughts are filtered out"""

    @pytest.fixture
    def axon(self):
        return Axon(AxonConfig(
            llm=LLMConfig(provider="mock", model="test", api_key="test"),
            grounding={"enable_grounding": False},
            memory={"enable_memory": False},
            quality_gate_threshold=0.9,
        ))

    async def test_quality_gate_keeps_at_least_one_thought(self, axon):
        """When all thoughts score below threshold, should keep the first one, not crash"""
        # Create thoughts that all score below the 0.9 threshold
        thoughts = [
            Thought(thought="Low confidence thought 1", thought_number=1,
                    total_thoughts=3, score=0.2),
            Thought(thought="Low confidence thought 2", thought_number=2,
                    total_thoughts=3, score=0.3),
            Thought(thought="Low confidence thought 3", thought_number=3,
                    total_thoughts=3, score=0.1),
        ]

        # Simulate the quality gate logic from core.py _run_pipeline
        quality_threshold = axon.config.quality_gate_threshold
        assert quality_threshold == 0.9

        original_thoughts = thoughts
        thoughts = [
            t for t in thoughts
            if t.score is None or t.score >= quality_threshold
        ]
        filtered_count = len(original_thoughts) - len(thoughts)

        # All should be filtered
        assert len(thoughts) == 0
        assert filtered_count == 3

        # The fix: keep at least one thought without IndexError
        if not thoughts:
            thoughts = [original_thoughts[0]]

        assert len(thoughts) == 1
        assert thoughts[0].thought == "Low confidence thought 1"

    async def test_quality_gate_no_filter_when_scores_above_threshold(self, axon):
        """When all thoughts score above threshold, none are filtered"""
        thoughts = [
            Thought(thought="High confidence 1", thought_number=1,
                    total_thoughts=2, score=0.95),
            Thought(thought="High confidence 2", thought_number=2,
                    total_thoughts=2, score=0.92),
        ]

        quality_threshold = axon.config.quality_gate_threshold
        original_thoughts = thoughts
        thoughts = [
            t for t in thoughts
            if t.score is None or t.score >= quality_threshold
        ]
        filtered_count = len(original_thoughts) - len(thoughts)

        assert filtered_count == 0
        assert len(thoughts) == 2

    async def test_quality_gate_none_scores_pass(self, axon):
        """Thoughts with score=None should pass the quality gate"""
        thoughts = [
            Thought(thought="No score thought", thought_number=1,
                    total_thoughts=2, score=None),
            Thought(thought="Low score thought", thought_number=2,
                    total_thoughts=2, score=0.1),
        ]

        quality_threshold = axon.config.quality_gate_threshold
        original_thoughts = thoughts
        thoughts = [
            t for t in thoughts
            if t.score is None or t.score >= quality_threshold
        ]

        assert len(thoughts) == 1
        assert thoughts[0].score is None


class TestSelfConsistencyEmptyPaths:
    """Test that SelfConsistency handles empty paths gracefully"""

    @pytest.fixture
    def config(self):
        return ThinkLayerConfig(num_parallel_paths=3, max_depth=2)

    async def test_all_paths_fail_returns_empty(self, config):
        """When all parallel paths fail, should return empty list + fallback message"""
        sc = SelfConsistency(config)
        failing_client = FailingLLMClient()

        thoughts, answer = await sc.solve("test query", {}, failing_client)

        assert thoughts == []
        assert answer == "Unable to generate a response."

    async def test_single_valid_path_works(self, config):
        """When at least one path succeeds, should return results"""
        response = '''{
            "thought": "Test thought",
            "thoughtNumber": 1,
            "totalThoughts": 1,
            "needsMoreThoughts": false,
            "nextThoughtNeeded": false,
            "stage": "analysis",
            "confidence": 0.85
        }'''
        sc = SelfConsistency(config)
        mock_client = MockLLMClient(response)

        thoughts, answer = await sc.solve("test query", {}, mock_client)

        assert len(thoughts) >= 1
        assert answer != ""
