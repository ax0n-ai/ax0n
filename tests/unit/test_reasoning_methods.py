"""
Tests for multiple reasoning methods
"""

import pytest
import asyncio
from typing import Dict, Any
from axon.reasoning import (
    ReasoningMethod,
    ChainOfThoughts,
    SelfConsistency,
    AlgorithmOfThoughts,
    ReasoningOrchestrator
)
from axon.core import ThinkLayerConfig, LLMConfig
from axon.core import Thought, ThoughtStage


class MockLLMClient:
    """Mock LLM client for testing"""
    
    def __init__(self, responses: Dict[str, str] = None):
        self.responses = responses or {}
        self.call_count = 0
    
    async def generate(self, prompt: str) -> str:
        """Generate mock response"""
        self.call_count += 1
        
        # Return predefined response if available
        if self.responses:
            return list(self.responses.values())[(self.call_count - 1) % len(self.responses)]
        
        # Default response
        return '''
{
    "thought": "This is a test thought",
    "thoughtNumber": 1,
    "totalThoughts": 3,
    "needsMoreThoughts": true,
    "nextThoughtNeeded": true,
    "stage": "analysis",
    "confidence": 0.8
}'''


@pytest.fixture
def config():
    """Test configuration"""
    return ThinkLayerConfig(
        max_depth=3,
        enable_parallel=True,
        auto_iterate=True
    )


@pytest.fixture
def llm_config():
    """Test LLM configuration"""
    return LLMConfig(
        provider="mock",
        model="test-model",
        api_key="test-key"
    )


@pytest.fixture
def context():
    """Test context"""
    return {
        "vector_results": [{"id": "1", "content": "test content"}],
        "user_attributes": {"user_id": "test_user"}
    }


class TestChainOfThoughts:
    """Test Chain of Thoughts reasoning"""
    
    @pytest.mark.asyncio
    async def test_cot_basic_flow(self, config, llm_config, context):
        """Test basic Chain of Thoughts flow"""
        cot = ChainOfThoughts(config, llm_config)
        llm_client = MockLLMClient()
        
        thoughts, answer = await cot.solve("Test query", context, llm_client)
        
        assert isinstance(thoughts, list)
        assert isinstance(answer, str)
        assert len(thoughts) > 0
        assert all(isinstance(thought, Thought) for thought in thoughts)
    
    @pytest.mark.asyncio
    async def test_cot_early_termination(self, config, llm_config, context):
        """Test CoT with early termination"""
        responses = {
            "step1": '''
{
    "thought": "First thought",
    "thoughtNumber": 1,
    "totalThoughts": 3,
    "needsMoreThoughts": false,
    "nextThoughtNeeded": false,
    "stage": "conclusion",
    "confidence": 0.9
}'''
        }
        
        cot = ChainOfThoughts(config, llm_config)
        llm_client = MockLLMClient(responses)
        
        thoughts, answer = await cot.solve("Test query", context, llm_client)
        
        assert len(thoughts) == 1
        assert not thoughts[0].needs_more_thoughts
    
    @pytest.mark.asyncio
    async def test_cot_prompt_generation(self, config, llm_config, context):
        """Test CoT prompt generation"""
        cot = ChainOfThoughts(config, llm_config)
        
        prompt = cot._build_cot_prompt("Test query", context, [], 1)
        
        assert "Chain of Thoughts" in prompt
        assert "Test query" in prompt
        assert "JSON" in prompt


class TestSelfConsistency:
    """Test Self-Consistency reasoning"""
    
    @pytest.mark.asyncio
    async def test_sc_basic_flow(self, config, llm_config, context):
        """Test basic Self-Consistency flow"""
        sc = SelfConsistency(config, llm_config)
        llm_client = MockLLMClient()
        
        thoughts, answer = await sc.solve("Test query", context, llm_client)
        
        assert isinstance(thoughts, list)
        assert isinstance(answer, str)
        assert len(thoughts) > 0
    
    @pytest.mark.asyncio
    async def test_sc_parallel_paths(self, config, llm_config, context):
        """Test Self-Consistency with multiple parallel paths"""
        sc = SelfConsistency(config, llm_config)
        sc.num_paths = 2  # Reduce for testing
        
        responses = {
            "path1": '''
{
    "thought": "Path 1 thought",
    "thoughtNumber": 1,
    "totalThoughts": 2,
    "needsMoreThoughts": false,
    "nextThoughtNeeded": false,
    "stage": "conclusion",
    "confidence": 0.8
}''',
            "path2": '''
{
    "thought": "Path 2 thought",
    "thoughtNumber": 1,
    "totalThoughts": 2,
    "needsMoreThoughts": false,
    "nextThoughtNeeded": false,
    "stage": "conclusion",
    "confidence": 0.9
}'''
        }
        
        llm_client = MockLLMClient(responses)
        
        thoughts, answer = await sc.solve("Test query", context, llm_client)
        
        assert isinstance(thoughts, list)
        assert isinstance(answer, str)
    
    @pytest.mark.asyncio
    async def test_sc_voting(self, config, llm_config, context):
        """Test Self-Consistency voting mechanism"""
        sc = SelfConsistency(config, llm_config)
        
        answers = ["Answer 1", "Answer 2", "Answer 3"]
        llm_client = MockLLMClient()
        
        final_answer = await sc._vote_on_answers("Test query", answers, llm_client)
        
        assert isinstance(final_answer, str)
        assert len(final_answer) > 0


class TestAlgorithmOfThoughts:
    """Test Algorithm of Thoughts reasoning"""
    
    @pytest.mark.asyncio
    async def test_aot_basic_flow(self, config, llm_config, context):
        """Test basic Algorithm of Thoughts flow"""
        aot = AlgorithmOfThoughts(config, llm_config)
        llm_client = MockLLMClient()
        
        thoughts, answer = await aot.solve("Test query", context, llm_client)
        
        assert isinstance(thoughts, list)
        assert isinstance(answer, str)
    
    @pytest.mark.asyncio
    async def test_aot_algorithm_generation(self, config, llm_config, context):
        """Test AoT algorithm generation"""
        aot = AlgorithmOfThoughts(config, llm_config)

        # Provide a mock that returns algorithm-shaped JSON
        responses = {
            "algo": '''{
    "algorithm_name": "Test Algorithm",
    "description": "A test approach",
    "steps": [
        {"step_number": 1, "description": "Step one", "input": "query", "output": "result", "method": "analysis"}
    ],
    "estimated_steps": 1
}'''
        }
        llm_client = MockLLMClient(responses)

        algorithm = await aot._generate_algorithm("Test query", context, llm_client)

        assert isinstance(algorithm, dict)
        assert "steps" in algorithm
    
    @pytest.mark.asyncio
    async def test_aot_algorithm_execution(self, config, llm_config, context):
        """Test AoT algorithm execution"""
        aot = AlgorithmOfThoughts(config, llm_config)
        
        algorithm = {
            "steps": [
                {
                    "step_number": 1,
                    "description": "Test step",
                    "input": "test input",
                    "output": "test output",
                    "method": "test method"
                }
            ],
            "estimated_steps": 1
        }
        
        llm_client = MockLLMClient()
        
        thoughts = await aot._execute_algorithm(algorithm, "Test query", context, llm_client)
        
        assert isinstance(thoughts, list)


class TestReasoningOrchestrator:
    """Test Reasoning Orchestrator"""
    
    @pytest.mark.asyncio
    async def test_orchestrator_initialization(self, config, llm_config):
        """Test orchestrator initialization"""
        orchestrator = ReasoningOrchestrator(config, llm_config)
        
        assert orchestrator.cot is not None
        assert orchestrator.self_consistency is not None
        assert orchestrator.aot is not None
        assert orchestrator.tot is not None
    
    @pytest.mark.asyncio
    async def test_orchestrator_cot_method(self, config, llm_config, context):
        """Test orchestrator with CoT method"""
        orchestrator = ReasoningOrchestrator(config, llm_config)
        llm_client = MockLLMClient()
        
        thoughts, answer = await orchestrator.solve(
            "Test query", context, ReasoningMethod.COT, llm_client
        )
        
        assert isinstance(thoughts, list)
        assert isinstance(answer, str)
    
    @pytest.mark.asyncio
    async def test_orchestrator_sc_method(self, config, llm_config, context):
        """Test orchestrator with Self-Consistency method"""
        orchestrator = ReasoningOrchestrator(config, llm_config)
        llm_client = MockLLMClient()
        
        thoughts, answer = await orchestrator.solve(
            "Test query", context, ReasoningMethod.SELF_CONSISTENCY, llm_client
        )
        
        assert isinstance(thoughts, list)
        assert isinstance(answer, str)
    
    @pytest.mark.asyncio
    async def test_orchestrator_aot_method(self, config, llm_config, context):
        """Test orchestrator with AoT method"""
        orchestrator = ReasoningOrchestrator(config, llm_config)
        llm_client = MockLLMClient()
        
        thoughts, answer = await orchestrator.solve(
            "Test query", context, ReasoningMethod.AOT, llm_client
        )
        
        assert isinstance(thoughts, list)
        assert isinstance(answer, str)
    
    @pytest.mark.asyncio
    async def test_orchestrator_tot_method(self, config, llm_config, context):
        """Test orchestrator with ToT method"""
        orchestrator = ReasoningOrchestrator(config, llm_config)
        llm_client = MockLLMClient()
        
        thoughts, answer = await orchestrator.solve(
            "Test query", context, ReasoningMethod.TOT, llm_client
        )
        
        assert isinstance(thoughts, list)
        assert isinstance(answer, str)
    
    def test_orchestrator_method_info(self, config, llm_config):
        """Test orchestrator method information"""
        orchestrator = ReasoningOrchestrator(config, llm_config)
        
        for method in ReasoningMethod:
            description = orchestrator.get_method_description(method)
            complexity = orchestrator.get_method_complexity(method)
            
            assert isinstance(description, str)
            assert isinstance(complexity, str)
            assert len(description) > 0
            assert len(complexity) > 0
    
    @pytest.mark.asyncio
    async def test_orchestrator_invalid_method(self, config, llm_config, context):
        """Test orchestrator with invalid method raises an error"""
        orchestrator = ReasoningOrchestrator(config, llm_config)
        llm_client = MockLLMClient()

        with pytest.raises((ValueError, AttributeError)):
            await orchestrator.solve(
                "Test query", context, "invalid_method", llm_client
            )


class TestReasoningMethodEnum:
    """Test ReasoningMethod enum"""
    
    def test_enum_values(self):
        """Test enum values"""
        assert ReasoningMethod.COT == "chain_of_thought"
        assert ReasoningMethod.SELF_CONSISTENCY == "self_consistency"
        assert ReasoningMethod.AOT == "algorithm_of_thoughts"
        assert ReasoningMethod.TOT == "tree_of_thoughts"
        assert ReasoningMethod.GOT == "graph_of_thoughts"
    
    def test_enum_list(self):
        """Test getting all enum values"""
        methods = list(ReasoningMethod)
        assert len(methods) == 5
        assert ReasoningMethod.COT in methods
        assert ReasoningMethod.SELF_CONSISTENCY in methods
        assert ReasoningMethod.AOT in methods
        assert ReasoningMethod.TOT in methods
        assert ReasoningMethod.GOT in methods


class TestIntegration:
    """Integration tests"""
    
    @pytest.mark.asyncio
    async def test_all_methods_with_same_query(self, config, llm_config, context):
        """Test all reasoning methods with the same query return valid types"""
        orchestrator = ReasoningOrchestrator(config, llm_config)
        llm_client = MockLLMClient()

        query = "What is the capital of France?"

        for method in [ReasoningMethod.COT, ReasoningMethod.SELF_CONSISTENCY, ReasoningMethod.AOT]:
            thoughts, answer = await orchestrator.solve(query, context, method, llm_client)

            assert isinstance(thoughts, list)
            assert isinstance(answer, str)
            assert len(answer) > 0
    
    @pytest.mark.asyncio
    async def test_method_comparison(self, config, llm_config):
        """Test comparing different methods"""
        orchestrator = ReasoningOrchestrator(config, llm_config)
        
        # Test that all methods have different descriptions
        descriptions = set()
        complexities = set()
        
        for method in ReasoningMethod:
            description = orchestrator.get_method_description(method)
            complexity = orchestrator.get_method_complexity(method)
            
            descriptions.add(description)
            complexities.add(complexity)
        
        # Should have unique descriptions and complexities
        assert len(descriptions) == len(ReasoningMethod)
        assert len(complexities) > 0  # Some methods might have same complexity


if __name__ == "__main__":
    pytest.main([__file__]) 