"""
Basic tests for Ax0n functionality
Tests for the updated API matching PLAN-1.md specifications
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from axon import Axon, AxonConfig, ReasoningMethod, LLMConfig
from axon.core.models import Thought, ThoughtStage, MemoryEntry


class TestAxonBasic:
    """Test basic Axon functionality"""
    
    @pytest.fixture
    def axon_config(self):
        """Create test configuration"""
        return AxonConfig(
            think_layer=dict(
                max_depth=3,
                enable_parallel=True,
                auto_iterate=True
            ),
            llm=LLMConfig(
                provider="openai",
                model="gpt-4",
                api_key="test-key"
            ),
            memory=dict(enable_memory=False),
            grounding=dict(enable_grounding=False)
        )
    
    @pytest.fixture
    def mock_axon(self, axon_config):
        """Create a mock Axon instance for testing"""
        ax = Axon(axon_config)
        
        # Mock the LLM client
        mock_client = AsyncMock()
        mock_client.generate = AsyncMock(return_value='''
        {
            "thought": "This is a test thought about AI reasoning",
            "thoughtNumber": 1,
            "totalThoughts": 3,
            "nextThoughtNeeded": true,
            "needsMoreThoughts": true,
            "stage": "analysis",
            "tags": ["test", "reasoning"],
            "score": 0.85
        }
        ''')
        
        ax._get_llm_client = AsyncMock(return_value=mock_client)
        
        return ax
    
    @pytest.mark.asyncio
    async def test_axon_initialization(self, axon_config):
        """Test that Axon initializes correctly"""
        ax = Axon(axon_config)
        
        assert ax is not None
        assert ax.config.llm.provider == "openai"
        assert ax.config.llm.model == "gpt-4"
        assert ax.config.grounding.enable_grounding is False
        assert ax.config.memory.enable_memory is False
        assert ax.config.think_layer.max_depth == 3
        assert ax.config.think_layer.enable_parallel is True
    
    @pytest.mark.asyncio
    async def test_thought_generation(self, mock_axon):
        """Test basic thought generation with the new API"""
        query = "What is artificial intelligence?"
        
        result = await mock_axon.think(query, ReasoningMethod.COT)
        
        assert result is not None
        assert isinstance(result, dict)
        assert 'thoughts' in result
        assert 'answer' in result
        assert 'method' in result
        assert 'metadata' in result
        assert result['method'] == ReasoningMethod.COT.value
    
    def test_thought_model(self):
        """Test Thought model creation"""
        thought = Thought(
            thought="AI systems can benefit from structured reasoning",
            thought_number=1,
            total_thoughts=3,
            next_thought_needed=True,
            needs_more_thoughts=True,
            stage=ThoughtStage.ANALYSIS,
            tags=["ai", "reasoning"],
            score=0.9
        )
        
        assert thought is not None
        assert thought.thought == "AI systems can benefit from structured reasoning"
        assert thought.thought_number == 1
        assert thought.total_thoughts == 3
        assert thought.stage == ThoughtStage.ANALYSIS
        assert "ai" in thought.tags
        assert thought.score == 0.9
    
    def test_memory_entry_model(self):
        """Test MemoryEntry model with new fields"""
        memory = MemoryEntry(
            id="test-1",
            content="Test memory content",
            confidence=0.9,
            salience_score=0.8,
            agent_id="agent-1",
            is_shared=False,
            memory_type="semantic"
        )
        
        assert memory is not None
        assert memory.id == "test-1"
        assert memory.confidence == 0.9
        assert memory.salience_score == 0.8
        assert memory.agent_id == "agent-1"
        assert memory.is_shared is False
        assert memory.memory_type == "semantic"
    
    def test_thought_stage_enum(self):
        """Test ThoughtStage enum values"""
        assert ThoughtStage.PROBLEM_DEFINITION.value == "problem_definition"
        assert ThoughtStage.RESEARCH.value == "research"
        assert ThoughtStage.ANALYSIS.value == "analysis"
        assert ThoughtStage.SYNTHESIS.value == "synthesis"
        assert ThoughtStage.CONCLUSION.value == "conclusion"
        assert ThoughtStage.VERIFICATION.value == "verification"
    
    def test_reasoning_method_enum(self):
        """Test ReasoningMethod enum values"""
        assert ReasoningMethod.COT.value == "chain_of_thought"
        assert ReasoningMethod.SELF_CONSISTENCY.value == "self_consistency"
        assert ReasoningMethod.AOT.value == "algorithm_of_thoughts"
        assert ReasoningMethod.TOT.value == "tree_of_thoughts"
        assert ReasoningMethod.GOT.value == "graph_of_thoughts"
    
    def test_config_defaults(self):
        """Test configuration defaults"""
        config = AxonConfig(
            llm=LLMConfig(provider="openai", model="gpt-4", api_key="test")
        )
        
        # Check defaults
        assert config.think_layer.max_depth == 5
        assert config.think_layer.enable_parallel is True
        assert config.think_layer.auto_iterate is True
        assert config.memory.enable_memory is True
        assert config.grounding.enable_grounding is True
        assert config.memory.storage_provider == "weaviate"
        assert config.grounding.search_provider == "google"
    
    @pytest.mark.asyncio
    async def test_get_available_methods(self, mock_axon):
        """Test getting available reasoning methods"""
        methods = mock_axon.get_available_methods()
        
        assert methods is not None
        assert isinstance(methods, list)
        assert len(methods) > 0
        
        # Check that each method has required fields
        for method in methods:
            assert 'name' in method
            assert 'description' in method
            assert 'complexity' in method
    
    @pytest.mark.asyncio
    async def test_method_info(self, mock_axon):
        """Test getting specific method information"""
        info = mock_axon.get_method_info(ReasoningMethod.TOT)
        
        assert info is not None
        assert isinstance(info, dict)
        assert 'description' in info
        assert 'complexity' in info


if __name__ == "__main__":
    pytest.main([__file__]) 