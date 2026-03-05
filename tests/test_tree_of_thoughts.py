"""
Tests for Tree of Thoughts functionality
"""

import pytest
from axon.reasoning import TreeOfThoughts, ThoughtNode
from axon.reasoning.tree_of_thoughts import EvaluationLevel
from axon.core.models import Thought, ThoughtStage
from axon.core.config import ThinkLayerConfig, LLMConfig


class TestThoughtNode:
    """Test ThoughtNode functionality"""

    def test_thought_node_creation(self):
        """Test creating a ThoughtNode"""
        node = ThoughtNode(
            thought="Test thought",
            thought_number=1,
            branch_id="test_branch",
            evaluation=EvaluationLevel.SURE,
            score=0.9
        )

        assert node.thought == "Test thought"
        assert node.thought_number == 1
        assert node.branch_id == "test_branch"
        assert node.evaluation == EvaluationLevel.SURE
        assert node.score == 0.9
        assert node.children == []

    def test_thought_node_add_child(self):
        """Test adding child nodes"""
        parent = ThoughtNode("Parent thought", 1)
        child = ThoughtNode("Child thought", 2)

        parent.add_child(child)

        assert len(parent.children) == 1
        assert parent.children[0] == child
        assert child.parent_id == parent.id
        assert child.branch_id == parent.branch_id

    def test_thought_node_to_dict(self):
        """Test converting node to dictionary"""
        node = ThoughtNode(
            thought="Test thought",
            thought_number=1,
            evaluation=EvaluationLevel.MAYBE,
            score=0.7
        )

        node_dict = node.to_dict()

        assert node_dict['thought'] == "Test thought"
        assert node_dict['thought_number'] == 1
        assert node_dict['evaluation'] == "maybe"
        assert node_dict['score'] == 0.7
        assert 'children' in node_dict


class TestTreeOfThoughts:
    """Test TreeOfThoughts functionality"""

    @pytest.fixture
    def mock_config(self):
        """Create mock configuration"""
        llm_config = LLMConfig(
            provider="openai",
            model="gpt-4",
            api_key="test-key"
        )

        think_config = ThinkLayerConfig(
            max_depth=3,
            enable_parallel=True,
            max_parallel_branches=3
        )

        return think_config, llm_config

    @pytest.fixture
    def tree_of_thoughts(self, mock_config):
        """Create TreeOfThoughts instance"""
        think_config, llm_config = mock_config
        return TreeOfThoughts(think_config, llm_config)

    def test_tree_of_thoughts_initialization(self, tree_of_thoughts):
        """Test TreeOfThoughts initialization"""
        assert tree_of_thoughts.max_branches == 3
        assert tree_of_thoughts.max_depth == 3
        assert tree_of_thoughts.evaluation_threshold == 0.7
        assert tree_of_thoughts.root_nodes == []
        assert tree_of_thoughts.current_branches == {}

    def test_build_candidate_prompt(self, tree_of_thoughts):
        """Test building candidate generation prompt"""
        query = "What is 2+2?"
        context = {"vector_results": [{"content": "Math facts"}], "user_attributes": {"level": "basic"}}

        prompt = tree_of_thoughts._build_candidate_prompt(query, context, 1)

        assert "What is 2+2?" in prompt
        assert "Generate 3 different thought candidates" in prompt
        assert "JSON array" in prompt

    def test_parse_candidate_response(self, tree_of_thoughts):
        """Test parsing candidate response"""
        response = '''
        [
            {
                "thought": "First candidate",
                "approach": "direct method",
                "confidence": 0.8
            },
            {
                "thought": "Second candidate",
                "approach": "alternative method",
                "confidence": 0.6
            }
        ]
        '''

        candidates = tree_of_thoughts._parse_candidate_response(response)

        assert len(candidates) == 2
        assert candidates[0]['thought'] == "First candidate"
        assert candidates[0]['confidence'] == 0.8
        assert candidates[1]['thought'] == "Second candidate"
        assert candidates[1]['confidence'] == 0.6

    def test_convert_branch_to_thoughts(self, tree_of_thoughts):
        """Test converting branch to Thought objects"""
        # Create a simple branch
        root = ThoughtNode("Root thought", 1, score=0.8)
        child1 = ThoughtNode("Child 1", 2, score=0.7)
        child2 = ThoughtNode("Child 2", 3, score=0.9)

        root.add_child(child1)
        child1.add_child(child2)

        # Register nodes in current_branches so _find_node_by_id can find parents
        tree_of_thoughts.current_branches[root.branch_id] = [root, child1, child2]

        thoughts = tree_of_thoughts._convert_branch_to_thoughts(child2)

        assert len(thoughts) == 3
        assert thoughts[0].thought == "Root thought"
        assert thoughts[0].thought_number == 1
        assert thoughts[1].thought == "Child 1"
        assert thoughts[1].thought_number == 2
        assert thoughts[2].thought == "Child 2"
        assert thoughts[2].thought_number == 3

        # Check that total_thoughts and next_thought_needed are set correctly
        for thought in thoughts:
            assert thought.total_thoughts == 3

        assert thoughts[0].next_thought_needed is True
        assert thoughts[1].next_thought_needed is True
        assert thoughts[2].next_thought_needed is False


if __name__ == "__main__":
    pytest.main([__file__])
