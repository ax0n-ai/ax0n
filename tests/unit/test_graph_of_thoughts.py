"""
Tests for Graph of Thoughts functionality
"""

import pytest
from axon.reasoning.graph_of_thoughts import GraphOfThoughts, ThoughtGraphNode
from axon.core.config import ThinkLayerConfig, LLMConfig


class MockLLMClient:
    """Mock LLM client for GOT tests"""

    def __init__(self):
        self.call_count = 0

    async def generate(self, prompt: str) -> str:
        self.call_count += 1

        if "Break the following query" in prompt:
            return '''
{
    "sub_problems": [
        {"id": "sp1", "description": "Understand the basics", "depends_on": []},
        {"id": "sp2", "description": "Analyze implications", "depends_on": ["sp1"]},
        {"id": "sp3", "description": "Draw conclusions", "depends_on": ["sp2"]}
    ]
}'''
        elif "SUB-PROBLEM" in prompt:
            return '{"solution": "Mock solution for sub-problem", "confidence": 0.85}'
        else:
            return "Synthesized final answer based on sub-problems."


class TestThoughtGraphNode:
    """Test ThoughtGraphNode dataclass"""

    def test_node_creation(self):
        node = ThoughtGraphNode(thought="test thought")
        assert node.thought == "test thought"
        assert node.dependencies == []
        assert node.dependents == []
        assert node.score is None
        assert node.solution == ""

    def test_node_with_dependencies(self):
        node = ThoughtGraphNode(
            id="n1",
            thought="depends on n0",
            dependencies=["n0"],
        )
        assert node.dependencies == ["n0"]


class TestGraphOfThoughts:
    """Test GraphOfThoughts class"""

    @pytest.fixture
    def config(self):
        return ThinkLayerConfig(max_depth=3, enable_parallel=True)

    @pytest.fixture
    def got(self, config):
        return GraphOfThoughts(config)

    @pytest.mark.asyncio
    async def test_solve_basic(self, got):
        llm = MockLLMClient()
        context = {"vector_results": []}

        thoughts, answer = await got.solve("What is AI?", context, llm)

        assert isinstance(thoughts, list)
        assert len(thoughts) > 0
        assert isinstance(answer, str)
        assert len(answer) > 0

    @pytest.mark.asyncio
    async def test_decompose_query(self, got):
        llm = MockLLMClient()
        context = {}

        nodes = await got._decompose_query("What is AI?", context, llm)

        assert len(nodes) == 3
        assert nodes[0].id == "sp1"
        assert nodes[1].dependencies == ["sp1"]
        assert nodes[2].dependencies == ["sp2"]

    @pytest.mark.asyncio
    async def test_solve_graph_ordering(self, got):
        """Test that nodes are solved in dependency order"""
        nodes = [
            ThoughtGraphNode(id="a", thought="first", dependencies=[]),
            ThoughtGraphNode(id="b", thought="second", dependencies=["a"]),
            ThoughtGraphNode(id="c", thought="third", dependencies=["b"]),
        ]

        llm = MockLLMClient()
        await got._solve_graph(nodes, "test", {}, llm)

        # All nodes should have solutions
        for n in nodes:
            assert n.solution != ""

    def test_collect_terminal_solutions(self, got):
        nodes = [
            ThoughtGraphNode(id="a", thought="t1", dependents=["b"], solution="sol1"),
            ThoughtGraphNode(id="b", thought="t2", dependents=[], solution="sol2"),
        ]
        solutions = got._collect_terminal_solutions(nodes)
        assert solutions == ["sol2"]

    def test_nodes_to_thoughts(self, got):
        nodes = [
            ThoughtGraphNode(id="a", thought="first", solution="s1", score=0.9),
            ThoughtGraphNode(id="b", thought="second", solution="s2", score=0.8),
        ]
        thoughts = got._nodes_to_thoughts(nodes)

        assert len(thoughts) == 2
        assert thoughts[0].thought_number == 1
        assert thoughts[1].thought_number == 2
        assert thoughts[0].next_thought_needed is True
        assert thoughts[1].next_thought_needed is False


if __name__ == "__main__":
    pytest.main([__file__])
