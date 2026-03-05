import asyncio
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import structlog

from ..core.models import Thought, ThoughtStage
from ..core.config import ThinkLayerConfig, LLMConfig
from ..utils import parse_json_object_from_response

logger = structlog.get_logger(__name__)


@dataclass
class ThoughtGraphNode:
    """A node in the Graph of Thoughts"""

    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    thought: str = ""
    dependencies: List[str] = field(default_factory=list)
    dependents: List[str] = field(default_factory=list)
    score: Optional[float] = None
    evaluation: Optional[str] = None
    solution: str = ""


class GraphOfThoughts:
    """
    Graph of Thoughts - Dependency-aware reasoning with parallel sub-problem solving.

    Steps:
    1. Decompose the query into sub-problems via LLM
    2. Identify dependency graph between sub-problems
    3. Topological sort, solve independent sub-problems in parallel
    4. Aggregate terminal node solutions
    5. Synthesize final answer
    """

    def __init__(self, config: ThinkLayerConfig, llm_config: Optional[LLMConfig] = None):
        self.config = config
        self.llm_config = llm_config
        self.logger = logger.bind(component="graph_of_thoughts")

    async def solve(
        self,
        query: str,
        context: Dict[str, Any],
        llm_client: Any,
    ) -> Tuple[List[Thought], str]:
        """
        Solve a problem using Graph of Thoughts.

        Args:
            query: The problem to solve
            context: Retrieved context
            llm_client: LLM client for generation

        Returns:
            Tuple of (thoughts, final_answer)
        """
        self.logger.info("Starting Graph of Thoughts solving", query=query[:100])

        try:
            nodes = await self._decompose_query(query, context, llm_client)

            if not nodes:
                nodes = [ThoughtGraphNode(thought=query)]

            await self._solve_graph(nodes, query, context, llm_client)

            terminal_solutions = self._collect_terminal_solutions(nodes)

            final_answer = await self._synthesize_answer(
                query, nodes, terminal_solutions, llm_client
            )

            thoughts = self._nodes_to_thoughts(nodes)

            self.logger.info(
                "Graph of Thoughts solving completed",
                node_count=len(nodes),
                thought_count=len(thoughts),
            )

            return thoughts, final_answer

        except Exception as e:
            self.logger.error("Graph of Thoughts solving failed", error=str(e))
            raise


    async def _decompose_query(
        self,
        query: str,
        context: Dict[str, Any],
        llm_client: Any,
    ) -> List[ThoughtGraphNode]:
        """Decompose query into sub-problems with dependency relationships."""

        context_summary = ""
        if context.get("vector_results"):
            context_summary += f"Relevant documents: {len(context['vector_results'])} found\n"

        prompt = f"""
You are solving a problem using Graph of Thoughts reasoning.
Break the following query into smaller sub-problems and specify which sub-problems depend on which.

QUERY: {query}

CONTEXT:
{context_summary}

Return your response as a JSON object:

{{
    "sub_problems": [
        {{
            "id": "sp1",
            "description": "First sub-problem",
            "depends_on": []
        }},
        {{
            "id": "sp2",
            "description": "Second sub-problem that depends on sp1",
            "depends_on": ["sp1"]
        }}
    ]
}}

IMPORTANT: Return ONLY valid JSON. No additional text.
"""

        try:
            response = await llm_client.generate(prompt.strip())
            data = parse_json_object_from_response(response)

            if not data or "sub_problems" not in data:
                return []

            nodes: Dict[str, ThoughtGraphNode] = {}
            for sp in data["sub_problems"]:
                node_id = sp.get("id", uuid.uuid4().hex[:8])
                node = ThoughtGraphNode(
                    id=node_id,
                    thought=sp.get("description", ""),
                    dependencies=sp.get("depends_on", []),
                )
                nodes[node_id] = node

            for node in nodes.values():
                for dep_id in node.dependencies:
                    if dep_id in nodes:
                        nodes[dep_id].dependents.append(node.id)

            return list(nodes.values())

        except Exception as e:
            self.logger.error("Failed to decompose query", error=str(e))
            return []


    async def _solve_graph(
        self,
        nodes: List[ThoughtGraphNode],
        query: str,
        context: Dict[str, Any],
        llm_client: Any,
    ) -> None:
        """Solve graph nodes in topological order using Kahn's algorithm.

        Properly detects cycles by tracking in-degree counts. Independent nodes
        at each level are solved in parallel via asyncio.gather.
        """
        node_map: Dict[str, ThoughtGraphNode] = {n.id: n for n in nodes}
        valid_ids = set(node_map.keys())

        for node in nodes:
            node.dependencies = [d for d in node.dependencies if d in valid_ids]

        in_degree: Dict[str, int] = {n.id: len(n.dependencies) for n in nodes}

        solved: set = set()

        while True:
            ready = [
                node_map[nid]
                for nid, deg in in_degree.items()
                if deg == 0 and nid not in solved
            ]
            if not ready:
                break

            tasks = [
                self._solve_single_node(node, node_map, query, context, llm_client)
                for node in ready
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            for node, result in zip(ready, results):
                if isinstance(result, Exception):
                    self.logger.warning(
                        "Node solving failed", node_id=node.id, error=str(result)
                    )
                    node.solution = f"Failed to solve: {node.thought}"
                solved.add(node.id)

                for dependent_id in node.dependents:
                    if dependent_id in in_degree:
                        in_degree[dependent_id] -= 1

            for node in ready:
                in_degree[node.id] = -1

        unsolved = [n for n in nodes if n.id not in solved]
        if unsolved:
            self.logger.warning(
                "Dependency cycle detected, force-solving remaining nodes",
                unsolved_count=len(unsolved),
                unsolved_ids=[n.id for n in unsolved],
            )
            for node in unsolved:
                node.solution = f"[cycle-broken] {node.thought}"
                solved.add(node.id)

    async def _solve_single_node(
        self,
        node: ThoughtGraphNode,
        node_map: Dict[str, ThoughtGraphNode],
        query: str,
        context: Dict[str, Any],
        llm_client: Any,
    ) -> None:
        """Solve a single graph node using LLM."""

        dep_context = ""
        for dep_id in node.dependencies:
            dep_node = node_map.get(dep_id)
            if dep_node and dep_node.solution:
                dep_context += f"- {dep_node.thought}: {dep_node.solution}\n"

        prompt = f"""
You are solving a sub-problem as part of a larger Graph of Thoughts reasoning process.

ORIGINAL QUERY: {query}

SUB-PROBLEM: {node.thought}

{"DEPENDENCY SOLUTIONS:" + chr(10) + dep_context if dep_context else ""}

Solve this sub-problem. Return your response as a JSON object:

{{
    "solution": "Your solution to this sub-problem",
    "confidence": 0.85
}}

IMPORTANT: Return ONLY valid JSON. No additional text.
"""

        try:
            response = await llm_client.generate(prompt.strip())
            data = parse_json_object_from_response(response)

            if data:
                node.solution = data.get("solution", response.strip())
                node.score = data.get("confidence")
            else:
                node.solution = response.strip()

        except Exception as e:
            self.logger.error(
                "Failed to solve node", node_id=node.id, error=str(e)
            )
            node.solution = f"Failed: {str(e)}"


    def _collect_terminal_solutions(
        self, nodes: List[ThoughtGraphNode]
    ) -> List[str]:
        """Collect solutions from terminal (leaf) nodes."""
        terminal = [n for n in nodes if not n.dependents]
        if not terminal:
            terminal = nodes  # fallback: use all
        return [n.solution for n in terminal if n.solution]

    async def _synthesize_answer(
        self,
        query: str,
        nodes: List[ThoughtGraphNode],
        terminal_solutions: List[str],
        llm_client: Any,
    ) -> str:
        """Synthesize a final answer from all sub-problem solutions."""

        all_solutions = "\n".join(
            f"- {n.thought}: {n.solution}" for n in nodes if n.solution
        )

        prompt = f"""
Based on the following sub-problem solutions, provide a clear and comprehensive answer to the original query.

QUERY: {query}

SUB-PROBLEM SOLUTIONS:
{all_solutions}

TASK: Synthesize a final answer that:
1. Directly addresses the original query
2. Integrates insights from all sub-problems
3. Is clear, concise, and well-structured

FINAL ANSWER:
"""

        try:
            response = await llm_client.generate(prompt.strip())
            return response.strip()
        except Exception as e:
            self.logger.error("Failed to synthesize answer", error=str(e))
            return " ".join(terminal_solutions) if terminal_solutions else "Unable to generate a response."


    def _nodes_to_thoughts(self, nodes: List[ThoughtGraphNode]) -> List[Thought]:
        """Convert graph nodes to Thought objects."""
        thoughts = []
        total = len(nodes)

        for i, node in enumerate(nodes):
            thought = Thought(
                thought=f"{node.thought}: {node.solution}" if node.solution else node.thought,
                thought_number=i + 1,
                total_thoughts=total,
                next_thought_needed=(i < total - 1),
                needs_more_thoughts=(i < total - 1),
                stage=ThoughtStage.ANALYSIS,
                score=node.score,
                metadata={
                    "node_id": node.id,
                    "dependencies": node.dependencies,
                    "dependents": node.dependents,
                },
            )
            thoughts.append(thought)

        return thoughts
