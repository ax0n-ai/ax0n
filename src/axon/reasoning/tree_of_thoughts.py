"""
Tree of Thoughts (ToT) implementation for Ax0n
"""

import asyncio
import json
import uuid
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum
import structlog
from ..core.models import Thought, ThoughtStage
from ..core.config import ThinkLayerConfig, LLMConfig

logger = structlog.get_logger(__name__)


class EvaluationLevel(str, Enum):
    """Evaluation levels for thought candidates"""
    SURE = "sure"
    MAYBE = "maybe"
    IMPOSSIBLE = "impossible"


class ThoughtNode:
    """Represents a node in the Tree of Thoughts"""
    
    def __init__(
        self,
        thought: str,
        thought_number: int,
        branch_id: Optional[str] = None,
        parent_id: Optional[str] = None,
        evaluation: Optional[EvaluationLevel] = None,
        score: Optional[float] = None,
        **kwargs
    ):
        self.id = str(uuid.uuid4())
        self.thought = thought
        self.thought_number = thought_number
        self.branch_id = branch_id or f"branch_{uuid.uuid4().hex[:8]}"
        self.parent_id = parent_id
        self.evaluation = evaluation
        self.score = score
        self.children: List[ThoughtNode] = []
        self.metadata = kwargs
        
    def add_child(self, child: 'ThoughtNode') -> None:
        """Add a child node"""
        child.parent_id = self.id
        child.branch_id = self.branch_id
        self.children.append(child)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert node to dictionary"""
        return {
            'id': self.id,
            'thought': self.thought,
            'thought_number': self.thought_number,
            'branch_id': self.branch_id,
            'parent_id': self.parent_id,
            'evaluation': self.evaluation.value if self.evaluation else None,
            'score': self.score,
            'children': [child.to_dict() for child in self.children],
            'metadata': self.metadata
        }


class TreeOfThoughts:
    """
    Tree of Thoughts implementation with branching, evaluation, and backtracking
    """
    
    def __init__(self, config: ThinkLayerConfig, llm_config: LLMConfig):
        self.config = config
        self.llm_config = llm_config
        self.logger = logger.bind(component="tree_of_thoughts")
        
        # Tree state
        self.root_nodes: List[ThoughtNode] = []
        self.current_branches: Dict[str, List[ThoughtNode]] = {}
        self.evaluated_nodes: Dict[str, EvaluationLevel] = {}
        
        # Configuration
        self.max_branches = config.max_parallel_branches
        self.max_depth = config.max_depth
        self.evaluation_threshold = 0.7  # Minimum score to continue a branch
        
        self.logger.info("Tree of Thoughts initialized", 
                        max_branches=self.max_branches, 
                        max_depth=self.max_depth)
    
    async def solve(
        self,
        query: str,
        context: Dict[str, Any],
        llm_client: Any
    ) -> Tuple[List[Thought], str]:
        """
        Solve a problem using Tree of Thoughts
        
        Args:
            query: The problem to solve
            context: Retrieved context
            llm_client: LLM client for generation
            
        Returns:
            Tuple of (thoughts, final_answer)
        """
        self.logger.info("Starting Tree of Thoughts solving", query=query[:100])
        
        try:
            # Step 1: Generate initial thought candidates
            initial_thoughts = await self._generate_thought_candidates(
                query, context, llm_client, 1
            )
            
            # Step 2: Evaluate initial candidates
            evaluated_thoughts = await self._evaluate_thoughts(
                query, initial_thoughts, llm_client
            )
            
            # Step 3: Create root nodes from promising thoughts
            for thought_data in evaluated_thoughts:
                if thought_data['evaluation'] in [EvaluationLevel.SURE, EvaluationLevel.MAYBE]:
                    node = ThoughtNode(
                        thought=thought_data['thought'],
                        thought_number=1,
                        evaluation=thought_data['evaluation'],
                        score=thought_data['score']
                    )
                    self.root_nodes.append(node)
                    self.current_branches[node.branch_id] = [node]
            
            # Step 4: Expand promising branches
            final_branch = await self._expand_branches(query, context, llm_client)
            
            # Step 5: Extract final answer
            final_answer = await self._extract_final_answer(
                query, final_branch, llm_client
            )
            
            # Step 6: Convert to Thought objects
            thoughts = self._convert_branch_to_thoughts(final_branch)
            
            self.logger.info("Tree of Thoughts solving completed", 
                           final_branch_id=final_branch.branch_id,
                           thought_count=len(thoughts))
            
            return thoughts, final_answer
            
        except Exception as e:
            self.logger.error("Tree of Thoughts solving failed", error=str(e))
            raise
    
    async def _generate_thought_candidates(
        self,
        query: str,
        context: Dict[str, Any],
        llm_client: Any,
        step: int,
        parent_thought: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Generate thought candidates for a given step"""
        
        prompt = self._build_candidate_prompt(query, context, step, parent_thought)
        
        try:
            response = await llm_client.generate(prompt)
            candidates = self._parse_candidate_response(response)
            
            self.logger.debug("Generated thought candidates", 
                            step=step, 
                            candidate_count=len(candidates))
            
            return candidates
            
        except Exception as e:
            self.logger.error("Failed to generate thought candidates", error=str(e))
            return []
    
    def _build_candidate_prompt(
        self,
        query: str,
        context: Dict[str, Any],
        step: int,
        parent_thought: Optional[str] = None
    ) -> str:
        """Build prompt for generating thought candidates"""
        
        context_summary = ""
        if context.get('vector_results'):
            context_summary += f"Relevant documents: {len(context['vector_results'])} found\n"
        if context.get('user_attributes'):
            context_summary += f"User context: {json.dumps(context['user_attributes'], indent=2)}\n"
        
        parent_context = ""
        if parent_thought:
            parent_context = f"\nPrevious thought: {parent_thought}"
        
        prompt = f"""
You are solving a problem using Tree of Thoughts reasoning.

QUERY: {query}

CONTEXT:
{context_summary}

{parent_context}

TASK: Generate {self.max_branches} different thought candidates for step {step}.

Each candidate should represent a different approach or perspective to advance the reasoning.

Return your response as a JSON array:

[
    {{
        "thought": "First candidate thought",
        "approach": "brief description of this approach",
        "confidence": 0.8
    }},
    {{
        "thought": "Second candidate thought", 
        "approach": "brief description of this approach",
        "confidence": 0.7
    }},
    ...
]

IMPORTANT: Return ONLY valid JSON array. No additional text.
"""
        return prompt.strip()
    
    def _parse_candidate_response(self, response: str) -> List[Dict[str, Any]]:
        """Parse candidate generation response"""
        try:
            # Extract JSON array
            json_start = response.find('[')
            json_end = response.rfind(']') + 1
            if json_start == -1 or json_end == 0:
                return []
            
            json_str = response[json_start:json_end]
            candidates = json.loads(json_str)
            
            # Validate candidates
            valid_candidates = []
            for candidate in candidates:
                if isinstance(candidate, dict) and 'thought' in candidate:
                    valid_candidates.append({
                        'thought': candidate['thought'],
                        'approach': candidate.get('approach', ''),
                        'confidence': candidate.get('confidence', 0.5)
                    })
            
            return valid_candidates
            
        except (json.JSONDecodeError, KeyError) as e:
            self.logger.error("Failed to parse candidate response", error=str(e))
            return []
    
    async def _evaluate_thoughts(
        self,
        query: str,
        thoughts: List[Dict[str, Any]],
        llm_client: Any
    ) -> List[Dict[str, Any]]:
        """Evaluate thought candidates"""
        
        prompt = self._build_evaluation_prompt(query, thoughts)
        
        try:
            response = await llm_client.generate(prompt)
            evaluations = self._parse_evaluation_response(response, thoughts)
            
            self.logger.debug("Evaluated thoughts", 
                            thought_count=len(evaluations),
                            evaluations=[e['evaluation'].value for e in evaluations])
            
            return evaluations
            
        except Exception as e:
            self.logger.error("Failed to evaluate thoughts", error=str(e))
            return thoughts  # Return original thoughts if evaluation fails
    
    def _build_evaluation_prompt(
        self,
        query: str,
        thoughts: List[Dict[str, Any]]
    ) -> str:
        """Build prompt for evaluating thoughts"""
        
        thoughts_text = ""
        for i, thought in enumerate(thoughts, 1):
            thoughts_text += f"{i}. {thought['thought']}\n"
        
        prompt = f"""
You are evaluating thought candidates for a Tree of Thoughts reasoning process.

QUERY: {query}

THOUGHT CANDIDATES:
{thoughts_text}

TASK: Evaluate each thought candidate as "sure", "maybe", or "impossible".

"sure" = This thought is clearly correct and should be pursued
"maybe" = This thought might be useful, worth exploring
"impossible" = This thought is clearly wrong or unhelpful

Return your response as a JSON array:

[
    {{
        "thought": "{thoughts[0]['thought'] if thoughts else ''}",
        "evaluation": "sure|maybe|impossible",
        "reasoning": "brief explanation of evaluation",
        "score": 0.9
    }},
    ...
]

IMPORTANT: Return ONLY valid JSON array. No additional text.
"""
        return prompt.strip()
    
    def _parse_evaluation_response(
        self,
        response: str,
        original_thoughts: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Parse evaluation response"""
        try:
            # Extract JSON array
            json_start = response.find('[')
            json_end = response.rfind(']') + 1
            if json_start == -1 or json_end == 0:
                return original_thoughts
            
            json_str = response[json_start:json_end]
            evaluations = json.loads(json_str)
            
            # Match evaluations with original thoughts
            evaluated_thoughts = []
            for i, evaluation in enumerate(evaluations):
                if i < len(original_thoughts):
                    original = original_thoughts[i]
                    evaluated_thoughts.append({
                        'thought': original['thought'],
                        'evaluation': EvaluationLevel(evaluation.get('evaluation', 'maybe')),
                        'reasoning': evaluation.get('reasoning', ''),
                        'score': evaluation.get('score', 0.5),
                        'approach': original.get('approach', '')
                    })
            
            return evaluated_thoughts
            
        except (json.JSONDecodeError, ValueError) as e:
            self.logger.error("Failed to parse evaluation response", error=str(e))
            return original_thoughts
    
    async def _expand_branches(
        self,
        query: str,
        context: Dict[str, Any],
        llm_client: Any
    ) -> ThoughtNode:
        """Expand promising branches to find the best solution"""
        
        current_depth = 1
        
        while current_depth < self.max_depth:
            self.logger.debug("Expanding branches", depth=current_depth)
            
            # Get active branches
            active_branches = [
                branch_id for branch_id, nodes in self.current_branches.items()
                if nodes and len(nodes) == current_depth
            ]
            
            if not active_branches:
                break
            
            # Expand each active branch
            new_branches = []
            for branch_id in active_branches:
                branch_nodes = self.current_branches[branch_id]
                if not branch_nodes:
                    continue
                
                latest_node = branch_nodes[-1]
                
                # Check if this branch should continue
                if (latest_node.evaluation == EvaluationLevel.IMPOSSIBLE or
                    latest_node.score and latest_node.score < self.evaluation_threshold):
                    continue
                
                # Generate next thoughts for this branch
                candidates = await self._generate_thought_candidates(
                    query, context, llm_client, current_depth + 1, latest_node.thought
                )
                
                if candidates:
                    # Evaluate candidates
                    evaluated = await self._evaluate_thoughts(query, candidates, llm_client)
                    
                    # Add promising candidates as children
                    for candidate in evaluated:
                        if candidate['evaluation'] in [EvaluationLevel.SURE, EvaluationLevel.MAYBE]:
                            child_node = ThoughtNode(
                                thought=candidate['thought'],
                                thought_number=current_depth + 1,
                                evaluation=candidate['evaluation'],
                                score=candidate['score']
                            )
                            latest_node.add_child(child_node)
                            new_branches.append(child_node)
            
            # Update current branches
            for branch_id in list(self.current_branches.keys()):
                if branch_id in active_branches:
                    # Extend existing branch
                    branch_nodes = self.current_branches[branch_id]
                    if branch_nodes:
                        latest_node = branch_nodes[-1]
                        self.current_branches[branch_id].extend(latest_node.children)
                else:
                    # Remove inactive branches
                    del self.current_branches[branch_id]
            
            current_depth += 1
            
            # Check if we have a clear winner
            best_branch = self._find_best_branch()
            if best_branch and best_branch.score and best_branch.score > 0.9:
                break
        
        # Return the best branch
        best_branch = self._find_best_branch()
        if not best_branch:
            # Fallback to first available branch
            for branch_nodes in self.current_branches.values():
                if branch_nodes:
                    best_branch = branch_nodes[-1]
                    break
        
        return best_branch or ThoughtNode("No solution found", 1)
    
    def _find_best_branch(self) -> Optional[ThoughtNode]:
        """Find the best branch based on evaluation scores"""
        best_node = None
        best_score = 0.0
        
        for branch_nodes in self.current_branches.values():
            if not branch_nodes:
                continue
            
            # Find the best node in this branch
            for node in branch_nodes:
                if node.score and node.score > best_score:
                    best_score = node.score
                    best_node = node
        
        return best_node
    
    async def _extract_final_answer(
        self,
        query: str,
        final_branch: ThoughtNode,
        llm_client: Any
    ) -> str:
        """Extract final answer from the best branch"""
        
        # Collect all thoughts in the branch
        branch_thoughts = []
        current_node = final_branch
        while current_node:
            branch_thoughts.insert(0, current_node.thought)
            # Find parent node
            if current_node.parent_id:
                current_node = self._find_node_by_id(current_node.parent_id)
            else:
                break
        
        thoughts_text = "\n".join([f"Step {i+1}: {thought}" for i, thought in enumerate(branch_thoughts)])
        
        prompt = f"""
Based on the following reasoning chain, provide a clear and comprehensive answer to the original query.

QUERY: {query}

REASONING CHAIN:
{thoughts_text}

TASK: Synthesize a final answer that:
1. Directly addresses the original query
2. Incorporates key insights from the reasoning chain
3. Is clear, concise, and well-structured

FINAL ANSWER:
"""
        
        try:
            response = await llm_client.generate(prompt)
            return response.strip()
        except Exception as e:
            self.logger.error("Failed to extract final answer", error=str(e))
            return f"Based on the reasoning: {' '.join(branch_thoughts)}"
    
    def _find_node_by_id(self, node_id: str) -> Optional[ThoughtNode]:
        """Find a node by its ID"""
        for branch_nodes in self.current_branches.values():
            for node in branch_nodes:
                if node.id == node_id:
                    return node
        return None
    
    def _convert_branch_to_thoughts(self, final_branch: ThoughtNode) -> List[Thought]:
        """Convert a branch to a list of Thought objects"""
        thoughts = []
        current_node = final_branch
        
        while current_node:
            thought = Thought(
                thought=current_node.thought,
                thought_number=current_node.thought_number,
                total_thoughts=current_node.thought_number,  # Will be updated
                next_thought_needed=False,  # Will be updated
                needs_more_thoughts=False,
                stage=ThoughtStage.ANALYSIS,
                score=current_node.score,
                branch_id=current_node.branch_id,
                metadata={
                    'evaluation': current_node.evaluation.value if current_node.evaluation else None,
                    'node_id': current_node.id
                }
            )
            thoughts.insert(0, thought)
            
            # Find parent node
            if current_node.parent_id:
                current_node = self._find_node_by_id(current_node.parent_id)
            else:
                break
        
        # Update total_thoughts and next_thought_needed
        total_thoughts = len(thoughts)
        for i, thought in enumerate(thoughts):
            thought.total_thoughts = total_thoughts
            thought.next_thought_needed = i < total_thoughts - 1
        
        return thoughts
    
    def get_tree_summary(self) -> Dict[str, Any]:
        """Get a summary of the tree structure"""
        return {
            'root_nodes': len(self.root_nodes),
            'active_branches': len(self.current_branches),
            'total_nodes': sum(len(nodes) for nodes in self.current_branches.values()),
            'max_depth': self.max_depth,
            'max_branches': self.max_branches
        } 