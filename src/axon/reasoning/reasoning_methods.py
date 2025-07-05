"""
Multiple reasoning methods for Ax0n
"""

import asyncio
import json
import uuid
from typing import Any, Dict, List, Optional, Tuple, Union
from enum import Enum
import structlog
from ..core.models import Thought, ThoughtStage
from ..core.config import ThinkLayerConfig, LLMConfig

logger = structlog.get_logger(__name__)


class ReasoningMethod(str, Enum):
    """Available reasoning methods"""
    COT = "chain_of_thought"           # Linear reasoning
    SELF_CONSISTENCY = "self_consistency"  # Parallel CoT with voting
    AOT = "algorithm_of_thoughts"      # Implicit algorithmic
    TOT = "tree_of_thoughts"           # Tree-based reasoning
    GOT = "graph_of_thoughts"          # Graph-based reasoning


class ChainOfThoughts:
    """Chain of Thoughts (CoT) - Linear reasoning"""
    
    def __init__(self, config: ThinkLayerConfig, llm_config: LLMConfig):
        self.config = config
        self.llm_config = llm_config
        self.logger = logger.bind(component="chain_of_thoughts")
    
    async def solve(
        self,
        query: str,
        context: Dict[str, Any],
        llm_client: Any
    ) -> Tuple[List[Thought], str]:
        """Solve using linear chain of thoughts"""
        
        self.logger.info("Starting Chain of Thoughts reasoning", query=query[:100])
        
        thoughts = []
        current_answer = ""
        
        try:
            for step in range(1, self.config.max_depth + 1):
                # Build prompt with previous thoughts
                prompt = self._build_cot_prompt(query, context, thoughts, step)
                
                response = await llm_client.generate(prompt)
                thought = self._parse_cot_response(response, step, len(thoughts) + 1)
                
                if thought:
                    thoughts.append(thought)
                    
                    # Check if we should continue
                    if not thought.needs_more_thoughts:
                        break
                else:
                    break
            
            # Generate final answer
            final_answer = await self._generate_final_answer(query, thoughts, llm_client)
            
            return thoughts, final_answer
            
        except Exception as e:
            self.logger.error("Chain of Thoughts failed", error=str(e))
            raise
    
    def _build_cot_prompt(
        self,
        query: str,
        context: Dict[str, Any],
        previous_thoughts: List[Thought],
        step: int
    ) -> str:
        """Build prompt for next thought in chain"""
        
        context_summary = ""
        if context.get('vector_results'):
            context_summary += f"Relevant documents: {len(context['vector_results'])} found\n"
        if context.get('user_attributes'):
            context_summary += f"User context: {json.dumps(context['user_attributes'], indent=2)}\n"
        
        thought_history = ""
        if previous_thoughts:
            thought_history = "Previous thoughts:\n"
            for i, thought in enumerate(previous_thoughts, 1):
                thought_history += f"{i}. {thought.thought}\n"
        
        prompt = f"""
You are solving a problem step by step using Chain of Thoughts reasoning.

QUERY: {query}

CONTEXT:
{context_summary}

{thought_history}

TASK: Generate the next thought in the reasoning chain (Step {step}).

Think through this step carefully and provide your reasoning. If you think you have enough information to provide a final answer, indicate that no more thoughts are needed.

Return your response as a JSON object:

{{
    "thought": "Your reasoning for this step",
    "thoughtNumber": {step},
    "totalThoughts": {self.config.max_depth},
    "needsMoreThoughts": true/false,
    "nextThoughtNeeded": true/false,
    "stage": "problem_definition|research|analysis|synthesis|conclusion",
    "confidence": 0.85
}}

IMPORTANT: Return ONLY valid JSON. No additional text.
"""
        return prompt.strip()
    
    def _parse_cot_response(self, response: str, step: int, total_steps: int) -> Optional[Thought]:
        """Parse CoT response into Thought object"""
        try:
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            if json_start == -1 or json_end == 0:
                return None
            
            json_str = response[json_start:json_end]
            data = json.loads(json_str)
            
            if 'thought' not in data:
                return None
            
            # Convert stage string to enum
            stage_str = data.get('stage', 'analysis')
            try:
                stage = ThoughtStage(stage_str)
            except ValueError:
                stage = ThoughtStage.ANALYSIS
            
            thought = Thought(
                thought=data['thought'],
                thought_number=step,
                total_thoughts=total_steps,
                next_thought_needed=data.get('nextThoughtNeeded', True),
                needs_more_thoughts=data.get('needsMoreThoughts', True),
                stage=stage,
                score=data.get('confidence')
            )
            
            return thought
            
        except (json.JSONDecodeError, KeyError) as e:
            self.logger.error("Failed to parse CoT response", error=str(e))
            return None
    
    async def _generate_final_answer(
        self,
        query: str,
        thoughts: List[Thought],
        llm_client: Any
    ) -> str:
        """Generate final answer from thought chain"""
        
        if not thoughts:
            return "Unable to generate a response."
        
        thoughts_text = "\n".join([f"Step {i+1}: {thought.thought}" for i, thought in enumerate(thoughts)])
        
        prompt = f"""
Based on the following reasoning chain, provide a clear and comprehensive answer to the original query.

QUERY: {query}

REASONING CHAIN:
{thoughts_text}

FINAL ANSWER:
"""
        
        try:
            response = await llm_client.generate(prompt)
            return response.strip()
        except Exception as e:
            self.logger.error("Failed to generate final answer", error=str(e))
            return f"Based on the reasoning: {' '.join([t.thought for t in thoughts])}"


class SelfConsistency:
    """Self-Consistency - Parallel CoT with voting"""
    
    def __init__(self, config: ThinkLayerConfig, llm_config: LLMConfig):
        self.config = config
        self.llm_config = llm_config
        self.logger = logger.bind(component="self_consistency")
        self.num_paths = 5  # Number of parallel reasoning paths
    
    async def solve(
        self,
        query: str,
        context: Dict[str, Any],
        llm_client: Any
    ) -> Tuple[List[Thought], str]:
        """Solve using self-consistency with multiple parallel paths"""
        
        self.logger.info("Starting Self-Consistency reasoning", query=query[:100])
        
        try:
            # Generate multiple parallel reasoning paths
            paths = await self._generate_parallel_paths(query, context, llm_client)
            
            # Extract final answers from each path
            answers = [path['answer'] for path in paths if path['answer']]
            
            # Vote on the best answer
            final_answer = await self._vote_on_answers(query, answers, llm_client)
            
            # Use the most common path as the main thought chain
            main_path = max(paths, key=lambda p: len(p['thoughts']))
            thoughts = main_path['thoughts']
            
            return thoughts, final_answer
            
        except Exception as e:
            self.logger.error("Self-Consistency failed", error=str(e))
            raise
    
    async def _generate_parallel_paths(
        self,
        query: str,
        context: Dict[str, Any],
        llm_client: Any
    ) -> List[Dict[str, Any]]:
        """Generate multiple parallel reasoning paths"""
        
        tasks = []
        for i in range(self.num_paths):
            task = self._generate_single_path(query, context, llm_client, i)
            tasks.append(task)
        
        try:
            paths = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Filter out failed paths
            valid_paths = []
            for path in paths:
                if isinstance(path, Exception):
                    self.logger.warning("Path generation failed", error=str(path))
                elif path and path.get('thoughts'):
                    valid_paths.append(path)
            
            return valid_paths
            
        except Exception as e:
            self.logger.error("Parallel path generation failed", error=str(e))
            return []
    
    async def _generate_single_path(
        self,
        query: str,
        context: Dict[str, Any],
        llm_client: Any,
        path_id: int
    ) -> Dict[str, Any]:
        """Generate a single reasoning path"""
        
        thoughts = []
        
        for step in range(1, self.config.max_depth + 1):
            prompt = self._build_sc_prompt(query, context, thoughts, step, path_id)
            
            try:
                response = await llm_client.generate(prompt)
                thought = self._parse_sc_response(response, step, len(thoughts) + 1, path_id)
                
                if thought:
                    thoughts.append(thought)
                    
                    if not thought.needs_more_thoughts:
                        break
                else:
                    break
                    
            except Exception as e:
                self.logger.warning(f"Step {step} failed for path {path_id}", error=str(e))
                break
        
        # Generate answer for this path
        answer = await self._generate_path_answer(query, thoughts, llm_client)
        
        return {
            'path_id': path_id,
            'thoughts': thoughts,
            'answer': answer
        }
    
    def _build_sc_prompt(
        self,
        query: str,
        context: Dict[str, Any],
        previous_thoughts: List[Thought],
        step: int,
        path_id: int
    ) -> str:
        """Build prompt for self-consistency path"""
        
        context_summary = ""
        if context.get('vector_results'):
            context_summary += f"Relevant documents: {len(context['vector_results'])} found\n"
        
        thought_history = ""
        if previous_thoughts:
            thought_history = "Previous thoughts:\n"
            for i, thought in enumerate(previous_thoughts, 1):
                thought_history += f"{i}. {thought.thought}\n"
        
        prompt = f"""
You are solving a problem using reasoning path {path_id + 1} of {self.num_paths}.

QUERY: {query}

CONTEXT:
{context_summary}

{thought_history}

TASK: Generate the next thought in your reasoning path (Step {step}).

Think through this step independently. Your path may differ from other approaches, but focus on logical reasoning.

Return your response as a JSON object:

{{
    "thought": "Your reasoning for this step",
    "thoughtNumber": {step},
    "totalThoughts": {self.config.max_depth},
    "needsMoreThoughts": true/false,
    "nextThoughtNeeded": true/false,
    "stage": "problem_definition|research|analysis|synthesis|conclusion",
    "confidence": 0.85
}}

IMPORTANT: Return ONLY valid JSON. No additional text.
"""
        return prompt.strip()
    
    def _parse_sc_response(
        self,
        response: str,
        step: int,
        total_steps: int,
        path_id: int
    ) -> Optional[Thought]:
        """Parse self-consistency response"""
        try:
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            if json_start == -1 or json_end == 0:
                return None
            
            json_str = response[json_start:json_end]
            data = json.loads(json_str)
            
            if 'thought' not in data:
                return None
            
            stage_str = data.get('stage', 'analysis')
            try:
                stage = ThoughtStage(stage_str)
            except ValueError:
                stage = ThoughtStage.ANALYSIS
            
            thought = Thought(
                thought=data['thought'],
                thought_number=step,
                total_thoughts=total_steps,
                next_thought_needed=data.get('nextThoughtNeeded', True),
                needs_more_thoughts=data.get('needsMoreThoughts', True),
                stage=stage,
                score=data.get('confidence'),
                metadata={'path_id': path_id}
            )
            
            return thought
            
        except (json.JSONDecodeError, KeyError) as e:
            self.logger.error("Failed to parse SC response", error=str(e))
            return None
    
    async def _generate_path_answer(
        self,
        query: str,
        thoughts: List[Thought],
        llm_client: Any
    ) -> str:
        """Generate answer for a single path"""
        
        if not thoughts:
            return ""
        
        thoughts_text = "\n".join([f"Step {i+1}: {thought.thought}" for i, thought in enumerate(thoughts)])
        
        prompt = f"""
Based on this reasoning path, provide a concise answer to the query.

QUERY: {query}

REASONING:
{thoughts_text}

ANSWER:
"""
        
        try:
            response = await llm_client.generate(prompt)
            return response.strip()
        except Exception as e:
            self.logger.error("Failed to generate path answer", error=str(e))
            return ""
    
    async def _vote_on_answers(
        self,
        query: str,
        answers: List[str],
        llm_client: Any
    ) -> str:
        """Vote on the best answer from multiple paths"""
        
        if not answers:
            return "Unable to generate a response."
        
        if len(answers) == 1:
            return answers[0]
        
        # Create voting prompt
        answers_text = "\n".join([f"{i+1}. {answer}" for i, answer in enumerate(answers)])
        
        prompt = f"""
Multiple reasoning paths have generated different answers to the same query.

QUERY: {query}

ANSWERS:
{answers_text}

TASK: Analyze these answers and provide the most accurate and comprehensive response. 
You may combine insights from multiple answers if they are complementary.

FINAL ANSWER:
"""
        
        try:
            response = await llm_client.generate(prompt)
            return response.strip()
        except Exception as e:
            self.logger.error("Voting failed", error=str(e))
            # Return the most common answer or first answer
            return answers[0] if answers else "Unable to generate a response."


class AlgorithmOfThoughts:
    """Algorithm of Thoughts (AoT) - Implicit algorithmic reasoning"""
    
    def __init__(self, config: ThinkLayerConfig, llm_config: LLMConfig):
        self.config = config
        self.llm_config = llm_config
        self.logger = logger.bind(component="algorithm_of_thoughts")
    
    async def solve(
        self,
        query: str,
        context: Dict[str, Any],
        llm_client: Any
    ) -> Tuple[List[Thought], str]:
        """Solve using algorithmic thinking"""
        
        self.logger.info("Starting Algorithm of Thoughts reasoning", query=query[:100])
        
        try:
            # Generate algorithmic approach
            algorithm = await self._generate_algorithm(query, context, llm_client)
            
            # Execute the algorithm step by step
            thoughts = await self._execute_algorithm(algorithm, query, context, llm_client)
            
            # Generate final answer
            final_answer = await self._generate_final_answer(query, thoughts, llm_client)
            
            return thoughts, final_answer
            
        except Exception as e:
            self.logger.error("Algorithm of Thoughts failed", error=str(e))
            raise
    
    async def _generate_algorithm(
        self,
        query: str,
        context: Dict[str, Any],
        llm_client: Any
    ) -> Dict[str, Any]:
        """Generate an algorithmic approach to the problem"""
        
        context_summary = ""
        if context.get('vector_results'):
            context_summary += f"Relevant documents: {len(context['vector_results'])} found\n"
        
        prompt = f"""
You are solving a problem using Algorithm of Thoughts (AoT) - an approach that breaks down problems into algorithmic steps.

QUERY: {query}

CONTEXT:
{context_summary}

TASK: Design an algorithmic approach to solve this problem. Break it down into clear, sequential steps.

Return your response as a JSON object:

{{
    "algorithm_name": "Brief name for this approach",
    "description": "Description of the algorithmic approach",
    "steps": [
        {{
            "step_number": 1,
            "description": "What this step does",
            "input": "What information is needed",
            "output": "What this step produces",
            "method": "How to perform this step"
        }},
        ...
    ],
    "estimated_steps": 3
}}

IMPORTANT: Return ONLY valid JSON. No additional text.
"""
        
        try:
            response = await llm_client.generate(prompt)
            
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            if json_start == -1 or json_end == 0:
                return {"steps": [], "estimated_steps": 3}
            
            json_str = response[json_start:json_end]
            algorithm = json.loads(json_str)
            
            return algorithm
            
        except Exception as e:
            self.logger.error("Failed to generate algorithm", error=str(e))
            return {"steps": [], "estimated_steps": 3}
    
    async def _execute_algorithm(
        self,
        algorithm: Dict[str, Any],
        query: str,
        context: Dict[str, Any],
        llm_client: Any
    ) -> List[Thought]:
        """Execute the algorithm step by step"""
        
        thoughts = []
        steps = algorithm.get('steps', [])
        
        for i, step in enumerate(steps, 1):
            prompt = self._build_aot_step_prompt(query, context, step, i, thoughts)
            
            try:
                response = await llm_client.generate(prompt)
                thought = self._parse_aot_response(response, i, len(steps), step)
                
                if thought:
                    thoughts.append(thought)
                    
                    if not thought.needs_more_thoughts:
                        break
                else:
                    break
                    
            except Exception as e:
                self.logger.warning(f"Algorithm step {i} failed", error=str(e))
                break
        
        return thoughts
    
    def _build_aot_step_prompt(
        self,
        query: str,
        context: Dict[str, Any],
        step: Dict[str, Any],
        step_number: int,
        previous_thoughts: List[Thought]
    ) -> str:
        """Build prompt for algorithm step execution"""
        
        context_summary = ""
        if context.get('vector_results'):
            context_summary += f"Relevant documents: {len(context['vector_results'])} found\n"
        
        thought_history = ""
        if previous_thoughts:
            thought_history = "Previous results:\n"
            for i, thought in enumerate(previous_thoughts, 1):
                thought_history += f"Step {i}: {thought.thought}\n"
        
        prompt = f"""
You are executing step {step_number} of an algorithmic approach to solve a problem.

ORIGINAL QUERY: {query}

CONTEXT:
{context_summary}

{thought_history}

CURRENT STEP:
- Description: {step.get('description', 'Execute this step')}
- Input: {step.get('input', 'Previous results')}
- Output: {step.get('output', 'Result of this step')}
- Method: {step.get('method', 'Apply the specified method')}

TASK: Execute this algorithmic step and provide the result.

Return your response as a JSON object:

{{
    "thought": "Your execution of this algorithmic step",
    "thoughtNumber": {step_number},
    "totalThoughts": {len(previous_thoughts) + 1},
    "needsMoreThoughts": true/false,
    "nextThoughtNeeded": true/false,
    "stage": "analysis",
    "confidence": 0.85,
    "algorithm_step": {step_number}
}}

IMPORTANT: Return ONLY valid JSON. No additional text.
"""
        return prompt.strip()
    
    def _parse_aot_response(
        self,
        response: str,
        step_number: int,
        total_steps: int,
        algorithm_step: Dict[str, Any]
    ) -> Optional[Thought]:
        """Parse AoT response"""
        try:
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            if json_start == -1 or json_end == 0:
                return None
            
            json_str = response[json_start:json_end]
            data = json.loads(json_str)
            
            if 'thought' not in data:
                return None
            
            thought = Thought(
                thought=data['thought'],
                thought_number=step_number,
                total_thoughts=total_steps,
                next_thought_needed=data.get('nextThoughtNeeded', True),
                needs_more_thoughts=data.get('needsMoreThoughts', True),
                stage=ThoughtStage.ANALYSIS,
                score=data.get('confidence'),
                metadata={
                    'algorithm_step': step_number,
                    'step_description': algorithm_step.get('description', ''),
                    'method': algorithm_step.get('method', '')
                }
            )
            
            return thought
            
        except (json.JSONDecodeError, KeyError) as e:
            self.logger.error("Failed to parse AoT response", error=str(e))
            return None
    
    async def _generate_final_answer(
        self,
        query: str,
        thoughts: List[Thought],
        llm_client: Any
    ) -> str:
        """Generate final answer from algorithmic execution"""
        
        if not thoughts:
            return "Unable to generate a response."
        
        thoughts_text = "\n".join([f"Step {i+1}: {thought.thought}" for i, thought in enumerate(thoughts)])
        
        prompt = f"""
Based on the algorithmic execution, provide a clear and comprehensive answer to the original query.

QUERY: {query}

ALGORITHMIC EXECUTION:
{thoughts_text}

FINAL ANSWER:
"""
        
        try:
            response = await llm_client.generate(prompt)
            return response.strip()
        except Exception as e:
            self.logger.error("Failed to generate final answer", error=str(e))
            return f"Based on the algorithmic execution: {' '.join([t.thought for t in thoughts])}"


class ReasoningOrchestrator:
    """Orchestrates different reasoning methods"""
    
    def __init__(self, config: ThinkLayerConfig, llm_config: LLMConfig):
        self.config = config
        self.llm_config = llm_config
        self.logger = logger.bind(component="reasoning_orchestrator")
        
        # Initialize all reasoning methods
        self.cot = ChainOfThoughts(config, llm_config)
        self.self_consistency = SelfConsistency(config, llm_config)
        self.aot = AlgorithmOfThoughts(config, llm_config)
        
        # Import TreeOfThoughts to avoid circular imports
        from .tree_of_thoughts import TreeOfThoughts
        self.tot = TreeOfThoughts(config, llm_config)
    
    async def solve(
        self,
        query: str,
        context: Dict[str, Any],
        method: ReasoningMethod,
        llm_client: Any
    ) -> Tuple[List[Thought], str]:
        """Solve using the specified reasoning method"""
        
        self.logger.info(f"Using reasoning method: {method.value}")
        
        if method == ReasoningMethod.COT:
            return await self.cot.solve(query, context, llm_client)
        elif method == ReasoningMethod.SELF_CONSISTENCY:
            return await self.self_consistency.solve(query, context, llm_client)
        elif method == ReasoningMethod.AOT:
            return await self.aot.solve(query, context, llm_client)
        elif method == ReasoningMethod.TOT:
            return await self.tot.solve(query, context, llm_client)
        else:
            raise ValueError(f"Unsupported reasoning method: {method}")
    
    def get_method_description(self, method: ReasoningMethod) -> str:
        """Get description of reasoning method"""
        descriptions = {
            ReasoningMethod.COT: "Linear step-by-step reasoning",
            ReasoningMethod.SELF_CONSISTENCY: "Multiple parallel paths with voting",
            ReasoningMethod.AOT: "Algorithmic problem decomposition",
            ReasoningMethod.TOT: "Tree-based exploration with evaluation",
            ReasoningMethod.GOT: "Graph-based reasoning with dependencies"
        }
        return descriptions.get(method, "Unknown method")
    
    def get_method_complexity(self, method: ReasoningMethod) -> str:
        """Get complexity level of reasoning method"""
        complexities = {
            ReasoningMethod.COT: "Low",
            ReasoningMethod.SELF_CONSISTENCY: "Medium",
            ReasoningMethod.AOT: "Low-Moderate",
            ReasoningMethod.TOT: "Medium",
            ReasoningMethod.GOT: "High"
        }
        return complexities.get(method, "Unknown") 