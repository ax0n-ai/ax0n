"""
Think Layer - Structured thought generation with multiple reasoning methods
"""

import asyncio
import json
import structlog
from typing import Any, Dict, List, Optional, Tuple, Union
from .models import Thought, ThoughtStage
from .config import ThinkLayerConfig, LLMConfig
from ..reasoning import ReasoningMethod, ReasoningOrchestrator

logger = structlog.get_logger(__name__)


class ThinkLayer:
    """Think Layer - Generates structured thoughts using various reasoning methods"""
    
    def __init__(self, config: ThinkLayerConfig, llm_config: LLMConfig):
        self.config = config
        self.llm_config = llm_config
        self.logger = logger.bind(component="think_layer")
        
        # Initialize reasoning orchestrator
        self.reasoning_orchestrator = ReasoningOrchestrator(config, llm_config)
    
    async def generate_thoughts(
        self,
        query: str,
        context: Dict[str, Any],
        llm_client: Any,
        method: ReasoningMethod = ReasoningMethod.COT
    ) -> Tuple[List[Thought], str]:
        """
        Generate thoughts using the specified reasoning method
        
        Args:
            query: User query
            context: Context from retriever
            llm_client: LLM client instance
            method: Reasoning method to use
            
        Returns:
            Tuple of (thoughts, final_answer)
        """
        
        self.logger.info(
            "Generating thoughts",
            method=method.value,
            query_length=len(query),
            context_keys=list(context.keys())
        )
        
        try:
            # Use the reasoning orchestrator to solve with the specified method
            thoughts, final_answer = await self.reasoning_orchestrator.solve(
                query, context, method, llm_client
            )
            
            self.logger.info(
                "Thought generation completed",
                method=method.value,
                num_thoughts=len(thoughts),
                answer_length=len(final_answer)
            )
            
            return thoughts, final_answer
            
        except Exception as e:
            self.logger.error("Thought generation failed", error=str(e), method=method.value)
            raise
    
    async def generate_thoughts_sequential(
        self,
        query: str,
        context: Dict[str, Any],
        llm_client: Any
    ) -> Tuple[List[Thought], str]:
        """
        Generate thoughts using sequential reasoning (legacy method)
        
        Args:
            query: User query
            context: Context from retriever
            llm_client: LLM client instance
            
        Returns:
            Tuple of (thoughts, final_answer)
        """
        
        self.logger.info("Generating thoughts using sequential reasoning")
        
        try:
            # Use Chain of Thoughts for sequential reasoning
            thoughts, final_answer = await self.reasoning_orchestrator.solve(
                query, context, ReasoningMethod.COT, llm_client
            )
            
            self.logger.info(
                "Sequential thought generation completed",
                num_thoughts=len(thoughts),
                answer_length=len(final_answer)
            )
            
            return thoughts, final_answer
            
        except Exception as e:
            self.logger.error("Sequential thought generation failed", error=str(e))
            raise
    
    def get_available_methods(self) -> List[ReasoningMethod]:
        """Get list of available reasoning methods"""
        return list(ReasoningMethod)
    
    def get_method_info(self, method: ReasoningMethod) -> Dict[str, str]:
        """Get information about a reasoning method"""
        return {
            "name": method.value,
            "description": self.reasoning_orchestrator.get_method_description(method),
            "complexity": self.reasoning_orchestrator.get_method_complexity(method)
        }
    
    def get_method_comparison(self) -> List[Dict[str, str]]:
        """Get comparison of all reasoning methods"""
        comparison = []
        for method in ReasoningMethod:
            comparison.append(self.get_method_info(method))
        return comparison

    def _get_config_summary(self) -> Dict[str, Any]:
        """Get configuration summary for logging"""
        return {
            'max_depth': self.config.max_depth,
            'enable_parallel': self.config.enable_parallel,
            'auto_iterate': self.config.auto_iterate,
            'llm_provider': self.llm_config.provider,
            'llm_model': self.llm_config.model
        }
    
    async def synthesize_answer(
        self,
        thoughts: List[Thought],
        citations: List[Any]
    ) -> str:
        """Synthesize final answer from thoughts and citations"""
        try:
            if not thoughts:
                return "Unable to generate a response based on the available information."
            
            # Simple synthesis for now
            answer_parts = []
            for i, thought in enumerate(thoughts, 1):
                answer_parts.append(f"Step {i}: {thought.thought}")
            
            if citations:
                answer_parts.append(f"\nSupporting evidence: {len(citations)} sources found")
            
            return "\n\n".join(answer_parts)
            
        except Exception as e:
            self.logger.error("Answer synthesis failed", error=str(e))
            return "Unable to synthesize answer due to an error." 