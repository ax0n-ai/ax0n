"""
Main Axon class - orchestrates all modules for structured reasoning
"""

import asyncio
import time
from typing import Any, Dict, List, Optional, Union
import structlog

from .config import AxonConfig, ThinkLayerConfig, LLMConfig
from .models import Thought, ThoughtResult, MemoryEntry
from ..retrieval import Retriever
from .think_layer import ThinkLayer
from ..grounding import GroundingModule
from ..memory import MemoryManager
from ..rendering import Renderer
from ..reasoning import ReasoningMethod


logger = structlog.get_logger(__name__)


class Axon:
    """
    Main Axon class that orchestrates structured reasoning and memory.
    
    This class coordinates all modules:
    - Retriever: Context fetching
    - Think Layer: Structured reasoning
    - Grounding: Fact verification
    - Memory: Knowledge persistence
    - Renderer: Output formatting
    """
    
    def __init__(self, config: Optional[AxonConfig] = None):
        """Initialize Axon with configuration"""
        self.config = config or AxonConfig()
        self.logger = logger.bind(component="axon")
        
        # Initialize components
        self.think_layer = ThinkLayer(self.config.think_layer, self.config.llm)
        self.retriever = Retriever(self.config.retriever)
        self.grounding = GroundingModule(self.config.grounding)
        self.memory = MemoryManager(self.config.memory)
        self.renderer = Renderer(self.config.renderer)
        
        self.logger.info("Axon initialized", config=self._get_config_summary())
    
    def _get_config_summary(self) -> Dict[str, Any]:
        """Get configuration summary for logging"""
        return {
            'max_depth': self.config.think_layer.max_depth,
            'enable_parallel': self.config.think_layer.enable_parallel,
            'llm_provider': self.config.llm.provider,
            'llm_model': self.config.llm.model,
            'enable_grounding': self.config.grounding.enable_grounding,
            'enable_memory': self.config.memory.enable_memory
        }
    
    async def think(
        self,
        query: str,
        method: ReasoningMethod = ReasoningMethod.COT,
        context: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Main thinking method - processes query through the full pipeline
        
        Args:
            query: User query to process
            method: Reasoning method to use (CoT, ToT, Self-Consistency, AoT, etc.)
            context: Optional additional context
            **kwargs: Additional arguments passed to components
            
        Returns:
            Dictionary containing thoughts, answer, and metadata
        """
        
        self.logger.info(
            "Starting thinking process",
            query=query[:100] + "..." if len(query) > 100 else query,
            method=method.value
        )
        
        try:
            # Step 1: Retrieve context
            retrieved_context = await self.retriever.retrieve(query, context or {})
            
            # Step 2: Generate thoughts using specified method
            thoughts, final_answer = await self.think_layer.generate_thoughts(
                query, retrieved_context, self._get_llm_client(), method
            )
            
            # Step 3: Ground claims (if enabled)
            grounded_thoughts = thoughts
            citations = []
            if self.config.grounding.enable_grounding:
                grounded_thoughts, citations = await self.grounding.ground_claims(
                    thoughts, query, retrieved_context
                )
            
            # Step 4: Extract and update memory (if enabled)
            memory_updates = []
            if self.config.memory.enable_memory:
                memory_updates = await self.memory.extract_and_update(
                    thoughts, query, retrieved_context
                )
            
            # Step 5: Render final output
            rendered_output = await self.renderer.render(
                query, thoughts, final_answer, citations, memory_updates
            )
            
            # Prepare response
            response = {
                "query": query,
                "method": method.value,
                "thoughts": [thought.dict() for thought in thoughts],
                "answer": final_answer,
                "citations": citations,
                "memory_updates": memory_updates,
                "rendered_output": rendered_output,
                "metadata": {
                    "num_thoughts": len(thoughts),
                    "method_complexity": self.think_layer.get_method_info(method)["complexity"],
                    "grounding_enabled": self.config.grounding.enable_grounding,
                    "memory_enabled": self.config.memory.enable_memory
                }
            }
            
            self.logger.info(
                "Thinking process completed",
                method=method.value,
                num_thoughts=len(thoughts),
                answer_length=len(final_answer)
            )
            
            return response
            
        except Exception as e:
            self.logger.error("Thinking process failed", error=str(e), method=method.value)
            raise
    
    async def think_sequential(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Legacy method for sequential thinking (uses Chain of Thoughts)
        
        Args:
            query: User query to process
            context: Optional additional context
            **kwargs: Additional arguments
            
        Returns:
            Dictionary containing thoughts, answer, and metadata
        """
        
        return await self.think(query, ReasoningMethod.COT, context, **kwargs)
    
    async def think_tree_of_thoughts(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Use Tree of Thoughts reasoning method
        
        Args:
            query: User query to process
            context: Optional additional context
            **kwargs: Additional arguments
            
        Returns:
            Dictionary containing thoughts, answer, and metadata
        """
        
        return await self.think(query, ReasoningMethod.TOT, context, **kwargs)
    
    async def think_self_consistency(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Use Self-Consistency reasoning method (parallel paths with voting)
        
        Args:
            query: User query to process
            context: Optional additional context
            **kwargs: Additional arguments
            
        Returns:
            Dictionary containing thoughts, answer, and metadata
        """
        
        return await self.think(query, ReasoningMethod.SELF_CONSISTENCY, context, **kwargs)
    
    async def think_algorithm_of_thoughts(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Use Algorithm of Thoughts reasoning method
        
        Args:
            query: User query to process
            context: Optional additional context
            **kwargs: Additional arguments
            
        Returns:
            Dictionary containing thoughts, answer, and metadata
        """
        
        return await self.think(query, ReasoningMethod.AOT, context, **kwargs)
    
    def get_available_methods(self) -> List[Dict[str, str]]:
        """Get information about all available reasoning methods"""
        return self.think_layer.get_method_comparison()
    
    def get_method_info(self, method: ReasoningMethod) -> Dict[str, str]:
        """Get information about a specific reasoning method"""
        return self.think_layer.get_method_info(method)
    
    async def _get_llm_client(self):
        """Get LLM client for internal use"""
        # This is a simplified version - in practice, you'd want to implement
        # a proper LLM client factory based on the configuration
        return None  # Placeholder - actual implementation would return configured client
    
    async def retrieve_memory(self, query: str, limit: int = 10) -> List[MemoryEntry]:
        """Retrieve relevant memories"""
        return await self.memory.retrieve_relevant(query, limit)
    
    async def update_memory(self, content: str, confidence: float = 1.0, **kwargs) -> MemoryEntry:
        """Manually add a memory entry"""
        return await self.memory.add_memory(content, confidence, **kwargs)
    
    async def ground_claim(self, claim: str) -> List[Any]:
        """Ground a specific claim with evidence"""
        return await self.grounding.ground_claim(claim)
    
    def get_config(self) -> AxonConfig:
        """Get the current configuration"""
        return self.config
    
    def update_config(self, **kwargs) -> None:
        """Update configuration parameters"""
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
            else:
                self.config.custom_settings[key] = value
        
        self.logger.info("Configuration updated", updates=kwargs) 