"""
Main Axon class - orchestrates all modules for structured reasoning
"""

import asyncio
import time
from typing import Any, Dict, List, Optional, Union
import structlog

from .config import AxonConfig
from .models import Thought, ThoughtResult, MemoryEntry, LLMConfig
from .retriever import Retriever
from .think_layer import ThinkLayer
from .grounding import GroundingModule
from .memory import MemoryManager
from .renderer import Renderer


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
    
    def __init__(
        self,
        llm_client: Optional[str] = None,
        api_key: Optional[str] = None,
        config: Optional[AxonConfig] = None,
        **kwargs
    ):
        """
        Initialize Axon with configuration.
        
        Args:
            llm_client: LLM provider name (e.g., "openai", "anthropic")
            api_key: API key for the LLM provider
            config: Full configuration object
            **kwargs: Additional configuration parameters
        """
        self.config = self._build_config(llm_client, api_key, config, **kwargs)
        self.logger = logger.bind(component="axon")
        
        # Initialize modules
        self.retriever = Retriever(self.config.retriever)
        self.think_layer = ThinkLayer(self.config.think_layer, self.config.llm)
        self.grounding = GroundingModule(self.config.grounding)
        self.memory = MemoryManager(self.config.memory)
        self.renderer = Renderer(self.config.renderer)
        
        self.logger.info("Axon initialized", config_summary=self._get_config_summary())
    
    def _build_config(
        self,
        llm_client: Optional[str],
        api_key: Optional[str],
        config: Optional[AxonConfig],
        **kwargs
    ) -> AxonConfig:
        """Build configuration from various sources"""
        if config is not None:
            return config
        
        # Build LLM config
        llm_config = LLMConfig(
            provider=llm_client or "openai",
            model=kwargs.get("model", "gpt-4"),
            api_key=api_key or kwargs.get("api_key"),
            temperature=kwargs.get("temperature", 0.7),
            max_tokens=kwargs.get("max_tokens", 1000),
            timeout=kwargs.get("timeout", 30),
        )
        
        # Create full config
        return AxonConfig(
            llm=llm_config,
            **{k: v for k, v in kwargs.items() if k not in ["model", "api_key", "temperature", "max_tokens", "timeout"]}
        )
    
    def _get_config_summary(self) -> Dict[str, Any]:
        """Get a summary of the configuration for logging"""
        return {
            "llm_provider": self.config.llm.provider,
            "llm_model": self.config.llm.model,
            "max_depth": self.config.think_layer.max_depth,
            "enable_grounding": self.config.grounding.enable_grounding,
            "enable_memory": self.config.memory.enable_memory,
            "enable_parallel": self.config.think_layer.enable_parallel,
        }
    
    async def think(
        self,
        query: str,
        max_depth: Optional[int] = None,
        enable_grounding: Optional[bool] = None,
        enable_memory: Optional[bool] = None,
        context: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> ThoughtResult:
        """
        Generate structured thoughts for a query.
        
        Args:
            query: The input query to reason about
            max_depth: Maximum depth of thought chains
            enable_grounding: Whether to enable fact grounding
            enable_memory: Whether to enable memory operations
            context: Additional context for the query
            **kwargs: Additional parameters for thought generation
            
        Returns:
            ThoughtResult containing thoughts, answer, and metadata
        """
        start_time = time.time()
        self.logger.info("Starting thought generation", query=query[:100] + "..." if len(query) > 100 else query)
        
        try:
            # 1. Retrieve relevant context
            context_data = await self._retrieve_context(query, context or {})
            
            # 2. Generate thoughts
            thoughts = await self._generate_thoughts(
                query, context_data, max_depth, **kwargs
            )
            
            # 3. Ground thoughts (if enabled)
            citations = []
            if enable_grounding or (enable_grounding is None and self.config.grounding.enable_grounding):
                citations = await self._ground_thoughts(thoughts, query)
            
            # 4. Update memory (if enabled)
            memory_updates = []
            if enable_memory or (enable_memory is None and self.config.memory.enable_memory):
                memory_updates = await self._update_memory(thoughts, query)
            
            # 5. Generate final answer
            answer = await self._synthesize_answer(thoughts, citations)
            
            # 6. Render output
            result = await self._render_result(
                thoughts, answer, citations, memory_updates, time.time() - start_time
            )
            
            self.logger.info(
                "Thought generation completed",
                execution_time=result.execution_time,
                thought_count=len(thoughts),
                citation_count=len(citations)
            )
            
            return result
            
        except Exception as e:
            self.logger.error("Thought generation failed", error=str(e), exc_info=True)
            raise
    
    async def _retrieve_context(self, query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Retrieve relevant context for the query"""
        try:
            # Get vector search results
            vector_results = await self.retriever.search(query)
            
            # Get user attributes from KV store
            user_attrs = await self.retriever.get_user_attributes(context.get("user_id"))
            
            # Get relevant memories
            memories = []
            if self.config.memory.enable_memory:
                memories = await self.memory.retrieve_relevant(query)
            
            return {
                "vector_results": vector_results,
                "user_attributes": user_attrs,
                "memories": memories,
                "additional_context": context
            }
        except Exception as e:
            self.logger.warning("Context retrieval failed", error=str(e))
            return {"vector_results": [], "user_attributes": {}, "memories": [], "additional_context": context}
    
    async def _generate_thoughts(
        self,
        query: str,
        context: Dict[str, Any],
        max_depth: Optional[int] = None,
        **kwargs
    ) -> List[Thought]:
        """Generate structured thoughts"""
        depth = max_depth or self.config.think_layer.max_depth
        
        return await self.think_layer.generate_thoughts(
            query=query,
            context=context,
            max_depth=depth,
            **kwargs
        )
    
    async def _ground_thoughts(self, thoughts: List[Thought], query: str) -> List[Any]:
        """Ground thoughts with real-world evidence"""
        try:
            return await self.grounding.ground_thoughts(thoughts, query)
        except Exception as e:
            self.logger.warning("Grounding failed", error=str(e))
            return []
    
    async def _update_memory(self, thoughts: List[Thought], query: str) -> List[Dict[str, Any]]:
        """Update memory with new knowledge"""
        try:
            return await self.memory.extract_and_store(thoughts, query)
        except Exception as e:
            self.logger.warning("Memory update failed", error=str(e))
            return []
    
    async def _synthesize_answer(self, thoughts: List[Thought], citations: List[Any]) -> str:
        """Synthesize final answer from thoughts and citations"""
        return await self.think_layer.synthesize_answer(thoughts, citations)
    
    async def _render_result(
        self,
        thoughts: List[Thought],
        answer: str,
        citations: List[Any],
        memory_updates: List[Dict[str, Any]],
        execution_time: float
    ) -> ThoughtResult:
        """Render the final result"""
        return await self.renderer.render_result(
            thoughts=thoughts,
            answer=answer,
            citations=citations,
            memory_updates=memory_updates,
            execution_time=execution_time
        )
    
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