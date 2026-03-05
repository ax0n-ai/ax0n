import asyncio
import time
from typing import Any, Callable, Dict, List, Optional
import structlog

from .config import AxonConfig
from .models import Thought, MemoryEntry
from ..retrieval import Retriever
from .think_layer import ThinkLayer
from ..grounding import GroundingModule
from ..memory import MemoryManager
from ..rendering import Renderer
from ..reasoning import ReasoningMethod
from .revision import RevisionLoop


logger = structlog.get_logger(__name__)

OnThoughtCallback = Callable[[Thought, int], None]


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

        if not self.config.llm:
            self.logger.warning("No LLM configuration provided. Some features may not work.")

        self.think_layer = ThinkLayer(self.config.think_layer, self.config.llm)
        self.retriever = Retriever(self.config.retriever)
        self.grounding = GroundingModule(self.config.grounding)
        self.memory = MemoryManager(self.config.memory)
        self.renderer = Renderer(self.config.renderer)
        self.revision_loop = RevisionLoop(self.config.think_layer, self.grounding)
        self._llm_client = None

        self._metrics: Dict[str, Any] = {
            "total_queries": 0,
            "total_thoughts_generated": 0,
            "total_revision_iterations": 0,
            "method_usage": {},
            "avg_thoughts_per_query": 0.0,
            "total_time_seconds": 0.0,
            "quality_gate_filtered": 0,
        }

        self.logger.info("Axon initialized", config=self._get_config_summary())

    def _get_config_summary(self) -> Dict[str, Any]:
        """Get configuration summary for logging (sensitive values masked)"""
        return {
            'max_depth': self.config.think_layer.max_depth,
            'enable_parallel': self.config.think_layer.enable_parallel,
            'llm_provider': self.config.llm.provider if self.config.llm else 'none',
            'llm_model': self.config.llm.model if self.config.llm else 'none',
            'enable_grounding': self.config.grounding.enable_grounding,
            'enable_memory': self.config.memory.enable_memory,
            'enable_hierarchical': self.config.think_layer.enable_hierarchical,
            'agent_hierarchy': self.config.think_layer.agent_hierarchy
        }

    async def think(
        self,
        query: str,
        method: Optional[ReasoningMethod] = None,
        context: Optional[Dict[str, Any]] = None,
        on_thought: Optional[OnThoughtCallback] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Main thinking method - processes query through the full pipeline

        Args:
            query: User query to process
            method: Reasoning method (None = auto-select based on query)
            context: Optional additional context
            on_thought: Optional callback invoked for each generated thought
            **kwargs: Additional arguments passed to components

        Returns:
            Dictionary containing thoughts, answer, and metadata
        """

        if not query or not query.strip():
            raise ValueError("Query cannot be empty")
        if len(query) > self.config.max_query_length:
            raise ValueError(
                f"Query too long ({len(query)} chars). "
                f"Maximum is {self.config.max_query_length} chars."
            )

        if method is None:
            method = self._auto_select_method(query)
        elif isinstance(method, str):
            method_str = method.strip().lower()
            name_map = {m.name.lower(): m for m in ReasoningMethod}
            value_map = {m.value.lower(): m for m in ReasoningMethod}
            if method_str in name_map:
                method = name_map[method_str]
            elif method_str in value_map:
                method = value_map[method_str]
            else:
                raise ValueError(
                    f"Unknown reasoning method '{method}'. "
                    f"Valid: {[m.name.lower() for m in ReasoningMethod]}"
                )

        self.logger.info(
            "Starting thinking process",
            query=query[:100] + "..." if len(query) > 100 else query,
            method=method.value
        )

        start_time = time.monotonic()

        try:
            result = await asyncio.wait_for(
                self._run_pipeline(query, method, context or {}, on_thought),
                timeout=self.config.request_timeout,
            )

            elapsed = time.monotonic() - start_time
            self._update_metrics(method, result, elapsed)

            return result

        except asyncio.TimeoutError:
            elapsed = time.monotonic() - start_time
            self.logger.error(
                "Thinking process timed out",
                method=method.value,
                timeout=self.config.request_timeout,
                elapsed=elapsed,
            )
            raise TimeoutError(
                f"Think pipeline exceeded {self.config.request_timeout}s timeout"
            )
        except Exception as e:
            self.logger.error("Thinking process failed", error=str(e), method=method.value)
            raise

    async def _run_pipeline(
        self,
        query: str,
        method: ReasoningMethod,
        context: Dict[str, Any],
        on_thought: Optional[OnThoughtCallback],
    ) -> Dict[str, Any]:
        """Execute the full think pipeline (called under timeout)."""

        retrieved_context = await self.retriever.retrieve(query, context)

        llm_client = await self._get_llm_client()
        thoughts, final_answer = await self.think_layer.generate_thoughts(
            query, retrieved_context, llm_client, method
        )

        if on_thought:
            for i, thought in enumerate(thoughts):
                try:
                    on_thought(thought, i)
                except Exception as cb_err:
                    self.logger.warning("on_thought callback error", error=str(cb_err))

        quality_threshold = self.config.quality_gate_threshold
        filtered_count = 0
        if quality_threshold > 0.0 and len(thoughts) > 1:
            original_thoughts = thoughts
            thoughts = [
                t for t in thoughts
                if t.score is None or t.score >= quality_threshold
            ]
            filtered_count = len(original_thoughts) - len(thoughts)
            if filtered_count > 0:
                self.logger.info(
                    "Quality gate filtered thoughts",
                    filtered=filtered_count,
                    threshold=quality_threshold,
                    remaining=len(thoughts),
                )
            if not thoughts:
                thoughts = [original_thoughts[0]]

        grounded_thoughts = thoughts
        citations = []
        if self.config.grounding.enable_grounding:
            grounded_thoughts, citations = await self.grounding.ground_claims(
                thoughts, query, retrieved_context
            )

        revision_iterations = 0
        if (self.config.grounding.enable_grounding and
                self.config.think_layer.enable_revision and
                self.config.think_layer.max_revision_iterations > 0):
            grounded_thoughts, citations, revision_iterations = (
                await self.revision_loop.run(
                    grounded_thoughts, query, retrieved_context,
                    llm_client, citations
                )
            )

        memory_updates = []
        if self.config.memory.enable_memory:
            memory_updates = await self.memory.extract_and_update(
                grounded_thoughts, query, retrieved_context
            )

        rendered_output = await self.renderer.render(
            query, grounded_thoughts, final_answer, citations, memory_updates
        )

        response = {
            "query": query,
            "method": method.value,
            "thoughts": [thought.model_dump() for thought in grounded_thoughts],
            "answer": final_answer,
            "citations": citations,
            "memory_updates": memory_updates,
            "rendered_output": rendered_output,
            "metadata": {
                "num_thoughts": len(grounded_thoughts),
                "method_complexity": self.think_layer.get_method_info(method)["complexity"],
                "grounding_enabled": self.config.grounding.enable_grounding,
                "memory_enabled": self.config.memory.enable_memory,
                "revision_iterations": revision_iterations,
                "quality_gate_filtered": filtered_count,
            }
        }

        self.logger.info(
            "Thinking process completed",
            method=method.value,
            num_thoughts=len(grounded_thoughts),
            answer_length=len(final_answer)
        )

        return response

    def _auto_select_method(self, query: str) -> ReasoningMethod:
        """Auto-select reasoning method based on query characteristics."""
        words = query.split()
        word_count = len(words)
        query_lower = query.lower()

        comparison_keywords = {"compare", "versus", "vs", "difference", "pros and cons", "trade-off"}
        if any(kw in query_lower for kw in comparison_keywords):
            return ReasoningMethod.TOT

        decomposition_keywords = {"steps", "how to", "plan", "design", "architect", "build"}
        if any(kw in query_lower for kw in decomposition_keywords) and word_count > 10:
            return ReasoningMethod.GOT

        ambiguity_keywords = {"debate", "controversial", "opinion", "argue", "perspectives"}
        if any(kw in query_lower for kw in ambiguity_keywords):
            return ReasoningMethod.SELF_CONSISTENCY

        algorithmic_keywords = {"algorithm", "optimize", "calculate", "formula", "equation", "solve"}
        if any(kw in query_lower for kw in algorithmic_keywords):
            return ReasoningMethod.AOT

        return ReasoningMethod.COT

    def _update_metrics(self, method: ReasoningMethod, result: Dict[str, Any], elapsed: float) -> None:
        """Update internal metrics counters."""
        if not self.config.enable_metrics:
            return

        self._metrics["total_queries"] += 1
        num_thoughts = result["metadata"]["num_thoughts"]
        self._metrics["total_thoughts_generated"] += num_thoughts
        self._metrics["total_revision_iterations"] += result["metadata"].get("revision_iterations", 0)
        self._metrics["total_time_seconds"] += elapsed
        self._metrics["quality_gate_filtered"] += result["metadata"].get("quality_gate_filtered", 0)

        method_key = method.value
        self._metrics["method_usage"][method_key] = self._metrics["method_usage"].get(method_key, 0) + 1

        total_q = self._metrics["total_queries"]
        self._metrics["avg_thoughts_per_query"] = self._metrics["total_thoughts_generated"] / total_q

    def get_metrics(self) -> Dict[str, Any]:
        """Return current metrics snapshot."""
        return dict(self._metrics)

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
        """
        Get LLM client for internal use (cached after first creation).

        Creates and returns an LLM client based on configuration.
        This is a factory method that handles different LLM providers.
        The returned client is wrapped with retry logic.
        """
        if self._llm_client is not None:
            return self._llm_client

        if not self.config.llm:
            self.logger.warning("No LLM configuration provided")
            return None

        provider = self.config.llm.provider.lower()

        try:
            if provider == "openai":
                raw_client = await self._create_openai_client()
            elif provider == "anthropic":
                raw_client = await self._create_anthropic_client()
            elif provider == "local":
                raw_client = await self._create_local_client()
            else:
                self.logger.warning("Unknown provider, returning mock client", provider=provider)
                raw_client = MockLLMClient()
        except Exception as e:
            self.logger.error("Failed to create LLM client", error=str(e))
            raw_client = MockLLMClient()

        self._llm_client = RetryLLMClient(
            raw_client,
            max_attempts=self.config.llm_retry_attempts,
            base_delay=self.config.llm_retry_base_delay,
        )

        return self._llm_client

    async def _create_openai_client(self):
        """Create OpenAI client"""
        try:
            import openai

            request_timeout = self.config.request_timeout

            class OpenAIClient:
                def __init__(self, config):
                    self.config = config
                    self.client = openai.AsyncOpenAI(
                        api_key=config.api_key,
                        base_url=config.base_url,
                        timeout=config.timeout
                    )

                async def generate(self, prompt: str) -> str:
                    response = await asyncio.wait_for(
                        self.client.chat.completions.create(
                            model=self.config.model,
                            messages=[{"role": "user", "content": prompt}],
                            temperature=self.config.temperature,
                            max_tokens=self.config.max_tokens
                        ),
                        timeout=request_timeout,
                    )
                    return response.choices[0].message.content

            return OpenAIClient(self.config.llm)
        except ImportError:
            self.logger.warning("OpenAI library not installed, using mock client")
            return MockLLMClient()

    async def _create_anthropic_client(self):
        """Create Anthropic client"""
        try:
            import anthropic

            request_timeout = self.config.request_timeout

            class AnthropicClient:
                def __init__(self, config):
                    self.config = config
                    self.client = anthropic.AsyncAnthropic(
                        api_key=config.api_key,
                        timeout=config.timeout
                    )

                async def generate(self, prompt: str) -> str:
                    response = await asyncio.wait_for(
                        self.client.messages.create(
                            model=self.config.model,
                            max_tokens=self.config.max_tokens,
                            temperature=self.config.temperature,
                            messages=[{"role": "user", "content": prompt}]
                        ),
                        timeout=request_timeout,
                    )
                    return response.content[0].text

            return AnthropicClient(self.config.llm)
        except ImportError:
            self.logger.warning("Anthropic library not installed, using mock client")
            return MockLLMClient()

    async def _create_local_client(self):
        """Create local LLM client (e.g., for LM Studio, Ollama)"""
        try:
            import aiohttp

            class LocalLLMClient:
                def __init__(self, config):
                    self.config = config
                    self.base_url = config.base_url or "http://localhost:1234/v1"

                async def generate(self, prompt: str) -> str:
                    async with aiohttp.ClientSession() as session:
                        async with session.post(
                            f"{self.base_url}/chat/completions",
                            json={
                                "model": self.config.model,
                                "messages": [{"role": "user", "content": prompt}],
                                "temperature": self.config.temperature,
                                "max_tokens": self.config.max_tokens
                            },
                            timeout=aiohttp.ClientTimeout(total=self.config.timeout)
                        ) as response:
                            data = await response.json()
                            return data["choices"][0]["message"]["content"]

            return LocalLLMClient(self.config.llm)
        except ImportError:
            self.logger.warning("aiohttp not installed, using mock client")
            return MockLLMClient()

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

        self._llm_client = None
        self.logger.info("Configuration updated", updates=kwargs)

    async def close(self) -> None:
        """Clean up resources (grounding session, etc.)."""
        await self.grounding.close()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()
        return False


class RetryLLMClient:
    """Wraps an LLM client with exponential backoff retry logic."""

    def __init__(self, client: Any, max_attempts: int = 3, base_delay: float = 1.0):
        self._client = client
        self._max_attempts = max_attempts
        self._base_delay = base_delay
        self._logger = logger.bind(component="retry_llm")

    async def generate(self, prompt: str) -> str:
        last_error: Optional[Exception] = None
        for attempt in range(1, self._max_attempts + 1):
            try:
                return await self._client.generate(prompt)
            except Exception as e:
                last_error = e
                if attempt < self._max_attempts:
                    delay = self._base_delay * (2 ** (attempt - 1))
                    self._logger.warning(
                        "LLM call failed, retrying",
                        attempt=attempt,
                        max_attempts=self._max_attempts,
                        delay=delay,
                        error=str(e),
                    )
                    await asyncio.sleep(delay)
        raise last_error  # type: ignore[misc]


class MockLLMClient:
    """Mock LLM client for testing/development"""

    async def generate(self, prompt: str) -> str:
        """Generate mock response"""
        return "This is a mock response. Please configure a real LLM provider."
