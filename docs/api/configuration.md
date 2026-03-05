# Configuration Classes

## `AxonConfig`
Main configuration class for the Ax0n system. Contains all component configs.

- `think_layer: ThinkLayerConfig` — Think layer configuration
- `llm: LLMConfig` — LLM provider configuration
- `retriever: RetrieverConfig` — Retrieval configuration
- `grounding: GroundingConfig` — Grounding module configuration
- `memory: MemoryConfig` — Memory manager configuration
- `renderer: RendererConfig` — Output formatting configuration

## `ThinkLayerConfig`
Configuration for the hierarchical think layer (reasoning):
- `max_depth: int` — Maximum thought depth (default: 5)
- `enable_parallel: bool` — Enable parallel reasoning (default: True)
- `max_parallel_branches: int` — Max parallel branches (default: 3)
- `max_branches: int` — Max branches for ToT (default: 5)
- `auto_iterate: bool` — Auto-iterate through thoughts (default: True)
- `enable_revision: bool` — Allow thought revision (default: True)
- `enable_branching: bool` — Allow thought branching (default: True)
- `thought_timeout: int` — Timeout for thoughts in seconds (default: 30)
- `evaluation_threshold: float` — Pruning threshold (default: 0.7)
- `enable_hierarchical: bool` — Enable hierarchical agent structure (default: False)
- `agent_hierarchy: str` — Agent hierarchy mode: flat, strategic, coordinated (default: "flat")

## `LLMConfig`
Configuration for model-agnostic LLM providers:
- `provider: str` — Provider name (openai, anthropic, local, etc.)
- `model: str` — Model name
- `api_key: Optional[str]` — API key (optional for local models)
- `base_url: Optional[str]` — Base URL for API calls (optional)
- `temperature: float` — Generation temperature (default: 0.7)
- `max_tokens: int` — Maximum tokens per response (default: 1000)
- `timeout: int` — Request timeout in seconds (default: 30)
- `supports_streaming: bool` — Whether model supports streaming (default: True)
- `supports_function_calling: bool` — Whether model supports function calling (default: False)

## `MemoryConfig`
Configuration for Mem0-inspired hybrid memory architecture:
- `enable_memory: bool` — Enable persistent memory (default: True)
- `storage_provider: str` — Storage provider: weaviate, pinecone, qdrant (default: "weaviate")
- `memory_provider: str` — Memory type: vector, kv, graph, hybrid (default: "hybrid")
- `extraction_threshold: float` — Threshold for memory extraction (default: 0.7)
- `deduplication_threshold: float` — Threshold for deduplication (default: 0.9)
- `similarity_threshold: float` — Similarity threshold for retrieval (default: 0.8)
- `max_memory_entries: int` — Max memory entries to store (default: 10000)
- `memory_cleanup_interval: int` — Cleanup interval in seconds (default: 3600)
- `enable_agent_memory: bool` — Enable agent-scoped memory (default: True)
- `enable_shared_memory: bool` — Enable shared multi-agent memory (default: True)
- `salience_scoring: bool` — Enable salience scoring for importance (default: True)

## `GroundingConfig`
Configuration for the grounding module:
- `enable_grounding: bool` — Enable real-world grounding (default: True)
- `search_provider: str` — Primary search provider: google, bing, duckduckgo (default: "google")
- `search_providers: List[str]` — List of search providers to use (default: ["google", "bing"])
- `max_search_results: int` — Max search results per query (default: 5)
- `max_citations: int` — Max citations to include (default: 5)
- `citation_threshold: float` — Min confidence for citations (default: 0.8)
- `enable_fact_checking: bool` — Enable automated fact checking (default: True)
- `search_timeout: int` — Search timeout in seconds (default: 10)
- `trusted_sources: List[str]` — List of trusted source domains (default: [])

## `RetrieverConfig`
Configuration for the retriever module:
- `vector_db_provider: str` — Vector DB provider: weaviate, pinecone (default: "weaviate")
- `vector_db_url: Optional[str]` — Vector DB connection URL
- `vector_db_api_key: Optional[str]` — Vector DB API key
- `max_results: int` — Maximum results to retrieve (default: 10)
- `similarity_threshold: float` — Minimum similarity score (default: 0.7)
- `enable_kv_store: bool` — Enable key-value store for user attributes (default: True)
- `kv_store_url: Optional[str]` — KV store connection URL
- `enable_graph_engine: bool` — Enable graph-based retrieval (default: False)

## `RendererConfig`
Configuration for the renderer module:
- `include_trace: bool` — Include reasoning trace in output (default: True)
- `include_citations: bool` — Include citations in output (default: True)
- `trace_verbosity: str` — Trace verbosity: minimal, medium, detailed (default: "medium")
- `citation_format: str` — Citation format: markdown, html, plain (default: "markdown")
- `enable_metadata: bool` — Include metadata in output (default: True)

## `AgentConfig`
Configuration for agents in hierarchical system:
- `agent_id: str` — Unique agent identifier
- `agent_type: str` — Agent type: strategic, coordination, operational (default: "operational")
- `agent_role: str` — Specific role description (default: "general")
- `max_depth: int` — Maximum reasoning depth for this agent (default: 3)
- `enable_memory: bool` — Whether this agent has memory (default: True)
- `parent_agent_id: Optional[str]` — Parent agent in hierarchy

---

## Example Usage

### Basic Configuration
```python
from axon import AxonConfig, LLMConfig

config = AxonConfig(
    think_layer=dict(max_depth=5, enable_parallel=True, auto_iterate=True),
    llm=LLMConfig(provider="openai", model="gpt-4", api_key="your-api-key"),
    memory=dict(enable_memory=True, storage_provider="weaviate"),
    grounding=dict(enable_grounding=True, search_provider="google")
)
```

### Advanced Configuration with Hierarchical Agents
```python
from axon import AxonConfig, ThinkLayerConfig, LLMConfig, MemoryConfig, GroundingConfig

config = AxonConfig(
    think_layer=ThinkLayerConfig(
        max_depth=10,
        enable_parallel=True,
        enable_hierarchical=True,
        agent_hierarchy="strategic",
        max_branches=5,
        evaluation_threshold=0.7
    ),
    llm=LLMConfig(
        provider="anthropic",
        model="claude-3-sonnet-20240229",
        api_key="your-key",
        temperature=0.8,
        max_tokens=2000
    ),
    memory=MemoryConfig(
        enable_memory=True,
        storage_provider="weaviate",
        enable_agent_memory=True,
        enable_shared_memory=True,
        salience_scoring=True
    ),
    grounding=GroundingConfig(
        enable_grounding=True,
        search_provider="google",
        max_citations=5,
        enable_fact_checking=True
    )
)
``` 