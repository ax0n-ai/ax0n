from typing import Any, Dict, List, Optional
from pydantic import BaseModel, ConfigDict, Field, model_validator
from .models import LLMConfig


class RetrieverConfig(BaseModel):
    """Configuration for the retriever module"""

    vector_db_provider: str = Field("weaviate", description="Vector DB provider (weaviate, pinecone)")
    vector_db_url: Optional[str] = Field(None, description="Vector DB connection URL")
    vector_db_api_key: Optional[str] = Field(None, description="Vector DB API key")
    max_results: int = Field(10, gt=0, description="Maximum results to retrieve")
    similarity_threshold: float = Field(0.7, ge=0.0, le=1.0, description="Minimum similarity score")
    enable_kv_store: bool = Field(True, description="Enable key-value store for user attributes")
    kv_store_url: Optional[str] = Field(None, description="KV store connection URL")
    enable_graph_engine: bool = Field(False, description="Enable graph-based retrieval")


class ThinkLayerConfig(BaseModel):
    """Configuration for the hierarchical think layer"""

    max_depth: int = Field(5, gt=0, description="Maximum depth of thought chains")
    enable_parallel: bool = Field(True, description="Enable parallel thought execution")
    max_parallel_branches: int = Field(3, gt=0, description="Maximum parallel branches")
    max_branches: int = Field(5, gt=0, description="Maximum branches for ToT")
    auto_iterate: bool = Field(True, description="Automatically continue thinking")
    enable_revision: bool = Field(True, description="Allow thought revision")
    max_revision_iterations: int = Field(3, ge=0, le=10, description="Maximum revision loop iterations (0 to disable)")
    revision_score_threshold: float = Field(0.7, ge=0.0, le=1.0, description="Stop revising when all thoughts score above this")
    max_revisions_per_thought: int = Field(2, ge=1, le=5, description="Maximum times a single thought can be revised")
    enable_branching: bool = Field(True, description="Allow thought branching")
    thought_timeout: int = Field(30, gt=0, description="Timeout for individual thoughts (seconds)")
    prompt_template_path: Optional[str] = Field(None, description="Path to custom prompt templates")
    evaluation_threshold: float = Field(0.7, ge=0.0, le=1.0, description="Threshold for thought evaluation")
    num_parallel_paths: int = Field(5, ge=1, le=20, description="Number of parallel reasoning paths for self-consistency")
    enable_hierarchical: bool = Field(False, description="Enable hierarchical agent structure")
    agent_hierarchy: str = Field("flat", description="Agent hierarchy mode (flat, strategic, coordinated)")


class GroundingConfig(BaseModel):
    """Configuration for the grounding module"""

    enable_grounding: bool = Field(True, description="Enable real-world grounding")
    search_provider: str = Field("google", description="Primary search provider (google, bing, duckduckgo)")
    search_providers: List[str] = Field(["google", "bing"], description="Search providers to use")
    max_search_results: int = Field(5, gt=0, description="Maximum search results per query")
    max_citations: int = Field(5, gt=0, description="Maximum citations to include")
    citation_threshold: float = Field(0.8, ge=0.0, le=1.0, description="Minimum confidence for citations")
    enable_fact_checking: bool = Field(True, description="Enable automated fact checking")
    search_timeout: int = Field(10, gt=0, description="Search timeout in seconds")
    trusted_sources: List[str] = Field(default_factory=list, description="List of trusted source domains")
    support_similarity_threshold: float = Field(0.2, ge=0.0, le=1.0, description="Minimum similarity for claim-citation overlap")
    contradiction_similarity_threshold: float = Field(0.3, ge=0.0, le=1.0, description="Minimum similarity to flag contradictions")


class MemoryConfig(BaseModel):
    """Configuration for the memory manager (Mem0-inspired hybrid architecture)"""

    enable_memory: bool = Field(True, description="Enable persistent memory")
    storage_provider: str = Field("weaviate", description="Storage provider (weaviate, pinecone, qdrant)")
    memory_provider: str = Field("hybrid", description="Memory storage provider (vector, kv, graph, hybrid)")
    extraction_threshold: float = Field(0.7, ge=0.0, le=1.0, description="Threshold for memory extraction")
    deduplication_threshold: float = Field(0.9, ge=0.0, le=1.0, description="Threshold for deduplication")
    similarity_threshold: float = Field(0.8, ge=0.0, le=1.0, description="Similarity threshold for memory retrieval")
    max_memory_entries: int = Field(10000, gt=0, description="Maximum memory entries to store")
    memory_cleanup_interval: int = Field(3600, gt=0, description="Memory cleanup interval (seconds)")
    memory_ttl_seconds: Optional[int] = Field(None, ge=0, description="Time-to-live for memories in seconds (None = no expiry)")
    enable_agent_memory: bool = Field(True, description="Enable agent-scoped memory")
    enable_shared_memory: bool = Field(True, description="Enable shared multi-agent memory")
    salience_scoring: bool = Field(True, description="Enable salience scoring for memory importance")


class RendererConfig(BaseModel):
    """Configuration for the renderer module"""

    include_trace: bool = Field(True, description="Include reasoning trace in output")
    include_citations: bool = Field(True, description="Include citations in output")
    trace_verbosity: str = Field("medium", description="Trace verbosity (minimal, medium, detailed)")
    citation_format: str = Field("markdown", description="Citation format (markdown, html, plain)")
    enable_metadata: bool = Field(True, description="Include metadata in output")


class AxonConfig(BaseModel):
    """Main configuration for Ax0n"""

    llm: Optional[LLMConfig] = Field(None, description="LLM configuration")
    retriever: RetrieverConfig = Field(default_factory=RetrieverConfig, description="Retriever configuration")
    think_layer: ThinkLayerConfig = Field(default_factory=ThinkLayerConfig, description="Think layer configuration")
    grounding: GroundingConfig = Field(default_factory=GroundingConfig, description="Grounding configuration")
    memory: MemoryConfig = Field(default_factory=MemoryConfig, description="Memory configuration")
    renderer: RendererConfig = Field(default_factory=RendererConfig, description="Renderer configuration")

    log_level: str = Field("INFO", description="Logging level")
    enable_async: bool = Field(True, description="Enable async execution")
    max_concurrent_requests: int = Field(10, gt=0, description="Maximum concurrent requests")
    request_timeout: int = Field(60, gt=0, description="Global request timeout (seconds)")
    max_query_length: int = Field(10000, gt=0, description="Maximum query length in characters")
    max_json_response_size: int = Field(10_000_000, gt=0, description="Maximum JSON response size in bytes (10MB)")

    quality_gate_threshold: float = Field(0.0, ge=0.0, le=1.0, description="Minimum thought confidence to pass quality gate (0.0 = disabled)")
    llm_retry_attempts: int = Field(3, ge=1, le=10, description="Number of retry attempts for LLM calls")
    llm_retry_base_delay: float = Field(1.0, gt=0, description="Base delay in seconds between LLM retries (exponential backoff)")

    debug_mode: bool = Field(False, description="Enable debug mode")
    enable_metrics: bool = Field(False, description="Enable performance metrics")
    cache_enabled: bool = Field(True, description="Enable response caching")

    custom_settings: Dict[str, Any] = Field(default_factory=dict, description="Custom configuration settings")

    @model_validator(mode='after')
    def validate_timeouts(self) -> 'AxonConfig':
        """Ensure request_timeout >= think_layer.thought_timeout"""
        if self.request_timeout < self.think_layer.thought_timeout:
            raise ValueError(
                f"request_timeout ({self.request_timeout}s) must be >= "
                f"think_layer.thought_timeout ({self.think_layer.thought_timeout}s)"
            )
        return self

    model_config = ConfigDict(validate_assignment=True, extra="forbid")
