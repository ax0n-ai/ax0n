"""
Configuration management for Ax0n
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
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
    """Configuration for the think layer"""
    
    max_depth: int = Field(5, gt=0, description="Maximum depth of thought chains")
    enable_parallel: bool = Field(True, description="Enable parallel thought execution")
    max_parallel_branches: int = Field(3, gt=0, description="Maximum parallel branches")
    auto_iterate: bool = Field(True, description="Automatically continue thinking")
    enable_revision: bool = Field(True, description="Allow thought revision")
    enable_branching: bool = Field(True, description="Allow thought branching")
    thought_timeout: int = Field(30, gt=0, description="Timeout for individual thoughts (seconds)")
    prompt_template_path: Optional[str] = Field(None, description="Path to custom prompt templates")


class GroundingConfig(BaseModel):
    """Configuration for the grounding module"""
    
    enable_grounding: bool = Field(True, description="Enable real-world grounding")
    search_providers: List[str] = Field(["google", "bing"], description="Search providers to use")
    max_search_results: int = Field(5, gt=0, description="Maximum search results per query")
    citation_threshold: float = Field(0.8, ge=0.0, le=1.0, description="Minimum confidence for citations")
    enable_fact_checking: bool = Field(True, description="Enable automated fact checking")
    search_timeout: int = Field(10, gt=0, description="Search timeout in seconds")


class MemoryConfig(BaseModel):
    """Configuration for the memory manager"""
    
    enable_memory: bool = Field(True, description="Enable persistent memory")
    memory_provider: str = Field("vector", description="Memory storage provider (vector, kv, hybrid)")
    extraction_threshold: float = Field(0.7, ge=0.0, le=1.0, description="Threshold for memory extraction")
    deduplication_threshold: float = Field(0.9, ge=0.0, le=1.0, description="Threshold for deduplication")
    max_memory_entries: int = Field(10000, gt=0, description="Maximum memory entries to store")
    memory_cleanup_interval: int = Field(3600, gt=0, description="Memory cleanup interval (seconds)")


class RendererConfig(BaseModel):
    """Configuration for the renderer module"""
    
    include_trace: bool = Field(True, description="Include reasoning trace in output")
    include_citations: bool = Field(True, description="Include citations in output")
    trace_verbosity: str = Field("medium", description="Trace verbosity (minimal, medium, detailed)")
    citation_format: str = Field("markdown", description="Citation format (markdown, html, plain)")
    enable_metadata: bool = Field(True, description="Include metadata in output")


class AxonConfig(BaseModel):
    """Main configuration for Ax0n"""
    
    # Core components
    llm: LLMConfig = Field(..., description="LLM configuration")
    retriever: RetrieverConfig = Field(default_factory=RetrieverConfig, description="Retriever configuration")
    think_layer: ThinkLayerConfig = Field(default_factory=ThinkLayerConfig, description="Think layer configuration")
    grounding: GroundingConfig = Field(default_factory=GroundingConfig, description="Grounding configuration")
    memory: MemoryConfig = Field(default_factory=MemoryConfig, description="Memory configuration")
    renderer: RendererConfig = Field(default_factory=RendererConfig, description="Renderer configuration")
    
    # Global settings
    log_level: str = Field("INFO", description="Logging level")
    enable_async: bool = Field(True, description="Enable async execution")
    max_concurrent_requests: int = Field(10, gt=0, description="Maximum concurrent requests")
    request_timeout: int = Field(60, gt=0, description="Global request timeout (seconds)")
    
    # Development settings
    debug_mode: bool = Field(False, description="Enable debug mode")
    enable_metrics: bool = Field(False, description="Enable performance metrics")
    cache_enabled: bool = Field(True, description="Enable response caching")
    
    # Custom settings
    custom_settings: Dict[str, Any] = Field(default_factory=dict, description="Custom configuration settings")
    
    class Config:
        """Pydantic configuration"""
        validate_assignment = True
        extra = "forbid" 