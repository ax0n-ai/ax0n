# Configuration

Ax0n is highly configurable to suit different use cases and requirements.

## Overview

Configuration in Ax0n is handled through the `AxonConfig` class, which contains settings for all components:

- **LLM Configuration**: Model provider, API keys, parameters
- **Reasoning Configuration**: Method-specific settings
- **Memory Configuration**: Storage and retrieval settings
- **Grounding Configuration**: Fact-checking settings
- **Retrieval Configuration**: Vector database settings

## Basic Configuration

### Minimal Setup

```python
from axon import AxonConfig, LLMConfig

# Minimal configuration
config = AxonConfig(
    llm=LLMConfig(
        provider="openai",
        model="gpt-4",
        api_key="your-api-key"
    )
)
```

### Environment Variables

You can use environment variables instead of hardcoding API keys:

```bash
export OPENAI_API_KEY="your-openai-api-key"
export ANTHROPIC_API_KEY="your-anthropic-api-key"
```

```python
from axon import AxonConfig, LLMConfig

# Uses environment variables automatically
config = AxonConfig(
    llm=LLMConfig(
        provider="openai",
        model="gpt-4"
        # api_key will be read from OPENAI_API_KEY
    )
)
```

## LLM Configuration

### OpenAI

```python
from axon import LLMConfig

llm_config = LLMConfig(
    provider="openai",
    model="gpt-4",
    api_key="your-api-key",
    temperature=0.7,
    max_tokens=2000,
    timeout=30
)
```

### Anthropic

```python
llm_config = LLMConfig(
    provider="anthropic",
    model="claude-3-sonnet-20240229",
    api_key="your-api-key",
    temperature=0.7,
    max_tokens=2000
)
```

### Local Models

```python
llm_config = LLMConfig(
    provider="local",
    model="llama-2-7b",
    base_url="http://localhost:11434/v1",
    api_key="ollama",  # or your local API key
    temperature=0.7
)
```

## Reasoning Configuration

### General Settings

```python
from axon import ReasoningConfig

reasoning_config = ReasoningConfig(
    max_depth=5,              # Maximum reasoning depth
    max_branches=3,           # Maximum parallel branches
    evaluation_threshold=0.7, # Minimum score for good thoughts
    auto_iterate=True,        # Continue until conclusion
    return_full_history=True  # Include all thoughts in result
)
```

### Method-Specific Settings

```python
reasoning_config = ReasoningConfig(
    # Tree of Thoughts specific
    tot_config={
        "max_depth": 5,
        "max_branches": 3,
        "evaluation_threshold": 0.7,
        "beam_width": 2
    },
    
    # Self-Consistency specific
    self_consistency_config={
        "num_paths": 5,
        "voting_threshold": 0.6
    },
    
    # Algorithm of Thoughts specific
    aot_config={
        "max_steps": 10,
        "step_timeout": 30
    }
)
```

## Memory Configuration

### Basic Memory

```python
from axon import MemoryConfig

memory_config = MemoryConfig(
    enabled=True,
    storage_type="in_memory",  # or "redis", "weaviate"
    max_entries=1000,
    similarity_threshold=0.8
)
```

### Redis Memory

```python
memory_config = MemoryConfig(
    enabled=True,
    storage_type="redis",
    redis_url="redis://localhost:6379",
    redis_db=0,
    max_entries=10000,
    similarity_threshold=0.8
)
```

### Vector Database Memory

```python
memory_config = MemoryConfig(
    enabled=True,
    storage_type="weaviate",
    weaviate_url="http://localhost:8080",
    weaviate_api_key="your-key",
    collection_name="axon_memory",
    max_entries=50000
)
```

## Grounding Configuration

### Basic Grounding

```python
from axon import GroundingConfig

grounding_config = GroundingConfig(
    enabled=True,
    search_provider="duckduckgo",  # or "google", "bing"
    max_search_results=5,
    citation_format="markdown"
)
```

### Advanced Grounding

```python
grounding_config = GroundingConfig(
    enabled=True,
    search_provider="google",
    google_api_key="your-google-api-key",
    google_cse_id="your-cse-id",
    max_search_results=10,
    citation_format="html",
    fact_check_threshold=0.8,
    require_citations=True
)
```

## Retrieval Configuration

### Vector Search

```python
from axon import RetrievalConfig

retrieval_config = RetrievalConfig(
    enabled=True,
    vector_provider="weaviate",
    weaviate_url="http://localhost:8080",
    weaviate_api_key="your-key",
    collection_name="axon_vectors",
    top_k=5,
    similarity_threshold=0.7
)
```

### Pinecone

```python
retrieval_config = RetrievalConfig(
    enabled=True,
    vector_provider="pinecone",
    pinecone_api_key="your-key",
    pinecone_environment="us-east1-gcp",
    index_name="axon-index",
    top_k=5
)
```

## Complete Configuration Example

```python
from axon import (
    AxonConfig, LLMConfig, ReasoningConfig, 
    MemoryConfig, GroundingConfig, RetrievalConfig
)

config = AxonConfig(
    llm=LLMConfig(
        provider="openai",
        model="gpt-4",
        temperature=0.7,
        max_tokens=2000
    ),
    
    reasoning=ReasoningConfig(
        max_depth=5,
        max_branches=3,
        evaluation_threshold=0.7,
        tot_config={
            "max_depth": 5,
            "max_branches": 3,
            "evaluation_threshold": 0.7
        }
    ),
    
    memory=MemoryConfig(
        enabled=True,
        storage_type="redis",
        redis_url="redis://localhost:6379",
        max_entries=10000,
        similarity_threshold=0.8
    ),
    
    grounding=GroundingConfig(
        enabled=True,
        search_provider="duckduckgo",
        max_search_results=5,
        citation_format="markdown"
    ),
    
    retrieval=RetrievalConfig(
        enabled=True,
        vector_provider="weaviate",
        weaviate_url="http://localhost:8080",
        top_k=5
    )
)

axon = Axon(config)
```

## Configuration Validation

Ax0n validates configuration at initialization:

```python
try:
    config = AxonConfig(...)
    axon = Axon(config)
except ConfigurationError as e:
    print(f"Configuration error: {e}")
```

## Runtime Configuration Overrides

You can override configuration at runtime:

```python
# Override reasoning depth for this query
result = await axon.think(
    "Complex problem here",
    ReasoningMethod.TOT,
    max_depth=10,
    evaluation_threshold=0.8
)

# Override LLM parameters
result = await axon.think(
    "Creative task",
    ReasoningMethod.COT,
    temperature=0.9,
    max_tokens=3000
)
```

## Configuration Best Practices

### For Development
```python
config = AxonConfig(
    llm=LLMConfig(
        provider="openai",
        model="gpt-4",
        temperature=0.7
    ),
    reasoning=ReasoningConfig(
        max_depth=3,  # Shorter for faster iteration
        return_full_history=True  # More debugging info
    )
)
```

### For Production
```python
config = AxonConfig(
    llm=LLMConfig(
        provider="openai",
        model="gpt-4",
        temperature=0.3,  # More consistent
        timeout=60
    ),
    reasoning=ReasoningConfig(
        max_depth=5,
        evaluation_threshold=0.8,  # Higher quality
        return_full_history=False  # Less memory usage
    ),
    memory=MemoryConfig(
        enabled=True,
        storage_type="redis",
        max_entries=50000
    )
)
```

### For Research
```python
config = AxonConfig(
    llm=LLMConfig(
        provider="openai",
        model="gpt-4",
        temperature=0.0  # Deterministic
    ),
    reasoning=ReasoningConfig(
        max_depth=10,
        return_full_history=True,
        auto_iterate=False  # Manual control
    ),
    grounding=GroundingConfig(
        enabled=True,
        require_citations=True
    )
)
```

## Related

- [API Reference](../api/axon.md) - Main API documentation
- [Data Models](../api/models.md) - Configuration data structures
- [Examples](../examples/basic-usage.md) - Working examples 