# API Reference

Complete API reference for Ax0n Think & Memory Layer.

## Core Components

### Main Classes
- **[Axon](axon.md)**: Main orchestrator class
- **[Configuration](configuration.md)**: Configuration classes
- **[Models](models.md)**: Data models and types
- **[Reasoning Methods](reasoning-methods.md)**: Available reasoning methods

## Overview

Ax0n provides a clean, intuitive API for structured reasoning and memory management:

```python
from axon import Axon, AxonConfig, ReasoningMethod, LLMConfig

# Initialize with configuration
config = AxonConfig(
    llm=LLMConfig(provider="openai", model="gpt-4", api_key="key"),
    think_layer=dict(max_depth=5, enable_parallel=True),
    memory=dict(enable_memory=True, storage_provider="weaviate"),
    grounding=dict(enable_grounding=True, search_provider="google")
)

axon = Axon(config)

# Use reasoning methods
result = await axon.think(query, ReasoningMethod.TOT)
```

## Quick Reference

### Reasoning Methods
- `ReasoningMethod.COT` - Chain of Thoughts
- `ReasoningMethod.SELF_CONSISTENCY` - Self-Consistency
- `ReasoningMethod.AOT` - Algorithm of Thoughts
- `ReasoningMethod.TOT` - Tree of Thoughts
- `ReasoningMethod.GOT` - Graph of Thoughts *(coming soon)*

### Core Methods
- `axon.think(query, method)` - Main reasoning method
- `axon.think_sequential(query)` - CoT convenience method
- `axon.think_tree_of_thoughts(query)` - ToT convenience method
- `axon.think_self_consistency(query)` - Self-Consistency convenience method
- `axon.think_algorithm_of_thoughts(query)` - AoT convenience method
- `axon.get_available_methods()` - List all methods
- `axon.get_method_info(method)` - Get method details

### Memory Methods
- `axon.retrieve_memory(query, limit)` - Retrieve relevant memories
- `axon.update_memory(content, confidence)` - Add memory entry
- `axon.ground_claim(claim)` - Ground a specific claim

## Configuration Options

See [Configuration](configuration.md) for detailed configuration options including:
- Think Layer configuration
- LLM configuration
- Memory configuration
- Grounding configuration
- Retrieval configuration
- Renderer configuration

## Models

See [Models](models.md) for data model specifications including:
- Thought
- ThoughtResult
- MemoryEntry
- GroundingEvidence
- AgentConfig
