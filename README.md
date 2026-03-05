# Ax0n: Model-Agnostic Think & Memory Layer for LLMs

[![PyPI version](https://badge.fury.io/py/ax0n-ai.svg)](https://badge.fury.io/py/ax0n-ai)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

**Ax0n** is a modular, scalable, and model-agnostic Think & Memory layer designed to empower AI agents with advanced hierarchical reasoning and persistent adaptive memory. It combines multiple advanced reasoning frameworks with a Mem0-inspired hybrid memory architecture to deliver production-grade contextual understanding, multi-agent collaboration, and real-world grounding—all without requiring a master control program (MCP).

Ax0n works by running each query through a full think-ground-revise pipeline: it generates structured thoughts using methods like Chain of Thought or Tree of Thoughts, grounds them against real-world evidence to detect contradictions, and then iteratively revises any flagged claims through the LLM until they converge—giving you self-correcting reasoning out of the box.

##  Features

###  Multiple Reasoning Methods
Ax0n supports multiple advanced reasoning frameworks:

- **Chain of Thoughts (CoT)** - Sequential decomposition with stepwise inference
- **Self-Consistency** - Multiple parallel reasoning paths with consensus voting
- **Algorithm of Thoughts (AoT)** - Algorithmic problem decomposition using procedural generation
- **Tree of Thoughts (ToT)** - Search trees with state evaluation and pruning
- **Graph of Thoughts (GoT)** - Directed graphs with dependency management *(coming soon)*

###  Hierarchical Thinking Layer
- **Strategic (High-Level) Agents**: Decompose abstract goals, evaluate global options
- **Coordination (Mid-Level) Agents**: Manage subtasks and inter-agent communication
- **Operational (Low-Level) Agents**: Execute atomic tasks and local data gathering
- JSON-format thoughts with traceability metadata (confidence score, source, timestamp)
- Configurable depth limits and parallelism for speed vs completeness tradeoffs
- Asynchronous execution and auto-iteration through thought layers

###  Multi-Agent Memory Layer (Mem0-Inspired)
- **Agent-Scoped Memory**: Context-rich semantic embeddings per agent with long-term persistence
- **Shared Joint Memory**: Centralized knowledge graph connecting entities and shared states
- **Hybrid Storage Backend**:
  - Semantic Vector Database for efficient nearest neighbor querying
  - Graph Database for complex relationships and multi-hop reasoning
  - Key-Value Store for rapid atomic fact lookups
- Adaptive memory consolidation with conflict resolution
- Salience scoring and intelligent memory pruning

###  Real-World Grounding
- Fact verification with citations from trusted sources
- Dynamic web search integration for up-to-date information
- Evidence-based reasoning with source metadata
- Contradiction detection and re-evaluation

###  Model Agnostic & Production-Ready
- Compatible with OpenAI GPT, Anthropic Claude, local LLMs (LLaMA, Falcon)
- Configurable prompting strategies and parameters
- Clean async API bindings and SDKs
- Low latency with token budget management
- Security features: encryption, zero-trust access

##  Installation

```bash
pip install ax0n-ai
```

##  Quick Start

### Basic Usage

```python
import asyncio
from axon import Axon, AxonConfig, ReasoningMethod, LLMConfig

# Initialize Axon with configuration
config = AxonConfig(
    think_layer=dict(max_depth=5, enable_parallel=True, auto_iterate=True),
    llm=LLMConfig(provider="openai", model="gpt-4", api_key="your-api-key"),
    memory=dict(enable_memory=True, storage_provider="weaviate", similarity_threshold=0.8),
    grounding=dict(enable_grounding=True, search_provider="google", max_citations=5)
)

axon = Axon(config)

async def main():
    query = "Plan a sustainable itinerary for Kyoto including historical sites."
    result = await axon.think(query, ReasoningMethod.TOT)  # Tree of Thoughts
    print(f"Answer:\n{result['answer']}")

asyncio.run(main())
```

### Using Different Reasoning Methods

```python
# Chain of Thoughts (CoT) - Sequential reasoning
result = await axon.think(query, ReasoningMethod.COT)

# Self-Consistency - Parallel paths with voting
result = await axon.think(query, ReasoningMethod.SELF_CONSISTENCY)

# Algorithm of Thoughts (AoT) - Algorithmic decomposition
result = await axon.think(query, ReasoningMethod.AOT)

# Tree of Thoughts (ToT) - Tree-based exploration
result = await axon.think(query, ReasoningMethod.TOT)

print(f"Answer: {result['answer']}")
print(f"Thoughts: {len(result['thoughts'])} generated")
print(f"Citations: {len(result['citations'])} sources")
```

### Convenience Methods

```python
# Use convenience methods for specific reasoning approaches
result = await axon.think_sequential(query)           # Chain of Thoughts
result = await axon.think_tree_of_thoughts(query)     # Tree of Thoughts
result = await axon.think_self_consistency(query)     # Self-Consistency
result = await axon.think_algorithm_of_thoughts(query) # Algorithm of Thoughts
```

### Method Comparison

```python
# Get information about all available methods
methods = axon.get_available_methods()
for method in methods:
    print(f"{method['name']}: {method['description']} (Complexity: {method['complexity']})")

# Get specific method info
info = axon.get_method_info(ReasoningMethod.TOT)
print(f"ToT: {info['description']}")
```

##  Configuration

### Basic Configuration

```python
from axon import AxonConfig, ThinkLayerConfig, LLMConfig

config = AxonConfig(
    think_layer=ThinkLayerConfig(
        max_depth=5,              # Maximum thought depth
        enable_parallel=True,     # Enable parallel reasoning
        auto_iterate=True         # Auto-iterate through thoughts
    ),
    llm=LLMConfig(
        provider="openai",        # LLM provider
        model="gpt-4",           # Model name
        api_key="your-key",      # API key
        temperature=0.7,         # Generation temperature
        max_tokens=1000          # Max tokens per response
    )
)
```

### Advanced Configuration

```python
from axon import AxonConfig, ThinkLayerConfig, LLMConfig, MemoryConfig, GroundingConfig

config = AxonConfig(
    think_layer=ThinkLayerConfig(
        max_depth=10,
        enable_parallel=True,
        auto_iterate=True,
        evaluation_threshold=0.7,
        max_branches=5,
        enable_hierarchical=True,
        agent_hierarchy="strategic"
    ),
    llm=LLMConfig(
        provider="anthropic",
        model="claude-3-sonnet-20240229",
        api_key="your-key",
        temperature=0.8,
        max_tokens=2000,
        timeout=60
    ),
    grounding=GroundingConfig(
        enable_grounding=True,
        search_provider="google",
        max_citations=5,
        enable_fact_checking=True
    ),
    memory=MemoryConfig(
        enable_memory=True,
        storage_provider="weaviate",
        similarity_threshold=0.8,
        enable_agent_memory=True,
        enable_shared_memory=True,
        salience_scoring=True
    )
)

axon = Axon(config)
```

##  Examples

### Multiple Reasoning Methods Demo

```python
# See examples/multiple_reasoning_methods_demo.py for a comprehensive demo
# showcasing all reasoning methods with the same query
```

### Tree of Thoughts Demo

```python
# See examples/tree_of_thoughts_demo.py for advanced ToT reasoning
```

##  API Reference

### Core Classes

#### `Axon`
Main class for the Think & Memory layer.

**Methods:**
- `think(query, method=ReasoningMethod.COT, context=None)` - Main thinking method
- `think_sequential(query, context=None)` - Chain of Thoughts
- `think_tree_of_thoughts(query, context=None)` - Tree of Thoughts
- `think_self_consistency(query, context=None)` - Self-Consistency
- `think_algorithm_of_thoughts(query, context=None)` - Algorithm of Thoughts
- `get_available_methods()` - Get all reasoning methods
- `get_method_info(method)` - Get method information

#### `ReasoningMethod` Enum
Available reasoning methods:
- `COT` - Chain of Thoughts
- `SELF_CONSISTENCY` - Self-Consistency
- `AOT` - Algorithm of Thoughts
- `TOT` - Tree of Thoughts
- `GOT` - Graph of Thoughts (coming soon)

### Configuration Classes

#### `AxonConfig`
Main configuration class containing all component configs.

#### `ThinkLayerConfig`
Configuration for the think layer:
- `max_depth: int` - Maximum thought depth
- `enable_parallel: bool` - Enable parallel reasoning
- `auto_iterate: bool` - Auto-iterate through thoughts

#### `LLMConfig`
Configuration for LLM providers:
- `provider: str` - Provider name (openai, anthropic, etc.)
- `model: str` - Model name
- `api_key: str` - API key
- `temperature: float` - Generation temperature
- `max_tokens: int` - Maximum tokens per response

##  Reasoning Methods Deep Dive

### Chain of Thoughts (CoT)
**Best for:** Simple tasks, step-by-step reasoning
**Complexity:** Low
**Structure:** Linear

```python
result = await axon.think(query, ReasoningMethod.COT)
```

### Self-Consistency
**Best for:** High-confidence tasks, multiple perspectives
**Complexity:** Medium
**Structure:** Parallel CoT with voting

```python
result = await axon.think(query, ReasoningMethod.SELF_CONSISTENCY)
```

### Algorithm of Thoughts (AoT)
**Best for:** Math, path search, algorithmic problems
**Complexity:** Low-Moderate
**Structure:** Implicit algorithmic

```python
result = await axon.think(query, ReasoningMethod.AOT)
```

### Tree of Thoughts (ToT)
**Best for:** Planning, problem solving, exploration
**Complexity:** Medium
**Structure:** Tree with evaluation

```python
result = await axon.think(query, ReasoningMethod.TOT)
```

##  Testing

Run the test suite:

```bash
pytest tests/
```

Run specific test categories:

```bash
pytest tests/test_reasoning_methods.py  # Test reasoning methods
pytest tests/test_tree_of_thoughts.py   # Test ToT specifically
pytest tests/test_basic.py              # Test basic functionality
```

##  Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

##  License

MIT License - see [LICENSE](LICENSE) for details.

##  Acknowledgments

- Inspired by research on Chain of Thoughts, Tree of Thoughts, and related reasoning methods
- Built with modern Python async/await patterns
- Designed for extensibility and model-agnostic operation

---

**Ax0n** - Think smarter, remember better, reason deeper.  