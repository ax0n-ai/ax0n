# Axon Class

The main entry point for the Ax0n Think & Memory layer.

## Overview

The `Axon` class orchestrates the full reasoning, grounding, and memory pipeline. It provides a unified interface for different reasoning methods and manages the interaction between all components.

## Constructor

```python
Axon(config: Optional[AxonConfig] = None)
```

**Parameters:**
- `config` (Optional[AxonConfig]): Configuration object. If not provided, uses defaults.

**Example:**
```python
from axon import Axon, AxonConfig, LLMConfig

config = AxonConfig(
    llm=LLMConfig(
        provider="openai",
        model="gpt-4",
        api_key="your-api-key"
    )
)
axon = Axon(config)
```

## Core Methods

### `think()`

Main method for running the reasoning pipeline.

```python
async think(
    query: str,
    method: ReasoningMethod = ReasoningMethod.COT,
    context: Optional[Dict[str, Any]] = None,
    **kwargs
) -> Dict[str, Any]
```

**Parameters:**
- `query` (str): The input question or problem to solve
- `method` (ReasoningMethod): The reasoning method to use (default: COT)
- `context` (Optional[Dict]): Additional context for the query
- `**kwargs`: Additional configuration overrides

**Returns:**
- `Dict[str, Any]`: Result containing answer, thoughts, metadata, etc.

**Example:**
```python
result = await axon.think(
    "What's the best time to visit Kyoto?",
    ReasoningMethod.TOT,
    context={"user_preferences": "cultural sites"}
)
print(result['answer'])
```

### `think_sequential()`

Run Chain of Thoughts (CoT) reasoning.

```python
async think_sequential(
    query: str,
    context: Optional[Dict[str, Any]] = None,
    **kwargs
) -> Dict[str, Any]
```

**Example:**
```python
result = await axon.think_sequential("What are the benefits of renewable energy?")
```

### `think_tree_of_thoughts()`

Run Tree of Thoughts (ToT) reasoning.

```python
async think_tree_of_thoughts(
    query: str,
    context: Optional[Dict[str, Any]] = None,
    **kwargs
) -> Dict[str, Any]
```

**Example:**
```python
result = await axon.think_tree_of_thoughts(
    "A company has $3M. Should they expand, improve products, or acquire?"
)
```

### `think_self_consistency()`

Run Self-Consistency reasoning.

```python
async think_self_consistency(
    query: str,
    context: Optional[Dict[str, Any]] = None,
    **kwargs
) -> Dict[str, Any]
```

**Example:**
```python
result = await axon.think_self_consistency("What's the capital of France?")
```

### `think_algorithm_of_thoughts()`

Run Algorithm of Thoughts (AoT) reasoning.

```python
async think_algorithm_of_thoughts(
    query: str,
    context: Optional[Dict[str, Any]] = None,
    **kwargs
) -> Dict[str, Any]
```

**Example:**
```python
result = await axon.think_algorithm_of_thoughts("Solve: 2x + 5 = 13")
```

## Utility Methods

### `get_available_methods()`

Get all available reasoning methods.

```python
get_available_methods() -> List[Dict[str, str]]
```

**Returns:**
- List of dictionaries containing method information

**Example:**
```python
methods = axon.get_available_methods()
for method in methods:
    print(f"{method['name']}: {method['description']}")
```

### `get_method_info()`

Get information about a specific reasoning method.

```python
get_method_info(method: ReasoningMethod) -> Dict[str, str]
```

**Parameters:**
- `method` (ReasoningMethod): The reasoning method to get info for

**Returns:**
- Dictionary containing method information

**Example:**
```python
info = axon.get_method_info(ReasoningMethod.TOT)
print(f"Description: {info['description']}")
print(f"Best for: {info['best_for']}")
```

## Return Format

All `think()` methods return a dictionary with the following structure:

```python
{
    "answer": str,                    # Final answer
    "thoughts": List[Dict],           # All generated thoughts
    "method": str,                    # Method used
    "execution_time": float,          # Time taken (seconds)
    "max_depth": int,                 # Maximum depth explored
    "memory_used": List[Dict],        # Memory entries used
    "grounding": List[Dict],          # Grounding evidence
    "metadata": Dict[str, Any]        # Additional metadata
}
```

### Thought Structure

Each thought in the `thoughts` list has this structure:

```python
{
    "thought": str,                   # The thought content
    "thought_number": int,            # Sequential number
    "depth": int,                     # Depth in reasoning tree
    "branch_id": Optional[str],       # Branch identifier
    "score": Optional[float],         # Evaluation score (0-1)
    "is_hypothesis": bool,            # Whether it's a hypothesis
    "is_verification": bool,          # Whether it's verification
    "needs_revision": bool,           # Whether revision is needed
    "metadata": Dict[str, Any]        # Additional thought metadata
}
```

## Error Handling

The Axon class raises specific exceptions:

```python
from axon import AxonError, ConfigurationError, ReasoningError

try:
    result = await axon.think("What's the weather?")
except AxonError as e:
    print(f"Axon error: {e}")
except ConfigurationError as e:
    print(f"Configuration error: {e}")
except ReasoningError as e:
    print(f"Reasoning error: {e}")
```

## Configuration

See [Configuration](../getting-started/configuration.md) for detailed configuration options.

## Examples

### Basic Usage
```python
from axon import Axon, AxonConfig, LLMConfig, ReasoningMethod

config = AxonConfig(
    llm=LLMConfig(
        provider="openai",
        model="gpt-4",
        api_key="your-api-key"
    )
)

axon = Axon(config)

# Simple question
result = await axon.think("What's the capital of France?")
print(result['answer'])

# Complex problem with Tree of Thoughts
result = await axon.think(
    "A company has $3M and must choose between expansion, "
    "product improvement, or acquisition. What should they do?",
    ReasoningMethod.TOT
)
print(f"Answer: {result['answer']}")
print(f"Thoughts explored: {len(result['thoughts'])}")
```

### Comparing Methods
```python
query = "What are the benefits of renewable energy?"

methods = [
    ReasoningMethod.COT,
    ReasoningMethod.SELF_CONSISTENCY,
    ReasoningMethod.TOT
]

for method in methods:
    result = await axon.think(query, method)
    print(f"\n{method.value}:")
    print(f"  Answer: {result['answer']}")
    print(f"  Thoughts: {len(result['thoughts'])}")
    print(f"  Time: {result['execution_time']:.2f}s")
```

### With Context and Memory
```python
# First query - stores in memory
result1 = await axon.think(
    "What are the latest developments in AI?",
    ReasoningMethod.COT
)

# Second query - uses memory for context
result2 = await axon.think(
    "How do these developments compare to last year?",
    ReasoningMethod.TOT,
    context={"previous_query": result1['answer']}
)

print(f"Memory entries used: {len(result2['memory_used'])}")
```

## Related

- [Reasoning Methods](reasoning-methods.md) - Detailed explanation of each method
- [Configuration](configuration.md) - Configuration options
- [Data Models](models.md) - Data structures and types
- [Components](components.md) - Individual component APIs 