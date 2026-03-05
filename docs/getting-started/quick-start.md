# Quick Start

Get up and running with Ax0n in minutes!

## Basic Usage

### 1. Import and Initialize

```python
import asyncio
from axon import Axon, AxonConfig, LLMConfig, ReasoningMethod

# Create configuration
config = AxonConfig(
    llm=LLMConfig(
        provider="openai",
        model="gpt-4",
        api_key="your-api-key"  # Or use environment variable
    )
)

# Initialize Axon
axon = Axon(config)
```

### 2. Simple Chain of Thoughts

```python
async def basic_example():
    query = "What are the benefits of renewable energy?"
    
    result = await axon.think(query, ReasoningMethod.COT)
    
    print(f"Answer: {result['answer']}")
    print(f"Thoughts generated: {len(result['thoughts'])}")
    
    # View the reasoning process
    for i, thought in enumerate(result['thoughts']):
        print(f"\nThought {i+1}: {thought['thought']}")

asyncio.run(basic_example())
```

### 3. Tree of Thoughts for Complex Problems

```python
async def complex_problem():
    query = """
    A company has $3M and must choose between:
    1. Expanding to new markets
    2. Improving existing products
    3. Acquiring a competitor
    
    What should they do and why?
    """
    
    result = await axon.think(query, ReasoningMethod.TOT)
    
    print(f"Final Answer: {result['answer']}")
    print(f"Exploration depth: {result['max_depth']}")
    print(f"Total thoughts explored: {len(result['thoughts'])}")

asyncio.run(complex_problem())
```

## Advanced Features

### Multiple Reasoning Methods

```python
async def compare_methods():
    query = "What's the capital of France?"
    
    methods = [
        ReasoningMethod.COT,
        ReasoningMethod.SELF_CONSISTENCY,
        ReasoningMethod.TOT
    ]
    
    for method in methods:
        print(f"\n=== {method.value} ===")
        result = await axon.think(query, method)
        print(f"Answer: {result['answer']}")
        print(f"Thoughts: {len(result['thoughts'])}")
        print(f"Time: {result.get('execution_time', 'N/A')}s")

asyncio.run(compare_methods())
```

### Custom Configuration

```python
from axon import AxonConfig, LLMConfig, ReasoningConfig

# Detailed configuration
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
        evaluation_threshold=0.7
    ),
    memory=MemoryConfig(
        enabled=True,
        storage_type="redis"
    )
)

axon = Axon(config)
```

### With Memory and Grounding

```python
async def advanced_example():
    # First query - stores in memory
    result1 = await axon.think(
        "What are the latest developments in AI?",
        ReasoningMethod.COT
    )
    
    # Second query - uses memory for context
    result2 = await axon.think(
        "How do these developments compare to last year?",
        ReasoningMethod.TOT
    )
    
    print(f"First answer: {result1['answer']}")
    print(f"Second answer: {result2['answer']}")
    print(f"Memory entries: {len(result2.get('memory_used', []))}")

asyncio.run(advanced_example())
```

## Error Handling

```python
import asyncio
from axon import AxonError

async def robust_example():
    try:
        result = await axon.think(
            "What's the weather like?",
            ReasoningMethod.COT,
            timeout=30
        )
        print(f"Success: {result['answer']}")
        
    except AxonError as e:
        print(f"Axon error: {e}")
    except asyncio.TimeoutError:
        print("Request timed out")
    except Exception as e:
        print(f"Unexpected error: {e}")

asyncio.run(robust_example())
```

## Next Steps

- [Configuration](configuration.md) - Learn about all configuration options
- [API Reference](../api/axon.md) - Complete API documentation
- [Examples](../examples/basic-usage.md) - More working examples
- [Features](../features/reasoning.md) - Deep dive into reasoning methods 