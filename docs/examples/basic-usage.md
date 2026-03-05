# Basic Usage Examples

Simple examples to get you started with Ax0n.

## Simple Chain of Thoughts

```python
import asyncio
from axon import Axon, AxonConfig, LLMConfig, ReasoningMethod

async def simple_cot():
    # Basic configuration
    config = AxonConfig(
        llm=LLMConfig(
            provider="openai",
            model="gpt-4",
            api_key="your-api-key"
        )
    )
    
    axon = Axon(config)
    
    # Simple question
    result = await axon.think(
        "What are the benefits of renewable energy?",
        ReasoningMethod.COT
    )
    
    print(f"Answer: {result['answer']}")
    print(f"Thoughts generated: {len(result['thoughts'])}")
    
    # Show the reasoning process
    for i, thought in enumerate(result['thoughts']):
        print(f"\nThought {i+1}: {thought['thought']}")

asyncio.run(simple_cot())
```

## Tree of Thoughts for Complex Problems

```python
async def complex_tot():
    config = AxonConfig(
        llm=LLMConfig(
            provider="openai",
            model="gpt-4",
            api_key="your-api-key"
        ),
        reasoning=ReasoningConfig(
            max_depth=5,
            max_branches=3,
            evaluation_threshold=0.7
        )
    )
    
    axon = Axon(config)
    
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
    
    # Show the best path
    best_thoughts = [t for t in result['thoughts'] if t.get('score', 0) > 0.7]
    print(f"\nHigh-scoring thoughts: {len(best_thoughts)}")

asyncio.run(complex_tot())
```

## Comparing Different Methods

```python
async def compare_methods():
    config = AxonConfig(
        llm=LLMConfig(
            provider="openai",
            model="gpt-4",
            api_key="your-api-key"
        )
    )
    
    axon = Axon(config)
    
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

## With Memory

```python
async def with_memory():
    config = AxonConfig(
        llm=LLMConfig(
            provider="openai",
            model="gpt-4",
            api_key="your-api-key"
        ),
        memory=MemoryConfig(
            enabled=True,
            storage_type="in_memory",
            max_entries=1000
        )
    )
    
    axon = Axon(config)
    
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
    print(f"Memory entries used: {len(result2.get('memory_used', []))}")

asyncio.run(with_memory())
```

## Error Handling

```python
import asyncio
from axon import AxonError, ConfigurationError

async def robust_example():
    config = AxonConfig(
        llm=LLMConfig(
            provider="openai",
            model="gpt-4",
            api_key="your-api-key"
        )
    )
    
    axon = Axon(config)
    
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

## Custom Configuration

```python
async def custom_config():
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
                "evaluation_threshold": 0.7,
                "beam_width": 2
            }
        ),
        memory=MemoryConfig(
            enabled=True,
            storage_type="in_memory",
            max_entries=1000,
            similarity_threshold=0.8
        )
    )
    
    axon = Axon(config)
    
    result = await axon.think(
        "Explain quantum computing in simple terms",
        ReasoningMethod.TOT
    )
    
    print(f"Answer: {result['answer']}")
    print(f"Configuration used: {result.get('config_used', 'default')}")

asyncio.run(custom_config())
```

## Batch Processing

```python
async def batch_processing():
    config = AxonConfig(
        llm=LLMConfig(
            provider="openai",
            model="gpt-4",
            api_key="your-api-key"
        )
    )
    
    axon = Axon(config)
    
    queries = [
        "What is machine learning?",
        "How does photosynthesis work?",
        "What are the benefits of exercise?",
        "Explain the water cycle"
    ]
    
    results = []
    
    for query in queries:
        result = await axon.think(query, ReasoningMethod.COT)
        results.append({
            'query': query,
            'answer': result['answer'],
            'thoughts': len(result['thoughts']),
            'time': result.get('execution_time', 0)
        })
    
    # Summary
    total_time = sum(r['time'] for r in results)
    total_thoughts = sum(r['thoughts'] for r in results)
    
    print(f"Processed {len(queries)} queries")
    print(f"Total time: {total_time:.2f}s")
    print(f"Total thoughts: {total_thoughts}")
    
    for result in results:
        print(f"\nQ: {result['query']}")
        print(f"A: {result['answer'][:100]}...")

asyncio.run(batch_processing())
```

## Next Steps

- [Multiple Methods](multiple-methods.md) - Advanced reasoning examples
- [Tree of Thoughts](tree-of-thoughts.md) - Complex problem solving
- [API Reference](../api/axon.md) - Complete API documentation
- [Configuration](../getting-started/configuration.md) - Configuration options 