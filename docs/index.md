# Ax0n: Model-Agnostic Think & Memory Layer for LLMs

Welcome to the Ax0n documentation! Ax0n is a modular, scalable, and model-agnostic Think & Memory layer designed to empower AI agents with advanced hierarchical reasoning and persistent adaptive memory.

##  What is Ax0n?

Ax0n combines multiple advanced reasoning frameworks with a Mem0-inspired hybrid memory architecture to deliver production-grade contextual understanding, multi-agent collaboration, and real-world grounding—all without requiring a master control program (MCP).

##  Key Features

### Multiple Reasoning Methods
- **Chain of Thoughts (CoT)**: Sequential decomposition with stepwise inference
- **Self-Consistency**: Multiple parallel reasoning paths with consensus voting
- **Algorithm of Thoughts (AoT)**: Algorithmic problem decomposition
- **Tree of Thoughts (ToT)**: Search trees with state evaluation and pruning
- **Graph of Thoughts (GoT)**: Graph-based reasoning *(coming soon)*

### Hierarchical Thinking Layer
- Strategic, coordination, and operational agent levels
- JSON-structured thoughts with traceability
- Asynchronous execution and auto-iteration
- Configurable depth and parallelism

### Multi-Agent Memory Layer
- Agent-scoped and shared memory
- Hybrid storage (vector, graph, key-value)
- Adaptive memory consolidation
- Salience scoring and intelligent pruning

### Real-World Grounding
- Fact verification with citations
- Dynamic web search integration
- Evidence-based reasoning
- Contradiction detection

### Model Agnostic & Production-Ready
- Compatible with OpenAI, Anthropic, local LLMs
- Clean async API
- Low latency with token budget management
- Security: encryption, zero-trust access

##  Installation

```bash
pip install ax0n-ai
```

##  Quick Start

```python
import asyncio
from axon import Axon, AxonConfig, ReasoningMethod, LLMConfig

config = AxonConfig(
    think_layer=dict(max_depth=5, enable_parallel=True, auto_iterate=True),
    llm=LLMConfig(provider="openai", model="gpt-4", api_key="your-api-key"),
    memory=dict(enable_memory=True, storage_provider="weaviate"),
    grounding=dict(enable_grounding=True, search_provider="google")
)

axon = Axon(config)

async def main():
    result = await axon.think(
        "Plan a sustainable itinerary for Kyoto",
        ReasoningMethod.TOT
    )
    print(f"Answer: {result['answer']}")

asyncio.run(main())
```

##  Documentation Structure

- **[Getting Started](getting-started/installation.md)**: Installation and setup
- **[API Reference](api/README.md)**: Detailed API documentation
- **[Examples](examples/basic-usage.md)**: Usage examples and demos
- **[Guides](guides/)**: In-depth guides and tutorials

##  Architecture

Ax0n is built on five core components:

1. **Retriever**: Context fetching with vector search and graph queries
2. **Think Layer**: Structured reasoning with multiple methods
3. **Grounding Module**: Fact verification and citation
4. **Memory Manager**: Persistent knowledge with Mem0 architecture
5. **Renderer**: Output formatting with traces and citations

##  Contributing

Contributions are welcome! Please see our contributing guidelines.

##  License

MIT License - see [LICENSE](../LICENSE) for details.

##  Acknowledgments

Inspired by research on Chain of Thoughts, Tree of Thoughts, Algorithm of Thoughts, and the Mem0 memory architecture.

---

**Ax0n** - Think smarter, remember better, reason deeper. 
