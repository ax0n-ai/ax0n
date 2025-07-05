# Ax0n: Model-Agnostic Think & Memory Layer for LLMs

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

Ax0n is a **model-agnostic Think & Memory layer** for LLMs. It enables structured, parallel reasoning with real-world grounding and persistent memory—no MCP needed.

## Features

- **Structured Reasoning**: Multi-step thought processes with JSON meta-control
- **Parallel Execution**: Tree of Thoughts / APR-style branching and merging
- **Real-world Grounding**: Fact verification with citations and evidence
- **Persistent Memory**: Mem0-inspired knowledge extraction and storage
- **Model Agnostic**: Works with any LLM (OpenAI, Anthropic, local models)

## Quick Start

```bash
pip install axon
```

```python
from axon import Axon

# Initialize with your preferred LLM
ax = Axon(llm_client="openai", api_key="your-key")

# Generate structured thoughts
result = await ax.think(
    "What's the best time to visit Kyoto?",
    max_depth=3,
    enable_grounding=True
)

print(result.answer)
print(result.trace)  # Full reasoning trace
print(result.citations)  # Evidence sources
```

## Architecture

Ax0n consists of 7 core modules:

1. **Retriever** - Context fetching via embeddings and KV lookup
2. **Think Layer** - Structured, parallel thought generation
3. **Grounding Module** - Real-world fact validation
4. **Memory Manager** - Knowledge extraction and persistence
5. **Renderer** - Output formatting with traces and citations
6. **Orchestrator** - Module coordination and flow control
7. **Testing & Validation** - Comprehensive test suite

## Documentation

- [Installation Guide](docs/installation.md)
- [API Reference](docs/api.md)
- [Architecture Overview](docs/architecture.md)
- [Examples](examples/)

## Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Inspired by Mem0's memory extraction patterns
- Built on Tree of Thoughts and APR research
- Community feedback and testing 