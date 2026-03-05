# Installation

## Prerequisites

Ax0n requires Python 3.8 or higher.

## Basic Installation

Install Ax0n from PyPI:

```bash
pip install ax0n-ai
```

## Development Installation

For development or to get the latest features:

```bash
# Clone the repository
git clone https://github.com/axon-ai/axon.git
cd axon

# Install in development mode
pip install -e .

# Install development dependencies
pip install -e .[dev]
```

## Dependencies

Ax0n includes support for various LLM providers and storage backends:

### LLM Providers
- **OpenAI**: `openai>=1.0.0`
- **Anthropic**: `anthropic>=0.7.0`
- **Local Models**: Via compatible APIs

### Vector Databases
- **Weaviate**: `weaviate-client>=3.25.0`
- **Pinecone**: `pinecone-client>=2.2.0`

### Key-Value Stores
- **Redis**: `redis>=4.5.0`

### Other Dependencies
- **HTTP Client**: `aiohttp>=3.8.0`
- **Data Validation**: `pydantic>=2.0.0`
- **Logging**: `structlog>=23.0.0`
- **Retry Logic**: `tenacity>=8.2.0`

## Environment Setup

### API Keys

Set up your LLM provider API keys:

```bash
# OpenAI
export OPENAI_API_KEY="your-openai-api-key"

# Anthropic
export ANTHROPIC_API_KEY="your-anthropic-api-key"
```

### Optional: Vector Database Setup

For advanced features, you can set up vector databases:

```bash
# Weaviate
export WEAVIATE_URL="http://localhost:8080"
export WEAVIATE_API_KEY="your-weaviate-key"

# Pinecone
export PINECONE_API_KEY="your-pinecone-key"
export PINECONE_ENVIRONMENT="us-east1-gcp"
```

### Optional: Redis Setup

For memory features:

```bash
export REDIS_URL="redis://localhost:6379"
```

## Verification

Test your installation:

```python
import asyncio
from axon import Axon, ReasoningMethod

# Test basic import
print(" Ax0n imported successfully!")

# Test configuration
config = AxonConfig()
axon = Axon(config)
print(" Axon initialized successfully!")
```

## Next Steps

- [Quick Start](quick-start.md) - Get up and running in minutes
- [Configuration](configuration.md) - Learn about configuration options
- [Examples](../examples/basic-usage.md) - See working examples 