# Custom LLM Integration Guide

This guide shows you how to integrate custom LLM providers with Ax0n.

## Overview

Ax0n supports multiple LLM providers out of the box:
- OpenAI (GPT-4, GPT-3.5)
- Anthropic (Claude)
- Local models (via API)

You can also integrate your own custom LLM provider.

## Using Built-in Providers

### OpenAI

```python
from axon import Axon, AxonConfig, LLMConfig

config = AxonConfig(
    llm=LLMConfig(
        provider="openai",
        model="gpt-4",
        api_key="your-api-key",
        temperature=0.7,
        max_tokens=2000
    )
)

axon = Axon(config)
```

### Anthropic (Claude)

```python
config = AxonConfig(
    llm=LLMConfig(
        provider="anthropic",
        model="claude-3-opus-20240229",
        api_key="your-api-key",
        temperature=0.7,
        max_tokens=2000
    )
)

axon = Axon(config)
```

### Local Models

```python
config = AxonConfig(
    llm=LLMConfig(
        provider="local",
        model="llama-2-70b",
        base_url="http://localhost:8000",
        api_key="not-required"
    )
)

axon = Axon(config)
```

## Creating a Custom LLM Client

### Step 1: Define Your Client Class

Create a new file `src/axon/core/custom_llm.py`:

```python
from typing import List, Dict, Any, Optional
import structlog

logger = structlog.get_logger(__name__)


class CustomLLMClient:
    """Custom LLM client implementation"""
    
    def __init__(
        self,
        api_key: str,
        model: str,
        base_url: Optional[str] = None,
        **kwargs
    ):
        self.api_key = api_key
        self.model = model
        self.base_url = base_url or "https://api.your-provider.com"
        self.logger = logger.bind(component="custom_llm")
        
        # Initialize your client
        self._setup_client()
    
    def _setup_client(self):
        """Initialize the LLM client"""
        # Your initialization code here
        pass
    
    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1000,
        **kwargs
    ) -> str:
        """
        Generate a response from the LLM.
        
        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            
        Returns:
            Generated text response
        """
        try:
            # Your generation logic here
            response = await self._call_api(
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=temperature,
                max_tokens=max_tokens
            )
            return response
        except Exception as e:
            self.logger.error("generation_failed", error=str(e))
            raise
    
    async def _call_api(
        self,
        prompt: str,
        system_prompt: Optional[str],
        temperature: float,
        max_tokens: int
    ) -> str:
        """Make API call to your LLM provider"""
        # Implement your API call logic
        # Example structure:
        # response = await self.client.completions.create(
        #     model=self.model,
        #     messages=[...],
        #     temperature=temperature,
        #     max_tokens=max_tokens
        # )
        # return response.choices[0].message.content
        pass
```

### Step 2: Register Your Client

Update `src/axon/core/core.py` to include your custom client:

```python
from .custom_llm import CustomLLMClient

class Axon:
    # ... existing code ...
    
    def _get_llm_client(self, config: LLMConfig) -> Any:
        """Create LLM client based on configuration"""
        provider = config.provider.lower()
        
        if provider == "openai":
            return self._create_openai_client(config)
        elif provider == "anthropic":
            return self._create_anthropic_client(config)
        elif provider == "local":
            return self._create_local_client(config)
        elif provider == "custom":  # Add this
            return self._create_custom_client(config)
        else:
            self.logger.warning(
                "unknown_provider",
                provider=provider,
                using_mock=True
            )
            return MockLLMClient()
    
    def _create_custom_client(self, config: LLMConfig):
        """Create custom LLM client"""
        try:
            return CustomLLMClient(
                api_key=config.api_key,
                model=config.model,
                base_url=config.base_url
            )
        except Exception as e:
            self.logger.error("custom_client_init_failed", error=str(e))
            return MockLLMClient()
```

### Step 3: Use Your Custom Client

```python
from axon import Axon, AxonConfig, LLMConfig

config = AxonConfig(
    llm=LLMConfig(
        provider="custom",
        model="your-model-name",
        api_key="your-api-key",
        base_url="https://api.your-provider.com"
    )
)

axon = Axon(config)
result = await axon.think("Your question here")
```

## Advanced: Streaming Responses

If your LLM supports streaming, you can implement it:

```python
class CustomLLMClient:
    # ... existing code ...
    
    async def generate_stream(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs
    ):
        """Generate streaming response"""
        async for chunk in self._stream_api(prompt, system_prompt):
            yield chunk
    
    async def _stream_api(self, prompt: str, system_prompt: Optional[str]):
        """Stream API call"""
        # Implement streaming logic
        # async for chunk in self.client.completions.create(..., stream=True):
        #     yield chunk.choices[0].delta.content
        pass
```

## Testing Your Custom Client

Create tests in `tests/unit/test_custom_llm.py`:

```python
import pytest
from axon.core.custom_llm import CustomLLMClient

@pytest.mark.asyncio
async def test_custom_llm_generation():
    """Test custom LLM generation"""
    client = CustomLLMClient(
        api_key="test-key",
        model="test-model"
    )
    
    response = await client.generate("Test prompt")
    assert response is not None
    assert isinstance(response, str)
```

## Best Practices

1. **Error Handling**: Always handle API errors gracefully
2. **Logging**: Use structured logging for debugging
3. **Rate Limiting**: Implement rate limiting if needed
4. **Retry Logic**: Add retry logic for transient failures
5. **Timeouts**: Set appropriate timeouts
6. **Token Counting**: Implement token counting if available
7. **Cost Tracking**: Track API costs if applicable

## Example: Hugging Face Integration

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

class HuggingFaceLLMClient:
    """Hugging Face model client"""
    
    def __init__(self, model: str, **kwargs):
        self.model_name = model
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.model = AutoModelForCausalLM.from_pretrained(model)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
    
    async def generate(
        self,
        prompt: str,
        max_tokens: int = 1000,
        temperature: float = 0.7,
        **kwargs
    ) -> str:
        """Generate response using Hugging Face model"""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature,
            do_sample=True
        )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response
```

## Troubleshooting

### API Key Issues
- Verify your API key is correct
- Check if API key has necessary permissions
- Ensure API key is not expired

### Connection Issues
- Verify base URL is correct
- Check network connectivity
- Verify firewall settings

### Model Issues
- Ensure model name is correct
- Check if model is available for your account
- Verify model supports required features

## Resources

- [OpenAI API Documentation](https://platform.openai.com/docs)
- [Anthropic API Documentation](https://docs.anthropic.com)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers)

## Next Steps

- [Configuration Guide](../getting-started/configuration.md)
- [API Reference](../api/README.md)
- [Examples](../../examples/)

