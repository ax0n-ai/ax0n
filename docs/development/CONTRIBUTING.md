# Contributing to Ax0n

Thank you for your interest in contributing to Ax0n! This document provides guidelines for contributing to the project.

## Getting Started

### Development Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/YOUR_USERNAME/Ax0n.git
   cd Ax0n
   ```

2. **Install dependencies**
   ```bash
   python -m pip install -e ".[dev]"
   ```

3. **Run setup script**
   ```bash
   python scripts/setup_dev.py
   ```

4. **Run tests**
   ```bash
   pytest tests/ -v
   ```

## Development Workflow

### 1. Create a Branch

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/your-bug-fix
```

### 2. Make Changes

- Write clean, readable code
- Follow Python PEP 8 style guidelines
- Add docstrings to all public functions/classes
- Add type hints where appropriate

### 3. Write Tests

- Add unit tests for new functions
- Add integration tests for new features
- Ensure all tests pass: `pytest tests/ -v`

### 4. Update Documentation

- Update relevant `.md` files in `docs/`
- Update docstrings
- Add examples if applicable

### 5. Commit Changes

```bash
git add .
git commit -m "feat: Add new feature"
```

**Commit Message Format:**
- `feat:` - New feature
- `fix:` - Bug fix
- `docs:` - Documentation changes
- `test:` - Test additions/changes
- `refactor:` - Code refactoring
- `chore:` - Maintenance tasks

### 6. Push and Create PR

```bash
git push origin feature/your-feature-name
```

Then create a Pull Request on GitHub.

## Code Style

### Python Style Guide

- Follow PEP 8
- Use Black for formatting (line length: 100)
- Use isort for import sorting
- Use type hints
- Write descriptive docstrings

**Example:**
```python
from typing import Optional, List

def process_thoughts(
    thoughts: List[Thought],
    config: Optional[AxonConfig] = None
) -> ThoughtResult:
    """
    Process a list of thoughts and return consolidated result.
    
    Args:
        thoughts: List of Thought objects to process
        config: Optional Axon configuration
        
    Returns:
        Consolidated ThoughtResult
        
    Raises:
        ValueError: If thoughts list is empty
    """
    if not thoughts:
        raise ValueError("Thoughts list cannot be empty")
    # ... implementation
```

### Documentation Style

- Use clear, concise language
- Include code examples
- Add links to related docs
- Keep docs up-to-date with code

## Testing Guidelines

### Unit Tests

- Test individual functions/methods
- Mock external dependencies
- Fast execution (<1s per test)
- Located in `tests/unit/`

### Integration Tests

- Test module interactions
- Test end-to-end workflows
- May use external services (mocked)
- Located in `tests/integration/`

### Test Structure

```python
import pytest
from axon import Axon, AxonConfig

class TestFeatureName:
    """Test suite for feature name"""
    
    @pytest.fixture
    def config(self):
        """Test configuration"""
        return AxonConfig(...)
    
    def test_specific_behavior(self, config):
        """Test specific behavior"""
        # Arrange
        axon = Axon(config)
        
        # Act
        result = axon.some_method()
        
        # Assert
        assert result.is_valid
```

## Pull Request Guidelines

### Before Submitting

- [ ] All tests pass
- [ ] Code is formatted (Black, isort)
- [ ] Docstrings are complete
- [ ] Documentation is updated
- [ ] CHANGELOG is updated (for significant changes)
- [ ] No merge conflicts

### PR Description Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
How was this tested?

## Checklist
- [ ] Tests pass
- [ ] Code formatted
- [ ] Docs updated
```

## Project Structure

```
Ax0n/
├── src/axon/           # Main source code
│   ├── core/           # Core functionality
│   ├── memory/         # Memory system
│   ├── grounding/      # Grounding module
│   ├── reasoning/      # Reasoning methods
│   ├── retrieval/      # Retrieval module
│   └── rendering/      # Output rendering
├── tests/              # All tests
│   ├── unit/           # Unit tests
│   └── integration/    # Integration tests
├── docs/               # Documentation
└── examples/           # Example scripts
```

## Adding New Features

### 1. New Reasoning Method

1. Add implementation to `src/axon/reasoning/`
2. Register in `ReasoningMethod` enum
3. Update `ThinkLayer` to handle new method
4. Add tests in `tests/unit/test_reasoning_methods.py`
5. Add example in `examples/`
6. Document in `docs/api/reasoning-methods.md`

### 2. New LLM Provider

1. Create client in `src/axon/core/core.py`
2. Add to `LLMConfig` provider options
3. Update `_get_llm_client()` factory
4. Add tests
5. Document configuration

### 3. New Storage Backend

1. Implement `MemoryStorage` interface
2. Add to `MemoryManager` initialization
3. Add configuration options
4. Add tests
5. Document setup

## Code Review Process

1. **Automated Checks**: CI runs tests and linters
2. **Peer Review**: Maintainer reviews code
3. **Feedback**: Address review comments
4. **Approval**: At least one approval required
5. **Merge**: Squash and merge to main

## Community

### Communication

- **Issues**: Bug reports and feature requests
- **Discussions**: Questions and ideas
- **Discord**: Real-time chat (if available)

### Getting Help

- Check existing documentation
- Search existing issues
- Ask in discussions
- Contact maintainers

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

## Recognition

Contributors are recognized in:
- CONTRIBUTORS.md file
- Release notes
- Project README

Thank you for contributing to Ax0n! 

