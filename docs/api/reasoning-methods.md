# Reasoning Methods

## Enum: `ReasoningMethod`

Enumeration of available reasoning methods for Ax0n.

- `COT` - Chain of Thoughts (linear reasoning)
- `SELF_CONSISTENCY` - Self-Consistency (parallel paths with voting)
- `AOT` - Algorithm of Thoughts (algorithmic reasoning)
- `TOT` - Tree of Thoughts (tree-based reasoning)
- `GOT` - Graph of Thoughts (graph-based, coming soon)

---

## Classes

### `ChainOfThoughts`
Linear, step-by-step reasoning. See `src/axon/reasoning/reasoning_methods.py`.

### `SelfConsistency`
Parallel Chain of Thoughts with voting. See `src/axon/reasoning/reasoning_methods.py`.

### `AlgorithmOfThoughts`
Algorithmic problem decomposition. See `src/axon/reasoning/reasoning_methods.py`.

### `TreeOfThoughts`
Tree-based exploration and evaluation. See `src/axon/reasoning/tree_of_thoughts.py`.

### `ReasoningOrchestrator`
Orchestrates all reasoning methods and provides a unified interface.

---

## Example Usage

```python
from axon import ReasoningMethod

method = ReasoningMethod.TOT
print(method.value)  # 'tree_of_thoughts'
``` 