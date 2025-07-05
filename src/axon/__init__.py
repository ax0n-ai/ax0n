"""
Ax0n: Model-Agnostic Think & Memory Layer for LLMs

A structured reasoning and memory system that wraps any LLM with:
- Multi-step thought processes
- Real-world grounding and fact verification
- Persistent memory extraction and storage
- Parallel execution capabilities
"""

from .core import Axon
from .models import Thought, ThoughtResult, MemoryEntry
from .config import AxonConfig

__version__ = "0.1.0"
__all__ = ["Axon", "Thought", "ThoughtResult", "MemoryEntry", "AxonConfig"] 