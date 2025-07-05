"""
Ax0n: Model-Agnostic Think & Memory Layer for LLMs

A comprehensive library for structured reasoning, grounding, and memory management.
"""

__version__ = "0.1.0"
__author__ = "Ax0n Team"
__email__ = "install.py@gmail.com"

# Core exports
from .core import (
    Axon,
    ThinkLayer,
    AxonConfig,
    ThinkLayerConfig,
    LLMConfig,
    Thought,
    ThoughtStage,
    ThoughtResult,
    MemoryEntry
)

# Reasoning exports
from .reasoning import (
    ReasoningMethod,
    ChainOfThoughts,
    SelfConsistency,
    AlgorithmOfThoughts,
    ReasoningOrchestrator,
    TreeOfThoughts,
    ThoughtNode
)

# Component exports
from .retrieval import Retriever
from .memory import MemoryManager
from .grounding import GroundingModule
from .rendering import Renderer

__all__ = [
    # Core
    'Axon',
    'ThinkLayer',
    'AxonConfig',
    'ThinkLayerConfig', 
    'LLMConfig',
    'Thought',
    'ThoughtStage',
    'ThoughtResult',
    'MemoryEntry',
    
    # Reasoning
    'ReasoningMethod',
    'ChainOfThoughts',
    'SelfConsistency',
    'AlgorithmOfThoughts',
    'ReasoningOrchestrator',
    'TreeOfThoughts',
    'ThoughtNode',
    
    # Components
    'Retriever',
    'MemoryManager',
    'GroundingModule',
    'Renderer'
] 