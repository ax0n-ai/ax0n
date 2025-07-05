"""
Core data models for Ax0n Think Layer
"""

from typing import Any, Dict, List, Optional, Union
from enum import Enum
from pydantic import BaseModel, Field
from datetime import datetime


class ThoughtStage(str, Enum):
    """Stages of thought development"""
    PROBLEM_DEFINITION = "problem_definition"
    RESEARCH = "research"
    ANALYSIS = "analysis"
    SYNTHESIS = "synthesis"
    CONCLUSION = "conclusion"
    VERIFICATION = "verification"


class Thought(BaseModel):
    """Represents a single thought in the reasoning chain"""
    
    thought: str = Field(..., description="The actual thought content")
    thought_number: int = Field(..., description="Sequential number of this thought")
    total_thoughts: int = Field(..., description="Total number of thoughts in this chain")
    next_thought_needed: bool = Field(True, description="Whether another thought is needed")
    is_revision: bool = Field(False, description="Whether this revises a previous thought")
    revises_thought: Optional[int] = Field(None, description="Thought number being revised")
    branch_from_thought: Optional[int] = Field(None, description="Thought number this branches from")
    branch_id: Optional[str] = Field(None, description="Unique identifier for this branch")
    needs_more_thoughts: bool = Field(True, description="Whether more thoughts are needed")
    is_hypothesis: bool = Field(False, description="Whether this is a hypothesis")
    is_verification: bool = Field(False, description="Whether this is a verification step")
    return_full_history: bool = Field(False, description="Whether to return full thought history")
    auto_iterate: bool = Field(True, description="Whether to automatically continue thinking")
    max_depth: int = Field(5, description="Maximum depth of thought chain")
    stage: ThoughtStage = Field(ThoughtStage.ANALYSIS, description="Current stage of reasoning")
    tags: List[str] = Field(default_factory=list, description="Tags for categorization")
    axioms_used: List[str] = Field(default_factory=list, description="Axioms or principles used")
    assumptions_challenged: List[str] = Field(default_factory=list, description="Assumptions being challenged")
    score: Optional[float] = Field(None, ge=0.0, le=1.0, description="Confidence score (0-1)")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="When this thought was generated")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class GroundingEvidence(BaseModel):
    """Evidence for grounding a thought in real-world facts"""
    
    source_url: str = Field(..., description="URL of the source")
    snippet: str = Field(..., description="Relevant text snippet")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence in this evidence")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="When evidence was found")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class ThoughtResult(BaseModel):
    """Result of a thought generation process"""
    
    thoughts: List[Thought] = Field(..., description="List of generated thoughts")
    answer: str = Field(..., description="Final synthesized answer")
    trace: List[Dict[str, Any]] = Field(default_factory=list, description="Full reasoning trace")
    citations: List[GroundingEvidence] = Field(default_factory=list, description="Evidence and citations")
    memory_updates: List[Dict[str, Any]] = Field(default_factory=list, description="Memory entries to update")
    execution_time: float = Field(..., description="Total execution time in seconds")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional result metadata")


class MemoryEntry(BaseModel):
    """A memory entry for persistent storage"""
    
    id: str = Field(..., description="Unique identifier")
    content: str = Field(..., description="The memory content")
    embedding: Optional[List[float]] = Field(None, description="Vector embedding")
    source_thoughts: List[int] = Field(default_factory=list, description="Thought numbers that generated this")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence in this memory")
    tags: List[str] = Field(default_factory=list, description="Categorization tags")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp")
    updated_at: datetime = Field(default_factory=datetime.utcnow, description="Last update timestamp")
    access_count: int = Field(0, description="Number of times accessed")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class LLMConfig(BaseModel):
    """Configuration for LLM clients"""
    
    provider: str = Field(..., description="LLM provider (openai, anthropic, etc.)")
    model: str = Field(..., description="Model name")
    api_key: Optional[str] = Field(None, description="API key")
    base_url: Optional[str] = Field(None, description="Base URL for API calls")
    temperature: float = Field(0.7, ge=0.0, le=2.0, description="Sampling temperature")
    max_tokens: int = Field(1000, gt=0, description="Maximum tokens to generate")
    timeout: int = Field(30, gt=0, description="Request timeout in seconds")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional provider-specific config") 