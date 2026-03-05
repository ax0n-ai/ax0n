from typing import Any, Dict, List, Optional
from enum import Enum
from pydantic import BaseModel, Field
from datetime import datetime, timezone


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
    revision_count: int = Field(0, ge=0, description="Number of times this thought has been revised")
    score: Optional[float] = Field(None, ge=0.0, le=1.0, description="Confidence score (0-1)")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="When this thought was generated")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class GroundingEvidence(BaseModel):
    """Evidence for grounding a thought in real-world facts"""

    source_url: str = Field(..., description="URL of the source")
    snippet: str = Field(..., description="Relevant text snippet")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence in this evidence")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="When evidence was found")
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
    """A memory entry for persistent storage (Mem0-inspired)"""

    id: str = Field(..., description="Unique identifier")
    content: str = Field(..., description="The memory content")
    embedding: Optional[List[float]] = Field(None, description="Vector embedding")
    source_thoughts: List[int] = Field(default_factory=list, description="Thought numbers that generated this")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence in this memory")
    salience_score: float = Field(0.5, ge=0.0, le=1.0, description="Importance/salience score")
    tags: List[str] = Field(default_factory=list, description="Categorization tags")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="Creation timestamp")
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="Last update timestamp")
    access_count: int = Field(0, description="Number of times accessed")
    agent_id: Optional[str] = Field(None, description="Agent this memory belongs to (agent-scoped)")
    is_shared: bool = Field(False, description="Whether this is shared across agents")
    relationships: List[str] = Field(default_factory=list, description="Related memory IDs")
    memory_type: str = Field("semantic", description="Memory type (semantic, episodic, procedural)")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class LLMConfig(BaseModel):
    """Configuration for LLM clients (model-agnostic)"""

    provider: str = Field(..., description="LLM provider (openai, anthropic, local, etc.)")
    model: str = Field(..., description="Model name")
    api_key: Optional[str] = Field(None, description="API key")
    base_url: Optional[str] = Field(None, description="Base URL for API calls")
    temperature: float = Field(0.7, ge=0.0, le=2.0, description="Sampling temperature")
    max_tokens: int = Field(1000, gt=0, description="Maximum tokens to generate")
    timeout: int = Field(30, gt=0, description="Request timeout in seconds")
    supports_streaming: bool = Field(True, description="Whether the model supports streaming")
    supports_function_calling: bool = Field(False, description="Whether the model supports function calling")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional provider-specific config")


class AgentConfig(BaseModel):
    """Configuration for an agent in the hierarchical system"""

    agent_id: str = Field(..., description="Unique agent identifier")
    agent_type: str = Field("operational", description="Agent type (strategic, coordination, operational)")
    agent_role: str = Field("general", description="Specific role description")
    max_depth: int = Field(3, gt=0, description="Maximum reasoning depth for this agent")
    enable_memory: bool = Field(True, description="Whether this agent has memory")
    parent_agent_id: Optional[str] = Field(None, description="Parent agent in hierarchy")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional agent metadata")
