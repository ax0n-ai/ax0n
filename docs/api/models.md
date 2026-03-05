# Data Models

## `Thought`
Represents a single structured thought in the reasoning process.
- `thought: str` — The content of the thought
- `thought_number: int` — Step number
- `total_thoughts: int` — Total steps in the chain/branch
- `next_thought_needed: bool` — Whether more thoughts are needed
- `needs_more_thoughts: bool` — Whether to continue
- `stage: ThoughtStage` — Reasoning stage
- `tags: List[str]` — Optional tags
- `score: float` — Confidence or evaluation score
- `metadata: dict` — Additional metadata

## `ThoughtStage`
Enum of reasoning stages:
- `PROBLEM_DEFINITION`
- `RESEARCH`
- `ANALYSIS`
- `SYNTHESIS`
- `CONCLUSION`
- `VERIFICATION`

## `ThoughtResult`
Result object containing the full output of a reasoning run.
- `thoughts: List[Thought]`
- `answer: str`
- `trace: List[dict]`
- `citations: List[dict]`
- `memory_updates: List[dict]`
- `execution_time: float`

## `MemoryEntry`
Represents a memory fact or knowledge entry.
- `content: str`
- `metadata: dict`
- `embedding: List[float]`

---

## Example Usage

```python
from axon import Thought, ThoughtStage

thought = Thought(
    thought="Kyoto is best in spring.",
    thought_number=1,
    total_thoughts=3,
    next_thought_needed=True,
    needs_more_thoughts=True,
    stage=ThoughtStage.ANALYSIS,
    tags=["travel"],
    score=0.9
)
``` 