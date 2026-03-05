from typing import Any, Dict, List, Tuple
import structlog

from .config import ThinkLayerConfig
from .models import Thought, GroundingEvidence
from ..grounding.grounding import GroundingModule

logger = structlog.get_logger(__name__)


class RevisionLoop:
    """
    Iterative think-ground-revise loop.

    After grounding flags contradicted thoughts (via needs_revision in validation
    metadata), this class sends them back through the LLM for revision using the
    contradicting evidence, re-grounds the revised thoughts, and repeats until
    convergence or a hard limit.

    Three safeguards prevent runaway loops:
    - Hard iteration limit (max_revision_iterations)
    - Per-thought revision cap (max_revisions_per_thought)
    - Diminishing returns detection (stop if avg score doesn't improve)
    """

    def __init__(self, config: ThinkLayerConfig, grounding: GroundingModule):
        self.config = config
        self.grounding = grounding
        self.logger = logger.bind(component="revision_loop")

    async def run(
        self,
        thoughts: List[Thought],
        query: str,
        context: Dict[str, Any],
        llm_client: Any,
        citations: List[GroundingEvidence],
    ) -> Tuple[List[Thought], List[GroundingEvidence], int]:
        """
        Run the revision loop on grounded thoughts.

        Args:
            thoughts: Grounded thoughts (with validation metadata)
            query: Original user query
            context: Retrieved context
            llm_client: LLM client with async generate() method
            citations: Existing grounding citations

        Returns:
            Tuple of (revised thoughts, updated citations, number of iterations run)
        """
        max_iters = self.config.max_revision_iterations
        if max_iters <= 0:
            return thoughts, citations, 0

        prev_avg_score = self._avg_validation_score(thoughts)
        iterations = 0

        for iteration in range(1, max_iters + 1):
            needs_revision = self._find_thoughts_needing_revision(thoughts)

            if not needs_revision:
                self.logger.info("Revision loop converged", iteration=iteration)
                break

            self.logger.info(
                "Revision iteration",
                iteration=iteration,
                thoughts_to_revise=len(needs_revision),
            )

            for idx, thought in needs_revision:
                revised = await self._revise_thought(
                    thought, query, llm_client, citations
                )
                thoughts[idx] = revised

            revised_only = [thoughts[idx] for idx, _ in needs_revision]
            revised_only, new_citations = await self.grounding.ground_claims(
                revised_only, query, context
            )

            for (idx, _), revised_thought in zip(needs_revision, revised_only):
                thoughts[idx] = revised_thought

            existing_urls = {c.source_url for c in citations}
            for c in new_citations:
                if c.source_url not in existing_urls:
                    citations.append(c)
                    existing_urls.add(c.source_url)

            iterations = iteration

            current_avg_score = self._avg_validation_score(thoughts)
            if current_avg_score <= prev_avg_score:
                self.logger.info(
                    "Revision loop stopped (no score improvement)",
                    prev_score=prev_avg_score,
                    current_score=current_avg_score,
                    iteration=iteration,
                )
                break
            prev_avg_score = current_avg_score

            if self._all_above_threshold(thoughts):
                self.logger.info(
                    "All thoughts above revision threshold",
                    iteration=iteration,
                    threshold=self.config.revision_score_threshold,
                )
                break

        return thoughts, citations, iterations

    def _find_thoughts_needing_revision(
        self, thoughts: List[Thought]
    ) -> List[Tuple[int, Thought]]:
        """Find thoughts that need revision based on validation metadata."""
        result = []
        for idx, thought in enumerate(thoughts):
            validation = thought.metadata.get("validation", {})
            needs_revision = validation.get("needs_revision", False)

            if not needs_revision:
                continue

            if thought.revision_count >= self.config.max_revisions_per_thought:
                self.logger.debug(
                    "Thought at revision cap, skipping",
                    thought_number=thought.thought_number,
                    revision_count=thought.revision_count,
                )
                continue

            result.append((idx, thought))

        return result

    async def _revise_thought(
        self,
        thought: Thought,
        query: str,
        llm_client: Any,
        citations: List[GroundingEvidence],
    ) -> Thought:
        """Revise a single contradicted thought using the LLM with evidence snippets."""
        validation = thought.metadata.get("validation", {})
        contradicted = validation.get("contradicted_claims", [])
        supported = validation.get("supported_claims", [])

        prompt = self._build_revision_prompt(
            thought.thought, query, contradicted, supported, citations
        )

        try:
            revised_text = await llm_client.generate(prompt)
        except Exception as e:
            self.logger.error(
                "LLM revision failed, keeping original",
                error=str(e),
                thought_number=thought.thought_number,
            )
            return thought

        revised = Thought(
            thought=revised_text,
            thought_number=thought.thought_number,
            total_thoughts=thought.total_thoughts,
            next_thought_needed=thought.next_thought_needed,
            is_revision=True,
            revises_thought=thought.thought_number,
            branch_from_thought=thought.branch_from_thought,
            branch_id=thought.branch_id,
            revision_count=thought.revision_count + 1,
            needs_more_thoughts=thought.needs_more_thoughts,
            is_hypothesis=thought.is_hypothesis,
            is_verification=thought.is_verification,
            stage=thought.stage,
            tags=thought.tags,
            axioms_used=thought.axioms_used,
            assumptions_challenged=thought.assumptions_challenged,
            score=thought.score,
            max_depth=thought.max_depth,
            auto_iterate=thought.auto_iterate,
            metadata={
                "original_thought": thought.thought,
                "revision_from_count": thought.revision_count,
            },
        )
        return revised

    def _build_revision_prompt(
        self,
        original_text: str,
        query: str,
        contradicted_claims: List[str],
        supported_claims: List[str],
        citations: List[GroundingEvidence],
    ) -> str:
        """Build a prompt for the LLM to revise a contradicted thought with evidence."""
        parts = [
            f"Original query: {query}",
            f"\nOriginal thought:\n{original_text}",
        ]

        if contradicted_claims:
            parts.append(
                "\nThe following claims were contradicted by evidence:\n"
                + "\n".join(f"- {c}" for c in contradicted_claims)
            )

        if supported_claims:
            parts.append(
                "\nThe following claims were supported by evidence:\n"
                + "\n".join(f"- {c}" for c in supported_claims)
            )

        if citations:
            evidence_parts = []
            for i, cit in enumerate(citations[:5], 1):
                title = cit.metadata.get("title", "")
                source = f" ({title})" if title else ""
                evidence_parts.append(f"  {i}. {cit.snippet[:300]}{source}")
            parts.append(
                "\nRelevant evidence from sources:\n" + "\n".join(evidence_parts)
            )

        parts.append(
            "\nPlease revise the thought to correct the contradicted claims "
            "while preserving the supported ones. Use the evidence above to "
            "inform your corrections. Output only the revised thought text."
        )

        return "\n".join(parts)

    def _avg_validation_score(self, thoughts: List[Thought]) -> float:
        """Calculate average validation score across thoughts."""
        scores = []
        for thought in thoughts:
            validation = thought.metadata.get("validation", {})
            score = validation.get("validation_score")
            if score is not None:
                scores.append(score)
        return sum(scores) / len(scores) if scores else 0.0

    def _all_above_threshold(self, thoughts: List[Thought]) -> bool:
        """Check if all thoughts are above the revision score threshold."""
        for thought in thoughts:
            validation = thought.metadata.get("validation", {})
            score = validation.get("validation_score", 0.0)
            if score < self.config.revision_score_threshold:
                return False
        return True
