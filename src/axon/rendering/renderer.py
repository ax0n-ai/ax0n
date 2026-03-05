from typing import Any, Dict, List, Optional
import json
import structlog

from ..core.config import RendererConfig
from ..core.models import Thought, ThoughtResult, GroundingEvidence

logger = structlog.get_logger(__name__)


class ResponseFormatter:
    """Format final answer assembly with multiple verbosity levels"""

    def __init__(self, config: RendererConfig):
        self.config = config
        self.logger = logger.bind(component="response_formatter")

    def format_answer(
        self,
        answer: str,
        thoughts: List[Thought],
        citations: List[GroundingEvidence],
        execution_time: float
    ) -> str:
        """
        Format the final answer with optional enhancements

        Args:
            answer: The base answer
            thoughts: List of thoughts
            citations: List of citations
            execution_time: Time taken to generate

        Returns:
            Formatted answer string
        """
        if not self.config.include_trace and not self.config.include_citations:
            return answer

        formatted = answer

        if self.config.include_trace:
            formatted += f"\n\n---\n**Reasoning Summary**\n"
            formatted += f"- Steps: {len(thoughts)}\n"
            formatted += f"- Execution Time: {execution_time:.2f}s\n"

            avg_confidence = self._calculate_avg_confidence(thoughts)
            if avg_confidence is not None:
                formatted += f"- Average Confidence: {avg_confidence:.2f}\n"

        if self.config.include_citations and citations:
            formatted += f"\n**Sources** ({len(citations)} citations)\n"
            for i, citation in enumerate(citations[:5], 1):
                formatted += f"{i}. {self._format_citation_inline(citation)}\n"

        return formatted

    def _calculate_avg_confidence(self, thoughts: List[Thought]) -> Optional[float]:
        """Calculate average confidence from thoughts"""
        scores = [t.score for t in thoughts if t.score is not None]
        if not scores:
            return None
        return sum(scores) / len(scores)

    def _format_citation_inline(self, citation: GroundingEvidence) -> str:
        """Format a citation for inline display"""
        if self.config.citation_format == "markdown":
            return f"[{citation.metadata.get('title', 'Source')}]({citation.source_url})"
        elif self.config.citation_format == "html":
            return f'<a href="{citation.source_url}">{citation.metadata.get("title", "Source")}</a>'
        else:
            return citation.source_url


class TraceRenderer:
    """Render thought trace with multiple verbosity levels"""

    def __init__(self, config: RendererConfig):
        self.config = config
        self.logger = logger.bind(component="trace_renderer")

    def render_trace(self, thoughts: List[Thought]) -> List[Dict[str, Any]]:
        """
        Render the thought trace based on verbosity setting

        Args:
            thoughts: List of thoughts to render

        Returns:
            Formatted trace entries
        """
        if not self.config.include_trace:
            return []

        verbosity = self.config.trace_verbosity

        if verbosity == "minimal":
            return self._render_minimal_trace(thoughts)
        elif verbosity == "medium":
            return self._render_medium_trace(thoughts)
        elif verbosity == "detailed":
            return self._render_detailed_trace(thoughts)
        else:
            return self._render_medium_trace(thoughts)

    def _render_minimal_trace(self, thoughts: List[Thought]) -> List[Dict[str, Any]]:
        """Minimal trace - just thought numbers and content"""
        return [
            {
                'thought_number': thought.thought_number,
                'thought': thought.thought,
                'score': thought.score
            }
            for thought in thoughts
        ]

    def _render_medium_trace(self, thoughts: List[Thought]) -> List[Dict[str, Any]]:
        """Medium trace - includes stage, tags, and scores"""
        return [
            {
                'thought_number': thought.thought_number,
                'thought': thought.thought,
                'stage': thought.stage.value,
                'tags': thought.tags,
                'score': thought.score,
                'branch_id': thought.branch_id
            }
            for thought in thoughts
        ]

    def _render_detailed_trace(self, thoughts: List[Thought]) -> List[Dict[str, Any]]:
        """Detailed trace - includes all metadata"""
        trace = []

        for thought in thoughts:
            entry = {
                'thought_number': thought.thought_number,
                'thought': thought.thought,
                'stage': thought.stage.value,
                'tags': thought.tags,
                'score': thought.score,
                'branch_id': thought.branch_id,
                'timestamp': thought.timestamp.isoformat(),
                'axioms_used': thought.axioms_used,
                'assumptions_challenged': thought.assumptions_challenged,
                'is_hypothesis': thought.is_hypothesis,
                'is_verification': thought.is_verification,
                'is_revision': thought.is_revision
            }

            if thought.metadata:
                entry['metadata'] = thought.metadata

            if thought.revises_thought:
                entry['revises_thought'] = thought.revises_thought
            if thought.branch_from_thought:
                entry['branch_from_thought'] = thought.branch_from_thought

            trace.append(entry)

        return trace

    def render_trace_as_text(self, thoughts: List[Thought]) -> str:
        """Render trace as human-readable text"""
        if not thoughts:
            return "No reasoning trace available."

        lines = ["## Reasoning Trace\n"]

        for thought in thoughts:
            lines.append(f"### Step {thought.thought_number}: {thought.stage.value.title()}")
            lines.append(f"{thought.thought}")

            if thought.score:
                lines.append(f"*Confidence: {thought.score:.2f}*")

            if thought.tags:
                lines.append(f"*Tags: {', '.join(thought.tags)}*")

            lines.append("")  # Empty line between thoughts

        return "\n".join(lines)

    def render_trace_as_json(self, thoughts: List[Thought]) -> str:
        """Render trace as JSON"""
        trace = self.render_trace(thoughts)
        return json.dumps(trace, indent=2, default=str)


class CitationRenderer:
    """Embed citations in multiple readable formats"""

    def __init__(self, config: RendererConfig):
        self.config = config
        self.logger = logger.bind(component="citation_renderer")

    def render_citations(self, citations: List[GroundingEvidence]) -> List[Dict[str, Any]]:
        """
        Render citations in the specified format

        Args:
            citations: List of grounding evidence

        Returns:
            List of rendered citations
        """
        if not self.config.include_citations:
            return []

        rendered = []
        for i, citation in enumerate(citations, 1):
            rendered_citation = self._render_single_citation(citation, i)
            rendered.append(rendered_citation)

        self.logger.debug("Rendered citations", count=len(rendered))
        return rendered

    def _render_single_citation(self, citation: GroundingEvidence, index: int) -> Dict[str, Any]:
        """Render a single citation"""
        base = {
            'index': index,
            'source_url': citation.source_url,
            'snippet': citation.snippet,
            'confidence': citation.confidence,
            'timestamp': citation.timestamp.isoformat()
        }

        if citation.metadata:
            base['title'] = citation.metadata.get('title', 'Untitled')
            base['source'] = citation.metadata.get('source', 'unknown')
            base['is_trusted'] = citation.metadata.get('is_trusted', False)

        if self.config.citation_format == "markdown":
            base['formatted'] = self._format_as_markdown(citation, index)
        elif self.config.citation_format == "html":
            base['formatted'] = self._format_as_html(citation, index)
        else:  # plain
            base['formatted'] = self._format_as_plain(citation, index)

        return base

    def _format_as_markdown(self, citation: GroundingEvidence, index: int) -> str:
        """Format citation as Markdown"""
        title = citation.metadata.get('title', 'Source') if citation.metadata else 'Source'
        snippet = citation.snippet[:100] + '...' if len(citation.snippet) > 100 else citation.snippet

        formatted = f"**[{index}]** [{title}]({citation.source_url})\n"
        formatted += f"> {snippet}\n"
        formatted += f"*Confidence: {citation.confidence:.2f}*"

        return formatted

    def _format_as_html(self, citation: GroundingEvidence, index: int) -> str:
        """Format citation as HTML"""
        title = citation.metadata.get('title', 'Source') if citation.metadata else 'Source'
        snippet = citation.snippet[:100] + '...' if len(citation.snippet) > 100 else citation.snippet

        html = f'<div class="citation" id="citation-{index}">\n'
        html += f'  <strong>[{index}]</strong> '
        html += f'<a href="{citation.source_url}" target="_blank">{title}</a>\n'
        html += f'  <blockquote>{snippet}</blockquote>\n'
        html += f'  <em>Confidence: {citation.confidence:.2f}</em>\n'
        html += '</div>'

        return html

    def _format_as_plain(self, citation: GroundingEvidence, index: int) -> str:
        """Format citation as plain text"""
        title = citation.metadata.get('title', 'Source') if citation.metadata else 'Source'
        snippet = citation.snippet[:100] + '...' if len(citation.snippet) > 100 else citation.snippet

        formatted = f"[{index}] {title}\n"
        formatted += f"URL: {citation.source_url}\n"
        formatted += f"Snippet: {snippet}\n"
        formatted += f"Confidence: {citation.confidence:.2f}"

        return formatted

    def render_citations_as_text(self, citations: List[GroundingEvidence]) -> str:
        """Render all citations as readable text"""
        if not citations:
            return "No citations available."

        lines = ["## Citations\n"]

        for i, citation in enumerate(citations, 1):
            title = citation.metadata.get('title', 'Source') if citation.metadata else 'Source'
            lines.append(f"{i}. **{title}**")
            lines.append(f"   {citation.source_url}")
            lines.append(f"   > {citation.snippet[:150]}...")
            lines.append(f"   *Confidence: {citation.confidence:.2f}*\n")

        return "\n".join(lines)


class Renderer:
    """
    Main renderer that coordinates output formatting

    Features:
    - Multiple verbosity levels (minimal, medium, detailed)
    - Multiple output formats (markdown, HTML, plain)
    - Citation formatting with confidence scores
    - Metadata inclusion with configurable detail
    """

    def __init__(self, config: RendererConfig):
        self.config = config
        self.logger = logger.bind(component="renderer")

        self.response_formatter = ResponseFormatter(config)
        self.trace_renderer = TraceRenderer(config)
        self.citation_renderer = CitationRenderer(config)

        self.logger.info(
            "Renderer initialized",
            trace_verbosity=config.trace_verbosity,
            citation_format=config.citation_format
        )

    async def render(
        self,
        query: str,
        thoughts: List[Thought],
        answer: str,
        citations: List[GroundingEvidence],
        memory_updates: List[Dict[str, Any]],
        execution_time: Optional[float] = None
    ) -> str:
        """
        Render the complete output

        Args:
            query: Original query
            thoughts: List of thoughts
            answer: Final answer
            citations: List of citations
            memory_updates: Memory update actions
            execution_time: Optional execution time

        Returns:
            Formatted output string
        """
        exec_time = execution_time or 0.0

        formatted_answer = self.response_formatter.format_answer(
            answer,
            thoughts,
            citations,
            exec_time
        )

        return formatted_answer

    async def render_result(
        self,
        thoughts: List[Thought],
        answer: str,
        citations: List[GroundingEvidence],
        memory_updates: List[Dict[str, Any]],
        execution_time: float
    ) -> ThoughtResult:
        """
        Render the final result with all components

        Args:
            thoughts: List of thoughts
            answer: Final answer
            citations: List of citations
            memory_updates: Memory update actions
            execution_time: Execution time

        Returns:
            Complete ThoughtResult object
        """
        try:
            formatted_answer = self.response_formatter.format_answer(
                answer,
                thoughts,
                citations,
                execution_time
            )

            trace = self.trace_renderer.render_trace(thoughts)

            rendered_citations = self.citation_renderer.render_citations(citations)

            metadata = {}
            if self.config.enable_metadata:
                metadata = self._build_metadata(thoughts, citations, memory_updates, execution_time)

            metadata['rendered_citations'] = rendered_citations

            result = ThoughtResult(
                thoughts=thoughts,
                answer=formatted_answer,
                trace=trace,
                citations=citations,
                memory_updates=memory_updates,
                execution_time=execution_time,
                metadata=metadata
            )

            self.logger.info(
                "Result rendered successfully",
                thoughts=len(thoughts),
                citations=len(rendered_citations),
                memory_updates=len(memory_updates)
            )

            return result

        except Exception as e:
            self.logger.error("Result rendering failed", error=str(e))
            return ThoughtResult(
                thoughts=thoughts,
                answer=answer,
                trace=[],
                citations=[],
                memory_updates=[],
                execution_time=execution_time
            )

    def _build_metadata(
        self,
        thoughts: List[Thought],
        citations: List[GroundingEvidence],
        memory_updates: List[Dict[str, Any]],
        execution_time: float
    ) -> Dict[str, Any]:
        """Build metadata dictionary"""
        avg_confidence = None
        if thoughts:
            scores = [t.score for t in thoughts if t.score is not None]
            if scores:
                avg_confidence = sum(scores) / len(scores)

        stage_counts = {}
        for thought in thoughts:
            stage = thought.stage.value
            stage_counts[stage] = stage_counts.get(stage, 0) + 1

        branches = set(t.branch_id for t in thoughts if t.branch_id)

        metadata = {
            'execution_time': execution_time,
            'thought_count': len(thoughts),
            'citation_count': len(citations),
            'memory_update_count': len(memory_updates),
            'average_confidence': avg_confidence,
            'stage_distribution': stage_counts,
            'branch_count': len(branches),
            'renderer_config': {
                'trace_verbosity': self.config.trace_verbosity,
                'citation_format': self.config.citation_format,
                'include_metadata': self.config.enable_metadata
            }
        }

        return metadata

    def export_as_json(self, result: ThoughtResult) -> str:
        """Export result as JSON"""
        data = {
            'answer': result.answer,
            'thoughts': [t.model_dump() for t in result.thoughts],
            'trace': result.trace,
            'citations': result.citations,
            'memory_updates': result.memory_updates,
            'execution_time': result.execution_time,
            'metadata': result.metadata
        }
        return json.dumps(data, indent=2, default=str)

    def export_as_markdown(self, result: ThoughtResult) -> str:
        """Export result as Markdown"""
        lines = ["# Ax0n Response\n"]

        lines.append("## Answer\n")
        lines.append(result.answer)
        lines.append("")

        if result.trace and self.config.include_trace:
            lines.append(self.trace_renderer.render_trace_as_text(result.thoughts))
            lines.append("")

        if result.citations and self.config.include_citations:
            lines.append(self.citation_renderer.render_citations_as_text(
                [c for c in result.citations if isinstance(c, (GroundingEvidence, dict))]
            ))
            lines.append("")

        if result.metadata and self.config.enable_metadata:
            lines.append("## Metadata\n")
            lines.append(f"- Execution Time: {result.execution_time:.2f}s")
            lines.append(f"- Thoughts: {result.metadata.get('thought_count', 0)}")
            lines.append(f"- Citations: {result.metadata.get('citation_count', 0)}")
            if result.metadata.get('average_confidence'):
                lines.append(f"- Avg Confidence: {result.metadata['average_confidence']:.2f}")

        return "\n".join(lines)
