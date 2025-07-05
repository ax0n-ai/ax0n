"""
Renderer module for Ax0n - formats final outputs with reasoning trace and citations
"""

from typing import Any, Dict, List, Optional
import structlog
from ..core.config import RendererConfig
from ..core.models import Thought, ThoughtResult

logger = structlog.get_logger(__name__)


class ResponseFormatter:
    """Format final answer assembly"""
    
    def __init__(self, config: RendererConfig):
        self.config = config
        self.logger = logger.bind(component="response_formatter")
    
    def format_answer(self, answer: str, thoughts: List[Thought]) -> str:
        """Format the final answer"""
        if not self.config.include_trace:
            return answer
        
        # Add thought summary
        formatted = f"{answer}\n\nReasoning Steps: {len(thoughts)}"
        return formatted


class TraceRenderer:
    """Render thought trace with toggles"""
    
    def __init__(self, config: RendererConfig):
        self.config = config
        self.logger = logger.bind(component="trace_renderer")
    
    def render_trace(self, thoughts: List[Thought]) -> List[Dict[str, Any]]:
        """Render the thought trace"""
        trace = []
        
        for thought in thoughts:
            trace_entry = {
                'thought_number': thought.thought_number,
                'thought': thought.thought,
                'stage': thought.stage.value,
                'tags': thought.tags,
                'score': thought.score
            }
            
            if self.config.trace_verbosity == "detailed":
                trace_entry.update({
                    'timestamp': thought.timestamp.isoformat(),
                    'axioms_used': thought.axioms_used,
                    'assumptions_challenged': thought.assumptions_challenged
                })
            
            trace.append(trace_entry)
        
        return trace


class CitationRenderer:
    """Embed citations in readable form"""
    
    def __init__(self, config: RendererConfig):
        self.config = config
        self.logger = logger.bind(component="citation_renderer")
    
    def render_citations(self, citations: List[Any]) -> List[Dict[str, Any]]:
        """Render citations in the specified format"""
        if not self.config.include_citations:
            return []
        
        rendered = []
        for citation in citations:
            rendered_citation = {
                'source': citation.get('source_url', 'Unknown'),
                'snippet': citation.get('snippet', ''),
                'confidence': citation.get('confidence', 0.0)
            }
            
            if self.config.citation_format == "markdown":
                rendered_citation['formatted'] = f"[{rendered_citation['source']}]({rendered_citation['source']})"
            elif self.config.citation_format == "html":
                rendered_citation['formatted'] = f'<a href="{rendered_citation["source"]}">{rendered_citation["source"]}</a>'
            else:
                rendered_citation['formatted'] = rendered_citation['source']
            
            rendered.append(rendered_citation)
        
        return rendered


class Renderer:
    """Main renderer that coordinates output formatting"""
    
    def __init__(self, config: RendererConfig):
        self.config = config
        self.logger = logger.bind(component="renderer")
        
        # Initialize components
        self.response_formatter = ResponseFormatter(config)
        self.trace_renderer = TraceRenderer(config)
        self.citation_renderer = CitationRenderer(config)
        
        self.logger.info("Renderer initialized")
    
    async def render_result(
        self,
        thoughts: List[Thought],
        answer: str,
        citations: List[Any],
        memory_updates: List[Dict[str, Any]],
        execution_time: float
    ) -> ThoughtResult:
        """Render the final result with all components"""
        try:
            # Format the answer
            formatted_answer = self.response_formatter.format_answer(answer, thoughts)
            
            # Render the trace
            trace = self.trace_renderer.render_trace(thoughts)
            
            # Render citations
            rendered_citations = self.citation_renderer.render_citations(citations)
            
            # Create the result
            result = ThoughtResult(
                thoughts=thoughts,
                answer=formatted_answer,
                trace=trace,
                citations=rendered_citations,
                memory_updates=memory_updates,
                execution_time=execution_time
            )
            
            return result
            
        except Exception as e:
            self.logger.error("Result rendering failed", error=str(e))
            # Return a basic result
            return ThoughtResult(
                thoughts=thoughts,
                answer=answer,
                trace=[],
                citations=[],
                memory_updates=[],
                execution_time=execution_time
            ) 