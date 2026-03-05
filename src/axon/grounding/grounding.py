from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime, timezone
from urllib.parse import urlparse
import structlog

from ..core.config import GroundingConfig
from ..core.models import Thought, GroundingEvidence
from ..utils import extract_claims
from ..utils.embeddings import EmbeddingProvider

logger = structlog.get_logger(__name__)


class SearchClient:
    """
    Web/API search client for fact verification
    Supports multiple search providers (Google, Bing, DuckDuckGo)
    """

    def __init__(self, config: GroundingConfig):
        self.config = config
        self.logger = logger.bind(component="search_client")
        self._session = None

    async def _get_session(self):
        """Get or create aiohttp session with connection limits"""
        if self._session is None or self._session.closed:
            try:
                import aiohttp
                connector = aiohttp.TCPConnector(
                    limit=20,          # max total connections
                    limit_per_host=5,  # max per host
                    ttl_dns_cache=300, # 5 min DNS cache
                )
                timeout = aiohttp.ClientTimeout(
                    total=self.config.search_timeout,
                    connect=5,
                )
                self._session = aiohttp.ClientSession(
                    connector=connector,
                    timeout=timeout,
                )
            except ImportError:
                self.logger.warning("aiohttp not available, using mock search")
                return None
        return self._session

    async def search(self, query: str, provider: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Search for information to ground claims

        Args:
            query: Search query
            provider: Optional specific provider to use

        Returns:
            List of search results with URLs and snippets
        """
        provider = provider or self.config.search_provider

        self.logger.info("Performing search", query=query[:100], provider=provider)

        try:
            if provider == "google":
                return await self._search_google(query)
            elif provider == "bing":
                return await self._search_bing(query)
            elif provider == "duckduckgo":
                return await self._search_duckduckgo(query)
            else:
                self.logger.warning(f"Unknown provider: {provider}, using mock")
                return self._mock_search_results(query)

        except Exception as e:
            self.logger.error("Search failed", error=str(e), provider=provider, exc_info=True)
            return self._mock_search_results(query)

    async def _search_google(self, query: str) -> List[Dict[str, Any]]:
        """
        Search using Google Custom Search API
        Requires GOOGLE_API_KEY and GOOGLE_CSE_ID environment variables
        """
        import os

        api_key = os.getenv("GOOGLE_API_KEY")
        cse_id = os.getenv("GOOGLE_CSE_ID")

        if not api_key or not cse_id:
            self.logger.warning("Google API credentials not found, using mock")
            return self._mock_search_results(query)

        session = await self._get_session()
        if not session:
            return self._mock_search_results(query)

        try:
            url = "https://www.googleapis.com/customsearch/v1"
            params = {
                'key': api_key,
                'cx': cse_id,
                'q': query,
                'num': self.config.max_search_results
            }

            async with session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    results = []

                    for item in data.get('items', []):
                        results.append({
                            'url': item.get('link', ''),
                            'title': item.get('title', ''),
                            'snippet': item.get('snippet', ''),
                            'source': 'google',
                            'timestamp': datetime.now(timezone.utc)
                        })

                    return results[:self.config.max_search_results]
                elif 400 <= response.status < 500:
                    self.logger.error(
                        "Google search auth/client error",
                        status=response.status,
                        body=await response.text(),
                    )
                    return self._mock_search_results(query)
                else:
                    self.logger.warning(
                        "Google search server error",
                        status=response.status,
                    )
                    return self._mock_search_results(query)

        except Exception as e:
            self.logger.error("Google search error", error=str(e))
            return self._mock_search_results(query)

    async def _search_bing(self, query: str) -> List[Dict[str, Any]]:
        """
        Search using Bing Search API
        Requires BING_API_KEY environment variable
        """
        import os

        api_key = os.getenv("BING_API_KEY")

        if not api_key:
            self.logger.warning("Bing API key not found, using mock")
            return self._mock_search_results(query)

        session = await self._get_session()
        if not session:
            return self._mock_search_results(query)

        try:
            url = "https://api.bing.microsoft.com/v7.0/search"
            headers = {'Ocp-Apim-Subscription-Key': api_key}
            params = {
                'q': query,
                'count': self.config.max_search_results
            }

            async with session.get(
                url,
                headers=headers,
                params=params,
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    results = []

                    for item in data.get('webPages', {}).get('value', []):
                        results.append({
                            'url': item.get('url', ''),
                            'title': item.get('name', ''),
                            'snippet': item.get('snippet', ''),
                            'source': 'bing',
                            'timestamp': datetime.now(timezone.utc)
                        })

                    return results[:self.config.max_search_results]
                elif 400 <= response.status < 500:
                    self.logger.error(
                        "Bing search auth/client error",
                        status=response.status,
                        body=await response.text(),
                    )
                    return self._mock_search_results(query)
                else:
                    self.logger.warning(
                        "Bing search server error",
                        status=response.status,
                    )
                    return self._mock_search_results(query)

        except Exception as e:
            self.logger.error("Bing search error", error=str(e))
            return self._mock_search_results(query)

    async def _search_duckduckgo(self, query: str) -> List[Dict[str, Any]]:
        """
        Search using DuckDuckGo Instant Answer API
        No API key required
        """
        session = await self._get_session()
        if not session:
            return self._mock_search_results(query)

        try:
            url = "https://api.duckduckgo.com/"
            params = {
                'q': query,
                'format': 'json',
                'no_html': 1,
                'skip_disambig': 1
            }

            async with session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    results = []

                    if data.get('Abstract'):
                        results.append({
                            'url': data.get('AbstractURL', ''),
                            'title': data.get('Heading', ''),
                            'snippet': data.get('Abstract', ''),
                            'source': 'duckduckgo',
                            'timestamp': datetime.now(timezone.utc)
                        })

                    for topic in data.get('RelatedTopics', [])[:self.config.max_search_results - 1]:
                        if isinstance(topic, dict) and 'Text' in topic:
                            results.append({
                                'url': topic.get('FirstURL', ''),
                                'title': topic.get('Text', '')[:100],
                                'snippet': topic.get('Text', ''),
                                'source': 'duckduckgo',
                                'timestamp': datetime.now(timezone.utc)
                            })

                    return results[:self.config.max_search_results]
                else:
                    self.logger.warning(
                        "DuckDuckGo search failed",
                        status=response.status,
                    )
                    return self._mock_search_results(query)

        except Exception as e:
            self.logger.error("DuckDuckGo search error", error=str(e))
            return self._mock_search_results(query)

    def _mock_search_results(self, query: str) -> List[Dict[str, Any]]:
        """Generate mock search results for development/testing"""
        return [
            {
                'url': f'https://example.com/article-{i}',
                'title': f'Mock result {i} for: {query[:50]}',
                'snippet': f'This is a mock search result containing information about {query}. '
                          f'It demonstrates the structure of search results.',
                'source': 'mock',
                'timestamp': datetime.now(timezone.utc)
            }
            for i in range(1, min(self.config.max_search_results + 1, 4))
        ]

    async def close(self):
        """Close the HTTP session"""
        if self._session:
            await self._session.close()


class CitationExtractor:
    """
    Extract snippets and sources from search results
    Create structured citations with confidence scores
    """

    def __init__(self, config: GroundingConfig):
        self.config = config
        self.logger = logger.bind(component="citation_extractor")

    async def extract_citations(
        self,
        search_results: List[Dict[str, Any]],
        query: Optional[str] = None
    ) -> List[GroundingEvidence]:
        """
        Extract citations from search results

        Args:
            search_results: List of search results
            query: Optional original query for relevance scoring

        Returns:
            List of structured citation objects
        """
        citations = []

        for result in search_results[:self.config.max_citations]:
            try:
                citation = await self._create_citation(result, query)
                if citation:
                    citations.append(citation)
            except Exception as e:
                self.logger.warning("Failed to create citation", error=str(e))

        self.logger.info("Extracted citations", count=len(citations))
        return citations

    async def _create_citation(
        self,
        result: Dict[str, Any],
        query: Optional[str]
    ) -> Optional[GroundingEvidence]:
        """Create a citation from a search result"""
        url = result.get('url', '')
        snippet = result.get('snippet', '')

        if not url or not snippet:
            return None

        is_trusted = self._is_trusted_source(url)

        confidence = self._calculate_confidence(result, query, is_trusted)

        if confidence < self.config.citation_threshold:
            return None

        citation = GroundingEvidence(
            source_url=url,
            snippet=snippet,
            confidence=confidence,
            timestamp=result.get('timestamp', datetime.now(timezone.utc)),
            metadata={
                'title': result.get('title', ''),
                'source': result.get('source', 'unknown'),
                'is_trusted': is_trusted
            }
        )

        return citation

    def _is_trusted_source(self, url: str) -> bool:
        """Check if URL is from a trusted source using proper domain matching"""
        if not self.config.trusted_sources:
            return True  # No trusted sources specified, trust all

        try:
            parsed = urlparse(url)
            hostname = (parsed.hostname or "").lower()
        except Exception:
            return False

        for trusted_domain in self.config.trusted_sources:
            td = trusted_domain.lower().lstrip(".")
            if hostname == td or hostname.endswith("." + td):
                return True

        return False

    def _calculate_confidence(
        self,
        result: Dict[str, Any],
        query: Optional[str],
        is_trusted: bool
    ) -> float:
        """
        Calculate confidence score for a citation

        Based on:
        - Source trustworthiness
        - Snippet relevance
        - Snippet length/quality
        """
        confidence = 0.5  # Base confidence

        if is_trusted:
            confidence += 0.2

        snippet = result.get('snippet', '')
        if len(snippet) > 100:
            confidence += 0.1
        if len(snippet) > 200:
            confidence += 0.1

        if query:
            query_words = set(query.lower().split())
            snippet_words = set(snippet.lower().split())
            overlap = len(query_words.intersection(snippet_words))
            relevance = overlap / max(len(query_words), 1)
            confidence += relevance * 0.2

        return min(confidence, 1.0)


class Validator:
    """
    Validate thoughts and annotate with evidence
    Detect contradictions and flag for revision
    """

    def __init__(self, config: GroundingConfig, embedding_provider: Optional[EmbeddingProvider] = None):
        self.config = config
        self.logger = logger.bind(component="validator")
        self._embedding_provider = embedding_provider or EmbeddingProvider()

    async def validate_thoughts(
        self,
        thoughts: List[Thought],
        citations: List[GroundingEvidence]
    ) -> List[Dict[str, Any]]:
        """
        Validate thoughts against citations

        Args:
            thoughts: List of thoughts to validate
            citations: List of grounding evidence

        Returns:
            List of validation results
        """
        validations = []

        for thought in thoughts:
            validation = await self._validate_single_thought(thought, citations)
            validations.append(validation)

        self.logger.info("Validated thoughts", count=len(validations))
        return validations

    async def _validate_single_thought(
        self,
        thought: Thought,
        citations: List[GroundingEvidence]
    ) -> Dict[str, Any]:
        """Validate a single thought"""
        claims = self._extract_claims(thought.thought)

        supported_claims = []
        unsupported_claims = []
        contradicted_claims = []

        for claim in claims:
            support = self._find_supporting_citation(claim, citations)

            if support['status'] == 'supported':
                supported_claims.append(claim)
            elif support['status'] == 'contradicted':
                contradicted_claims.append(claim)
            else:
                unsupported_claims.append(claim)

        total_claims = len(claims)
        if total_claims == 0:
            validation_score = 0.5
        else:
            validation_score = (
                len(supported_claims) - len(contradicted_claims) * 0.5
            ) / total_claims
            validation_score = max(0.0, min(1.0, validation_score))

        needs_revision = (
            len(contradicted_claims) > 0 or
            validation_score < 0.5
        )

        return {
            'thought_number': thought.thought_number,
            'validation_score': validation_score,
            'supported_claims': supported_claims,
            'unsupported_claims': unsupported_claims,
            'contradicted_claims': contradicted_claims,
            'needs_revision': needs_revision,
            'refuted': len(contradicted_claims) > 0
        }

    def _extract_claims(self, thought_content: str) -> List[str]:
        """Extract factual claims from thought content using shared utility."""
        return extract_claims(thought_content)

    def _find_supporting_citation(
        self,
        claim: str,
        citations: List[GroundingEvidence]
    ) -> Dict[str, Any]:
        """
        Find citations that support, contradict, or are neutral to a claim.
        Uses semantic similarity (sentence-transformers when available, n-gram fallback).
        """
        best_match: Dict[str, Any] = {
            'status': 'unsupported',
            'citation': None,
            'confidence': 0.0
        }

        single_word_indicators = {
            'never', 'false', 'incorrect', 'wrong',
            'disproven', 'debunked', 'myth', 'untrue', 'inaccurate',
            'contrary', 'refuted',
        }
        multi_word_indicators = [
            'no evidence', 'does not', 'is not', 'cannot',
            'fails to', 'not true', 'not accurate',
        ]

        support_threshold = self.config.support_similarity_threshold
        contradiction_threshold = self.config.contradiction_similarity_threshold

        for citation in citations:
            similarity = self._embedding_provider.similarity(claim, citation.snippet)

            if similarity > support_threshold:  # Some meaningful overlap
                snippet_lower = citation.snippet.lower()
                snippet_words = set(snippet_lower.split())

                has_contradiction = bool(single_word_indicators & snippet_words)
                if not has_contradiction:
                    has_contradiction = any(
                        phrase in snippet_lower
                        for phrase in multi_word_indicators
                    )

                if has_contradiction and similarity > contradiction_threshold:
                    return {
                        'status': 'contradicted',
                        'citation': citation,
                        'confidence': citation.confidence * similarity,
                    }
                elif similarity > best_match['confidence']:
                    best_match = {
                        'status': 'supported',
                        'citation': citation,
                        'confidence': citation.confidence * similarity,
                    }

        return best_match


class GroundingModule:
    """
    Main grounding module that coordinates fact verification

    Features:
    - Multi-provider search (Google, Bing, DuckDuckGo)
    - Citation extraction with confidence scoring
    - Thought validation with contradiction detection
    - Trusted source verification
    """

    def __init__(self, config: GroundingConfig, embedding_provider: Optional[EmbeddingProvider] = None):
        self.config = config
        self.logger = logger.bind(component="grounding")
        self._embedding_provider = embedding_provider or EmbeddingProvider()

        self.search_client = SearchClient(config)
        self.citation_extractor = CitationExtractor(config)
        self.validator = Validator(config, self._embedding_provider)

        self.logger.info(
            "Grounding module initialized",
            enable_grounding=config.enable_grounding,
            search_provider=config.search_provider
        )

    async def ground_claims(
        self,
        thoughts: List[Thought],
        query: str,
        context: Dict[str, Any]
    ) -> Tuple[List[Thought], List[GroundingEvidence]]:
        """
        Ground thoughts with real-world evidence

        Args:
            thoughts: List of thoughts to ground
            query: Original query
            context: Retrieved context

        Returns:
            Tuple of (grounded thoughts, citations)
        """
        if not self.config.enable_grounding:
            return thoughts, []

        self.logger.info("Grounding claims", num_thoughts=len(thoughts))

        try:
            search_results = await self.search_client.search(query)

            citations = await self.citation_extractor.extract_citations(
                search_results,
                query
            )

            if self.config.enable_fact_checking:
                validations = await self.validator.validate_thoughts(thoughts, citations)

                for thought, validation in zip(thoughts, validations):
                    thought.metadata['validation'] = validation

            self.logger.info(
                "Grounding completed",
                citations=len(citations),
                searches=len(search_results)
            )

            return thoughts, citations

        except Exception as e:
            self.logger.error("Grounding failed", error=str(e), exc_info=True)
            return thoughts, []

    async def ground_claim(self, claim: str) -> List[GroundingEvidence]:
        """
        Ground a specific claim with evidence

        Args:
            claim: Claim to ground

        Returns:
            List of supporting evidence
        """
        try:
            search_results = await self.search_client.search(claim)
            citations = await self.citation_extractor.extract_citations(
                search_results,
                claim
            )
            return citations
        except Exception as e:
            self.logger.error("Claim grounding failed", error=str(e))
            return []

    async def verify_fact(
        self,
        fact: str,
        min_confidence: float = 0.7
    ) -> Dict[str, Any]:
        """
        Verify a specific fact

        Args:
            fact: Fact to verify
            min_confidence: Minimum confidence threshold

        Returns:
            Verification result with confidence and evidence
        """
        citations = await self.ground_claim(fact)

        if not citations:
            return {
                'fact': fact,
                'verified': False,
                'confidence': 0.0,
                'evidence': []
            }

        avg_confidence = sum(c.confidence for c in citations) / len(citations)

        return {
            'fact': fact,
            'verified': avg_confidence >= min_confidence,
            'confidence': avg_confidence,
            'evidence': citations
        }

    async def close(self):
        """Clean up resources"""
        await self.search_client.close()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()
        return False
