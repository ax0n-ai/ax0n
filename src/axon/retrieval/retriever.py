import asyncio
import warnings
from collections import defaultdict, deque
from typing import Any, Callable, Dict, List, Optional
import structlog
from ..core.config import RetrieverConfig

logger = structlog.get_logger(__name__)


class VectorSearch:
    """Vector database search functionality"""

    def __init__(self, config: RetrieverConfig, embedding_function: Optional[Callable] = None):
        self.config = config
        self.logger = logger.bind(component="vector_search")
        self._client = None
        self.embedding_function = embedding_function

    async def _get_client(self):
        """Get vector DB client based on provider"""
        if self._client is not None:
            return self._client

        if self.config.vector_db_provider == "weaviate":
            try:
                import weaviate
                url = self.config.vector_db_url or "http://localhost:8080"
                api_key = self.config.vector_db_api_key

                if hasattr(weaviate, "connect_to_local"):
                    from urllib.parse import urlparse
                    parsed = urlparse(url)
                    host = parsed.hostname or "localhost"
                    port = parsed.port or 8080

                    if api_key:
                        auth = weaviate.auth.AuthApiKey(api_key)
                        self._client = weaviate.connect_to_local(
                            host=host, port=port, auth_credentials=auth
                        )
                    else:
                        self._client = weaviate.connect_to_local(
                            host=host, port=port
                        )
                else:
                    self._client = weaviate.Client(
                        url=url,
                        auth_client_secret=weaviate.AuthApiKey(api_key=api_key) if api_key else None
                    )
            except Exception as e:
                self.logger.warning("Weaviate client not available, using mock", error=str(e))
                self._client = MockVectorClient()

        elif self.config.vector_db_provider == "pinecone":
            try:
                from pinecone import Pinecone
                self._client = Pinecone(api_key=self.config.vector_db_api_key)
            except Exception as e:
                self.logger.warning("Pinecone client not available, using mock", error=str(e))
                self._client = MockVectorClient()
        else:
            self.logger.warning(f"Unknown vector DB provider: {self.config.vector_db_provider}, using mock")
            self._client = MockVectorClient()

        return self._client

    async def search(self, query: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Search for similar vectors

        Args:
            query: Search query
            limit: Maximum number of results

        Returns:
            List of search results with metadata
        """
        try:
            client = await self._get_client()
            limit = limit or self.config.max_results

            if hasattr(client, 'collections'):
                collection = client.collections.get("Document")
                response = collection.query.near_text(
                    query=query, limit=limit
                )
                results = []
                for obj in response.objects:
                    props = obj.properties
                    results.append({
                        'content': props.get('content', ''),
                        'metadata': props.get('metadata', {}),
                        'score': obj.metadata.distance if obj.metadata else 0.0,
                    })
                return results

            elif hasattr(client, 'query'):
                response = client.query.get("Document", ["content", "metadata", "score"]).with_near_text({
                    "concepts": [query]
                }).with_limit(limit).do()

                results = []
                for obj in response['data']['Get']['Document']:
                    results.append({
                        'content': obj['content'],
                        'metadata': obj.get('metadata', {}),
                        'score': obj.get('score', 0.0)
                    })
                return results

            elif hasattr(client, 'list_indexes'):
                index_name = "ax0n-index"
                indexes = client.list_indexes()
                index_names = [idx.name for idx in indexes] if hasattr(indexes, '__iter__') else []
                if index_name not in index_names:
                    return []

                index = client.Index(index_name)

                if self.embedding_function:
                    embedding = self.embedding_function(query)
                    response = index.query(vector=embedding, top_k=limit, include_metadata=True)
                    results = []
                    for match in response.get('matches', []):
                        results.append({
                            'content': match.get('metadata', {}).get('content', ''),
                            'metadata': match.get('metadata', {}),
                            'score': match.get('score', 0.0),
                        })
                    return results
                else:
                    warnings.warn(
                        "No embedding_function provided for Pinecone; returning mock results. "
                        "Pass embedding_function to VectorSearch or Retriever for real queries.",
                        stacklevel=2,
                    )
                    return self._mock_search_results(query, limit)

            else:
                return self._mock_search_results(query, limit)

        except Exception as e:
            self.logger.error("Vector search failed", error=str(e), exc_info=True)
            return []

    def _mock_search_results(self, query: str, limit: int) -> List[Dict[str, Any]]:
        """Mock search results for development"""
        return [
            {
                'content': f"Mock result for: {query}",
                'metadata': {'source': 'mock', 'type': 'document'},
                'score': 0.85
            }
        ] * min(limit, 3)


class KVStore:
    """Key-value store for user attributes"""

    def __init__(self, config: RetrieverConfig):
        self.config = config
        self.logger = logger.bind(component="kv_store")
        self._client = None

    async def _get_client(self):
        """Get KV store client"""
        if self._client is not None:
            return self._client

        if self.config.kv_store_url:
            try:
                import redis.asyncio as redis
                self._client = redis.from_url(
                    self.config.kv_store_url,
                    socket_connect_timeout=5,
                    socket_timeout=10,
                )
            except ImportError:
                self.logger.warning("Redis client not available, using mock")
                self._client = MockKVClient()
        else:
            self.logger.info("No KV store URL provided, using mock")
            self._client = MockKVClient()

        return self._client

    async def get_user_attributes(self, user_id: Optional[str]) -> Dict[str, Any]:
        """
        Get user attributes from KV store

        Args:
            user_id: User identifier

        Returns:
            Dictionary of user attributes
        """
        if not user_id:
            return {}

        try:
            client = await self._get_client()

            if hasattr(client, 'get'):
                key = f"user:{user_id}:attributes"
                data = await client.get(key)
                if data:
                    import json
                    return json.loads(data)
                return {}
            else:
                return self._mock_user_attributes(user_id)

        except Exception as e:
            self.logger.error("Failed to get user attributes", error=str(e), exc_info=True)
            return {}

    async def set_user_attributes(self, user_id: str, attributes: Dict[str, Any]) -> bool:
        """
        Set user attributes in KV store

        Args:
            user_id: User identifier
            attributes: User attributes to store

        Returns:
            Success status
        """
        import json as _json
        payload = _json.dumps(attributes, default=str)
        if len(payload) > 1_000_000:
            self.logger.warning(
                "User attributes too large",
                user_id=user_id,
                size=len(payload),
            )
            return False

        try:
            client = await self._get_client()

            if hasattr(client, 'set'):
                key = f"user:{user_id}:attributes"
                import json
                await client.set(key, json.dumps(attributes))
                return True
            else:
                return True

        except Exception as e:
            self.logger.error("Failed to set user attributes", error=str(e), exc_info=True)
            return False

    def _mock_user_attributes(self, user_id: str) -> Dict[str, Any]:
        """Mock user attributes for development"""
        return {
            'preferences': {
                'language': 'en',
                'timezone': 'UTC',
                'theme': 'dark'
            },
            'history': {
                'last_query': 'What is machine learning?',
                'query_count': 42
            }
        }


class GraphEngine:
    """In-memory concept graph with BFS traversal for relationship reasoning."""

    def __init__(self, config: RetrieverConfig):
        self.config = config
        self.logger = logger.bind(component="graph_engine")
        self._enabled = config.enable_graph_engine
        self._graph: Dict[str, set] = defaultdict(set)

    def add_relationship(
        self,
        concept_a: str,
        concept_b: str,
        relationship: str = "related_to",
        confidence: float = 1.0,
    ) -> None:
        """
        Add a bidirectional relationship between two concepts.

        Args:
            concept_a: First concept
            concept_b: Second concept
            relationship: Relationship type
            confidence: Confidence score for the relationship
        """
        self._graph[concept_a.lower()].add((concept_b.lower(), relationship, confidence))
        self._graph[concept_b.lower()].add((concept_a.lower(), relationship, confidence))

    async def get_related_concepts(
        self, concept: str, max_depth: int = 2
    ) -> List[Dict[str, Any]]:
        """
        Get related concepts through BFS graph traversal.

        Args:
            concept: Starting concept
            max_depth: Maximum traversal depth

        Returns:
            List of related concepts with relationships
        """
        if not self._enabled:
            return []

        try:
            return self._bfs_traverse(concept.lower(), max_depth)
        except Exception as e:
            self.logger.error("Graph traversal failed", error=str(e))
            return []

    def _bfs_traverse(self, start: str, max_depth: int) -> List[Dict[str, Any]]:
        """BFS traversal of the concept graph."""
        if start not in self._graph:
            return []

        visited: set = {start}
        queue: deque = deque()

        for related, relationship, confidence in self._graph[start]:
            queue.append((related, relationship, confidence, 1))

        results: List[Dict[str, Any]] = []

        while queue:
            concept, relationship, confidence, depth = queue.popleft()
            if concept in visited:
                continue
            visited.add(concept)

            results.append({
                'concept': concept,
                'relationship': relationship,
                'confidence': confidence,
                'depth': depth,
            })

            if depth < max_depth:
                for next_concept, next_rel, next_conf in self._graph.get(concept, set()):
                    if next_concept not in visited:
                        queue.append((next_concept, next_rel, next_conf, depth + 1))

        return results


class MockVectorClient:
    """Mock vector client for development"""

    def query(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        """Return empty query results."""
        return {'matches': []}

    def search(self, *args: Any, **kwargs: Any) -> List[Dict[str, Any]]:
        """Return empty search results."""
        return []


class MockKVClient:
    """Mock KV client for development (bounded to prevent unbounded growth)"""

    _MAX_ENTRIES = 10_000

    def __init__(self):
        self._store: Dict[str, Any] = {}

    async def get(self, key: str):
        return self._store.get(key)

    async def set(self, key: str, value: str):
        if key not in self._store and len(self._store) >= self._MAX_ENTRIES:
            oldest_key = next(iter(self._store))
            del self._store[oldest_key]
        self._store[key] = value


class Retriever:
    """
    Main retriever class that coordinates vector search, KV store, and graph engine
    """

    def __init__(self, config: RetrieverConfig):
        self.config = config
        self.logger = logger.bind(component="retriever")

        self.vector_search = VectorSearch(config)
        self.kv_store = KVStore(config)
        self.graph_engine = GraphEngine(config)

        self.logger.info("Retriever initialized", config=self._get_config_summary())

    def _get_config_summary(self) -> Dict[str, Any]:
        """Get configuration summary for logging"""
        return {
            'vector_db_provider': self.config.vector_db_provider,
            'enable_kv_store': self.config.enable_kv_store,
            'enable_graph_engine': self.config.enable_graph_engine,
            'max_results': self.config.max_results
        }

    async def search(self, query: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Perform vector search for relevant documents

        Args:
            query: Search query
            limit: Maximum number of results

        Returns:
            List of search results
        """
        return await self.vector_search.search(query, limit)

    async def get_user_attributes(self, user_id: Optional[str]) -> Dict[str, Any]:
        """
        Get user attributes from KV store

        Args:
            user_id: User identifier

        Returns:
            User attributes dictionary
        """
        if not self.config.enable_kv_store:
            return {}
        return await self.kv_store.get_user_attributes(user_id)

    async def set_user_attributes(self, user_id: str, attributes: Dict[str, Any]) -> bool:
        """
        Set user attributes in KV store

        Args:
            user_id: User identifier
            attributes: Attributes to store

        Returns:
            Success status
        """
        if not self.config.enable_kv_store:
            return False
        return await self.kv_store.set_user_attributes(user_id, attributes)

    async def get_related_concepts(self, concept: str, max_depth: int = 2) -> List[Dict[str, Any]]:
        """
        Get related concepts through graph traversal

        Args:
            concept: Starting concept
            max_depth: Maximum traversal depth

        Returns:
            List of related concepts
        """
        return await self.graph_engine.get_related_concepts(concept, max_depth)

    def add_relationship(
        self,
        concept_a: str,
        concept_b: str,
        relationship: str = "related_to",
        confidence: float = 1.0,
    ) -> None:
        """Add a relationship to the concept graph (pass-through to GraphEngine)."""
        self.graph_engine.add_relationship(concept_a, concept_b, relationship, confidence)

    async def retrieve(self, query: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Main retrieval method - retrieve comprehensive context for a query

        Args:
            query: The query to get context for
            context: Optional additional context with user_id, etc.

        Returns:
            Dictionary containing all relevant context
        """
        user_id = context.get('user_id') if context else None
        return await self.retrieve_context(query, user_id)

    async def retrieve_context(self, query: str, user_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Retrieve comprehensive context for a query

        Args:
            query: The query to get context for
            user_id: Optional user identifier

        Returns:
            Dictionary containing all relevant context
        """
        self.logger.info("Retrieving context", query=query[:100] + "..." if len(query) > 100 else query)

        tasks = [
            self.search(query),
            self.get_user_attributes(user_id),
            self.get_related_concepts(query, max_depth=1)
        ]

        try:
            vector_results, user_attrs, related_concepts = await asyncio.gather(*tasks, return_exceptions=True)

            if isinstance(vector_results, Exception):
                self.logger.warning("Vector search failed", error=str(vector_results))
                vector_results = []
            if isinstance(user_attrs, Exception):
                self.logger.warning("User attributes retrieval failed", error=str(user_attrs))
                user_attrs = {}
            if isinstance(related_concepts, Exception):
                self.logger.warning("Graph traversal failed", error=str(related_concepts))
                related_concepts = []

            context = {
                'vector_results': vector_results,
                'user_attributes': user_attrs,
                'related_concepts': related_concepts,
                'query': query,
                'user_id': user_id
            }

            self.logger.info(
                "Context retrieval completed",
                vector_results_count=len(vector_results),
                related_concepts_count=len(related_concepts)
            )

            return context

        except Exception as e:
            self.logger.error("Context retrieval failed", error=str(e))
            return {
                'vector_results': [],
                'user_attributes': {},
                'related_concepts': [],
                'query': query,
                'user_id': user_id
            }
