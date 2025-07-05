"""
Retriever module for Ax0n - handles context fetching via embeddings and KV lookup
"""

import asyncio
from typing import Any, Dict, List, Optional, Union
import structlog
from ..core.config import RetrieverConfig

logger = structlog.get_logger(__name__)


class VectorSearch:
    """Vector database search functionality"""
    
    def __init__(self, config: RetrieverConfig):
        self.config = config
        self.logger = logger.bind(component="vector_search")
        self._client = None
        
    async def _get_client(self):
        """Get vector DB client based on provider"""
        if self._client is not None:
            return self._client
            
        if self.config.vector_db_provider == "weaviate":
            try:
                import weaviate
                self._client = weaviate.Client(
                    url=self.config.vector_db_url or "http://localhost:8080",
                    auth_client_secret=weaviate.AuthApiKey(api_key=self.config.vector_db_api_key) if self.config.vector_db_api_key else None
                )
            except ImportError:
                self.logger.warning("Weaviate client not available, using mock")
                self._client = MockVectorClient()
                
        elif self.config.vector_db_provider == "pinecone":
            try:
                import pinecone
                pinecone.init(
                    api_key=self.config.vector_db_api_key,
                    environment=self.config.vector_db_url or "us-east1-gcp"
                )
                self._client = pinecone
            except ImportError:
                self.logger.warning("Pinecone client not available, using mock")
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
            
            if hasattr(client, 'query'):
                # Weaviate client
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
                # Pinecone client
                index_name = "ax0n-index"
                if index_name not in client.list_indexes():
                    return []
                    
                index = client.Index(index_name)
                # For now, return mock results since we need embeddings
                return self._mock_search_results(query, limit)
                
            else:
                # Mock client
                return self._mock_search_results(query, limit)
                
        except Exception as e:
            self.logger.error("Vector search failed", error=str(e))
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
                self._client = redis.from_url(self.config.kv_store_url)
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
                # Redis client
                key = f"user:{user_id}:attributes"
                data = await client.get(key)
                if data:
                    import json
                    return json.loads(data)
                return {}
            else:
                # Mock client
                return self._mock_user_attributes(user_id)
                
        except Exception as e:
            self.logger.error("Failed to get user attributes", error=str(e))
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
        try:
            client = await self._get_client()
            
            if hasattr(client, 'set'):
                # Redis client
                key = f"user:{user_id}:attributes"
                import json
                await client.set(key, json.dumps(attributes))
                return True
            else:
                # Mock client
                return True
                
        except Exception as e:
            self.logger.error("Failed to set user attributes", error=str(e))
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
    """Optional graph-based retrieval for relationship reasoning"""
    
    def __init__(self, config: RetrieverConfig):
        self.config = config
        self.logger = logger.bind(component="graph_engine")
        self._enabled = config.enable_graph_engine
        
    async def get_related_concepts(self, concept: str, max_depth: int = 2) -> List[Dict[str, Any]]:
        """
        Get related concepts through graph traversal
        
        Args:
            concept: Starting concept
            max_depth: Maximum traversal depth
            
        Returns:
            List of related concepts with relationships
        """
        if not self._enabled:
            return []
            
        try:
            # Mock implementation for now
            return self._mock_related_concepts(concept, max_depth)
        except Exception as e:
            self.logger.error("Graph traversal failed", error=str(e))
            return []
    
    def _mock_related_concepts(self, concept: str, max_depth: int) -> List[Dict[str, Any]]:
        """Mock related concepts for development"""
        return [
            {
                'concept': f'related_{concept}_1',
                'relationship': 'is_a',
                'confidence': 0.8
            },
            {
                'concept': f'related_{concept}_2',
                'relationship': 'part_of',
                'confidence': 0.6
            }
        ]


class MockVectorClient:
    """Mock vector client for development"""
    pass


class MockKVClient:
    """Mock KV client for development"""
    def __init__(self):
        self._store = {}
    
    async def get(self, key: str):
        return self._store.get(key)
    
    async def set(self, key: str, value: str):
        self._store[key] = value


class Retriever:
    """
    Main retriever class that coordinates vector search, KV store, and graph engine
    """
    
    def __init__(self, config: RetrieverConfig):
        self.config = config
        self.logger = logger.bind(component="retriever")
        
        # Initialize components
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
        
        # Parallel execution of different retrieval methods
        tasks = [
            self.search(query),
            self.get_user_attributes(user_id),
            self.get_related_concepts(query, max_depth=1)
        ]
        
        try:
            vector_results, user_attrs, related_concepts = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Handle exceptions
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