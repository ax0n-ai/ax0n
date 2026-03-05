import asyncio
import json
import os
import uuid
import hashlib
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime, timezone
from collections import defaultdict
import structlog

from ..core.config import MemoryConfig
from ..core.models import MemoryEntry, Thought
from ..utils import extract_claims
from ..utils.embeddings import EmbeddingProvider

logger = structlog.get_logger(__name__)


class MemoryExtractor:
    """Extract candidate memory facts from thoughts"""

    def __init__(self, config: MemoryConfig):
        self.config = config
        self.logger = logger.bind(component="memory_extractor")

    async def extract_candidates(
        self,
        thoughts: List[Thought],
        query: str,
        context: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Extract candidate memory facts from thoughts

        Args:
            thoughts: List of thoughts to analyze
            query: Original query
            context: Retrieved context

        Returns:
            List of candidate memory entries
        """
        candidates = []

        for thought in thoughts:
            if thought.score and thought.score < self.config.extraction_threshold:
                continue

            facts = self._extract_facts_from_thought(thought)

            for fact in facts:
                candidate = {
                    'content': fact,
                    'source_thought': thought.thought_number,
                    'confidence': thought.score or 0.7,
                    'stage': thought.stage.value,
                    'tags': thought.tags.copy() if thought.tags else [],
                    'timestamp': datetime.now(timezone.utc)
                }
                candidates.append(candidate)

        self.logger.info("Extracted memory candidates", count=len(candidates))
        return candidates

    def _extract_facts_from_thought(self, thought: Thought) -> List[str]:
        """
        Extract individual facts from a thought using shared claim extraction.
        """
        return extract_claims(thought.thought, min_words=4, min_chars=20, max_claims=5)


class SalienceScorer:
    """Score memory importance/salience using heuristics"""

    def __init__(self, config: MemoryConfig):
        self.config = config
        self.logger = logger.bind(component="salience_scorer")

    def score_memory(self, candidate: Dict[str, Any], context: Dict[str, Any]) -> float:
        """
        Calculate salience score for a memory candidate

        Args:
            candidate: Memory candidate
            context: Current context

        Returns:
            Salience score between 0 and 1
        """
        if not self.config.salience_scoring:
            return candidate.get('confidence', 0.5)

        score = 0.0

        score += candidate.get('confidence', 0.5) * 0.4

        content_len = len(candidate['content'])
        length_score = min(content_len / 200, 1.0)
        score += length_score * 0.2

        tags = candidate.get('tags', [])
        tag_score = min(len(tags) / 5, 1.0)
        score += tag_score * 0.2

        stage_weights = {
            'problem_definition': 0.8,
            'research': 0.7,
            'analysis': 0.9,
            'synthesis': 0.9,
            'conclusion': 1.0,
            'verification': 0.8
        }
        stage = candidate.get('stage', 'analysis')
        score += stage_weights.get(stage, 0.7) * 0.2

        return min(score, 1.0)


class MemoryDeduplicator:
    """Handle deduplication and similarity-based merging"""

    def __init__(self, config: MemoryConfig, embedding_provider: Optional[EmbeddingProvider] = None):
        self.config = config
        self.logger = logger.bind(component="memory_deduplicator")
        self._embedding_provider = embedding_provider or EmbeddingProvider()

    def compute_similarity(self, text1: str, text2: str) -> float:
        """
        Compute semantic similarity between two texts.

        Uses sentence-transformers cosine similarity when available,
        otherwise falls back to weighted n-gram Jaccard.
        """
        return self._embedding_provider.similarity(text1, text2)

    async def deduplicate(
        self,
        new_memory: Dict[str, Any],
        existing_memories: List[MemoryEntry]
    ) -> Tuple[str, Optional[str]]:
        """
        Determine if memory should be added, updated, or merged

        Args:
            new_memory: New memory candidate
            existing_memories: Existing memory entries

        Returns:
            Tuple of (action, memory_id) where action is 'add', 'update', or 'delete'
        """
        threshold = self.config.deduplication_threshold

        for existing in existing_memories:
            similarity = self.compute_similarity(
                new_memory['content'],
                existing.content
            )

            if similarity >= threshold:
                self.logger.debug(
                    "Found duplicate memory",
                    similarity=similarity,
                    existing_id=existing.id
                )
                return ('update', existing.id)

        return ('add', None)

    def merge_memories(
        self,
        new_memory: Dict[str, Any],
        existing_memory: MemoryEntry
    ) -> MemoryEntry:
        """
        Merge new memory with existing memory

        Args:
            new_memory: New memory data
            existing_memory: Existing memory entry

        Returns:
            Updated memory entry
        """
        new_conf = new_memory.get('confidence', 0.5)
        old_conf = existing_memory.confidence
        old_weight = existing_memory.access_count + 1  # +1 counts the original entry itself
        merged_conf = (old_conf * old_weight + new_conf) / (old_weight + 1)

        new_tags = set(new_memory.get('tags', []))
        old_tags = set(existing_memory.tags)
        merged_tags = list(old_tags.union(new_tags))

        existing_memory.confidence = merged_conf
        existing_memory.tags = merged_tags
        existing_memory.updated_at = datetime.now(timezone.utc)
        existing_memory.access_count += 1

        if 'salience_score' in new_memory:
            existing_memory.salience_score = max(
                existing_memory.salience_score,
                new_memory['salience_score']
            )

        return existing_memory


class MemoryStorage:
    """
    Hybrid storage backend for memories
    Supports vector, KV, and graph storage
    """

    def __init__(self, config: MemoryConfig, embedding_provider: Optional[EmbeddingProvider] = None):
        self.config = config
        self.logger = logger.bind(component="memory_storage")
        self._lock = asyncio.Lock()
        self._embedding_provider = embedding_provider or EmbeddingProvider()

        self._agent_memories: Dict[str, List[MemoryEntry]] = defaultdict(list)
        self._shared_memories: List[MemoryEntry] = []
        self._memory_graph: Dict[str, List[str]] = defaultdict(list)

        self.logger.info(
            "Memory storage initialized",
            provider=config.storage_provider,
            mode=config.memory_provider
        )

    def _total_memory_count(self) -> int:
        """Count all stored memories across agents and shared."""
        return len(self._shared_memories) + sum(
            len(m) for m in self._agent_memories.values()
        )

    def _evict_expired(self) -> int:
        """Evict memories past TTL. Returns count of evicted entries."""
        ttl = self.config.memory_ttl_seconds
        if ttl is None or ttl <= 0:
            return 0

        now = datetime.now(timezone.utc)
        evicted = 0

        before = len(self._shared_memories)
        self._shared_memories = [
            m for m in self._shared_memories
            if (now - m.created_at).total_seconds() <= ttl
        ]
        expired_shared = before - len(self._shared_memories)
        evicted += expired_shared

        for agent_id in list(self._agent_memories.keys()):
            before = len(self._agent_memories[agent_id])
            self._agent_memories[agent_id] = [
                m for m in self._agent_memories[agent_id]
                if (now - m.created_at).total_seconds() <= ttl
            ]
            evicted += before - len(self._agent_memories[agent_id])

        if evicted > 0:
            self.logger.info("Evicted expired memories", count=evicted, ttl_seconds=ttl)

        return evicted

    def _evict_oldest(self) -> None:
        """Evict expired memories first, then oldest when over the configured limit."""
        self._evict_expired()

        over = self._total_memory_count() - self.config.max_memory_entries
        if over <= 0:
            return
        self._shared_memories.sort(key=lambda m: m.created_at)
        while over > 0 and self._shared_memories:
            evicted = self._shared_memories.pop(0)
            self._memory_graph.pop(evicted.id, None)
            over -= 1
        if over > 0:
            for agent_id in list(self._agent_memories.keys()):
                self._agent_memories[agent_id].sort(key=lambda m: m.created_at)
                while over > 0 and self._agent_memories[agent_id]:
                    evicted = self._agent_memories[agent_id].pop(0)
                    self._memory_graph.pop(evicted.id, None)
                    over -= 1

    async def store_memory(
        self,
        memory: MemoryEntry,
        agent_id: Optional[str] = None
    ) -> bool:
        """
        Store a memory entry (thread-safe, bounded)

        Args:
            memory: Memory entry to store
            agent_id: Optional agent ID for agent-scoped storage

        Returns:
            Success status
        """
        async with self._lock:
            try:
                if memory.is_shared or not agent_id:
                    self._shared_memories.append(memory)
                    self.logger.debug("Stored shared memory", memory_id=memory.id)
                else:
                    self._agent_memories[agent_id].append(memory)
                    self.logger.debug(
                        "Stored agent memory",
                        memory_id=memory.id,
                        agent_id=agent_id
                    )

                for related_id in memory.relationships:
                    self._memory_graph[memory.id].append(related_id)
                    self._memory_graph[related_id].append(memory.id)

                self._evict_oldest()

                return True

            except Exception as e:
                self.logger.error("Failed to store memory", error=str(e), exc_info=True)
                return False

    async def retrieve_memories(
        self,
        query: str,
        agent_id: Optional[str] = None,
        limit: int = 10,
        min_salience: float = 0.0
    ) -> List[MemoryEntry]:
        """
        Retrieve relevant memories

        Args:
            query: Search query
            agent_id: Optional agent ID for agent-scoped retrieval
            limit: Maximum number of memories
            min_salience: Minimum salience score

        Returns:
            List of relevant memory entries
        """
        try:
            candidates = []

            if agent_id and self.config.enable_agent_memory:
                candidates.extend(self._agent_memories.get(agent_id, []))

            if self.config.enable_shared_memory:
                candidates.extend(self._shared_memories)

            candidates = [m for m in candidates if m.salience_score >= min_salience]

            scored_candidates = []

            for memory in candidates:
                relevance = self._embedding_provider.similarity(query, memory.content)

                score = (relevance * 0.6) + (memory.salience_score * 0.4)
                scored_candidates.append((score, memory))

            scored_candidates.sort(key=lambda x: x[0], reverse=True)
            results = [memory for score, memory in scored_candidates[:limit]]

            self.logger.info(
                "Retrieved memories",
                query=query[:50],
                count=len(results),
                agent_id=agent_id
            )

            return results

        except Exception as e:
            self.logger.error("Failed to retrieve memories", error=str(e))
            return []

    async def update_memory(self, memory: MemoryEntry) -> bool:
        """
        Update an existing memory (thread-safe)

        Args:
            memory: Updated memory entry

        Returns:
            Success status
        """
        async with self._lock:
            try:
                if memory.is_shared:
                    for i, m in enumerate(self._shared_memories):
                        if m.id == memory.id:
                            self._shared_memories[i] = memory
                            return True
                else:
                    if memory.agent_id:
                        memories = self._agent_memories.get(memory.agent_id, [])
                        for i, m in enumerate(memories):
                            if m.id == memory.id:
                                self._agent_memories[memory.agent_id][i] = memory
                                return True

                return False

            except Exception as e:
                self.logger.error("Failed to update memory", error=str(e), exc_info=True)
                return False

    async def delete_memory(self, memory_id: str) -> bool:
        """
        Delete a memory (thread-safe)

        Args:
            memory_id: ID of memory to delete

        Returns:
            Success status
        """
        async with self._lock:
            try:
                self._shared_memories = [
                    m for m in self._shared_memories if m.id != memory_id
                ]

                for agent_id in self._agent_memories:
                    self._agent_memories[agent_id] = [
                        m for m in self._agent_memories[agent_id] if m.id != memory_id
                    ]

                if memory_id in self._memory_graph:
                    del self._memory_graph[memory_id]

                return True

            except Exception as e:
                self.logger.error("Failed to delete memory", error=str(e), exc_info=True)
                return False

    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory storage statistics"""
        total_agent_memories = sum(len(memories) for memories in self._agent_memories.values())

        return {
            'total_shared_memories': len(self._shared_memories),
            'total_agent_memories': total_agent_memories,
            'agents_with_memories': len(self._agent_memories),
            'graph_nodes': len(self._memory_graph)
        }

    def save_to_file(self, path: str) -> bool:
        """
        Persist all memories to a JSON file.

        Args:
            path: File path for the JSON dump

        Returns:
            True on success
        """
        try:
            data = {
                'shared_memories': [m.model_dump() for m in self._shared_memories],
                'agent_memories': {
                    agent_id: [m.model_dump() for m in memories]
                    for agent_id, memories in self._agent_memories.items()
                },
                'memory_graph': dict(self._memory_graph),
            }
            os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
            with open(path, 'w') as f:
                json.dump(data, f, indent=2, default=str)
            self.logger.info("Memories saved to file", path=path)
            return True
        except Exception as e:
            self.logger.error("Failed to save memories", error=str(e))
            return False

    def load_from_file(self, path: str) -> bool:
        """
        Load memories from a JSON file.

        Args:
            path: File path of the JSON dump

        Returns:
            True on success
        """
        try:
            if not os.path.exists(path):
                self.logger.warning("Memory file not found", path=path)
                return False

            with open(path, 'r') as f:
                data = json.load(f)

            self._shared_memories = [
                MemoryEntry(**m) for m in data.get('shared_memories', [])
            ]
            self._agent_memories = defaultdict(list)
            for agent_id, memories in data.get('agent_memories', {}).items():
                self._agent_memories[agent_id] = [
                    MemoryEntry(**m) for m in memories
                ]
            self._memory_graph = defaultdict(list)
            for key, value in data.get('memory_graph', {}).items():
                self._memory_graph[key] = value

            self.logger.info("Memories loaded from file", path=path)
            return True
        except Exception as e:
            self.logger.error("Failed to load memories", error=str(e))
            return False


class MemoryManager:
    """
    Main memory manager implementing Mem0-inspired hybrid memory architecture

    Features:
    - Agent-scoped memory for personalized context
    - Shared joint memory for multi-agent collaboration
    - Salience scoring for importance ranking
    - Deduplication and intelligent merging
    - Hybrid storage (vector, graph, KV)
    """

    def __init__(self, config: MemoryConfig, embedding_provider: Optional[EmbeddingProvider] = None):
        self.config = config
        self.logger = logger.bind(component="memory_manager")

        self._embedding_provider = embedding_provider or EmbeddingProvider()

        self.extractor = MemoryExtractor(config)
        self.scorer = SalienceScorer(config)
        self.deduplicator = MemoryDeduplicator(config, self._embedding_provider)
        self.storage = MemoryStorage(config, self._embedding_provider)

        self.logger.info(
            "MemoryManager initialized",
            enable_agent_memory=config.enable_agent_memory,
            enable_shared_memory=config.enable_shared_memory,
            storage_provider=config.storage_provider
        )

    async def extract_and_update(
        self,
        thoughts: List[Thought],
        query: str,
        context: Dict[str, Any],
        agent_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Extract candidate memory facts from thoughts and update memory storage

        Args:
            thoughts: List of thoughts to extract memories from
            query: Original query
            context: Retrieved context
            agent_id: Optional agent identifier

        Returns:
            List of memory update actions (add/update/delete)
        """
        if not self.config.enable_memory:
            return []

        self.logger.info(
            "Extracting and updating memory",
            num_thoughts=len(thoughts),
            agent_id=agent_id
        )

        try:
            candidates = await self.extractor.extract_candidates(thoughts, query, context)

            if not candidates:
                return []

            for candidate in candidates:
                candidate['salience_score'] = self.scorer.score_memory(candidate, context)

            existing_memories = await self.storage.retrieve_memories(
                query,
                agent_id=agent_id,
                limit=100,
                min_salience=0.0
            )

            memory_updates = []

            for candidate in candidates:
                action, memory_id = await self.deduplicator.deduplicate(
                    candidate,
                    existing_memories
                )

                if action == 'add':
                    memory = self._create_memory_entry(candidate, agent_id)
                    success = await self.storage.store_memory(memory, agent_id)

                    if success:
                        memory_updates.append({
                            'action': 'add',
                            'memory_id': memory.id,
                            'content': memory.content[:100] + '...' if len(memory.content) > 100 else memory.content,
                            'salience': memory.salience_score
                        })

                elif action == 'update' and memory_id:
                    existing = next((m for m in existing_memories if m.id == memory_id), None)
                    if existing:
                        updated = self.deduplicator.merge_memories(candidate, existing)
                        success = await self.storage.update_memory(updated)

                        if success:
                            memory_updates.append({
                                'action': 'update',
                                'memory_id': updated.id,
                                'content': updated.content[:100] + '...' if len(updated.content) > 100 else updated.content,
                                'salience': updated.salience_score
                            })

            self.logger.info(
                "Memory extraction completed",
                updates=len(memory_updates),
                agent_id=agent_id
            )

            return memory_updates

        except Exception as e:
            self.logger.error("Memory extraction failed", error=str(e), exc_info=True)
            return [{'action': 'error', 'error': str(e)}]

    def _create_memory_entry(
        self,
        candidate: Dict[str, Any],
        agent_id: Optional[str]
    ) -> MemoryEntry:
        """Create a MemoryEntry from a candidate"""
        memory_id = self._generate_memory_id(candidate['content'])

        return MemoryEntry(
            id=memory_id,
            content=candidate['content'],
            confidence=candidate.get('confidence', 0.7),
            salience_score=candidate.get('salience_score', 0.5),
            agent_id=agent_id,
            is_shared=(agent_id is None),
            tags=candidate.get('tags', []),
            source_thoughts=[candidate.get('source_thought', 0)],
            memory_type=candidate.get('memory_type', 'semantic'),
            created_at=candidate.get('timestamp', datetime.now(timezone.utc)),
            updated_at=datetime.now(timezone.utc)
        )

    def _generate_memory_id(self, content: str) -> str:
        """Generate a unique memory ID based on content hash"""
        content_hash = hashlib.sha256(content.encode()).hexdigest()[:12]
        random_suffix = uuid.uuid4().hex[:8]
        return f"mem_{content_hash}_{random_suffix}"

    async def retrieve_relevant(
        self,
        query: str,
        agent_id: Optional[str] = None,
        limit: int = 10
    ) -> List[MemoryEntry]:
        """
        Retrieve relevant memories for a query

        Args:
            query: Search query
            agent_id: Optional agent identifier
            limit: Maximum number of memories

        Returns:
            List of relevant memory entries
        """
        if not self.config.enable_memory:
            return []

        return await self.storage.retrieve_memories(
            query,
            agent_id=agent_id,
            limit=limit,
            min_salience=self.config.similarity_threshold
        )

    async def add_memory(
        self,
        content: str,
        confidence: float = 1.0,
        agent_id: Optional[str] = None,
        **kwargs
    ) -> MemoryEntry:
        """
        Manually add a memory entry

        Args:
            content: Memory content
            confidence: Confidence score
            agent_id: Optional agent identifier
            **kwargs: Additional memory attributes

        Returns:
            Created memory entry
        """
        memory_id = self._generate_memory_id(content)

        memory = MemoryEntry(
            id=memory_id,
            content=content,
            confidence=confidence,
            salience_score=kwargs.get('salience_score', confidence),
            agent_id=agent_id,
            is_shared=(agent_id is None),
            tags=kwargs.get('tags', []),
            memory_type=kwargs.get('memory_type', 'semantic')
        )

        await self.storage.store_memory(memory, agent_id)
        return memory

    def get_stats(self) -> Dict[str, Any]:
        """Get memory manager statistics"""
        return self.storage.get_memory_stats()
