"""
Tests for memory module improvements
"""

import os
import json
import tempfile
import pytest

from axon.core.config import MemoryConfig
from axon.core.models import MemoryEntry, Thought, ThoughtStage
from axon.memory.memory import (
    MemoryExtractor,
    MemoryDeduplicator,
    MemoryStorage,
    MemoryManager,
)


class TestMemoryExtractor:
    """Test improved fact extraction"""

    @pytest.fixture
    def extractor(self):
        return MemoryExtractor(MemoryConfig())

    def test_extract_multiple_sentences(self, extractor):
        thought = Thought(
            thought="Machine learning is a subset of artificial intelligence. It uses statistical methods to learn. Deep learning extends this further with neural networks.",
            thought_number=1,
            total_thoughts=1,
            score=0.9,
        )
        facts = extractor._extract_facts_from_thought(thought)
        assert len(facts) >= 2

    def test_skip_questions(self, extractor):
        thought = Thought(
            thought="What is AI? This is a complex topic that requires careful analysis.",
            thought_number=1,
            total_thoughts=1,
            score=0.9,
        )
        facts = extractor._extract_facts_from_thought(thought)
        # Should not include the question
        for f in facts:
            assert not f.endswith("?")

    def test_skip_short_statements(self, extractor):
        thought = Thought(
            thought="Yes. No. OK. This is a meaningful statement about reasoning systems.",
            thought_number=1,
            total_thoughts=1,
            score=0.9,
        )
        facts = extractor._extract_facts_from_thought(thought)
        for f in facts:
            assert len(f.split()) >= 4


class TestMemoryDeduplicator:
    """Test improved similarity computation"""

    @pytest.fixture
    def dedup(self):
        return MemoryDeduplicator(MemoryConfig())

    def test_identical_strings(self, dedup):
        sim = dedup.compute_similarity("the cat sat on the mat", "the cat sat on the mat")
        assert sim == 1.0

    def test_similar_strings(self, dedup):
        sim = dedup.compute_similarity("the cat sat on the mat", "the cat sat on a mat")
        assert sim > 0.5

    def test_different_strings(self, dedup):
        sim = dedup.compute_similarity("hello world", "completely different text here")
        assert sim < 0.3

    def test_empty_strings(self, dedup):
        assert dedup.compute_similarity("", "hello") == 0.0
        assert dedup.compute_similarity("hello", "") == 0.0
        assert dedup.compute_similarity("", "") == 0.0


class TestMemoryPersistence:
    """Test save_to_file / load_from_file"""

    @pytest.fixture
    def storage(self):
        return MemoryStorage(MemoryConfig())

    @pytest.mark.asyncio
    async def test_save_and_load(self, storage):
        # Store a memory
        mem = MemoryEntry(
            id="test-1",
            content="Test memory content",
            confidence=0.9,
            salience_score=0.8,
        )
        await storage.store_memory(mem)

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name

        try:
            assert storage.save_to_file(path) is True
            assert os.path.exists(path)

            # Load into a new storage
            new_storage = MemoryStorage(MemoryConfig())
            assert new_storage.load_from_file(path) is True

            stats = new_storage.get_memory_stats()
            assert stats['total_shared_memories'] == 1
        finally:
            os.unlink(path)

    def test_load_nonexistent(self, storage):
        assert storage.load_from_file("/nonexistent/path.json") is False


if __name__ == "__main__":
    pytest.main([__file__])
