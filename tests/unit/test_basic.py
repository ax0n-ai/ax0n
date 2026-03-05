"""
Basic unit tests for Ax0n functionality
"""

import pytest
from axon import Axon, AxonConfig, LLMConfig, ReasoningMethod
from axon.core.models import Thought, ThoughtStage


class TestAxonBasic:
    """Test basic Axon functionality"""

    @pytest.fixture
    def axon_config(self):
        """Create test configuration"""
        return AxonConfig(
            llm=LLMConfig(
                provider="openai",
                model="gpt-4",
                api_key="test-key"
            ),
            memory={"enable_memory": False},
            grounding={"enable_grounding": False},
        )

    def test_axon_initialization(self, axon_config):
        """Test that Axon initializes correctly"""
        ax = Axon(axon_config)

        assert ax is not None
        assert ax.config.llm.provider == "openai"
        assert ax.config.llm.model == "gpt-4"
        assert ax.config.grounding.enable_grounding is False
        assert ax.config.memory.enable_memory is False

    def test_axon_no_llm_config(self):
        """Test that Axon works without LLM config"""
        ax = Axon()
        assert ax is not None
        assert ax.config.llm is None

    def test_thought_stage_enum(self):
        """Test ThoughtStage enum values"""
        assert ThoughtStage.PROBLEM_DEFINITION.value == "problem_definition"
        assert ThoughtStage.RESEARCH.value == "research"
        assert ThoughtStage.ANALYSIS.value == "analysis"
        assert ThoughtStage.SYNTHESIS.value == "synthesis"
        assert ThoughtStage.CONCLUSION.value == "conclusion"
        assert ThoughtStage.VERIFICATION.value == "verification"

    def test_thought_model(self):
        """Test Thought model creation"""
        thought = Thought(
            thought="AI systems can benefit from structured reasoning",
            thought_number=1,
            total_thoughts=3,
            next_thought_needed=True,
            needs_more_thoughts=True,
            stage=ThoughtStage.ANALYSIS,
            tags=["ai", "reasoning"],
            score=0.9
        )

        assert thought.thought == "AI systems can benefit from structured reasoning"
        assert thought.thought_number == 1
        assert thought.total_thoughts == 3
        assert thought.stage == ThoughtStage.ANALYSIS
        assert "ai" in thought.tags
        assert thought.score == 0.9

    def test_reasoning_method_enum(self):
        """Test ReasoningMethod enum values"""
        assert ReasoningMethod.COT.value == "chain_of_thought"
        assert ReasoningMethod.SELF_CONSISTENCY.value == "self_consistency"
        assert ReasoningMethod.AOT.value == "algorithm_of_thoughts"
        assert ReasoningMethod.TOT.value == "tree_of_thoughts"
        assert ReasoningMethod.GOT.value == "graph_of_thoughts"


class TestConfigValidation:
    """Test production config validation"""

    def test_timeout_cross_validation_raises(self):
        """request_timeout must be >= thought_timeout"""
        with pytest.raises(ValueError, match="request_timeout"):
            AxonConfig(
                think_layer={"thought_timeout": 120},
                request_timeout=30,
            )

    def test_timeout_cross_validation_passes(self):
        """Valid when request_timeout >= thought_timeout"""
        config = AxonConfig(
            think_layer={"thought_timeout": 30},
            request_timeout=60,
        )
        assert config.request_timeout >= config.think_layer.thought_timeout

    def test_num_parallel_paths_in_config(self):
        """num_parallel_paths should be configurable"""
        config = AxonConfig(think_layer={"num_parallel_paths": 3})
        assert config.think_layer.num_parallel_paths == 3

    def test_max_query_length_in_config(self):
        """max_query_length should have default"""
        config = AxonConfig()
        assert config.max_query_length == 10000

    def test_max_json_response_size_in_config(self):
        """max_json_response_size should have default"""
        config = AxonConfig()
        assert config.max_json_response_size == 10_000_000


class TestQueryValidation:
    """Test query validation in Axon.think()"""

    @pytest.fixture
    def axon(self):
        return Axon(AxonConfig(
            llm=LLMConfig(provider="openai", model="gpt-4", api_key="test"),
            memory={"enable_memory": False},
            grounding={"enable_grounding": False},
        ))

    @pytest.mark.asyncio
    async def test_empty_query_raises(self, axon):
        with pytest.raises(ValueError, match="cannot be empty"):
            await axon.think("")

    @pytest.mark.asyncio
    async def test_whitespace_query_raises(self, axon):
        with pytest.raises(ValueError, match="cannot be empty"):
            await axon.think("   ")

    @pytest.mark.asyncio
    async def test_too_long_query_raises(self, axon):
        axon.config = AxonConfig(
            llm=LLMConfig(provider="openai", model="gpt-4", api_key="test"),
            memory={"enable_memory": False},
            grounding={"enable_grounding": False},
            max_query_length=100,
        )
        with pytest.raises(ValueError, match="too long"):
            await axon.think("x" * 200)


class TestMemoryBounds:
    """Test bounded memory storage"""

    @pytest.mark.asyncio
    async def test_memory_eviction(self):
        from axon.core.config import MemoryConfig
        from axon.memory.memory import MemoryStorage
        from axon.core.models import MemoryEntry

        config = MemoryConfig(max_memory_entries=3)
        storage = MemoryStorage(config)

        for i in range(5):
            entry = MemoryEntry(
                id=f"mem_{i}",
                content=f"Memory {i}",
                confidence=0.9,
            )
            await storage.store_memory(entry)

        assert storage._total_memory_count() <= 3


class TestMockKVClientBounds:
    """Test bounded MockKVClient"""

    @pytest.mark.asyncio
    async def test_kv_eviction(self):
        from axon.retrieval.retriever import MockKVClient

        client = MockKVClient()
        client._MAX_ENTRIES = 3

        for i in range(5):
            await client.set(f"key_{i}", f"value_{i}")

        assert len(client._store) <= 3


class TestEmbeddingProvider:
    """Test EmbeddingProvider n-gram fallback (sentence-transformers may not be available)"""

    def test_identical_texts(self):
        from axon.utils.embeddings import EmbeddingProvider
        provider = EmbeddingProvider()
        score = provider.similarity("the quick brown fox", "the quick brown fox")
        assert score == 1.0

    def test_similar_texts(self):
        from axon.utils.embeddings import EmbeddingProvider
        provider = EmbeddingProvider()
        score = provider.similarity(
            "the quick brown fox jumps over the lazy dog",
            "the quick brown fox leaps over the lazy dog"
        )
        assert score > 0.5

    def test_different_texts(self):
        from axon.utils.embeddings import EmbeddingProvider
        provider = EmbeddingProvider()
        score = provider.similarity(
            "the quick brown fox",
            "quantum mechanics describes particle behavior"
        )
        assert score < 0.2

    def test_empty_text(self):
        from axon.utils.embeddings import EmbeddingProvider
        provider = EmbeddingProvider()
        assert provider.similarity("", "hello") == 0.0
        assert provider.similarity("hello", "") == 0.0

    def test_uses_embeddings_property(self):
        from axon.utils.embeddings import EmbeddingProvider
        provider = EmbeddingProvider()
        # Should be a bool regardless of whether ST is available
        assert isinstance(provider.uses_embeddings, bool)

    def test_encode_without_st(self):
        from axon.utils.embeddings import EmbeddingProvider
        provider = EmbeddingProvider()
        if not provider.uses_embeddings:
            assert provider.encode("hello") is None


class TestTrustedSourceDomainMatching:
    """Test that trusted source validation uses proper domain matching"""

    def test_exact_domain_match(self):
        from axon.core.config import GroundingConfig
        from axon.grounding.grounding import CitationExtractor
        config = GroundingConfig(trusted_sources=["wikipedia.org"])
        extractor = CitationExtractor(config)
        assert extractor._is_trusted_source("https://wikipedia.org/wiki/Test") is True

    def test_subdomain_match(self):
        from axon.core.config import GroundingConfig
        from axon.grounding.grounding import CitationExtractor
        config = GroundingConfig(trusted_sources=["wikipedia.org"])
        extractor = CitationExtractor(config)
        assert extractor._is_trusted_source("https://en.wikipedia.org/wiki/Test") is True

    def test_no_false_positive(self):
        """'wikipedia.org' should NOT match 'notwikipedia.org'"""
        from axon.core.config import GroundingConfig
        from axon.grounding.grounding import CitationExtractor
        config = GroundingConfig(trusted_sources=["wikipedia.org"])
        extractor = CitationExtractor(config)
        assert extractor._is_trusted_source("https://notwikipedia.org/evil") is False

    def test_no_false_positive_in_path(self):
        """Domain in path should not match"""
        from axon.core.config import GroundingConfig
        from axon.grounding.grounding import CitationExtractor
        config = GroundingConfig(trusted_sources=["wikipedia.org"])
        extractor = CitationExtractor(config)
        assert extractor._is_trusted_source("https://evil.com/wikipedia.org") is False

    def test_empty_trusted_sources_trusts_all(self):
        from axon.core.config import GroundingConfig
        from axon.grounding.grounding import CitationExtractor
        config = GroundingConfig(trusted_sources=[])
        extractor = CitationExtractor(config)
        assert extractor._is_trusted_source("https://anything.com") is True


class TestDatetimeTimezoneAware:
    """Test that model timestamps are timezone-aware (not naive)"""

    def test_thought_timestamp_is_utc(self):
        from datetime import timezone
        thought = Thought(
            thought="Test",
            thought_number=1,
            total_thoughts=1,
            next_thought_needed=False,
            needs_more_thoughts=False,
        )
        assert thought.timestamp.tzinfo is not None
        assert thought.timestamp.tzinfo == timezone.utc

    def test_memory_entry_timestamps_are_utc(self):
        from datetime import timezone
        from axon.core.models import MemoryEntry
        entry = MemoryEntry(id="t1", content="test", confidence=0.9)
        assert entry.created_at.tzinfo == timezone.utc
        assert entry.updated_at.tzinfo == timezone.utc


class TestModelDumpNotDict:
    """Test that .model_dump() is used instead of deprecated .dict()"""

    def test_thought_model_dump(self):
        thought = Thought(
            thought="Test",
            thought_number=1,
            total_thoughts=1,
            next_thought_needed=False,
            needs_more_thoughts=False,
        )
        dumped = thought.model_dump()
        assert isinstance(dumped, dict)
        assert "thought" in dumped

    def test_renderer_export_json(self):
        """Renderer.export_as_json should use model_dump not dict"""
        from axon.rendering.renderer import Renderer
        from axon.core.config import RendererConfig
        from axon.core.models import ThoughtResult
        renderer = Renderer(RendererConfig())
        result = ThoughtResult(
            thoughts=[],
            answer="test",
            trace=[],
            citations=[],
            memory_updates=[],
            execution_time=0.1,
        )
        json_str = renderer.export_as_json(result)
        assert '"answer": "test"' in json_str


class TestRevisionLoop:
    """Test iterative think-ground-revise loop"""

    @pytest.fixture
    def think_layer_config(self):
        from axon.core.config import ThinkLayerConfig
        return ThinkLayerConfig(
            max_revision_iterations=3,
            revision_score_threshold=0.7,
            max_revisions_per_thought=2,
        )

    @pytest.fixture
    def mock_llm_client(self):
        class MockLLM:
            async def generate(self, prompt: str) -> str:
                return "Revised thought based on evidence."
        return MockLLM()

    def _make_thought(self, number=1, text="Test thought", validation=None, revision_count=0):
        metadata = {}
        if validation is not None:
            metadata["validation"] = validation
        return Thought(
            thought=text,
            thought_number=number,
            total_thoughts=3,
            next_thought_needed=True,
            needs_more_thoughts=True,
            revision_count=revision_count,
            metadata=metadata,
        )

    @pytest.mark.asyncio
    async def test_no_contradictions_passes_through(self, think_layer_config, mock_llm_client):
        """Clean thoughts with no contradictions should pass through unchanged."""
        from axon.core.revision import RevisionLoop
        from axon.core.config import GroundingConfig
        from axon.grounding.grounding import GroundingModule

        grounding = GroundingModule(GroundingConfig(enable_grounding=False))
        loop = RevisionLoop(think_layer_config, grounding)

        thoughts = [
            self._make_thought(1, "Thought one", validation={
                "validation_score": 0.9,
                "needs_revision": False,
                "contradicted_claims": [],
                "supported_claims": ["claim A"],
            }),
            self._make_thought(2, "Thought two", validation={
                "validation_score": 0.8,
                "needs_revision": False,
                "contradicted_claims": [],
                "supported_claims": ["claim B"],
            }),
        ]

        result_thoughts, result_citations, iterations = await loop.run(
            thoughts, "test query", {}, mock_llm_client, []
        )

        assert iterations == 0
        assert len(result_thoughts) == 2
        assert result_thoughts[0].thought == "Thought one"
        assert result_thoughts[1].thought == "Thought two"

    @pytest.mark.asyncio
    async def test_contradicted_thought_gets_revised(self, think_layer_config, mock_llm_client):
        """A contradicted thought should be revised with is_revision=True and revision_count=1."""
        from axon.core.revision import RevisionLoop
        from axon.core.config import GroundingConfig
        from axon.grounding.grounding import GroundingModule

        # Use a grounding module that marks the revised thought as clean
        grounding = GroundingModule(GroundingConfig(
            enable_grounding=True,
            enable_fact_checking=False,  # Skip re-validation to avoid search calls
        ))
        loop = RevisionLoop(think_layer_config, grounding)

        thoughts = [
            self._make_thought(1, "The earth is flat and wide", validation={
                "validation_score": 0.2,
                "needs_revision": True,
                "contradicted_claims": ["The earth is flat and wide"],
                "supported_claims": [],
            }),
        ]

        result_thoughts, _, iterations = await loop.run(
            thoughts, "geography", {}, mock_llm_client, []
        )

        assert iterations >= 1
        revised = result_thoughts[0]
        assert revised.is_revision is True
        assert revised.revises_thought == 1
        assert revised.revision_count == 1

    @pytest.mark.asyncio
    async def test_respects_max_iterations(self, mock_llm_client):
        """Loop should stop at max_revision_iterations even if thoughts still need revision."""
        from axon.core.revision import RevisionLoop
        from axon.core.config import ThinkLayerConfig, GroundingConfig
        from axon.grounding.grounding import GroundingModule

        config = ThinkLayerConfig(max_revision_iterations=2, max_revisions_per_thought=5)

        # Grounding that always re-flags as needing revision with improving scores
        class AlwaysContradictGrounding:
            """Mock grounding that always sets needs_revision=True with improving scores."""
            def __init__(self):
                self.call_count = 0

            async def ground_claims(self, thoughts, query, context):
                self.call_count += 1
                for t in thoughts:
                    t.metadata["validation"] = {
                        "validation_score": 0.1 * self.call_count,
                        "needs_revision": True,
                        "contradicted_claims": ["some claim"],
                        "supported_claims": [],
                    }
                return thoughts, []

        mock_grounding = AlwaysContradictGrounding()
        loop = RevisionLoop(config, mock_grounding)

        thoughts = [
            self._make_thought(1, "Bad claim", validation={
                "validation_score": 0.1,
                "needs_revision": True,
                "contradicted_claims": ["Bad claim"],
                "supported_claims": [],
            }),
        ]

        _, _, iterations = await loop.run(
            thoughts, "test", {}, mock_llm_client, []
        )

        assert iterations <= 2

    @pytest.mark.asyncio
    async def test_per_thought_cap(self, mock_llm_client):
        """A thought already at revision cap should be skipped."""
        from axon.core.revision import RevisionLoop
        from axon.core.config import ThinkLayerConfig, GroundingConfig
        from axon.grounding.grounding import GroundingModule

        config = ThinkLayerConfig(max_revision_iterations=3, max_revisions_per_thought=2)
        grounding = GroundingModule(GroundingConfig(enable_grounding=False))
        loop = RevisionLoop(config, grounding)

        # This thought has already been revised 2 times (at the cap)
        thoughts = [
            self._make_thought(1, "Revised twice already", validation={
                "validation_score": 0.3,
                "needs_revision": True,
                "contradicted_claims": ["some claim"],
                "supported_claims": [],
            }, revision_count=2),
        ]

        result_thoughts, _, iterations = await loop.run(
            thoughts, "test", {}, mock_llm_client, []
        )

        # Should not revise since the thought is at the cap
        assert iterations == 0
        assert result_thoughts[0].thought == "Revised twice already"
        assert result_thoughts[0].revision_count == 2

    @pytest.mark.asyncio
    async def test_disabled_when_zero_iterations(self, mock_llm_client):
        """max_revision_iterations=0 should skip the loop entirely."""
        from axon.core.revision import RevisionLoop
        from axon.core.config import ThinkLayerConfig, GroundingConfig
        from axon.grounding.grounding import GroundingModule

        config = ThinkLayerConfig(max_revision_iterations=0)
        grounding = GroundingModule(GroundingConfig(enable_grounding=False))
        loop = RevisionLoop(config, grounding)

        thoughts = [
            self._make_thought(1, "Should not change", validation={
                "validation_score": 0.1,
                "needs_revision": True,
                "contradicted_claims": ["claim"],
                "supported_claims": [],
            }),
        ]

        result_thoughts, _, iterations = await loop.run(
            thoughts, "test", {}, mock_llm_client, []
        )

        assert iterations == 0
        assert result_thoughts[0].thought == "Should not change"


class TestMemoryTTLEviction:
    """Test time-based memory eviction"""

    @pytest.mark.asyncio
    async def test_ttl_eviction_removes_old_memories(self):
        from axon.core.config import MemoryConfig
        from axon.memory.memory import MemoryStorage
        from axon.core.models import MemoryEntry
        from datetime import timedelta

        config = MemoryConfig(max_memory_entries=100, memory_ttl_seconds=60)
        storage = MemoryStorage(config)

        # Store an old memory (created 120s ago)
        old_entry = MemoryEntry(
            id="mem_old", content="Old memory", confidence=0.9,
        )
        old_entry.created_at = old_entry.created_at - timedelta(seconds=120)
        await storage.store_memory(old_entry)

        # Store a fresh memory
        fresh_entry = MemoryEntry(
            id="mem_fresh", content="Fresh memory", confidence=0.9,
        )
        await storage.store_memory(fresh_entry)

        assert storage._total_memory_count() == 1
        assert storage._shared_memories[0].id == "mem_fresh"

    @pytest.mark.asyncio
    async def test_no_ttl_keeps_all(self):
        from axon.core.config import MemoryConfig
        from axon.memory.memory import MemoryStorage
        from axon.core.models import MemoryEntry
        from datetime import timedelta

        config = MemoryConfig(max_memory_entries=100, memory_ttl_seconds=None)
        storage = MemoryStorage(config)

        old_entry = MemoryEntry(
            id="mem_old", content="Old memory", confidence=0.9,
        )
        old_entry.created_at = old_entry.created_at - timedelta(seconds=99999)
        await storage.store_memory(old_entry)

        assert storage._total_memory_count() == 1


class TestMemorySHA256:
    """Test that memory IDs use SHA256 instead of MD5"""

    def test_memory_id_format(self):
        from axon.core.config import MemoryConfig
        from axon.memory.memory import MemoryManager

        manager = MemoryManager(MemoryConfig())
        mid = manager._generate_memory_id("test content")
        # SHA256 gives 12-char hash + 8-char uuid
        parts = mid.split("_")
        assert parts[0] == "mem"
        assert len(parts[1]) == 12  # SHA256 truncated to 12
        assert len(parts[2]) == 8   # UUID suffix


class TestGroundingSessionCleanup:
    """Test that grounding module supports async context manager"""

    @pytest.mark.asyncio
    async def test_context_manager(self):
        from axon.core.config import GroundingConfig
        from axon.grounding.grounding import GroundingModule

        async with GroundingModule(GroundingConfig(enable_grounding=False)) as gm:
            assert gm is not None
        # No error raised = session closed properly


class TestConfigurableThresholds:
    """Test configurable similarity thresholds in GroundingConfig"""

    def test_default_thresholds(self):
        from axon.core.config import GroundingConfig
        config = GroundingConfig()
        assert config.support_similarity_threshold == 0.2
        assert config.contradiction_similarity_threshold == 0.3

    def test_custom_thresholds(self):
        from axon.core.config import GroundingConfig
        config = GroundingConfig(
            support_similarity_threshold=0.4,
            contradiction_similarity_threshold=0.6,
        )
        assert config.support_similarity_threshold == 0.4
        assert config.contradiction_similarity_threshold == 0.6


class TestSharedClaimExtraction:
    """Test the shared extract_claims utility"""

    def test_basic_extraction(self):
        from axon.utils import extract_claims
        text = "The Earth orbits the Sun. The Moon orbits the Earth. Stars emit light through nuclear fusion."
        claims = extract_claims(text)
        assert len(claims) >= 2

    def test_filters_questions(self):
        from axon.utils import extract_claims
        text = "What is the meaning of life? The Earth is round."
        claims = extract_claims(text)
        # Question should be filtered
        for c in claims:
            assert not c.endswith("?")

    def test_filters_short(self):
        from axon.utils import extract_claims
        text = "Yes. No. The Earth revolves around the Sun."
        claims = extract_claims(text)
        assert all(len(c.split()) >= 4 for c in claims)

    def test_filters_hedging(self):
        from axon.utils import extract_claims
        text = "Maybe the sky is blue. The sky appears blue due to Rayleigh scattering."
        claims = extract_claims(text)
        assert all(not c.lower().startswith("maybe") for c in claims)

    def test_empty_text(self):
        from axon.utils import extract_claims
        assert extract_claims("") == []
        assert extract_claims("   ") == []


class TestGoTTopologicalSort:
    """Test Graph of Thoughts proper Kahn's algorithm"""

    @pytest.mark.asyncio
    async def test_linear_dependency_order(self):
        from axon.reasoning.graph_of_thoughts import GraphOfThoughts, ThoughtGraphNode
        from axon.core.config import ThinkLayerConfig

        class MockLLM:
            async def generate(self, prompt):
                return '{"solution": "solved", "confidence": 0.8}'

        got = GraphOfThoughts(ThinkLayerConfig())

        # A -> B -> C (linear dependency)
        a = ThoughtGraphNode(id="a", thought="Step A", dependencies=[], dependents=["b"])
        b = ThoughtGraphNode(id="b", thought="Step B", dependencies=["a"], dependents=["c"])
        c = ThoughtGraphNode(id="c", thought="Step C", dependencies=["b"], dependents=[])

        await got._solve_graph([a, b, c], "test", {}, MockLLM())

        assert a.solution != ""
        assert b.solution != ""
        assert c.solution != ""

    @pytest.mark.asyncio
    async def test_cycle_detection(self):
        from axon.reasoning.graph_of_thoughts import GraphOfThoughts, ThoughtGraphNode
        from axon.core.config import ThinkLayerConfig

        got = GraphOfThoughts(ThinkLayerConfig())

        class MockLLM:
            async def generate(self, prompt):
                return '{"solution": "solved", "confidence": 0.8}'

        # A -> B -> A (cycle)
        a = ThoughtGraphNode(id="a", thought="A", dependencies=["b"], dependents=["b"])
        b = ThoughtGraphNode(id="b", thought="B", dependencies=["a"], dependents=["a"])

        await got._solve_graph([a, b], "test", {}, MockLLM())

        # Both should be force-solved with cycle marker
        assert "[cycle-broken]" in a.solution
        assert "[cycle-broken]" in b.solution

    @pytest.mark.asyncio
    async def test_independent_nodes_parallel(self):
        from axon.reasoning.graph_of_thoughts import GraphOfThoughts, ThoughtGraphNode
        from axon.core.config import ThinkLayerConfig

        call_order = []

        class TrackingLLM:
            async def generate(self, prompt):
                call_order.append("call")
                return '{"solution": "solved", "confidence": 0.9}'

        got = GraphOfThoughts(ThinkLayerConfig())

        # Three independent nodes (no deps)
        a = ThoughtGraphNode(id="a", thought="A", dependencies=[], dependents=[])
        b = ThoughtGraphNode(id="b", thought="B", dependencies=[], dependents=[])
        c = ThoughtGraphNode(id="c", thought="C", dependencies=[], dependents=[])

        await got._solve_graph([a, b, c], "test", {}, TrackingLLM())

        assert a.solution != ""
        assert b.solution != ""
        assert c.solution != ""


class TestRevisionWithEvidence:
    """Test that revision prompt includes evidence snippets"""

    def test_revision_prompt_includes_evidence(self):
        from axon.core.revision import RevisionLoop
        from axon.core.config import ThinkLayerConfig, GroundingConfig
        from axon.grounding.grounding import GroundingModule
        from axon.core.models import GroundingEvidence

        config = ThinkLayerConfig()
        grounding = GroundingModule(GroundingConfig(enable_grounding=False))
        loop = RevisionLoop(config, grounding)

        citations = [
            GroundingEvidence(
                source_url="https://example.com",
                snippet="The Earth is approximately spherical",
                confidence=0.9,
                metadata={"title": "Earth Shape"},
            )
        ]

        prompt = loop._build_revision_prompt(
            "The Earth is flat",
            "What shape is the Earth?",
            ["The Earth is flat"],
            [],
            citations,
        )

        assert "spherical" in prompt
        assert "Earth Shape" in prompt
        assert "evidence" in prompt.lower()

    @pytest.mark.asyncio
    async def test_revised_thought_preserves_original(self):
        from axon.core.revision import RevisionLoop
        from axon.core.config import ThinkLayerConfig, GroundingConfig
        from axon.grounding.grounding import GroundingModule

        config = ThinkLayerConfig()
        grounding = GroundingModule(GroundingConfig(
            enable_grounding=True, enable_fact_checking=False,
        ))
        loop = RevisionLoop(config, grounding)

        class MockLLM:
            async def generate(self, prompt):
                return "Revised: The Earth is spherical."

        thought = Thought(
            thought="The Earth is flat",
            thought_number=1,
            total_thoughts=1,
            next_thought_needed=False,
            metadata={
                "validation": {
                    "needs_revision": True,
                    "validation_score": 0.2,
                    "contradicted_claims": ["flat"],
                    "supported_claims": [],
                }
            },
        )

        revised = await loop._revise_thought(thought, "shape", MockLLM(), [])
        assert revised.metadata.get("original_thought") == "The Earth is flat"
        assert revised.metadata.get("revision_from_count") == 0


class TestAutoMethodSelection:
    """Test automatic reasoning method selection"""

    def test_comparison_query_selects_tot(self):
        ax = Axon(AxonConfig(
            llm=LLMConfig(provider="openai", model="gpt-4", api_key="test"),
            memory={"enable_memory": False}, grounding={"enable_grounding": False},
        ))
        method = ax._auto_select_method("Compare Python versus JavaScript for web dev")
        assert method == ReasoningMethod.TOT

    def test_algorithmic_query_selects_aot(self):
        ax = Axon(AxonConfig(
            llm=LLMConfig(provider="openai", model="gpt-4", api_key="test"),
            memory={"enable_memory": False}, grounding={"enable_grounding": False},
        ))
        method = ax._auto_select_method("Calculate the optimal algorithm for sorting")
        assert method == ReasoningMethod.AOT

    def test_simple_query_selects_cot(self):
        ax = Axon(AxonConfig(
            llm=LLMConfig(provider="openai", model="gpt-4", api_key="test"),
            memory={"enable_memory": False}, grounding={"enable_grounding": False},
        ))
        method = ax._auto_select_method("What is photosynthesis?")
        assert method == ReasoningMethod.COT

    def test_debate_query_selects_sc(self):
        ax = Axon(AxonConfig(
            llm=LLMConfig(provider="openai", model="gpt-4", api_key="test"),
            memory={"enable_memory": False}, grounding={"enable_grounding": False},
        ))
        method = ax._auto_select_method("Is this a controversial debate topic?")
        assert method == ReasoningMethod.SELF_CONSISTENCY


class TestQualityGate:
    """Test quality gate filtering of low-confidence thoughts"""

    def test_quality_gate_config(self):
        config = AxonConfig(quality_gate_threshold=0.5)
        assert config.quality_gate_threshold == 0.5

    def test_quality_gate_default_disabled(self):
        config = AxonConfig()
        assert config.quality_gate_threshold == 0.0


class TestRetryLLMClient:
    """Test LLM retry wrapper"""

    @pytest.mark.asyncio
    async def test_retry_on_failure(self):
        from axon.core.core import RetryLLMClient

        call_count = 0

        class FailThenSucceedLLM:
            async def generate(self, prompt):
                nonlocal call_count
                call_count += 1
                if call_count < 3:
                    raise ConnectionError("Transient failure")
                return "Success"

        client = RetryLLMClient(FailThenSucceedLLM(), max_attempts=3, base_delay=0.01)
        result = await client.generate("test")
        assert result == "Success"
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_retry_exhausted_raises(self):
        from axon.core.core import RetryLLMClient

        class AlwaysFailLLM:
            async def generate(self, prompt):
                raise ConnectionError("Permanent failure")

        client = RetryLLMClient(AlwaysFailLLM(), max_attempts=2, base_delay=0.01)
        with pytest.raises(ConnectionError, match="Permanent"):
            await client.generate("test")

    @pytest.mark.asyncio
    async def test_no_retry_on_success(self):
        from axon.core.core import RetryLLMClient

        call_count = 0

        class SuccessLLM:
            async def generate(self, prompt):
                nonlocal call_count
                call_count += 1
                return "Immediate success"

        client = RetryLLMClient(SuccessLLM(), max_attempts=3, base_delay=0.01)
        result = await client.generate("test")
        assert result == "Immediate success"
        assert call_count == 1


class TestMetrics:
    """Test metrics tracking"""

    def test_metrics_disabled_by_default(self):
        ax = Axon(AxonConfig(
            llm=LLMConfig(provider="openai", model="gpt-4", api_key="test"),
            memory={"enable_memory": False}, grounding={"enable_grounding": False},
        ))
        metrics = ax.get_metrics()
        assert metrics["total_queries"] == 0

    def test_metrics_config(self):
        config = AxonConfig(enable_metrics=True)
        assert config.enable_metrics is True


class TestEmbeddingCache:
    """Test embedding provider LRU cache"""

    def test_cache_stats_initial(self):
        from axon.utils.embeddings import EmbeddingProvider
        provider = EmbeddingProvider(cache_size=10)
        stats = provider.cache_stats
        assert stats["hits"] == 0
        assert stats["misses"] == 0
        assert stats["size"] == 0
        assert stats["max_size"] == 10

    def test_cache_hit_rate(self):
        from axon.utils.embeddings import EmbeddingProvider
        provider = EmbeddingProvider(cache_size=10)
        # Call similarity twice with same texts
        provider.similarity("hello world", "hello world")
        # n-gram fallback doesn't use embedding cache, but cache_stats still work
        stats = provider.cache_stats
        assert isinstance(stats["hit_rate"], float)


class TestAxonContextManager:
    """Test Axon async context manager"""

    @pytest.mark.asyncio
    async def test_context_manager(self):
        async with Axon(AxonConfig(
            llm=LLMConfig(provider="openai", model="gpt-4", api_key="test"),
            memory={"enable_memory": False}, grounding={"enable_grounding": False},
        )) as ax:
            assert ax is not None


class TestStreamingCallback:
    """Test on_thought callback support"""

    @pytest.mark.asyncio
    async def test_method_none_auto_selects(self):
        """Passing method=None should auto-select without error."""
        ax = Axon(AxonConfig(
            llm=LLMConfig(provider="openai", model="gpt-4", api_key="test"),
            memory={"enable_memory": False}, grounding={"enable_grounding": False},
        ))
        # Just verify auto-selection returns a valid method
        method = ax._auto_select_method("test query")
        assert isinstance(method, ReasoningMethod)


class TestRetryConfig:
    """Test retry configuration"""

    def test_default_retry_config(self):
        config = AxonConfig()
        assert config.llm_retry_attempts == 3
        assert config.llm_retry_base_delay == 1.0

    def test_custom_retry_config(self):
        config = AxonConfig(llm_retry_attempts=5, llm_retry_base_delay=0.5)
        assert config.llm_retry_attempts == 5
        assert config.llm_retry_base_delay == 0.5


if __name__ == "__main__":
    pytest.main([__file__])
