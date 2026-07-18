"""Tests for retrieval.semantic_cache.SemanticCacheManager.

Uses a real (temp-file) DatabaseManager for persistence but a fake
embedding model (`_FakeModel`) - these tests verify cache hit/miss logic,
threshold behavior, overwrite, persistence, and repository isolation, not
real embedding inference.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest

from core.exceptions import RetrievalError
from database.sqlite_client import DatabaseManager
from ingestion.repository_metadata import RepositoryMetadata
from retrieval.semantic_cache import (
    SemanticCacheManager,
    _has_distinguishing_conflict,
    _is_lexically_compatible,
    _lexical_jaccard,
)


def _repository_metadata(owner: str = "acme", name: str = "demo") -> RepositoryMetadata:
    return RepositoryMetadata(
        name=name,
        owner=owner,
        clone_url=f"https://github.com/{owner}/{name}.git",
        default_branch="main",
        local_path=Path(f"/tmp/{owner}_{name}"),
        last_updated=datetime(2026, 1, 1, tzinfo=timezone.utc),
        commit_hash="a" * 40,
    )


class _FakeModel:
    """Deterministic fake: maps exact query text to a hand-picked vector.

    Any query not in `vectors` embeds to an all-zero vector, so its
    cosine similarity to everything cached is 0.0 (a clean miss).
    """

    def __init__(self, vectors: dict[str, np.ndarray]) -> None:
        self._vectors = vectors
        self.encode_calls: list[str] = []

    def encode(self, sentences: list[str], **kwargs: object) -> np.ndarray:
        (query,) = sentences
        self.encode_calls.append(query)
        vector = self._vectors.get(query, np.zeros(4, dtype=np.float32))
        return np.array([vector], dtype=np.float32)


@pytest.fixture
def db(tmp_path: Path) -> DatabaseManager:
    manager = DatabaseManager(db_path=tmp_path / "test.db")
    manager.initialize_database()
    return manager


@pytest.fixture
def repository_id(db: DatabaseManager) -> str:
    return db.store_repository(_repository_metadata())


class TestCacheMiss:
    def test_empty_cache_is_a_miss_without_embedding_the_query(
        self, db: DatabaseManager, repository_id: str
    ) -> None:
        model = _FakeModel({})
        cache = SemanticCacheManager(db, model=model, similarity_threshold=0.95)

        result = cache.lookup(repository_id, "how does auth work")

        assert result is None
        assert model.encode_calls == []  # no cached entries: never even embeds

    def test_dissimilar_query_is_a_miss(self, db: DatabaseManager, repository_id: str) -> None:
        model = _FakeModel(
            {
                "how does auth work": np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
                "what does the parser do": np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32),
            }
        )
        cache = SemanticCacheManager(db, model=model, similarity_threshold=0.95)
        cache.store(repository_id, "how does auth work", "Auth works via...", ["a"])

        result = cache.lookup(repository_id, "what does the parser do")

        assert result is None


class TestCacheHit:
    def test_similar_query_returns_cached_response(self, db: DatabaseManager, repository_id: str) -> None:
        # "how does auth work" / "how does the auth flow work" - real lexical overlap (shares
        # "auth"+"work"), unlike this test's pre-fix placeholder pair ("auth"/"authentication"
        # aren't the same token, so this real pair is also what exercises the new lexical gate
        # correctly rather than accidentally tripping it).
        model = _FakeModel(
            {
                "how does auth work": np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
                "how does the auth flow work": np.array([0.99, 0.01, 0.0, 0.0], dtype=np.float32),
            }
        )
        cache = SemanticCacheManager(db, model=model, similarity_threshold=0.95)
        cache.store(repository_id, "how does auth work", "Auth works via JWT.", ["a", "b"])

        result = cache.lookup(repository_id, "how does the auth flow work")

        assert result is not None
        assert result.response == "Auth works via JWT."
        assert result.retrieved_chunk_ids == ["a", "b"]
        assert result.query == "how does auth work"
        assert result.similarity > 0.95


class TestSimilarityThreshold:
    def test_similarity_just_below_threshold_is_a_miss(self, db: DatabaseManager, repository_id: str) -> None:
        # cos(theta) between [1,0] and [0.9,0.4359] ~= 0.9 - below 0.95.
        model = _FakeModel(
            {
                "cached query": np.array([1.0, 0.0], dtype=np.float32),
                "new query": np.array([0.9, 0.43589], dtype=np.float32),
            }
        )
        cache = SemanticCacheManager(db, model=model, similarity_threshold=0.95)
        cache.store(repository_id, "cached query", "response", [])

        assert cache.lookup(repository_id, "new query") is None

    def test_lower_threshold_accepts_the_same_pair(self, db: DatabaseManager, repository_id: str) -> None:
        # "explain the parser" / "please explain the parser" - real lexical overlap (shares
        # "explain"+"parser"), unlike this test's pre-fix placeholder pair ("cached query"/
        # "new query" only share the word "query", jaccard ~0.33, below the new lexical gate).
        model = _FakeModel(
            {
                "explain the parser": np.array([1.0, 0.0], dtype=np.float32),
                "please explain the parser": np.array([0.9, 0.43589], dtype=np.float32),
            }
        )
        cache = SemanticCacheManager(db, model=model, similarity_threshold=0.8)
        cache.store(repository_id, "explain the parser", "response", [])

        result = cache.lookup(repository_id, "please explain the parser")

        assert result is not None
        assert result.response == "response"

    def test_default_threshold_comes_from_settings(
        self, db: DatabaseManager, repository_id: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import retrieval.semantic_cache as semantic_cache_module

        monkeypatch.setattr(semantic_cache_module.settings, "CACHE_SIMILARITY_THRESHOLD", 0.5)
        model = _FakeModel(
            {
                "explain the parser": np.array([1.0, 0.0], dtype=np.float32),
                "please explain the parser": np.array([0.6, 0.8], dtype=np.float32),  # cos similarity = 0.6
            }
        )
        cache = SemanticCacheManager(db, model=model)
        cache.store(repository_id, "explain the parser", "response", [])

        assert cache.lookup(repository_id, "please explain the parser") is not None


class TestOverwrite:
    def test_storing_the_same_query_again_overwrites_the_cached_response(
        self, db: DatabaseManager, repository_id: str
    ) -> None:
        model = _FakeModel({"q": np.array([1.0, 0.0], dtype=np.float32)})
        cache = SemanticCacheManager(db, model=model, similarity_threshold=0.95)

        first_id = cache.store(repository_id, "q", "first response", ["a"])
        second_id = cache.store(repository_id, "q", "second response", ["b"])

        assert first_id == second_id
        result = cache.lookup(repository_id, "q")
        assert result.response == "second response"
        assert result.retrieved_chunk_ids == ["b"]


class TestPersistence:
    def test_cache_survives_a_new_manager_instance(self, db: DatabaseManager, repository_id: str) -> None:
        model = _FakeModel({"q": np.array([1.0, 0.0], dtype=np.float32)})
        writer = SemanticCacheManager(db, model=model, similarity_threshold=0.95)
        writer.store(repository_id, "q", "cached response", ["a"])

        reader = SemanticCacheManager(db, model=model, similarity_threshold=0.95)
        result = reader.lookup(repository_id, "q")

        assert result is not None
        assert result.response == "cached response"

    def test_clear_removes_entries(self, db: DatabaseManager, repository_id: str) -> None:
        model = _FakeModel({"q": np.array([1.0, 0.0], dtype=np.float32)})
        cache = SemanticCacheManager(db, model=model, similarity_threshold=0.95)
        cache.store(repository_id, "q", "response", [])

        removed = cache.clear(repository_id)

        assert removed == 1
        assert cache.lookup(repository_id, "q") is None


class TestRepositoryIsolation:
    def test_cache_entries_do_not_leak_across_repositories(self, db: DatabaseManager) -> None:
        repo_a = db.store_repository(_repository_metadata(owner="acme", name="a"))
        repo_b = db.store_repository(_repository_metadata(owner="acme", name="b"))
        model = _FakeModel({"q": np.array([1.0, 0.0], dtype=np.float32)})
        cache = SemanticCacheManager(db, model=model, similarity_threshold=0.95)

        cache.store(repo_a, "q", "response for a", [])

        assert cache.lookup(repo_a, "q") is not None
        assert cache.lookup(repo_b, "q") is None


class TestErrorHandling:
    def test_embedding_failure_raises_retrieval_error(self, db: DatabaseManager, repository_id: str) -> None:
        class _RaisingModel:
            def encode(self, sentences: list[str], **kwargs: object) -> None:
                raise RuntimeError("boom")

        cache = SemanticCacheManager(db, model=_RaisingModel())

        with pytest.raises(RetrievalError):
            cache.store(repository_id, "q", "response", [])


class TestLexicalCompatibilityHelpers:
    """`_is_lexically_compatible` and its two independent gates, tested as pure functions.

    Real-bug regression: "What is the backend tech stack?" vs "What is
    the frontend tech stack?" measured at cosine similarity 0.9974
    against the real CodeBERT model (see semantic_cache.py's module
    docstring) - well above any threshold that would also keep genuine
    near-duplicates (measured should-hit range 0.9861-0.9925 sits
    *inside* the should-miss range 0.9807-0.9987). This class is the
    fix: reject on lexical grounds, independent of cosine similarity.
    """

    def test_backend_vs_frontend_is_incompatible(self) -> None:
        assert not _is_lexically_compatible(
            "What is the backend tech stack?", "What is the frontend tech stack?"
        )

    def test_back_end_vs_front_end_phrasing_is_incompatible(self) -> None:
        assert not _is_lexically_compatible(
            "What tech stack is used for the back end?", "What tech stack is used for the front end?"
        )

    def test_register_vs_login_endpoint_is_incompatible(self) -> None:
        assert not _is_lexically_compatible(
            "What does the register endpoint do?", "What does the login endpoint do?"
        )

    def test_genuine_paraphrase_is_compatible(self) -> None:
        assert _is_lexically_compatible(
            "What is the backend tech stack?", "What tech stack does the backend use?"
        )

    def test_exact_repeat_is_compatible(self) -> None:
        assert _is_lexically_compatible("How does login work?", "How does login work?")

    def test_totally_unrelated_topics_are_incompatible(self) -> None:
        assert not _is_lexically_compatible(
            "What is the backend tech stack?", "Where is the database connection configured?"
        )

    def test_broader_query_is_not_a_conflict(self) -> None:
        """One query not mentioning a category at all is not a conflict - it's just less specific."""
        assert not _has_distinguishing_conflict("What is the tech stack used?", "What is the backend tech stack used?")
        assert _is_lexically_compatible("What is the tech stack used?", "What is the backend tech stack used?")

    def test_query_mentioning_both_sides_does_not_conflict_with_either(self) -> None:
        assert not _has_distinguishing_conflict(
            "Compare the backend and frontend tech stacks", "What is the backend tech stack?"
        )

    def test_lexical_jaccard_matches_hand_computed_value(self) -> None:
        # content words: {backend, tech, stack} vs {frontend, tech, stack} -> intersection 2, union 4
        assert _lexical_jaccard("What is the backend tech stack?", "What is the frontend tech stack?") == 0.5

    def test_dev_vs_production_is_a_genuine_conflict(self) -> None:
        assert _has_distinguishing_conflict("the dev environment", "the production environment")
        assert _has_distinguishing_conflict("the development environment", "the production environment")

    def test_dev_does_not_falsely_match_as_a_substring_of_other_words(self) -> None:
        """`\\bdev\\b` must not match inside "devops"/"device" etc. - only the standalone word "dev"."""
        assert not _has_distinguishing_conflict("the devops pipeline is broken", "the production environment")


class TestLookupRejectsHighSimilarityButIncompatible:
    """Integration tests: the new gate wired into `lookup()` itself, not just the pure helpers.

    Fake vectors are deliberately near-identical (cosine similarity well
    above any real threshold) so a pre-fix `lookup()` would return a
    hit - these confirm the real query *text* correctly overrides that.
    """

    def test_backend_frontend_miss_despite_near_identical_embeddings(
        self, db: DatabaseManager, repository_id: str
    ) -> None:
        model = _FakeModel(
            {
                "What is the backend tech stack?": np.array([1.0, 0.0], dtype=np.float32),
                "What is the frontend tech stack?": np.array([0.999, 0.045], dtype=np.float32),  # cos ~0.999
            }
        )
        cache = SemanticCacheManager(db, model=model, similarity_threshold=0.95)
        cache.store(repository_id, "What is the backend tech stack?", "Express/Node backend.", ["a"])

        result = cache.lookup(repository_id, "What is the frontend tech stack?")

        assert result is None

    def test_skips_incompatible_top_match_in_favor_of_compatible_lower_one(
        self, db: DatabaseManager, repository_id: str
    ) -> None:
        model = _FakeModel(
            {
                "What is the backend tech stack?": np.array([1.0, 0.0, 0.0], dtype=np.float32),
                # Higher cosine similarity to the incoming query than the compatible entry below,
                # but lexically incompatible - must be skipped, not returned.
                "What is the frontend tech stack?": np.array([0.999, 0.045, 0.0], dtype=np.float32),
                # Genuine paraphrase - lower raw cosine similarity, but the only compatible entry.
                "What tech stack does the backend use?": np.array([0.98, 0.199, 0.0], dtype=np.float32),
            }
        )
        cache = SemanticCacheManager(db, model=model, similarity_threshold=0.95)
        cache.store(repository_id, "What is the frontend tech stack?", "React frontend.", ["f"])
        cache.store(repository_id, "What tech stack does the backend use?", "Express/Node backend.", ["b"])

        result = cache.lookup(repository_id, "What is the backend tech stack?")

        assert result is not None
        assert result.response == "Express/Node backend."
        assert result.retrieved_chunk_ids == ["b"]

    def test_genuine_near_duplicate_still_hits_with_the_new_gate_active(
        self, db: DatabaseManager, repository_id: str
    ) -> None:
        model = _FakeModel(
            {
                "How does login work?": np.array([1.0, 0.0], dtype=np.float32),
                "How does the login flow work?": np.array([0.99, 0.141], dtype=np.float32),
            }
        )
        cache = SemanticCacheManager(db, model=model, similarity_threshold=0.95)
        cache.store(repository_id, "How does login work?", "Login uses JWT.", ["a"])

        result = cache.lookup(repository_id, "How does the login flow work?")

        assert result is not None
        assert result.response == "Login uses JWT."
