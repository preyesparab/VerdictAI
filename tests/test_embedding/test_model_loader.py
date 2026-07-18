"""Tests for embedding.model_loader.

Model construction is mocked throughout - these tests verify the CodeBERT
vs. MiniLM selection/assembly logic, not real model downloads/inference.
"""

from __future__ import annotations

import pytest

from core.constants import DEFAULT_EMBEDDING_MODEL, DEFAULT_MINILM_MODEL
from core.exceptions import EmbeddingError
from embedding import model_loader


class _FakeSentenceTransformer:
    def __init__(self, name_or_path: str | None = None, modules: list | None = None) -> None:
        self.name_or_path = name_or_path
        self.modules = modules


class _FakeTransformer:
    def __init__(self, name: str) -> None:
        self.name = name

    def get_word_embedding_dimension(self) -> int:
        return 768


class _FakePooling:
    def __init__(self, dimension: int, **kwargs: object) -> None:
        self.dimension = dimension


class TestActiveModelName:
    def test_uses_codebert_when_enabled(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(model_loader.settings, "USE_CODEBERT", True)
        assert model_loader.active_model_name() == DEFAULT_EMBEDDING_MODEL

    def test_uses_minilm_when_disabled(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(model_loader.settings, "USE_CODEBERT", False)
        assert model_loader.active_model_name() == DEFAULT_MINILM_MODEL


class TestLoadEmbeddingModel:
    def test_assembles_codebert_from_transformer_and_pooling(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(model_loader.models, "Transformer", _FakeTransformer)
        monkeypatch.setattr(model_loader.models, "Pooling", _FakePooling)
        monkeypatch.setattr(model_loader, "SentenceTransformer", _FakeSentenceTransformer)

        model = model_loader.load_embedding_model(DEFAULT_EMBEDDING_MODEL)

        assert isinstance(model, _FakeSentenceTransformer)
        assert model.name_or_path is None
        assert len(model.modules) == 2
        assert isinstance(model.modules[0], _FakeTransformer)
        assert isinstance(model.modules[1], _FakePooling)

    def test_loads_minilm_directly(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(model_loader, "SentenceTransformer", _FakeSentenceTransformer)

        model = model_loader.load_embedding_model(DEFAULT_MINILM_MODEL)

        assert isinstance(model, _FakeSentenceTransformer)
        assert model.name_or_path == DEFAULT_MINILM_MODEL
        assert model.modules is None

    def test_defaults_to_active_model_name(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(model_loader, "SentenceTransformer", _FakeSentenceTransformer)
        monkeypatch.setattr(model_loader.settings, "USE_CODEBERT", False)

        model = model_loader.load_embedding_model()

        assert model.name_or_path == DEFAULT_MINILM_MODEL

    def test_wraps_construction_failure_as_embedding_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def _raise(*args: object, **kwargs: object) -> None:
            raise RuntimeError("boom")

        monkeypatch.setattr(model_loader, "SentenceTransformer", _raise)

        with pytest.raises(EmbeddingError):
            model_loader.load_embedding_model(DEFAULT_MINILM_MODEL)
