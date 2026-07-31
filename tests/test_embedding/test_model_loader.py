"""Tests for embedding.model_loader.

`google.genai.Client` construction and `embed_content` calls are mocked
throughout - these tests verify client wiring, error handling, and the
`GeminiEmbeddingModel.encode(...)` adapter's shape/output, not real network
calls to the Gemini API.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from core.constants import DEFAULT_EMBEDDING_MODEL, EMBEDDING_OUTPUT_DIMENSIONALITY
from core.exceptions import EmbeddingError
from embedding import model_loader


class _FakeEmbedding:
    def __init__(self, values: list[float]) -> None:
        self.values = values


class _FakeEmbedContentResponse:
    def __init__(self, embeddings: list[_FakeEmbedding]) -> None:
        self.embeddings = embeddings


class _FakeModels:
    def __init__(self, dimension: int) -> None:
        self._dimension = dimension
        self.calls: list[dict[str, Any]] = []

    def embed_content(self, *, model: str, contents: list[str], config: Any) -> _FakeEmbedContentResponse:
        self.calls.append({"model": model, "contents": contents, "config": config})
        return _FakeEmbedContentResponse(
            [_FakeEmbedding([float(i)] * self._dimension) for i in range(len(contents))]
        )


class _FakeClient:
    def __init__(self, *, api_key: str | None = None, dimension: int = EMBEDDING_OUTPUT_DIMENSIONALITY) -> None:
        self.api_key = api_key
        self.models = _FakeModels(dimension)


class TestActiveModelName:
    def test_returns_the_default_model(self) -> None:
        assert model_loader.active_model_name() == DEFAULT_EMBEDDING_MODEL


class TestLoadEmbeddingModel:
    def test_constructs_client_and_returns_adapter(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(model_loader.settings, "GEMINI_API_KEY", "test-key")
        monkeypatch.setattr(model_loader.genai, "Client", _FakeClient)

        model = model_loader.load_embedding_model()

        assert isinstance(model, model_loader.GeminiEmbeddingModel)
        assert model._model_name == DEFAULT_EMBEDDING_MODEL  # noqa: SLF001 - white-box test
        assert model._output_dimensionality == EMBEDDING_OUTPUT_DIMENSIONALITY  # noqa: SLF001

    def test_uses_explicit_model_name(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(model_loader.settings, "GEMINI_API_KEY", "test-key")
        monkeypatch.setattr(model_loader.genai, "Client", _FakeClient)

        model = model_loader.load_embedding_model("gemini-embedding-2")

        assert model._model_name == "gemini-embedding-2"  # noqa: SLF001

    def test_raises_when_api_key_missing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(model_loader.settings, "GEMINI_API_KEY", None)

        with pytest.raises(EmbeddingError, match="GEMINI_API_KEY"):
            model_loader.load_embedding_model()

    def test_wraps_client_construction_failure_as_embedding_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(model_loader.settings, "GEMINI_API_KEY", "test-key")

        def _raise(*args: object, **kwargs: object) -> None:
            raise RuntimeError("boom")

        monkeypatch.setattr(model_loader.genai, "Client", _raise)

        with pytest.raises(EmbeddingError):
            model_loader.load_embedding_model()


class TestGeminiEmbeddingModelEncode:
    def test_encodes_sentences_into_expected_shape(self) -> None:
        client = _FakeClient(dimension=768)
        model = model_loader.GeminiEmbeddingModel(client, DEFAULT_EMBEDDING_MODEL, 768)

        vectors = model.encode(["def foo(): ...", "def bar(): ..."])

        assert isinstance(vectors, np.ndarray)
        assert vectors.shape == (2, 768)
        assert vectors.dtype == np.float32

    def test_passes_model_and_output_dimensionality_to_embed_content(self) -> None:
        client = _FakeClient(dimension=768)
        model = model_loader.GeminiEmbeddingModel(client, DEFAULT_EMBEDDING_MODEL, 768)

        model.encode(["hello"])

        assert len(client.models.calls) == 1
        call = client.models.calls[0]
        assert call["model"] == DEFAULT_EMBEDDING_MODEL
        assert call["contents"] == ["hello"]
        assert call["config"].output_dimensionality == 768

    def test_ignores_sentence_transformers_style_kwargs(self) -> None:
        client = _FakeClient(dimension=768)
        model = model_loader.GeminiEmbeddingModel(client, DEFAULT_EMBEDDING_MODEL, 768)

        vectors = model.encode(["hello"], convert_to_numpy=True, show_progress_bar=False)

        assert vectors.shape == (1, 768)

    def test_propagates_embed_content_failure_unwrapped(self) -> None:
        class _RaisingModels:
            def embed_content(self, **kwargs: object) -> None:
                raise RuntimeError("gemini api exploded")

        class _RaisingClient:
            models = _RaisingModels()

        model = model_loader.GeminiEmbeddingModel(_RaisingClient(), DEFAULT_EMBEDDING_MODEL, 768)

        with pytest.raises(RuntimeError, match="gemini api exploded"):
            model.encode(["hello"])
