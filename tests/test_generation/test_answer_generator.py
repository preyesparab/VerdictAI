"""Tests for generation.answer_generator.LLMService and match_citations.

`_FakeLLMClient`/`_FakeSemanticCache` stand in for
`generation.llm_client.LLMClient`/`retrieval.semantic_cache.SemanticCacheManager`
- these tests verify prompt construction, citation matching, cache
updates, and the empty/insufficient-context paths, never a real LLM call
or database.
"""

from __future__ import annotations

import pytest

from core.exceptions import RetrievalError
from generation.answer_generator import (
    INSUFFICIENT_CONTEXT_MESSAGE,
    LLMService,
    match_citations,
)
from generation.llm_client import LLMCompletion
from models.schemas import ChunkCitation, ContextDocument
from prompts.templates import SYSTEM_PROMPT, build_prompt


def _citation(chunk_id: str, file_path: str, function_name: str | None, class_name: str | None = None) -> ChunkCitation:
    return ChunkCitation(chunk_id=chunk_id, file_path=file_path, function_name=function_name, class_name=class_name)


def _context_document(
    context: str = "Repository: repo-1\nFile Path: src/a.py\n...",
    chunk_citations: list[ChunkCitation] | None = None,
    query: str = "how does auth work",
) -> ContextDocument:
    citations = chunk_citations if chunk_citations is not None else [_citation("a", "src/a.py", "authenticate")]
    return ContextDocument(
        repository_id="repo-1",
        query=query,
        context=context,
        included_chunks=[c.chunk_id for c in citations],
        chunk_citations=citations,
        total_tokens=10,
        truncated=False,
    )


class _FakeLLMClient:
    def __init__(self, completion: LLMCompletion | None = None, raise_error: Exception | None = None) -> None:
        self.completion = completion or LLMCompletion(
            text="default answer", prompt_tokens=1, completion_tokens=1, total_tokens=2, model_name="fake-model"
        )
        self.raise_error = raise_error
        self.calls: list[tuple[str, str]] = []

    def complete(self, system_prompt: str, user_prompt: str) -> LLMCompletion:
        self.calls.append((system_prompt, user_prompt))
        if self.raise_error:
            raise self.raise_error
        return self.completion


class _FakeSemanticCache:
    def __init__(self, raise_error: Exception | None = None) -> None:
        self.raise_error = raise_error
        self.store_calls: list[tuple[str, str, str, list[str]]] = []

    def store(self, repository_id: str, query: str, response: str, chunk_ids: list[str]) -> int:
        self.store_calls.append((repository_id, query, response, chunk_ids))
        if self.raise_error:
            raise self.raise_error
        return 1


class TestPromptCreation:
    def test_builds_prompt_from_context_document(self) -> None:
        llm_client = _FakeLLMClient()
        cache = _FakeSemanticCache()
        service = LLMService(semantic_cache=cache, llm_client=llm_client)
        context_document = _context_document(context="Repository Context Body")

        service.generate_answer("repo-1", "how does auth work", context_document)

        expected_system, expected_user = build_prompt("repo-1", "Repository Context Body", "how does auth work")
        assert llm_client.calls == [(expected_system, expected_user)]
        assert llm_client.calls[0][0] == SYSTEM_PROMPT


class TestCacheUpdate:
    def test_successful_generation_updates_cache_with_cited_chunks(self) -> None:
        citations = [
            _citation("a", "src/auth.py", "authenticate"),
            _citation("b", "src/other.py", "unrelated"),
        ]
        completion = LLMCompletion(
            text="Authentication happens in src/auth.py via authenticate.",
            prompt_tokens=5,
            completion_tokens=3,
            total_tokens=8,
            model_name="mistral",
        )
        llm_client = _FakeLLMClient(completion=completion)
        cache = _FakeSemanticCache()
        service = LLMService(semantic_cache=cache, llm_client=llm_client)
        context_document = _context_document(chunk_citations=citations)

        result = service.generate_answer("repo-1", "how does auth work", context_document)

        assert result.cited_chunks == ["a"]
        assert result.answer == completion.text
        assert result.prompt_tokens == 5
        assert result.completion_tokens == 3
        assert result.total_tokens == 8
        assert result.model_name == "mistral"
        assert cache.store_calls == [("repo-1", "how does auth work", completion.text, ["a"])]

    def test_cache_failure_does_not_raise_or_break_the_answer(self) -> None:
        cache = _FakeSemanticCache(raise_error=RetrievalError("cache backend down"))
        service = LLMService(semantic_cache=cache, llm_client=_FakeLLMClient())

        result = service.generate_answer("repo-1", "q", _context_document())

        assert result.answer == "default answer"  # caller still gets their answer
        # "default answer" cites nothing recognizable, so cited_chunks is empty regardless.
        assert cache.store_calls == [("repo-1", "q", "default answer", [])]


class TestEmptyContext:
    def test_skips_llm_and_cache_when_context_is_empty(self) -> None:
        llm_client = _FakeLLMClient()
        cache = _FakeSemanticCache()
        service = LLMService(semantic_cache=cache, llm_client=llm_client)
        context_document = _context_document(context="", chunk_citations=[])

        result = service.generate_answer("repo-1", "q", context_document)

        assert result.answer == INSUFFICIENT_CONTEXT_MESSAGE
        assert result.cited_chunks == []
        assert result.prompt_tokens == 0
        assert result.completion_tokens == 0
        assert result.total_tokens == 0
        assert result.model_name == ""
        assert llm_client.calls == []  # LLM never invoked
        assert cache.store_calls == []  # nothing to cache


class TestInsufficientContext:
    def test_llm_reported_insufficiency_yields_no_cited_chunks_but_still_caches(self) -> None:
        completion = LLMCompletion(
            text=INSUFFICIENT_CONTEXT_MESSAGE,
            prompt_tokens=4,
            completion_tokens=2,
            total_tokens=6,
            model_name="mistral",
        )
        llm_client = _FakeLLMClient(completion=completion)
        cache = _FakeSemanticCache()
        service = LLMService(semantic_cache=cache, llm_client=llm_client)
        context_document = _context_document()  # non-empty context, so the LLM is still invoked

        result = service.generate_answer("repo-1", "q", context_document)

        assert result.answer == INSUFFICIENT_CONTEXT_MESSAGE
        assert result.cited_chunks == []
        assert len(llm_client.calls) == 1  # unlike the empty-context case, the LLM WAS invoked
        assert cache.store_calls == [("repo-1", "q", INSUFFICIENT_CONTEXT_MESSAGE, [])]


class TestMatchCitations:
    def test_matches_file_path_and_function_name(self) -> None:
        citations = [_citation("a", "src/auth.py", "authenticate")]
        answer = "See `src/auth.py`, function `authenticate` for details."

        assert match_citations(answer, citations) == ["a"]

    def test_no_match_when_file_path_absent(self) -> None:
        citations = [_citation("a", "src/auth.py", "authenticate")]
        answer = "This mentions authenticate but not the file."

        assert match_citations(answer, citations) == []

    def test_no_match_when_function_name_absent(self) -> None:
        citations = [_citation("a", "src/auth.py", "authenticate")]
        answer = "This mentions src/auth.py but not the function."

        assert match_citations(answer, citations) == []

    def test_falls_back_to_class_name_when_function_name_is_none(self) -> None:
        citations = [_citation("a", "src/models.py", None, class_name="User")]
        answer = "See `src/models.py`, class `User`."

        assert match_citations(answer, citations) == ["a"]

    def test_matches_on_file_path_alone_when_no_function_or_class(self) -> None:
        citations = [_citation("a", "src/config.py", None, None)]
        answer = "Configuration lives in src/config.py."

        assert match_citations(answer, citations) == ["a"]

    def test_returns_multiple_matched_chunk_ids_in_order(self) -> None:
        citations = [
            _citation("a", "src/auth.py", "authenticate"),
            _citation("b", "src/session.py", "create_session"),
        ]
        answer = "Auth uses `src/auth.py` `authenticate` and `src/session.py` `create_session`."

        assert match_citations(answer, citations) == ["a", "b"]
