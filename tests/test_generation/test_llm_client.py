"""Tests for generation.llm_client.LLMClient.

Gemini, Groq, and Ollama SDKs are fully mocked via fake clients matching
`GeminiClient`/`GroqChatClient`/`OllamaChatClient`'s shape - these tests
verify provider selection, prompt/response plumbing, lazy client loading,
and error wrapping, never a real network call.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import generation.llm_client as llm_client_module
from core.exceptions import LLMGenerationError
from generation.llm_client import LLMClient, load_gemini_client, load_groq_client


class _FakeOllamaClient:
    def __init__(
        self,
        response_text: str = "An answer.",
        prompt_eval_count: int = 10,
        eval_count: int = 5,
        raise_error: bool = False,
    ) -> None:
        self.response_text = response_text
        self.prompt_eval_count = prompt_eval_count
        self.eval_count = eval_count
        self.raise_error = raise_error
        self.chat_calls: list[dict] = []

    def chat(self, model, messages, **kwargs):
        self.chat_calls.append({"model": model, "messages": messages, **kwargs})
        if self.raise_error:
            raise RuntimeError("ollama boom")
        return SimpleNamespace(
            message=SimpleNamespace(content=self.response_text),
            prompt_eval_count=self.prompt_eval_count,
            eval_count=self.eval_count,
        )


class _FakeGeminiModels:
    def __init__(
        self,
        response_text: str = "An answer.",
        prompt_token_count: int = 12,
        candidates_token_count: int = 6,
        total_token_count: int | None = None,
        raise_error: bool = False,
    ) -> None:
        self.response_text = response_text
        self.prompt_token_count = prompt_token_count
        self.candidates_token_count = candidates_token_count
        self.total_token_count = (
            total_token_count if total_token_count is not None else prompt_token_count + candidates_token_count
        )
        self.raise_error = raise_error
        self.generate_content_calls: list[dict] = []

    def generate_content(self, model, contents, config):
        self.generate_content_calls.append({"model": model, "contents": contents, "config": config})
        if self.raise_error:
            raise RuntimeError("gemini boom")
        return SimpleNamespace(
            text=self.response_text,
            usage_metadata=SimpleNamespace(
                prompt_token_count=self.prompt_token_count,
                candidates_token_count=self.candidates_token_count,
                total_token_count=self.total_token_count,
            ),
        )


class _FakeGeminiClient:
    def __init__(self, **kwargs) -> None:
        self.models = _FakeGeminiModels(**kwargs)


class _FakeGroqCompletions:
    def __init__(
        self,
        response_text: str = "An answer.",
        prompt_tokens: int = 15,
        completion_tokens: int = 7,
        total_tokens: int | None = None,
        raise_error: bool = False,
    ) -> None:
        self.response_text = response_text
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.total_tokens = total_tokens if total_tokens is not None else prompt_tokens + completion_tokens
        self.raise_error = raise_error
        self.create_calls: list[dict] = []

    def create(self, model, messages, **kwargs):
        self.create_calls.append({"model": model, "messages": messages, **kwargs})
        if self.raise_error:
            raise RuntimeError("groq boom")
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=self.response_text))],
            usage=SimpleNamespace(
                prompt_tokens=self.prompt_tokens,
                completion_tokens=self.completion_tokens,
                total_tokens=self.total_tokens,
            ),
        )


class _FakeGroqChat:
    def __init__(self, **kwargs) -> None:
        self.completions = _FakeGroqCompletions(**kwargs)


class _FakeGroqClient:
    def __init__(self, **kwargs) -> None:
        self.chat = _FakeGroqChat(**kwargs)


class TestGeminiInitialization:
    def test_load_gemini_client_constructs_with_api_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import google.genai

        constructed_with = {}

        class _FakeClient:
            def __init__(self, api_key):
                constructed_with["api_key"] = api_key

        monkeypatch.setattr(google.genai, "Client", _FakeClient)

        client = load_gemini_client("test-key")

        assert constructed_with["api_key"] == "test-key"
        assert isinstance(client, _FakeClient)

    def test_client_construction_failure_raises_llm_generation_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import google.genai

        class _RaisingClient:
            def __init__(self, api_key):
                raise RuntimeError("bad credentials")

        monkeypatch.setattr(google.genai, "Client", _RaisingClient)

        with pytest.raises(LLMGenerationError):
            load_gemini_client("test-key")


class TestGroqInitialization:
    def test_load_groq_client_constructs_with_api_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import groq

        constructed_with = {}

        class _FakeClient:
            def __init__(self, api_key):
                constructed_with["api_key"] = api_key

        monkeypatch.setattr(groq, "Groq", _FakeClient)

        client = load_groq_client("test-key")

        assert constructed_with["api_key"] == "test-key"
        assert isinstance(client, _FakeClient)

    def test_client_construction_failure_raises_llm_generation_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import groq

        class _RaisingClient:
            def __init__(self, api_key):
                raise RuntimeError("bad credentials")

        monkeypatch.setattr(groq, "Groq", _RaisingClient)

        with pytest.raises(LLMGenerationError):
            load_groq_client("test-key")


class TestMissingApiKey:
    def test_missing_api_key_raises_before_touching_sdk(self) -> None:
        with pytest.raises(LLMGenerationError, match="GEMINI_API_KEY"):
            load_gemini_client(None)

    def test_empty_api_key_raises(self) -> None:
        with pytest.raises(LLMGenerationError, match="GEMINI_API_KEY"):
            load_gemini_client("")

    def test_llm_client_surfaces_missing_key_on_complete(self) -> None:
        # An explicit empty string (not None, which would fall back to
        # settings.GEMINI_API_KEY - a real configured key in this project's
        # own .env) - this is what "no key configured" looks like.
        llm = LLMClient(use_gemini=True, gemini_api_key="")

        with pytest.raises(LLMGenerationError, match="GEMINI_API_KEY"):
            llm.complete("system", "user")

    def test_groq_missing_api_key_raises_before_touching_sdk(self) -> None:
        with pytest.raises(LLMGenerationError, match="GROQ_API_KEY"):
            load_groq_client(None)

    def test_llm_client_surfaces_missing_groq_key_on_complete(self) -> None:
        llm = LLMClient(use_gemini=False, use_groq=True, groq_api_key="")

        with pytest.raises(LLMGenerationError, match="GROQ_API_KEY"):
            llm.complete("system", "user")


class TestGeminiResponseGeneration:
    def test_completes_via_gemini_when_selected(self) -> None:
        client = _FakeGeminiClient(response_text="Auth uses JWT.", prompt_token_count=20, candidates_token_count=8)
        llm = LLMClient(gemini_client=client, use_gemini=True, gemini_model="gemini-2.5-flash")

        result = llm.complete("system prompt", "user prompt")

        assert result.text == "Auth uses JWT."
        assert result.prompt_tokens == 20
        assert result.completion_tokens == 8
        assert result.total_tokens == 28
        assert result.model_name == "gemini-2.5-flash"

        call = client.models.generate_content_calls[0]
        assert call["model"] == "gemini-2.5-flash"
        assert call["contents"] == "user prompt"
        assert call["config"].system_instruction == "system prompt"

    def test_default_model_comes_from_settings(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(llm_client_module.settings, "GEMINI_MODEL", "gemini-2.5-pro")
        client = _FakeGeminiClient()
        llm = LLMClient(gemini_client=client, use_gemini=True)

        result = llm.complete("system", "user")

        assert result.model_name == "gemini-2.5-pro"


class TestGroqResponseGeneration:
    def test_completes_via_groq_when_selected(self) -> None:
        client = _FakeGroqClient(response_text="Auth uses JWT.", prompt_tokens=25, completion_tokens=9)
        llm = LLMClient(use_gemini=False, groq_client=client, use_groq=True, groq_model="llama-3.3-70b-versatile")

        result = llm.complete("system prompt", "user prompt")

        assert result.text == "Auth uses JWT."
        assert result.prompt_tokens == 25
        assert result.completion_tokens == 9
        assert result.total_tokens == 34
        assert result.model_name == "llama-3.3-70b-versatile"

        call = client.chat.completions.create_calls[0]
        assert call["model"] == "llama-3.3-70b-versatile"
        assert call["messages"] == [
            {"role": "system", "content": "system prompt"},
            {"role": "user", "content": "user prompt"},
        ]

    def test_default_model_comes_from_settings(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(llm_client_module.settings, "GROQ_MODEL", "gemma2-9b-it")
        client = _FakeGroqClient()
        llm = LLMClient(use_gemini=False, groq_client=client, use_groq=True)

        result = llm.complete("system", "user")

        assert result.model_name == "gemma2-9b-it"

    def test_response_schema_requests_json_object_format(self) -> None:
        client = _FakeGroqClient()
        llm = LLMClient(use_gemini=False, groq_client=client, use_groq=True)

        llm.complete("system", "user", response_schema=list)

        assert client.chat.completions.create_calls[0]["response_format"] == {"type": "json_object"}

    def test_no_response_format_when_schema_not_requested(self) -> None:
        client = _FakeGroqClient()
        llm = LLMClient(use_gemini=False, groq_client=client, use_groq=True)

        llm.complete("system", "user")

        assert "response_format" not in client.chat.completions.create_calls[0]


class TestProviderSelection:
    def test_gemini_selected_when_use_gemini_true(self) -> None:
        gemini = _FakeGeminiClient()
        ollama = _FakeOllamaClient()
        llm = LLMClient(
            gemini_client=gemini, ollama_client=ollama, use_gemini=True, use_groq=False, use_ollama=True
        )

        llm.complete("system", "user")

        assert len(gemini.models.generate_content_calls) == 1
        assert ollama.chat_calls == []

    def test_ollama_selected_when_gemini_disabled(self) -> None:
        gemini = _FakeGeminiClient()
        ollama = _FakeOllamaClient()
        llm = LLMClient(
            gemini_client=gemini, ollama_client=ollama, use_gemini=False, use_groq=False, use_ollama=True
        )

        llm.complete("system", "user")

        assert ollama.chat_calls != []
        assert gemini.models.generate_content_calls == []

    def test_groq_selected_when_gemini_disabled_and_groq_enabled(self) -> None:
        gemini = _FakeGeminiClient()
        groq = _FakeGroqClient()
        ollama = _FakeOllamaClient()
        llm = LLMClient(
            gemini_client=gemini, groq_client=groq, ollama_client=ollama,
            use_gemini=False, use_groq=True, use_ollama=True,
        )

        llm.complete("system", "user")

        assert groq.chat.completions.create_calls != []
        assert gemini.models.generate_content_calls == []
        assert ollama.chat_calls == []

    def test_gemini_takes_priority_over_groq_when_both_enabled(self) -> None:
        gemini = _FakeGeminiClient()
        groq = _FakeGroqClient()
        llm = LLMClient(gemini_client=gemini, groq_client=groq, use_gemini=True, use_groq=True)

        llm.complete("system", "user")

        assert len(gemini.models.generate_content_calls) == 1
        assert groq.chat.completions.create_calls == []

    def test_raises_clear_error_when_no_provider_enabled(self) -> None:
        llm = LLMClient(use_gemini=False, use_groq=False, use_ollama=False)

        with pytest.raises(LLMGenerationError, match="No LLM provider is enabled"):
            llm.complete("system", "user")

    def test_defaults_to_settings_use_gemini(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(llm_client_module.settings, "USE_GEMINI", True)
        gemini = _FakeGeminiClient()
        llm = LLMClient(gemini_client=gemini)

        llm.complete("system", "user")

        assert len(gemini.models.generate_content_calls) == 1

    def test_defaults_to_settings_use_groq(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(llm_client_module.settings, "USE_GEMINI", False)
        monkeypatch.setattr(llm_client_module.settings, "USE_GROQ", True)
        groq = _FakeGroqClient()
        llm = LLMClient(groq_client=groq)

        llm.complete("system", "user")

        assert groq.chat.completions.create_calls != []


class TestGeminiFailureHandling:
    def test_gemini_generation_failure_raises_llm_generation_error(self) -> None:
        client = _FakeGeminiClient(raise_error=True)
        llm = LLMClient(gemini_client=client, use_gemini=True)

        with pytest.raises(LLMGenerationError):
            llm.complete("system", "user")

    def test_ollama_failure_raises_llm_generation_error(self) -> None:
        client = _FakeOllamaClient(raise_error=True)
        llm = LLMClient(ollama_client=client, use_gemini=False, use_groq=False, use_ollama=True)

        with pytest.raises(LLMGenerationError):
            llm.complete("system", "user")

    def test_groq_failure_raises_llm_generation_error(self) -> None:
        client = _FakeGroqClient(raise_error=True)
        llm = LLMClient(use_gemini=False, groq_client=client, use_groq=True)

        with pytest.raises(LLMGenerationError):
            llm.complete("system", "user")


class TestLazyClientLoading:
    def test_gemini_client_loaded_lazily_once(self, monkeypatch: pytest.MonkeyPatch) -> None:
        fake = _FakeGeminiClient()
        load_calls: list[str | None] = []

        def _fake_loader(api_key: str | None):
            load_calls.append(api_key)
            return fake

        monkeypatch.setattr(llm_client_module, "load_gemini_client", _fake_loader)
        llm = LLMClient(use_gemini=True, gemini_api_key="test-key")

        llm.complete("system", "user")
        llm.complete("system", "user2")

        assert load_calls == ["test-key"]  # loaded exactly once, then cached

    def test_ollama_client_loaded_lazily_once(self, monkeypatch: pytest.MonkeyPatch) -> None:
        fake = _FakeOllamaClient()
        load_calls: list[str] = []

        def _fake_loader(base_url: str):
            load_calls.append(base_url)
            return fake

        monkeypatch.setattr(llm_client_module, "load_ollama_client", _fake_loader)
        llm = LLMClient(use_gemini=False, use_groq=False, use_ollama=True, ollama_base_url="http://example:11434")

        llm.complete("system", "user")
        llm.complete("system", "user2")

        assert load_calls == ["http://example:11434"]

    def test_groq_client_loaded_lazily_once(self, monkeypatch: pytest.MonkeyPatch) -> None:
        fake = _FakeGroqClient()
        load_calls: list[str | None] = []

        def _fake_loader(api_key: str | None):
            load_calls.append(api_key)
            return fake

        monkeypatch.setattr(llm_client_module, "load_groq_client", _fake_loader)
        llm = LLMClient(use_gemini=False, use_groq=True, groq_api_key="test-key")

        llm.complete("system", "user")
        llm.complete("system", "user2")

        assert load_calls == ["test-key"]  # loaded exactly once, then cached
