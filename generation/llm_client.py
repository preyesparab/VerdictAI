"""Client wrapper for calling the underlying LLM provider (Phase 16; Gemini update; Groq added Phase 27 follow-up).

`LLMClient` is the *only* place in this system that imports the
`google-genai`, `groq`, or `ollama` SDKs, and the only place a real
network call to any provider happens. Provider selection is a static,
config-driven priority - not automatic retry-on-failure between
providers:

    if settings.USE_GEMINI: use Gemini (settings.GEMINI_MODEL, default
        ``gemini-2.5-flash``)
    elif settings.USE_GROQ: use Groq (settings.GROQ_MODEL, default
        ``llama-3.3-70b-versatile``) - a hosted fallback for when
        Gemini's free-tier quota is exhausted (see the Phase 27 follow-up
        entry in docs/state/PROGRESS.md)
    elif settings.USE_OLLAMA: use Ollama (a local model, settings.OLLAMA_MODEL,
        default ``mistral``)
    else: raise LLMGenerationError - no provider is configured

Every third-party SDK exception is caught here and re-raised as
`core.exceptions.LLMGenerationError` - no `google.genai`/`groq`/`ollama`
exception type is ever allowed to cross this module's boundary, matching
how `retrieval.reranker`/`embedding.model_loader` wrap
`sentence_transformers` failures at their own model-loading/inference
boundaries.

Streaming is not implemented yet (see Phase 16 scope). `LLMClient.complete`
returns one `LLMCompletion` per call; a future streaming variant would add
a separate `stream_complete` method yielding incremental chunks, without
changing this one - `generation.answer_generator.LLMService` is written
against `complete` alone so that addition would not require touching it.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Protocol

from config import settings
from core.exceptions import LLMGenerationError
from core.logging import get_logger

logger = get_logger(__name__)


@dataclasses.dataclass(frozen=True)
class LLMCompletion:
    """Raw result of one LLM completion call.

    Attributes:
        text: The generated answer text.
        prompt_tokens: Tokens consumed by the prompt, per the provider's
            own usage accounting.
        completion_tokens: Tokens consumed by the generated answer, per
            the provider's own usage accounting.
        total_tokens: `prompt_tokens + completion_tokens`.
        model_name: The specific model that generated `text`.
    """

    text: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    model_name: str


class OllamaChatClient(Protocol):
    """The subset of `ollama.Client` this module needs."""

    def chat(self, model: str, messages: list[dict[str, str]], **kwargs: Any) -> Any:
        """Run one chat completion."""
        ...


class _GeminiModels(Protocol):
    def generate_content(self, model: str, contents: str, config: Any) -> Any:
        """Run one Gemini generation call."""
        ...


class GeminiClient(Protocol):
    """The subset of `google.genai.Client` this module needs (`.models.generate_content`)."""

    models: _GeminiModels


class _GroqCompletions(Protocol):
    def create(self, model: str, messages: list[dict[str, str]], **kwargs: Any) -> Any:
        """Run one Groq chat completion."""
        ...


class _GroqChat(Protocol):
    completions: _GroqCompletions


class GroqChatClient(Protocol):
    """The subset of `groq.Groq` this module needs (`.chat.completions.create`) - Groq's API is OpenAI-compatible."""

    chat: _GroqChat


def load_ollama_client(base_url: str) -> OllamaChatClient:
    """Construct an `ollama.Client` pointed at `base_url`.

    Args:
        base_url: The local Ollama server's base URL
            (`settings.OLLAMA_BASE_URL`).

    Returns:
        A ready-to-use `ollama.Client`.

    Raises:
        LLMGenerationError: If the client cannot be constructed.
    """
    import ollama

    try:
        return ollama.Client(host=base_url)
    except Exception as exc:  # noqa: BLE001 - third-party client construction boundary
        raise LLMGenerationError(f"Failed to construct Ollama client for {base_url!r}: {exc}") from exc


def load_gemini_client(api_key: str | None) -> GeminiClient:
    """Construct a `google.genai.Client` authenticated with `api_key`.

    Args:
        api_key: Google Gemini API key (`settings.GEMINI_API_KEY`, read
            from the `GEMINI_API_KEY` environment variable/`.env` entry).

    Returns:
        A ready-to-use `genai.Client`.

    Raises:
        LLMGenerationError: If `api_key` is missing, or the client cannot
            be constructed.
    """
    if not api_key:
        raise LLMGenerationError(
            "GEMINI_API_KEY is not set. Add it to your .env file, or set USE_GEMINI=False "
            "and configure USE_OLLAMA instead."
        )

    from google import genai

    try:
        return genai.Client(api_key=api_key)
    except Exception as exc:  # noqa: BLE001 - third-party client construction boundary
        raise LLMGenerationError(f"Failed to construct Gemini client: {exc}") from exc


def load_groq_client(api_key: str | None) -> GroqChatClient:
    """Construct a `groq.Groq` client authenticated with `api_key`.

    Args:
        api_key: Groq API key (`settings.GROQ_API_KEY`, read from the
            `GROQ_API_KEY` environment variable/`.env` entry).

    Returns:
        A ready-to-use `groq.Groq` client.

    Raises:
        LLMGenerationError: If `api_key` is missing, or the client cannot
            be constructed.
    """
    if not api_key:
        raise LLMGenerationError(
            "GROQ_API_KEY is not set. Add it to your .env file, or set USE_GROQ=False "
            "and configure another provider instead."
        )

    from groq import Groq

    try:
        return Groq(api_key=api_key)
    except Exception as exc:  # noqa: BLE001 - third-party client construction boundary
        raise LLMGenerationError(f"Failed to construct Groq client: {exc}") from exc


def _ollama_format_for(response_schema: Any | None) -> str | dict[str, Any] | None:
    """Best-effort translation of `response_schema` into Ollama's `chat(..., format=...)` argument.

    Ollama's structured-output support (a JSON-schema `format` dict, or
    the string ``"json"`` for loose JSON mode) is weaker than Gemini's
    constrained decoding - a pydantic model exposes `.model_json_schema()`
    directly, but `response_schema` is more commonly `list[SomeModel]`
    (Gemini's own convention for "return a JSON array"), which has no
    such method. Rather than hand-building a matching array-of-objects
    JSON Schema, this falls back to `"json"` (loose mode: valid JSON,
    not schema-validated) for anything that isn't a bare model - not a
    full substitute for Gemini's guarantee, since Ollama is not the
    provider this project actually runs Adjudicate against yet
    (`USE_OLLAMA=False`), and this path is unverified against a live
    server.

    Args:
        response_schema: Whatever was passed to `LLMClient.complete`, or
            None if the caller wants plain text.

    Returns:
        `None` if `response_schema` is None (unchanged, plain-text
        behavior); the schema's own `model_json_schema()` dict if it
        exposes one; otherwise the string ``"json"``.
    """
    if response_schema is None:
        return None
    model_json_schema = getattr(response_schema, "model_json_schema", None)
    if callable(model_json_schema):
        return model_json_schema()
    return "json"


def _complete_with_ollama(
    client: OllamaChatClient,
    model: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float,
    max_tokens: int,
    response_schema: Any | None = None,
) -> LLMCompletion:
    """Run one completion through an Ollama chat client.

    Args:
        client: The Ollama client to call.
        model: The Ollama model to use.
        system_prompt: The system turn.
        user_prompt: The user turn.
        temperature: Sampling temperature.
        max_tokens: Maximum tokens to generate (Ollama's `num_predict`).
        response_schema: If given, requests structured JSON output - see
            `_ollama_format_for` for how faithfully this is honored.

    Returns:
        The completion.

    Raises:
        LLMGenerationError: If the request fails.
    """
    try:
        response = client.chat(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            options={"temperature": temperature, "num_predict": max_tokens},
            format=_ollama_format_for(response_schema),
        )
    except Exception as exc:  # noqa: BLE001 - third-party inference boundary, see module docstring
        raise LLMGenerationError(f"Ollama generation failed (model={model!r}): {exc}") from exc

    prompt_tokens = int(response.prompt_eval_count or 0)
    completion_tokens = int(response.eval_count or 0)
    return LLMCompletion(
        text=response.message.content,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
        model_name=model,
    )


def _complete_with_gemini(
    client: GeminiClient,
    model: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float,
    max_tokens: int,
    timeout: float,
    response_schema: Any | None = None,
) -> LLMCompletion:
    """Run one completion through a Gemini client.

    Args:
        client: The Gemini client to call.
        model: The Gemini model to use.
        system_prompt: The system turn (Gemini's `system_instruction`).
        user_prompt: The user turn (Gemini's `contents`).
        temperature: Sampling temperature.
        max_tokens: Maximum output tokens.
        timeout: Request timeout, in seconds.
        response_schema: If given, constrains Gemini's output to JSON
            matching this schema (a pydantic model, or `list[Model]` for
            a JSON array) via `response_mime_type="application/json"` -
            Gemini's own structured-output mode (constrained decoding),
            not a post-hoc parse of free text. None (the default) leaves
            generation as plain text, unchanged from before this
            parameter existed.

            Also disables Gemini 2.5's "thinking" tokens for this call
            (`ThinkingConfig(thinking_budget=0)`) - real reproduction
            against a Prosecutor call (mkocabas/VIBE, Phase 27
            follow-up) showed `max_output_tokens` is a *shared* budget
            with the model's own internal reasoning: one real call spent
            979 of a 1024-token budget on invisible "thinking", leaving
            28 tokens for the actual JSON and truncating it mid-string
            (`finish_reason=MAX_TOKENS`) - not a malformed-JSON/escaping
            bug, a truncation bug. A schema-constrained extraction task
            (claims/verdicts against grounding rules already spelled out
            in the prompt) doesn't need deep chain-of-thought reasoning
            the way an open-ended free-text answer might, so thinking is
            switched off only when `response_schema` is given - free-text
            completions (e.g. the Defender's justification) are
            unaffected.

    Returns:
        The completion. `text` is a JSON string when `response_schema`
        was given, matching that schema - parsing it into a typed object
        is the caller's responsibility (this module stays schema-agnostic).

    Raises:
        LLMGenerationError: If the request fails.
    """
    from google.genai import types

    config_kwargs: dict[str, Any] = {
        "system_instruction": system_prompt,
        "temperature": temperature,
        "max_output_tokens": max_tokens,
        "http_options": types.HttpOptions(timeout=int(timeout * 1000)),
    }
    if response_schema is not None:
        config_kwargs["response_mime_type"] = "application/json"
        config_kwargs["response_schema"] = response_schema
        config_kwargs["thinking_config"] = types.ThinkingConfig(thinking_budget=0)

    try:
        response = client.models.generate_content(
            model=model,
            contents=user_prompt,
            config=types.GenerateContentConfig(**config_kwargs),
        )
    except Exception as exc:  # noqa: BLE001 - third-party inference boundary, see module docstring
        raise LLMGenerationError(f"Gemini generation failed (model={model!r}): {exc}") from exc

    usage = response.usage_metadata
    prompt_tokens = int(usage.prompt_token_count or 0) if usage else 0
    completion_tokens = int(usage.candidates_token_count or 0) if usage else 0
    total_tokens = int(usage.total_token_count or 0) if usage else (prompt_tokens + completion_tokens)
    return LLMCompletion(
        text=response.text or "",
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        model_name=model,
    )


def _complete_with_groq(
    client: GroqChatClient,
    model: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float,
    max_tokens: int,
    response_schema: Any | None = None,
) -> LLMCompletion:
    """Run one completion through a Groq client (OpenAI-compatible chat completions API).

    Args:
        client: The Groq client to call.
        model: The Groq model to use.
        system_prompt: The system turn.
        user_prompt: The user turn.
        temperature: Sampling temperature.
        max_tokens: Maximum completion tokens.
        response_schema: If given, requests JSON output via Groq's
            `response_format={"type": "json_object"}` mode - a weaker
            guarantee than Gemini's schema-constrained decoding (valid
            JSON is enforced, but not conformance to `response_schema`'s
            actual shape - that's still the caller's own validation's
            job, e.g. `adjudicate.schemas.parse_claims`). Groq's JSON
            mode requires the word "json" to appear somewhere in the
            prompt; every caller that uses structured output already
            states this in its system prompt for Gemini's benefit, so
            this is satisfied without extra wiring.

    Returns:
        The completion.

    Raises:
        LLMGenerationError: If the request fails.
    """
    kwargs: dict[str, Any] = {}
    if response_schema is not None:
        kwargs["response_format"] = {"type": "json_object"}

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )
    except Exception as exc:  # noqa: BLE001 - third-party inference boundary, see module docstring
        raise LLMGenerationError(f"Groq generation failed (model={model!r}): {exc}") from exc

    usage = response.usage
    prompt_tokens = int(usage.prompt_tokens or 0) if usage else 0
    completion_tokens = int(usage.completion_tokens or 0) if usage else 0
    total_tokens = int(usage.total_tokens or 0) if usage else (prompt_tokens + completion_tokens)
    return LLMCompletion(
        text=response.choices[0].message.content or "",
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        model_name=model,
    )


class LLMClient:
    """Selects Gemini, Groq, or Ollama per `settings.USE_GEMINI`/`USE_GROQ`/`USE_OLLAMA` and completes prompts through it.

    All three provider clients are loaded lazily (on first `complete` call
    that needs them) and cached on the instance; tests can also inject
    fake clients directly via `gemini_client`/`groq_client`/`ollama_client`.
    """

    def __init__(
        self,
        gemini_client: GeminiClient | None = None,
        groq_client: GroqChatClient | None = None,
        ollama_client: OllamaChatClient | None = None,
        use_gemini: bool | None = None,
        use_groq: bool | None = None,
        use_ollama: bool | None = None,
        gemini_model: str | None = None,
        gemini_api_key: str | None = None,
        groq_model: str | None = None,
        groq_api_key: str | None = None,
        ollama_model: str | None = None,
        ollama_base_url: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        timeout: float | None = None,
    ) -> None:
        """Initialize the client.

        Args:
            gemini_client: Pre-constructed Gemini client. Overridable for
                testing; defaults to lazily loading one via
                `load_gemini_client` the first time it is needed.
            groq_client: Pre-constructed Groq client. Overridable for
                testing; defaults to lazily loading one via
                `load_groq_client` the first time it is needed.
            ollama_client: Pre-constructed Ollama client. Overridable for
                testing; defaults to lazily loading one via
                `load_ollama_client` the first time it is needed.
            use_gemini: Provider switch, checked first. Defaults to
                `settings.USE_GEMINI`.
            use_groq: Provider switch, checked only if `use_gemini` is
                False. Defaults to `settings.USE_GROQ`.
            use_ollama: Provider switch, checked only if both `use_gemini`
                and `use_groq` are False. Defaults to `settings.USE_OLLAMA`.
            gemini_model: Defaults to `settings.GEMINI_MODEL`.
            gemini_api_key: Defaults to `settings.GEMINI_API_KEY`.
            groq_model: Defaults to `settings.GROQ_MODEL`.
            groq_api_key: Defaults to `settings.GROQ_API_KEY`.
            ollama_model: Defaults to `settings.OLLAMA_MODEL`.
            ollama_base_url: Defaults to `settings.OLLAMA_BASE_URL`.
            temperature: Sampling temperature. Defaults to
                `settings.LLM_TEMPERATURE`.
            max_tokens: Maximum completion tokens. Defaults to
                `settings.LLM_MAX_TOKENS`.
            timeout: Request timeout, in seconds (Gemini only - neither
                the Groq nor the Ollama client is given an explicit
                timeout). Defaults to `settings.LLM_TIMEOUT_SECONDS`.
        """
        self._use_gemini = use_gemini if use_gemini is not None else settings.USE_GEMINI
        self._use_groq = use_groq if use_groq is not None else settings.USE_GROQ
        self._use_ollama = use_ollama if use_ollama is not None else settings.USE_OLLAMA
        self._gemini_model = gemini_model or settings.GEMINI_MODEL
        self._gemini_api_key = gemini_api_key if gemini_api_key is not None else settings.GEMINI_API_KEY
        self._groq_model = groq_model or settings.GROQ_MODEL
        self._groq_api_key = groq_api_key if groq_api_key is not None else settings.GROQ_API_KEY
        self._ollama_model = ollama_model or settings.OLLAMA_MODEL
        self._ollama_base_url = ollama_base_url or settings.OLLAMA_BASE_URL
        self._temperature = temperature if temperature is not None else settings.LLM_TEMPERATURE
        self._max_tokens = max_tokens or settings.LLM_MAX_TOKENS
        self._timeout = timeout if timeout is not None else settings.LLM_TIMEOUT_SECONDS
        self._gemini_client = gemini_client
        self._groq_client = groq_client
        self._ollama_client = ollama_client

    def _ensure_gemini_client(self) -> GeminiClient:
        """Lazily load the Gemini client on first use."""
        if self._gemini_client is None:
            self._gemini_client = load_gemini_client(self._gemini_api_key)
        return self._gemini_client

    def _ensure_groq_client(self) -> GroqChatClient:
        """Lazily load the Groq client on first use."""
        if self._groq_client is None:
            self._groq_client = load_groq_client(self._groq_api_key)
        return self._groq_client

    def _ensure_ollama_client(self) -> OllamaChatClient:
        """Lazily load the Ollama client on first use."""
        if self._ollama_client is None:
            self._ollama_client = load_ollama_client(self._ollama_base_url)
        return self._ollama_client

    def complete(
        self, system_prompt: str, user_prompt: str, response_schema: Any | None = None
    ) -> LLMCompletion:
        """Complete `(system_prompt, user_prompt)` through the configured provider.

        Args:
            system_prompt: The system turn - fixed instructions.
            user_prompt: The user turn - repository metadata, context,
                and the question.
            response_schema: If given, requests structured JSON output
                matching this schema instead of plain text (see
                `_complete_with_gemini`'s docstring for Gemini's
                constrained-decoding behavior, and `_complete_with_groq`'s/
                `_ollama_format_for`'s docstrings for Groq's/Ollama's
                weaker best-effort equivalents). None (the default) is
                plain text, exactly as before this parameter existed -
                every existing caller (`generation.answer_generator`) is
                unaffected.

        Returns:
            The provider's completion.

        Raises:
            LLMGenerationError: If none of `USE_GEMINI`, `USE_GROQ`, or
                `USE_OLLAMA` is enabled, the selected provider's client
                cannot be constructed, or the request fails.
        """
        if self._use_gemini:
            logger.info("Provider selected: gemini (model=%s)", self._gemini_model)
            client = self._ensure_gemini_client()
            return _complete_with_gemini(
                client,
                self._gemini_model,
                system_prompt,
                user_prompt,
                self._temperature,
                self._max_tokens,
                self._timeout,
                response_schema,
            )

        if self._use_groq:
            logger.info("Provider selected: groq (model=%s)", self._groq_model)
            client = self._ensure_groq_client()
            return _complete_with_groq(
                client, self._groq_model, system_prompt, user_prompt, self._temperature, self._max_tokens,
                response_schema,
            )

        if self._use_ollama:
            logger.info("Provider selected: ollama (model=%s)", self._ollama_model)
            client = self._ensure_ollama_client()
            return _complete_with_ollama(
                client, self._ollama_model, system_prompt, user_prompt, self._temperature, self._max_tokens,
                response_schema,
            )

        raise LLMGenerationError(
            "No LLM provider is enabled: set USE_GEMINI=True (with GEMINI_API_KEY), "
            "USE_GROQ=True (with GROQ_API_KEY), or USE_OLLAMA=True in your configuration."
        )
