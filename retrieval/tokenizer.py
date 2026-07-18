"""Tokenizer for keyword-based (BM25) retrieval over source code.

Splits code text - identifiers, function/class names, comments, and raw
source - into normalized, searchable tokens: punctuation is stripped,
snake_case and camelCase compound identifiers are split into their parts,
and every token is lowercased. This is what lets a query for ``fetch user``
match a chunk defining ``def fetch_user(...)`` or ``fetchUser(...)``.

`retrieval.sparse_retriever.BM25Manager` is the only current consumer.
"""

from __future__ import annotations

import re
from collections.abc import Callable

_NON_ALNUM = re.compile(r"[^A-Za-z0-9]+")
# Boundary between a lowercase/digit and an uppercase letter (fetchUser ->
# fetch|User), or between the last letter of an acronym and the start of a
# new capitalized word (HTTPServer -> HTTP|Server).
_CAMEL_CASE_BOUNDARY = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")

# Per-token normalization steps applied, in order, after splitting.
# Appending to this list (e.g. a stemmer or stopword filter) is the
# intended extension point for additional preprocessing - `tokenize`
# itself should not need to change to support it.
TOKEN_PREPROCESSORS: list[Callable[[str], str]] = [str.lower]


def _split_camel_case(word: str) -> list[str]:
    """Split one punctuation-free word on camelCase/acronym boundaries.

    Args:
        word: A word containing no punctuation (already split on
            non-alphanumeric characters).

    Returns:
        `word`'s camelCase-delimited parts, e.g. ``"fetchUserByID"`` ->
        ``["fetch", "User", "By", "ID"]``.
    """
    return [part for part in _CAMEL_CASE_BOUNDARY.split(word) if part]


def tokenize(text: str) -> list[str]:
    """Split `text` into normalized, keyword-searchable tokens.

    Args:
        text: Source text to tokenize - raw code, an identifier, a
            comment, or any other free text.

    Returns:
        Tokens with punctuation removed, snake_case/camelCase compound
        identifiers split into parts, and every token passed through
        `TOKEN_PREPROCESSORS` (lowercasing by default). Both
        ``fetch_user_by_id`` and ``fetchUserByID`` tokenize to
        ``["fetch", "user", "by", "id"]``. Empty for empty/whitespace-only
        input.
    """
    tokens: list[str] = []
    for word in _NON_ALNUM.split(text):
        if word:
            tokens.extend(_split_camel_case(word))

    for preprocessor in TOKEN_PREPROCESSORS:
        tokens = [preprocessor(token) for token in tokens]

    return [token for token in tokens if token]
