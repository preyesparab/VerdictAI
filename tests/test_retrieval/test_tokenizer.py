"""Tests for retrieval.tokenizer.tokenize."""

from __future__ import annotations

from retrieval.tokenizer import tokenize


class TestTokenize:
    def test_splits_snake_case(self) -> None:
        assert tokenize("fetch_user_by_id") == ["fetch", "user", "by", "id"]

    def test_splits_camel_case(self) -> None:
        assert tokenize("fetchUserByID") == ["fetch", "user", "by", "id"]

    def test_splits_acronym_boundary(self) -> None:
        assert tokenize("HTTPServer") == ["http", "server"]

    def test_removes_punctuation(self) -> None:
        assert tokenize("user.get_name()") == ["user", "get", "name"]

    def test_handles_comment_text(self) -> None:
        assert tokenize("# TODO: fix this_bug!!") == ["todo", "fix", "this", "bug"]

    def test_mixed_snake_and_camel_case(self) -> None:
        assert tokenize("parse_HTTPRequest_body") == ["parse", "http", "request", "body"]

    def test_empty_string_returns_empty_list(self) -> None:
        assert tokenize("") == []

    def test_whitespace_only_returns_empty_list(self) -> None:
        assert tokenize("   ") == []

    def test_punctuation_only_returns_empty_list(self) -> None:
        assert tokenize("!!! ... ???") == []
