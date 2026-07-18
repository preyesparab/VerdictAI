"""Tests for models.schemas."""

from __future__ import annotations

import dataclasses
import uuid

import pytest

from models.schemas import ChunkType, CodeChunk


def _make_chunk(**overrides: object) -> CodeChunk:
    defaults: dict[str, object] = {
        "chunk_id": uuid.uuid4(),
        "file_id": "file-1",
        "file_path": "src/app.py",
        "language": "python",
        "chunk_type": ChunkType.FUNCTION,
        "function_name": "run",
        "class_name": None,
        "parent_class": None,
        "start_line": 1,
        "end_line": 3,
        "raw_code": "def run():\n    return 1\n",
    }
    defaults.update(overrides)
    return CodeChunk(**defaults)  # type: ignore[arg-type]


def test_fields_round_trip() -> None:
    chunk = _make_chunk()
    assert chunk.function_name == "run"
    assert chunk.chunk_type == ChunkType.FUNCTION


def test_is_immutable() -> None:
    chunk = _make_chunk()
    with pytest.raises(dataclasses.FrozenInstanceError):
        chunk.function_name = "other"  # type: ignore[misc]


def test_chunk_type_values() -> None:
    assert {t.value for t in ChunkType} == {
        "function", "async_function", "class", "method", "arrow_function",
        "sliding", "parent",
    }


def test_parent_chunk_id_defaults_to_none() -> None:
    assert _make_chunk().parent_chunk_id is None


def test_parent_chunk_id_can_be_set() -> None:
    parent_id = uuid.uuid4()
    chunk = _make_chunk(parent_chunk_id=parent_id)
    assert chunk.parent_chunk_id == parent_id
