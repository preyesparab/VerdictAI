"""FastAPI HTTP layer for RepoMind (Phase 20).

Wraps `pipeline.Pipeline` (Phase 19) so the CLI, a future React frontend,
and Adjudicate can all call indexing/query/context/graph over HTTP
instead of importing Python modules directly. Nothing here duplicates
pipeline logic - see `api.main`'s module docstring.
"""
