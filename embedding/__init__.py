"""Embedding generation layer (Phase 8): chunk vector embeddings for retrieval.

Converts Level-1 AST chunks stored by `database.sqlite_client.DatabaseManager`
(Phase 7) into dense vectors and persists them back to SQLite. Has no
knowledge of FAISS, retrieval, or generation - those are later phases that
read embeddings back out via `DatabaseManager.load_embeddings`.

Pipeline: `embedding.model_loader.load_embedding_model` selects and
constructs either CodeBERT or MiniLM (per `settings.USE_CODEBERT`), and
`embedding.embedding_manager.EmbeddingManager.generate_embeddings` uses it
to embed a repository's AST chunks in configurable batches.
"""
