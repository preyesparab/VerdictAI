"""Ingestion layer: repository loading, Tree-sitter AST parsing, and code chunking.

Converts raw source code into structured, retrievable units. Has no
dependency on retrieval or generation layers.

`ingestion.repository_manager.RepositoryManager` is the only public
interface for repository acquisition — other modules (including the rest
of this package) must not import `ingestion.git_client` directly.

Pipeline order: `RepositoryManager.get_repository()` returns a local path
-> `ingestion.file_discovery.FileDiscovery.discover()` turns that path
into a filtered list of `SourceFile` objects ->
`ingestion.ast_parser.TreeSitterParser.parse_many()` parses each
`SourceFile` into `models.schemas.CodeChunk` (AST) objects ->
`ingestion.chunker.SemanticChunker.build_chunks()` enriches each file's
AST chunks with sliding-window and parent chunks, for embedding
(a future phase) and small-to-big retrieval.
"""
