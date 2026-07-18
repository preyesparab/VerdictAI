"""Knowledge graph construction: relationships between files, functions, classes, and methods.

Converts the flat `models.schemas.CodeChunk` list produced by ingestion
(Phases 4-5) into a directed graph capturing structural relationships that
plain vector similarity cannot: which function calls which, which class
inherits from which, which file imports which, and which class contains
which methods. Has no dependency on retrieval or generation layers — it
only builds and queries an in-memory `networkx.DiGraph`.

Pipeline: `graph.graph_builder.RepositoryGraphBuilder.build_graph()`
consumes the `SourceFile` list from `ingestion.file_discovery.FileDiscovery`
and the `CodeChunk` list from `ingestion.ast_parser.TreeSitterParser`,
using `graph.call_extractor` and `graph.import_extractor` internally to
detect call sites and import statements. The resulting graph is persisted
by `database.graph_store.save_graph`/`load_graph` and queried via
`graph.graph_queries` — both a future retrieval phase (graph expansion)
and ad-hoc inspection can use the query helpers directly.
"""
