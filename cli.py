"""Command-line entrypoint for RepoMind (Phase 19).

Thin entrypoint: this file only parses arguments and calls
`pipeline.Pipeline` - it contains no ingestion/retrieval/generation logic
of its own.

Examples:
    python cli.py index https://github.com/psf/requests
    python cli.py query <repository_id> "How does the Session class work?"
"""

from __future__ import annotations

import argparse
import sys

from core.exceptions import RepoMindError
from core.logging import get_logger
from pipeline import AskResult, IndexingSummary, Pipeline

logger = get_logger(__name__)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for the `index` and `query` subcommands.

    Args:
        argv: Argument list to parse. Defaults to `sys.argv[1:]`.

    Returns:
        The parsed arguments, exposing `command` plus each subcommand's
        own arguments (`repository_url`, or `repository_id`/`question`).
    """
    parser = argparse.ArgumentParser(
        description="Index a repository or ask a question about one, via RepoMind's pipeline.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    index_parser = subparsers.add_parser("index", help="Clone and fully index a GitHub repository.")
    index_parser.add_argument(
        "repository_url",
        help="GitHub repository URL, e.g. https://github.com/psf/requests",
    )

    query_parser = subparsers.add_parser("query", help="Ask a question about an already-indexed repository.")
    query_parser.add_argument("repository_id", help="The repository_id printed by the `index` command.")
    query_parser.add_argument("question", help="The natural-language question to ask.")

    return parser.parse_args(argv)


def _print_indexing_summary(summary: IndexingSummary) -> None:
    """Print a human-readable summary of an `IndexingSummary` to stdout.

    Args:
        summary: `Pipeline.index_repository`'s result.
    """
    print(f"Repository:      {summary.repository.owner}/{summary.repository.name}")
    print(f"Repository ID:   {summary.repository_id}")
    print(f"Commit:          {summary.repository.commit_hash}")
    print(f"Files discovered:{summary.files_discovered:>4}")
    print(f"Chunks indexed:  {summary.chunks_indexed}")
    print(f"Graph nodes/edges: {summary.graph_nodes} / {summary.graph_edges}")
    print(f"Chunks embedded: {summary.embedded_chunks}")


def _print_ask_result(result: AskResult) -> None:
    """Print a generated answer and its citations to stdout.

    Args:
        result: `Pipeline.query`'s result.
    """
    print(result.answer.answer)
    print()
    print(f"[model={result.llm_model} cache_hit={result.cache_hit} "
          f"retrieved={result.retrieved_count} graph_expanded={result.graph_expanded_count} "
          f"latency_ms={result.answer.latency_ms:.0f}]")
    if result.citations:
        print("\nCitations:")
        for citation in result.citations:
            location = f"{citation.file_path}"
            if citation.function_name:
                location += f"::{citation.function_name}"
            print(f"  - {location} ({citation.chunk_type}, source={citation.retrieval_source}, "
                  f"score={citation.relevance_score:.3f})")


def _run_index(pipeline: Pipeline, repository_url: str) -> int:
    """Run `Pipeline.index_repository` and print the result.

    Args:
        pipeline: The pipeline to run indexing through.
        repository_url: The repository URL to index.

    Returns:
        Process exit code: 0 on success, 1 on failure.
    """
    def _on_progress(stage: str, status: str) -> None:
        if status == "running":
            print(f"... {stage}")

    try:
        summary = pipeline.index_repository(repository_url, on_progress=_on_progress)
    except RepoMindError as exc:
        logger.error("Indexing failed: %s", exc)
        print(f"Indexing failed: {exc}", file=sys.stderr)
        return 1

    _print_indexing_summary(summary)
    return 0


def _run_query(pipeline: Pipeline, repository_id: str, question: str) -> int:
    """Run `Pipeline.query` and print the result.

    Args:
        pipeline: The pipeline to run the query through.
        repository_id: The repository to query.
        question: The natural-language question to ask.

    Returns:
        Process exit code: 0 on success, 1 on failure.
    """
    try:
        result = pipeline.query(question, repository_id)
    except RepoMindError as exc:
        logger.error("Query failed: %s", exc)
        print(f"Query failed: {exc}", file=sys.stderr)
        return 1

    _print_ask_result(result)
    return 0


def main(argv: list[str] | None = None) -> int:
    """Entry point: dispatch to `index` or `query` based on the command line.

    Args:
        argv: Argument list to parse. Defaults to `sys.argv[1:]`.

    Returns:
        Process exit code: 0 on success, 1 on failure.
    """
    args = _parse_args(argv)
    pipeline = Pipeline()

    if args.command == "index":
        return _run_index(pipeline, args.repository_url)
    return _run_query(pipeline, args.repository_id, args.question)


if __name__ == "__main__":
    sys.exit(main())
