"""Prompt template definitions used by the generation layer (Phase 16).

`SYSTEM_PROMPT` is the fixed instruction set every generation call sends
as the system turn - it is what makes the LLM answer strictly from the
supplied repository context, cite every claim, and admit when the
context is insufficient, instead of relying on its own training data.
`build_prompt` assembles the rest (repository metadata, the retrieved
context, the user's question) into the user turn, in the exact order
Phase 16 specifies: System Prompt -> Repository Metadata -> Repository
Context -> User Question.
"""

from __future__ import annotations

SYSTEM_PROMPT: str = """You are a repository-aware coding assistant. You answer questions about a \
specific software repository using ONLY the code context supplied to you below - you have no other \
knowledge of this repository.

Rules you must follow:
1. Answer ONLY using the information in the "Repository Context" section. Never use outside knowledge, \
assumptions, or general programming conventions to fill gaps the context does not cover.
2. Never hallucinate. Do not invent file paths, function names, classes, or behavior that is not \
explicitly present in the supplied context.
3. Cite every claim. Every factual statement you make must be attributed to the specific code it came \
from. Every citation must include both the file path and the function name it came from, in the form: \
(source: `<file_path>`, function `<function_name>`). If the cited code is a class rather than a function, \
cite the class name instead, in the form: (source: `<file_path>`, class `<class_name>`).
4. If the supplied context does not contain enough information to answer the question, explicitly say so \
- state plainly that the repository does not contain enough information to answer, rather than guessing \
or answering partially from outside knowledge.
"""


def build_user_prompt(repository_id: str, context: str, query: str) -> str:
    """Assemble the user turn: Repository Metadata -> Repository Context -> User Question.

    Args:
        repository_id: The repository the question is about.
        context: `models.schemas.ContextDocument.context` - the assembled,
            token-budgeted repository context from Phase 15.
        query: The user's question.

    Returns:
        The user turn text, in the fixed order Phase 16 specifies.
    """
    return (
        f"Repository Metadata:\nRepository ID: {repository_id}\n\n"
        f"Repository Context:\n{context}\n\n"
        f"User Question:\n{query}"
    )


def build_prompt(repository_id: str, context: str, query: str) -> tuple[str, str]:
    """Build the full prompt as a (system, user) pair for a chat-style LLM call.

    Args:
        repository_id: The repository the question is about.
        context: The assembled repository context from Phase 15.
        query: The user's question.

    Returns:
        A tuple of (`SYSTEM_PROMPT`, `build_user_prompt(...)`) - the
        System Prompt turn and the Repository Metadata/Context/Question
        turn, ready to hand to `generation.llm_client.LLMClient.complete`.
    """
    return SYSTEM_PROMPT, build_user_prompt(repository_id, context, query)
