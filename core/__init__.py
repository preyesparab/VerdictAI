"""Core foundational infrastructure: logging, exceptions, and constants.

Every other layer (ingestion, retrieval, generation, evaluation, database)
depends on `core`, but `core` depends on nothing outside the standard
library and `config`. This keeps foundational infrastructure reusable and
free of business logic.
"""
