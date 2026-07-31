# RepoMind container image
# Minimal scaffold — build steps will expand as dependencies are added.

FROM python:3.11-slim

WORKDIR /app

# git is a hard runtime dependency: ingestion.git_client shells out to it via GitPython.
RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

# Shell form (not exec-form CMD) is required here: Render injects the
# listen port via the $PORT env var at container start, and exec-form CMD
# (CMD ["uvicorn", ...]) never expands env vars - only a shell does that.
# `exec` hands PID 1 to uvicorn itself (rather than leaving it as a child
# of /bin/sh), so SIGTERM from Render's stop/redeploy still reaches it
# directly for a clean shutdown. ${PORT:-8000} falls back to 8000 for
# local `docker run`/docker-compose, where $PORT is unset.
CMD ["sh", "-c", "exec uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
