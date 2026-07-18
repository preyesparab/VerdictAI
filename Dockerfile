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

CMD ["python", "app.py"]
