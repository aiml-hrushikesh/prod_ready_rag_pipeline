FROM python:3.14-slim

WORKDIR /app

COPY pyproject.toml poetry.lock* /app/
RUN pip install --no-cache-dir poetry
RUN poetry config virtualenvs.create false && poetry install --no-root --no-interaction --no-ansi

COPY . /app

ENV OLLAMA_BASE_URL=http://host.docker.internal:11434
ENV PYTHONUNBUFFERED=1

ENTRYPOINT ["python", "-m", "src.main", "run"]
