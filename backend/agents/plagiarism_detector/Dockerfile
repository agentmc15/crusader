FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y gcc && rm -rf /var/lib/apt/lists/*

COPY ../../pyproject.toml ../../poetry.lock* ./
RUN pip install poetry && \
    poetry config virtualenvs.create false && \
    poetry install --no-dev

COPY . .

EXPOSE 8002

CMD ["uvicorn", "agents.plagiarism_detector.app.main:app", "--host", "0.0.0.0", "--port", "8002"]
