# BASE can point at a mirrored or internal Python 3.9+ image
ARG BASE=python:3.12-slim
FROM ${BASE}

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install ".[server,openai,anthropic,huggingface]" && useradd --create-home --uid 10001 fgao

COPY deploy/fgao.yaml /etc/fgao/fgao.yaml
USER fgao
EXPOSE 8000

HEALTHCHECK --interval=15s --timeout=3s --start-period=20s \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2).status == 200 else 1)"

# one worker per container; scale with replicas (metrics and admission control are per process)
CMD ["fgao", "serve", "--config", "/etc/fgao/fgao.yaml", "--host", "0.0.0.0", "--port", "8000"]
