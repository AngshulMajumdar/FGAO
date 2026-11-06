FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
RUN useradd --create-home --uid 1000 user
WORKDIR /home/user/app

COPY --chown=user pyproject.toml ./
COPY --chown=user src ./src
RUN pip install ".[server,openai]"
COPY --chown=user fgao.yaml ./
COPY --chown=user data ./data

USER user
EXPOSE 7860
CMD ["fgao", "serve", "--config", "fgao.yaml", "--host", "0.0.0.0", "--port", "7860"]
