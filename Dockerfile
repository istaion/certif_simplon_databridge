# syntax=docker/dockerfile:1
FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc \
        curl \
    && rm -rf /var/lib/apt/lists/*

# uv pour l'installation rapide des paquets
COPY --from=ghcr.io/astral-sh/uv:0.10.4 /uv /usr/local/bin/uv

WORKDIR /app

# Installer les dépendances (layer mis en cache tant que requirements-api.txt ne change pas)
COPY requirements-api.txt .
RUN uv pip install --system --no-cache -r requirements-api.txt

# Copier les sources
COPY pyproject.toml .
COPY data_process/ ./data_process/
COPY API/ ./API/
COPY prediction_passages/ ./prediction_passages/

# Installer le package local (data_process) sans ses dépendances
RUN uv pip install --system --no-cache --no-deps -e .

# Utilisateur non-root
ARG APP_USER="appuser"
RUN useradd --create-home --no-log-init ${APP_USER} && \
    chown -R ${APP_USER}:${APP_USER} /app

# Créer le volume
VOLUME /app/data_process/config
RUN chown -R ${APP_USER}:${APP_USER} /app/data_process/config

USER ${APP_USER}

EXPOSE 8000

CMD ["uvicorn", "API.app:app", "--host", "0.0.0.0", "--port", "8000"]
