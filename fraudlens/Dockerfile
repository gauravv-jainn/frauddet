# FraudLens: one image, two services (scoring API and Streamlit app). See docker-compose.yml.
# Train on the host first (python -m fraudkit.train) so artifacts/ holds a model; it is mounted read-only.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 curl \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 10001 fraudlens

WORKDIR /app
COPY requirements.txt requirements-api.txt ./
RUN pip install -r requirements.txt -r requirements-api.txt

COPY fraudkit ./fraudkit
COPY api ./api
COPY app ./app
COPY configs ./configs
COPY .streamlit ./.streamlit
RUN mkdir -p /app/data /app/artifacts && chown -R fraudlens:fraudlens /app/data

USER fraudlens
ENV FRAUDLENS_DB=/app/data/fraudlens.sqlite
EXPOSE 8000 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD curl -fsS http://localhost:8000/v1/health || exit 1
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers"]
