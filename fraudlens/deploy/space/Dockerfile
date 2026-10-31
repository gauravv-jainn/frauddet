# FraudLens on a Hugging Face Docker Space. Uploaded to the Space root by scripts/deploy_space.py.
# Artifacts are not in the image: fraudkit.bootstrap downloads them at start-up, when the Space secrets are available.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 1000 fraudlens

WORKDIR /app
COPY requirements.txt requirements-deploy.txt ./
RUN pip install -r requirements.txt -r requirements-deploy.txt

COPY fraudkit ./fraudkit
COPY app ./app
COPY configs ./configs
COPY .streamlit ./.streamlit
RUN mkdir -p /app/data /app/artifacts && chown -R fraudlens:fraudlens /app/data /app/artifacts

USER fraudlens
ENV FRAUDLENS_DB=/app/data/fraudlens.sqlite
EXPOSE 7860
# The app does not need the secrets once bootstrap has used them, so they are removed from its environment.
# XSRF is off because the huggingface.co page embeds the app in an iframe, where the check rejects file uploads.
CMD ["sh", "-c", "python -m fraudkit.bootstrap && exec env -u HF_TOKEN -u FRAUDLENS_ADMIN_PASSWORD -u FRAUDLENS_DEMO_PASSWORD streamlit run app/streamlit_app.py --server.address 0.0.0.0 --server.port 7860 --server.enableXsrfProtection false --server.enableCORS false --client.toolbarMode minimal"]
