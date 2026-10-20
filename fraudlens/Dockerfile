# FraudLens scoring API + Streamlit app.
# Train first on the host (python -m fraudkit.train) so artifacts/ contains the model, then:
#   docker build -t fraudlens .
#   docker run -p 8000:8000 -e FRAUDLENS_API_KEY=change-me fraudlens                     # API
#   docker run -p 8501:8501 fraudlens streamlit run app/streamlit_app.py --server.address 0.0.0.0   # app
FROM python:3.11-slim

WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
COPY requirements.txt requirements-api.txt ./
RUN pip install --no-cache-dir -r requirements.txt -r requirements-api.txt

COPY fraudkit ./fraudkit
COPY api ./api
COPY app ./app
COPY configs ./configs
COPY .streamlit ./.streamlit
COPY artifacts ./artifacts

EXPOSE 8000 8501
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
