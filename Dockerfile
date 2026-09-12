FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    NLTK_DATA=/usr/share/nltk_data \
    HF_HOME=/opt/huggingface \
    TRANSFORMER_MODEL=cardiffnlp/twitter-xlm-roberta-base-sentiment

# LightGBM needs the OpenMP runtime, which the slim image does not include
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt requirements-serving.txt ./
RUN pip install --no-cache-dir \
      --index-url https://download.pytorch.org/whl/cpu \
      --extra-index-url https://pypi.org/simple \
      -r requirements-serving.txt \
    && python -m nltk.downloader -d /usr/share/nltk_data stopwords wordnet

# Bake the sentiment model into the image so the container starts fast and needs no network
RUN python -c "import os; from transformers import AutoModelForSequenceClassification, AutoTokenizer; \
name = os.environ['TRANSFORMER_MODEL']; AutoTokenizer.from_pretrained(name); \
AutoModelForSequenceClassification.from_pretrained(name)" \
    && chmod -R a+rX /opt/huggingface

COPY app.py lgbm_model.pkl tfidf_vectorizer.pkl ./
COPY src/ ./src/

RUN useradd --create-home appuser
USER appuser

EXPOSE 8080
# One worker: each holds its own copy of the model in memory. Threads handle concurrent requests.
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "--workers", "1", "--threads", "4", "--timeout", "180", "--preload", "app:create_app()"]
