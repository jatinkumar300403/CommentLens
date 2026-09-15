FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app
ENV NLTK_DATA=/usr/share/nltk_data

# LightGBM needs the OpenMP runtime, which the slim image does not include
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# CPU-only PyTorch index: the default Linux build of torch pulls in CUDA libraries worth ~2 GB
COPY requirements.txt requirements-serving.txt ./
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple -r requirements-serving.txt && python -m nltk.downloader -d /usr/share/nltk_data stopwords wordnet

# Bake one copy of the sentiment model into the image. The Hub cache would hold the weights twice,
# so it lives in /tmp/hf and is deleted in the same layer.
COPY scripts/bake_model.py /tmp/bake_model.py
RUN HF_HOME=/tmp/hf python /tmp/bake_model.py cardiffnlp/twitter-xlm-roberta-base-sentiment /opt/models/cardiffnlp/twitter-xlm-roberta-base-sentiment && rm -rf /tmp/hf /tmp/bake_model.py && chmod -R a+rX /opt/models

# Serve the baked copy. Offline mode stops transformers checking the Hub for updates at startup.
ENV TRANSFORMER_MODEL=/opt/models/cardiffnlp/twitter-xlm-roberta-base-sentiment
ENV HF_HUB_OFFLINE=1
ENV TRANSFORMERS_OFFLINE=1

COPY app.py lgbm_model.pkl tfidf_vectorizer.pkl ./
COPY src/ ./src/

RUN useradd --create-home appuser
USER appuser

EXPOSE 8080
# Listen on $PORT, which Cloud Run sets (8080 when run anywhere else).
# One worker and no --preload: preloading loads the model in the parent process and forks it, which
# duplicated much of its memory and got the instance killed at Cloud Run's 2 GiB limit.
CMD ["sh", "-c", "exec gunicorn --bind 0.0.0.0:${PORT:-8080} --workers 1 --threads 4 --timeout 180 'app:create_app()'"]
