FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    NLTK_DATA=/usr/share/nltk_data

# LightGBM needs the OpenMP runtime, which the slim image does not include
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && python -m nltk.downloader -d /usr/share/nltk_data stopwords wordnet

COPY app.py lgbm_model.pkl tfidf_vectorizer.pkl ./
COPY src/ ./src/

RUN useradd --create-home appuser
USER appuser

EXPOSE 8080
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "--workers", "2", "--threads", "4", "--timeout", "120", "app:create_app()"]
