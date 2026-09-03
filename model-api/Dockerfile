# Serving image — one app, one env var (MODEL_URI). The same image runs as both A/B
# arms in docker-compose.yml; only MODEL_URI differs.
#
# recommender.py is NOT copied in here: it travels inside the logged pyfunc artifact via
# train.py's `code_paths`, which is the whole point of pyfunc packaging.
FROM python:3.12-slim

WORKDIR /app

COPY model-api/requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY model-api/app.py app.py

HEALTHCHECK --interval=10s --timeout=5s --retries=10 --start-period=15s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
