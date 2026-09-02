# A/B router image — proxies /recommend to model-v1 / model-v2, logs assignment +
# outcome to events-db (docs/PLAN.md 5.6).
FROM python:3.12-slim

WORKDIR /app

COPY ab-router/requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY ab-router/app.py app.py

HEALTHCHECK --interval=10s --timeout=5s --retries=10 --start-period=15s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
