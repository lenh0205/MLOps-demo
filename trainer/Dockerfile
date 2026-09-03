# One-shot training container — runs `python train.py` and exits.
# Build context is the repo root: the image needs both trainer/ and data/ (see
# DEFAULT_DATA in train.py, which is data/interactions.csv relative to the repo root).
FROM python:3.12-slim

WORKDIR /app

COPY trainer/requirements.txt trainer/requirements.txt
RUN pip install --no-cache-dir -r trainer/requirements.txt

COPY trainer/ trainer/
COPY data/ data/

CMD ["python", "trainer/train.py"]
