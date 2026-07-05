FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY backend/ ./backend/
COPY frontend/ ./frontend/
COPY models/encoder.json models/vocab.bpe ./models/

ENV PORT=7860
EXPOSE 7860

CMD ["python3", "backend/start.py"]
