# ── Dockerfile ──────────────────────────────────────────────────
# Build & run:
#   docker build -t urdu-story-api .
#   docker run -p 8000:8000 urdu-story-api
# ────────────────────────────────────────────────────────────────
FROM python:3.11-slim

# Prevent Python from buffering stdout/stderr
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install dependencies first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code + model + tokenizer assets
COPY tokenizer_utils.py .
COPY app.py .
COPY model/ model/
COPY tokenizer/ tokenizer/

# Expose the API port
EXPOSE 8000

# Start the server
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
