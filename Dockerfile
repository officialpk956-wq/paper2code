FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
# Render has no GPU. Install the CPU-only PyTorch wheel (~190 MB) explicitly first;
# the default CUDA build (~2 GB, pulled transitively by sentence-transformers) is
# what breaks the build (BrokenPipe / OOM on `pip install`). torch is then already
# satisfied for sentence-transformers, block-viz and labs.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
RUN addgroup --system --gid 1001 appgroup && \
    adduser --system --uid 1001 --gid 1001 --no-create-home appuser

RUN chown -R appuser:appgroup /app

USER appuser

EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:${PORT:-8080}/api/health')" || exit 1
CMD ["sh", "-c", "uvicorn backend.server:app --host 0.0.0.0 --port ${PORT:-8080}"]
