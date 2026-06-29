FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
RUN addgroup --system --gid 1001 appgroup && \
    adduser --system --uid 1001 --gid 1001 --no-create-home appuser

RUN chown -R appuser:appgroup /app

USER appuser

EXPOSE 8000
CMD ["uvicorn", "backend.server:app", "--host", "0.0.0.0", "--port", "8000"]
