FROM e2bdev/code-interpreter:latest
RUN pip install --no-cache-dir \
    torch --index-url https://download.pytorch.org/whl/cpu \
    numpy einops scipy matplotlib

RUN addgroup --system --gid 1001 appgroup && \
    adduser --system --uid 1001 --gid 1001 --no-create-home appuser

USER appuser
