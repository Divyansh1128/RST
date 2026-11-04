FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY server ./server
COPY web ./web
COPY artifacts/trained ./artifacts/trained
ENV RST_ARTIFACTS=/app/artifacts/trained PORT=8000
CMD ["sh", "-c", "uvicorn server.app:app --host 0.0.0.0 --port ${PORT} --proxy-headers --forwarded-allow-ips='*'"]
