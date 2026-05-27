# ─────────────────────────────────────────────────────────────
# Stage 1 — Builder: instala dependencias en un entorno limpio
# ─────────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

# Copiar solo el archivo de dependencias primero (mejor cache de capas)
COPY requirements.txt .

RUN pip install --upgrade pip \
    && pip install --no-cache-dir --prefix=/install -r requirements.txt


# ─────────────────────────────────────────────────────────────
# Stage 2 — Runtime: imagen mínima de producción
# ─────────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

# Metadatos de la imagen
LABEL maintainer="mlops-team"
LABEL description="Telco Churn Prediction API — XGBoost + FastAPI"
LABEL version="1.0.0"

# Usuario no-root para seguridad
RUN groupadd -r appuser && useradd -r -g appuser appuser

WORKDIR /service

# Copiar dependencias instaladas en el stage builder
COPY --from=builder /install /usr/local

# Copiar código de la aplicación y el modelo serializado
COPY app/ ./app/

# Cambiar propietario al usuario no-root
RUN chown -R appuser:appuser /service

USER appuser

# Variables de entorno con valores por defecto
ENV MODEL_PATH=/service/app/model.joblib \
    CHURN_THRESHOLD=0.5 \
    PORT=8000

# Puerto expuesto
EXPOSE 8000

# Health check integrado en Docker
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" \
    || exit 1

# Comando de arranque con Uvicorn (servidor ASGI de producción)
CMD ["sh", "-c", "uvicorn app.api:app --host 0.0.0.0 --port ${PORT} --workers 2 --log-level info"]
