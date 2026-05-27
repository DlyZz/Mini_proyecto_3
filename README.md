# Telco Churn Prediction — MLOps

Servicio de inferencia para predicción de abandono de clientes, basado en el modelo XGBoost entrenado en el Notebook 2 (AUC-ROC = 0.8476).

---

## Estructura del proyecto

```
.
├── app/
│   ├── main.py            # Servicio FastAPI (endpoints /predict, /predict/batch, /health)
│   └── model.joblib       # Pipeline serializado (copiado desde el Notebook 2)
├── tests/
│   └── test_api.py        # Pruebas unitarias con pytest (cobertura ≥ 80%)
├── .github/
│   └── workflows/
│       └── ci-cd.yml      # Pipeline CI/CD: lint → test → build & push Docker
├── 3_monitoring.ipynb     # Estrategia de monitoreo: deriva de datos y del modelo
├── Dockerfile             # Imagen multi-stage (builder + runtime)
├── requirements.txt       # Dependencias Python
└── README.md
```

---

## Quickstart — ejecución local

### 1. Sin Docker

```bash
# Instalar dependencias
pip install -r requirements.txt

# Copiar el modelo al directorio esperado
cp ruta/a/tu/model.joblib app/model.joblib

# Levantar el servidor
uvicorn app.api:app --reload --port 8000
```

La documentación interactiva estará disponible en:  
- Swagger UI: http://localhost:8000/docs  
- ReDoc: http://localhost:8000/redoc

### 2. Con Docker

```bash
# Construir la imagen
docker build -t telco-churn-api:latest .

# Ejecutar el contenedor
docker run -p 8000:8000 \
  -v $(pwd)/app/model.joblib:/service/app/model.joblib \
  telco-churn-api:latest
```

### 3. Usar la imagen desde GitHub Container Registry

```bash
docker pull ghcr.io/<tu-usuario>/telco-churn-api:latest

docker run -p 8000:8000 \
  -v $(pwd)/app/model.joblib:/service/app/model.joblib \
  ghcr.io/<tu-usuario>/telco-churn-api:latest
```

---

## Endpoints

### `GET /health`
Verifica que el servicio y el modelo estén operativos.

```json
{
  "status": "ok",
  "model_loaded": true,
  "threshold": 0.5
}
```

---

### `POST /predict`
Predicción para un cliente individual.

**Request:**
```json
{
  "gender": "Male",
  "SeniorCitizen": 0,
  "Partner": "Yes",
  "Dependents": "No",
  "tenure": 12,
  "PhoneService": "Yes",
  "MultipleLines": "No",
  "InternetService": "Fiber optic",
  "OnlineSecurity": "No",
  "OnlineBackup": "No",
  "DeviceProtection": "No",
  "TechSupport": "No",
  "StreamingTV": "Yes",
  "StreamingMovies": "Yes",
  "Contract": "Month-to-month",
  "PaperlessBilling": "Yes",
  "PaymentMethod": "Electronic check",
  "MonthlyCharges": 79.85,
  "TotalCharges": 958.20
}
```

**Response:**
```json
{
  "prediction": {
    "churn_probability": 0.7823,
    "churn_prediction": true,
    "risk_level": "ALTO"
  },
  "model_version": "xgboost-v1.0",
  "threshold_used": 0.5
}
```

---

### `POST /predict/batch`
Predicción masiva (hasta 1 000 clientes por request).

**Request:** `{ "customers": [ <CustomerFeatures>, ... ] }`

**Response:**
```json
{
  "predictions": [ ... ],
  "total_customers": 5,
  "predicted_churners": 3,
  "churn_rate_predicted": 0.6,
  "model_version": "xgboost-v1.0"
}
```

---

## Variables de entorno

| Variable | Valor por defecto | Descripción |
|---|---|---|
| `MODEL_PATH` | `app/model.joblib` | Ruta al pipeline serializado |
| `CHURN_THRESHOLD` | `0.5` | Umbral de probabilidad para clasificar como churner |
| `PORT` | `8000` | Puerto del servidor Uvicorn |

---

## Pruebas

```bash
# Ejecutar todas las pruebas con cobertura
pytest tests/ -v --cov=app --cov-report=term-missing

# Solo pruebas de un módulo específico
pytest tests/test_api.py::TestPredict -v
```

Cobertura mínima requerida: **80%** (configurada en el workflow CI/CD).

---

## CI/CD Pipeline

El flujo `.github/workflows/ci-cd.yml` se activa en cada push:

```
push a cualquier rama
    │
    ├─▶ [lint] ruff check + ruff format --check
    │         │
    │         ▼ (solo si lint pasa)
    ├─▶ [test] pytest + cobertura ≥ 80%
    │         │
    │         ▼ (solo en push a main)
    └─▶ [build & push] Docker → ghcr.io/<usuario>/telco-churn-api:latest
```

Los Pull Requests reciben automáticamente un comentario con el reporte de cobertura.

---

## Niveles de riesgo

| Probabilidad | Nivel | Acción recomendada |
|---|---|---|
| < 0.40 | 🟢 BAJO | Sin intervención |
| 0.40 – 0.69 | 🟡 MEDIO | Oferta de retención estándar |
| ≥ 0.70 | 🔴 ALTO | Contacto prioritario + oferta personalizada |

---

## Monitoreo

Ver `3_monitoring.ipynb` para la estrategia completa, incluyendo:
- Detección de deriva de datos (KS test + Chi-cuadrado)
- Monitoreo de AUC-ROC con ventana deslizante
- Métricas de producción del servicio (latencia P50/P95/P99, tasa de errores)
- Plan de alertas y umbrales de re-entrenamiento
