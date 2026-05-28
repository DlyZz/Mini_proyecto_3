#  Predicción de Churn en Telecomunicaciones

Sistema completo de Machine Learning para la predicción de abandono de clientes (*churn*) en una empresa de telecomunicaciones. Incluye análisis exploratorio, modelado con cuatro algoritmos de árboles, interpretabilidad con LIME y una arquitectura MLOps lista para producción.

---

##  Tabla de contenidos

1. [Descripción del problema](#descripción-del-problema)
2. [Dataset](#dataset)
3. [Estructura del proyecto](#estructura-del-proyecto)
4. [Notebooks](#notebooks)
5. [Resultados del modelado](#resultados-del-modelado)
6. [Arquitectura MLOps](#arquitectura-mlops)
7. [API de inferencia](#api-de-inferencia)
8. [Docker](#docker)
9. [CI/CD con GitHub Actions](#cicd-con-github-actions)
10. [Monitoreo en producción](#monitoreo-en-producción)
11. [Instalación y uso local](#instalación-y-uso-local)
12. [Dependencias](#dependencias)

---

## Descripción del problema

Las empresas de telecomunicaciones enfrentan tasas significativas de pérdida de clientes. Identificar de forma temprana a los clientes con mayor probabilidad de abandonar el servicio permite intervenir con estrategias de retención focalizadas, reduciendo el costo de adquisición de nuevos clientes.

Este proyecto construye un sistema de clasificación binaria que predice si un cliente hará *churn* (`1`) o no (`0`), con las siguientes características clave:

- **Desbalance de clases:** 26.5% churn vs 73.5% no-churn (ratio 2.77:1), gestionado con pesos de clase y `scale_pos_weight`.
- **Perfil de cliente de alto riesgo:** contrato month-to-month, pago por electronic check, antigüedad menor a 18 meses, cargo mensual superior a $74.
- **Métrica principal:** AUC-ROC, robusta ante el desbalance de clases.

---

## Dataset

**Nombre:** Telco Customer Churn  
**Fuente:** [Kaggle — blastchar/telco-customer-churn](https://www.kaggle.com/blastchar/telco-customer-churn)  
**Tamaño:** 7 043 registros × 21 columnas  
**Variable objetivo:** `Churn` (Yes/No → 1/0)

El dataset contiene tres tipos de variables:

| Tipo | Variables |
|---|---|
| Demográficas | `gender`, `SeniorCitizen`, `Partner`, `Dependents` |
| De contrato y facturación | `Contract`, `PaymentMethod`, `PaperlessBilling`, `MonthlyCharges`, `TotalCharges`, `tenure` |
| De servicios contratados | `PhoneService`, `MultipleLines`, `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies` |

**Tratamiento de datos:**
- `TotalCharges` se carga como `object` (contiene espacios en clientes nuevos con tenure=0); se convierte a numérico con `pd.to_numeric(..., errors='coerce')`.
- 11 valores faltantes (0.16%) en `TotalCharges`, imputados con la mediana.
- `customerID` descartado por ser identificador sin valor predictivo.

---

## Estructura del proyecto

```
telco-churn-mlops/
├── data/
│   ├── WA_Fn-UseC_-Telco-Customer-Churn.csv   # Dataset original (Kaggle)
│   └── telco_churn_clean.csv                   # Dataset limpio (generado por notebook 1)
│
├── notebooks/
│   ├── 1_eda_preprocessing.ipynb               # EDA completo + preprocesamiento
│   ├── 2_model_training.ipynb                  # Pipelines, GridSearchCV, evaluación
│   └── 3_interpretability.ipynb                # Explicabilidad con LIME
│
├── mlops/
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py                             # FastAPI — endpoints de inferencia
│   │   ├── schemas.py                          # Modelos Pydantic (entrada/salida)
│   │   └── model.joblib                        # Pipeline XGBoost serializado
│   │
│   ├── tests/
│   │   ├── __init__.py
│   │   └── test_api.py                         # Pruebas unitarias (pytest)
│   │
│   ├── .github/workflows/
│   │   └── ci-cd.yml                           # Pipeline CI/CD (GitHub Actions)
│   │
│   ├── 3_monitoring.ipynb                      # Estrategia de monitoreo MLOps
│   ├── Dockerfile                              # Imagen Docker multi-stage
│   ├── requirements.txt                        # Dependencias del proyecto
│   └── README.md                              # Este archivo
```

---

## Notebooks

### `1_eda_preprocessing.ipynb` — Análisis Exploratorio

Cubre el análisis completo del dataset antes del modelado:

- **Carga e inspección inicial:** shape, tipos de variables, estadísticas descriptivas numéricas y categóricas.
- **Valores faltantes:** detección y visualización (solo `TotalCharges`, 0.16%).
- **Desbalance de clases:** ratio 2.77:1; gráficos de barras y torta.
- **Distribuciones numéricas:** histogramas + KDE para `tenure`, `MonthlyCharges`, `TotalCharges`, `SeniorCitizen`. Se detecta bimodalidad en `tenure`.
- **Variables categóricas vs Churn:** tasas de churn por categoría para las 15 variables categóricas.
- **Matriz de correlación:** `tenure` tiene la correlación negativa más fuerte con churn (-0.35); `MonthlyCharges` correlaciona positivamente.
- **Análisis de outliers:** criterio IQR; `MonthlyCharges` presenta outliers leves (~3.7%), mantenidos porque son datos reales.
- **Servicios vs Churn:** `OnlineSecurity` y `TechSupport` reducen el churn de ~42% a ~15%; `StreamingTV` y `StreamingMovies` no diferencian.
- **Perfil del cliente de riesgo:**

| Característica | Cliente en riesgo | Cliente fidelizado |
|---|---|---|
| Contrato | Month-to-month (88.6%) | One/Two year |
| Pago | Electronic check (57.3%) | Automático/tarjeta |
| Antigüedad media | 18 meses | 37.6 meses |
| Cargo mensual | $74.44 | $61.27 |

**Salida:** `data/telco_churn_clean.csv` con `TotalCharges` imputada y `Churn` en formato binario (0/1).

---

### `2_model_training.ipynb` — Modelado y Evaluación

#### Preprocesamiento con Pipeline

```
ColumnTransformer
├── Numéricas (tenure, MonthlyCharges, TotalCharges, SeniorCitizen)
│   └── SimpleImputer(median) → StandardScaler
└── Categóricas (gender, Contract, PaymentMethod, ...)
    └── SimpleImputer(most_frequent) → OneHotEncoder(handle_unknown='ignore')
```

División estratificada: 80% entrenamiento (5 634 muestras) / 20% test (1 409 muestras), con `stratify=y` para preservar el ratio de clases en ambos conjuntos.

#### Modelos entrenados

| Modelo | Hiperparámetros ajustados |
|---|---|
| Random Forest | `n_estimators`, `max_depth`, `min_samples_split`, `min_samples_leaf`, `max_features`, `criterion` |
| XGBoost | `n_estimators`, `max_depth`, `learning_rate`, `subsample`, `colsample_bytree`, `reg_alpha`, `reg_lambda`, `scale_pos_weight` |
| CatBoost | `iterations`, `depth`, `learning_rate`, `l2_leaf_reg`, `bagging_temperature`, `subsample` |
| LightGBM | `n_estimators`, `num_leaves`, `max_depth`, `learning_rate`, `min_child_samples`, `subsample`, `colsample_bytree`, `reg_alpha`, `reg_lambda` |

Búsqueda con `GridSearchCV` + `StratifiedKFold(n_splits=5)`, optimizando AUC-ROC.

#### Resultados en test

| Modelo | Accuracy | Precision | Recall | F1 | AUC-ROC |
|---|---|---|---|---|---|
| Random Forest | 0.7630 | 0.5388 | 0.7433 | 0.6247 | 0.8417 |
| **XGBoost** | 0.7502 | 0.5193 | 0.7914 | 0.6271 | **0.8476** |
| CatBoost | 0.7480 | 0.5163 | 0.8048 | 0.6290 | 0.8447 |
| LightGBM | 0.7544 | 0.5255 | 0.7727 | 0.6255 | 0.8417 |

**Modelo seleccionado: XGBoost** — mejor AUC-ROC en test (0.8476) y alta estabilidad en validación cruzada (0.8482 ± 0.011, IC 95%: [0.826, 0.870]).

Variables más importantes (consistentes entre modelos): `tenure`, `Contract_Month-to-month`, `TotalCharges`, `MonthlyCharges`, `PaymentMethod_Electronic check`.

**Salida:** `app/model.joblib` — pipeline XGBoost completo serializado (122.9 KB).

---

### `3_interpretability.ipynb` — Explicabilidad con LIME

Aplica `LimeTabularExplainer` sobre el pipeline XGBoost entrenado para explicar predicciones individuales. Analiza al menos tres casos representativos:

- **Caso 1 — Alto riesgo (Churn = 1):** cliente con contrato month-to-month, tenure bajo y alto cargo mensual.
- **Caso 2 — Bajo riesgo (Churn = 0):** cliente con contrato de dos años, tenure alto y servicios de seguridad activos.
- **Caso 3 — Riesgo medio (frontera de decisión):** cliente con características mixtas donde el modelo tiene menor certeza.

Para cada caso se muestra: la probabilidad de churn, las variables que más empujan hacia cada clase y una interpretación del resultado en términos de negocio.

---

## Resultados del modelado

### ¿Por qué AUC-ROC como métrica principal?

En un problema de churn con desbalance 2.77:1, la *accuracy* es engañosa: un modelo que prediga siempre "No Churn" tendría 73.5% de accuracy sin detectar ningún cliente en riesgo. El AUC-ROC mide la capacidad discriminativa del modelo en todos los umbrales posibles, siendo agnóstica al desbalance.

### ¿Por qué XGBoost?

- Mejor AUC-ROC en test (0.8476) y en CV (0.8482).
- Alta estabilidad entre folds (σ = 0.011).
- Recall de 79.1% sobre los 374 clientes reales de churn en test: detecta correctamente a 296 de ellos.
- Maneja el desbalance directamente con `scale_pos_weight=2.77`.

---

## Arquitectura MLOps

```
┌─────────────────────────────────────────────────────────────────┐
│                        DESARROLLO                               │
│  Notebooks (EDA → Modelado → LIME) → model.joblib               │
└────────────────────────┬────────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────────┐
│                      SERVICIO API                               │
│  FastAPI (main.py)                                              │
│  ├── GET  /health          → estado del servicio                │
│  ├── POST /predict         → predicción individual              │
│  └── POST /predict/batch   → predicción por lote (hasta 1000)   │
└────────────────────────┬────────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────────┐
│                    CONTENERIZACIÓN                              │
│  Docker (multi-stage build)                                     │
│  ├── Stage 1 (builder): instala dependencias                    │
│  └── Stage 2 (runtime): imagen mínima, usuario no-root          │
└────────────────────────┬────────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────────┐
│                       CI/CD                                     │
│  GitHub Actions (ci-cd.yml)                                     │
│  ├── Job 1: Lint con ruff                                       │
│  ├── Job 2: Tests con pytest (cobertura ≥ 80%)                  │
│  └── Job 3: Build + Push imagen a ghcr.io (solo en main)        │
└────────────────────────┬────────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────────┐
│                     MONITOREO                                   │
│  ├── Data drift: KS-test (numéricas) + Chi² (categóricas)       │
│  ├── Model drift: AUC-ROC rolling por ventana temporal          │
│  ├── Distribución de predicciones: detección de sesgo           │
│  └── Métricas de servicio: latencia, throughput, errores        │
└─────────────────────────────────────────────────────────────────┘
```

---

## API de inferencia

La API expone tres endpoints:

### `GET /health`
Verifica el estado del servicio.

```json
{
  "status": "ok",
  "model_loaded": true,
  "threshold": 0.5
}
```

### `POST /predict`
Recibe las características de un cliente y devuelve la predicción.

**Body de ejemplo:**
```json
{
  "gender": "Male",
  "SeniorCitizen": 1,
  "Partner": "No",
  "Dependents": "No",
  "tenure": 5,
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
  "MonthlyCharges": 99.9,
  "TotalCharges": 499.5
}
```

**Respuesta:**
```json
{
  "prediction": {
    "churn_probability": 0.9612,
    "churn_prediction": true,
    "risk_level": "ALTO"
  },
  "model_version": "xgboost-v1.0",
  "threshold_used": 0.5
}
```

Los niveles de riesgo son: `BAJO` (< 0.40), `MEDIO` (0.40–0.69), `ALTO` (≥ 0.70).

### `POST /predict/batch`
Recibe una lista de clientes (máximo 1 000) y devuelve predicciones agregadas.

```json
{
  "customers": [ { ...cliente1... }, { ...cliente2... } ]
}
```

**Respuesta:**
```json
{
  "predictions": [...],
  "total_customers": 2,
  "predicted_churners": 1,
  "churn_rate_predicted": 0.5,
  "model_version": "xgboost-v1.0"
}
```

---

## Docker

### Construir y ejecutar

```bash
# Construir la imagen
docker build -t telco-churn-api .

# Ejecutar el contenedor (primera vez)
docker run -d --name telco-churn-container -p 8000:8000 telco-churn-api

# Iniciar / detener el contenedor
docker start telco-churn-container
docker stop telco-churn-container

# Ver estado
docker ps -a | grep telco-churn
```

### Probar la API desde Docker

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "gender": "Male", "SeniorCitizen": 1, "Partner": "No", "Dependents": "No",
    "tenure": 5, "PhoneService": "Yes", "MultipleLines": "No",
    "InternetService": "Fiber optic", "OnlineSecurity": "No", "OnlineBackup": "No",
    "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "Yes",
    "StreamingMovies": "Yes", "Contract": "Month-to-month",
    "PaperlessBilling": "Yes", "PaymentMethod": "Electronic check",
    "MonthlyCharges": 99.9, "TotalCharges": 499.5
  }'
```

El Dockerfile usa **multi-stage build** para minimizar el tamaño de la imagen final:
- **Stage builder:** instala todas las dependencias de Python.
- **Stage runtime:** copia solo los artefactos necesarios, ejecuta con usuario no-root (`appuser`) por seguridad.
- Incluye `HEALTHCHECK` integrado que consulta `/health` cada 30 segundos.

---

## CI/CD con GitHub Actions

El archivo `.github/workflows/ci-cd.yml` define tres jobs encadenados:

| Job | Herramienta | Condición de ejecución |
|---|---|---|
| 🔍 **Lint** | `ruff check` + `ruff format` | Todo push y PR |
| 🧪 **Tests** | `pytest` con cobertura ≥ 80% | Solo si lint pasa |
| 🐳 **Build & Push** | Docker → `ghcr.io` | Solo en push a `main` con tests exitosos |

El job de build genera automáticamente dos tags: `latest` y `sha-<commit>`, y publica un resumen del deploy en la pestaña Actions de GitHub.

Para ver los resultados: repositorio en GitHub → pestaña **Actions**.

---

## Monitoreo en producción

Documentado en `mlops/3_monitoring.ipynb`. La estrategia cubre cuatro capas:

**Deriva de datos (Data Drift)**
- Variables numéricas: prueba Kolmogorov-Smirnov (umbral: p-value < 0.05).
- Variables categóricas: prueba Chi-cuadrado.
- Frecuencia sugerida: semanal.

**Deriva del modelo (Model/Concept Drift)**
- AUC-ROC calculado en ventanas temporales deslizantes.
- Alerta si AUC-ROC cae más de 0.03 puntos respecto a la línea base.

**Distribución de predicciones**
- Monitoreo semanal del porcentaje de clientes clasificados como churn.
- Alertas si la tasa predicha se desvía más de 15% de la línea base histórica.

**Métricas de servicio**
- Latencia P95 (umbral: < 200 ms), tasa de errores (< 1%), throughput.
- Medibles con el header `X-Response-Time-Ms` que incluye cada respuesta.

**Plan de re-entrenamiento:** activado automáticamente si AUC-ROC < 0.80 en producción o si se detecta drift significativo en más de 3 variables clave.

---

## Instalación y uso local

### Requisitos previos

- Python 3.10 o 3.11
- Docker Desktop (para contenerización)
- Git

### Instalación

```bash
# Clonar el repositorio
git clone https://github.com/<usuario>/telco-churn-mlops.git
cd telco-churn-mlops

# Instalar dependencias
pip install -r mlops/requirements.txt
```

### Ejecutar la API localmente (sin Docker)

```bash
cd mlops
uvicorn app.main:app --reload --port 8001
```

La documentación interactiva estará disponible en `http://localhost:8001/docs`.

### Ejecutar las pruebas

```bash
cd mlops
pytest tests/ -v --cov=app --cov-report=term-missing
```

### Regenerar el modelo

Ejecutar los notebooks en orden desde la raíz del proyecto:

```
1_eda_preprocessing.ipynb  →  genera data/telco_churn_clean.csv
2_model_training.ipynb     →  genera mlops/app/model.joblib
3_interpretability.ipynb   →  genera explicaciones LIME
```

---

## Dependencias

```
fastapi>=0.95.0
uvicorn
scikit-learn
pandas
numpy
joblib
lime
xgboost
catboost
lightgbm
matplotlib
seaborn
pytest
ruff
```

Generadas automáticamente con `pip freeze > requirements.txt` desde el entorno virtual del proyecto.

---

## Herramientas y tecnologías

| Capa | Herramientas |
|---|---|
| Análisis y modelado | pandas, numpy, scikit-learn, xgboost, catboost, lightgbm |
| Visualización | matplotlib, seaborn |
| Interpretabilidad | lime |
| Servicio API | FastAPI, Pydantic, uvicorn |
| Contenerización | Docker (multi-stage) |
| CI/CD | GitHub Actions, ruff, pytest |
| Entorno de desarrollo | Jupyter Notebook / VS Code |