#  Predicción de Churn en Telecomunicaciones

Sistema completo de Machine Learning para la predicción de abandono de clientes (*churn*) en una empresa de telecomunicaciones. Incluye análisis exploratorio, modelado con cuatro algoritmos de árboles, interpretabilidad con LIME y una arquitectura MLOps lista para producción.

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
