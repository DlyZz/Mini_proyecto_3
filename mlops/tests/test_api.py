"""
Pruebas unitarias — Telco Churn Prediction API
Ejecutar con: pytest tests/ -v --cov=app --cov-report=term-missing
"""

from __future__ import annotations

import numpy as np
import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock


# Datos de prueba
VALID_CUSTOMER = {
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
    "TotalCharges": 958.20,
}


# Helpers
def _mock_single(prob: float):
    m = MagicMock()
    m.predict_proba.return_value = np.array([[1 - prob, prob]])
    return m


def _mock_batch(probs: list[float]):
    m = MagicMock()
    m.predict_proba.return_value = np.array([[1 - p, p] for p in probs])
    return m


# Fixture: bloquea joblib.load para TODOS los tests 
@pytest.fixture(autouse=True)
def no_disk_load(monkeypatch):
    """Impide que el lifespan lea model.joblib del disco en cualquier test."""
    monkeypatch.setattr("app.main.joblib.load", lambda path: _mock_single(0.75))


# Fixtures de cliente 
@pytest.fixture
def client():
    """Cliente estándar: modelo mock (prob=0.75) cargado por lifespan."""
    from mlops.app.api import app
    with TestClient(app) as c:
        yield c


@pytest.fixture
def client_no_model():
    """Cliente sin modelo: limpia model_store después de que lifespan corra."""
    from mlops.app.api import app, model_store
    with TestClient(app) as c:
        model_store.clear()   
        yield c
    model_store.clear()


# Tests:
class TestHealth:
    def test_health_ok_with_model(self, client):
        r = client.get("/health")
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "ok"
        assert body["model_loaded"] is True

    def test_health_model_not_loaded(self, client_no_model):
        r = client_no_model.get("/health")
        assert r.status_code == 200
        assert r.json()["model_loaded"] is False


# Tests: 
class TestPredict:
    def test_predict_returns_200(self, client):
        r = client.post("/predict", json=VALID_CUSTOMER)
        assert r.status_code == 200

    def test_predict_structure(self, client):
        pred = client.post("/predict", json=VALID_CUSTOMER).json()["prediction"]
        assert "churn_probability" in pred
        assert "churn_prediction" in pred
        assert "risk_level" in pred

    def test_predict_high_prob_is_churn(self, client):
        """prob=0.75 → churn=True, riesgo=ALTO"""
        pred = client.post("/predict", json=VALID_CUSTOMER).json()["prediction"]
        assert pred["churn_prediction"] is True
        assert pred["risk_level"] == "ALTO"
        assert 0.0 <= pred["churn_probability"] <= 1.0

    def test_predict_low_prob_no_churn(self):
        from mlops.app.api import app, model_store
        model_store["pipeline"] = _mock_single(0.20)
        with TestClient(app) as c:
            pred = c.post("/predict", json=VALID_CUSTOMER).json()["prediction"]
        assert pred["churn_prediction"] is False
        assert pred["risk_level"] == "BAJO"

    def test_predict_medium_risk(self):
        from mlops.app.api import app, model_store
        model_store["pipeline"] = _mock_single(0.55)
        with TestClient(app) as c:
            pred = c.post("/predict", json=VALID_CUSTOMER).json()["prediction"]
        assert pred["risk_level"] == "MEDIO"

    def test_predict_model_not_loaded_returns_503(self, client_no_model):
        r = client_no_model.post("/predict", json=VALID_CUSTOMER)
        assert r.status_code == 503

    def test_predict_invalid_gender_returns_422(self, client):
        r = client.post("/predict", json={**VALID_CUSTOMER, "gender": "X"})
        assert r.status_code == 422

    def test_predict_invalid_contract_returns_422(self, client):
        r = client.post("/predict", json={**VALID_CUSTOMER, "Contract": "Weekly"})
        assert r.status_code == 422

    def test_predict_negative_tenure_returns_422(self, client):
        r = client.post("/predict", json={**VALID_CUSTOMER, "tenure": -1})
        assert r.status_code == 422

    def test_predict_missing_field_returns_422(self, client):
        bad = {k: v for k, v in VALID_CUSTOMER.items() if k != "MonthlyCharges"}
        assert client.post("/predict", json=bad).status_code == 422

    def test_predict_latency_header(self, client):
        r = client.post("/predict", json=VALID_CUSTOMER)
        assert "x-response-time-ms" in r.headers


#  Tests: /predict/batch
class TestPredictBatch:
    def test_batch_single_customer(self):
        from mlops.app.api import app, model_store
        model_store["pipeline"] = _mock_batch([0.8])
        with TestClient(app) as c:
            body = c.post("/predict/batch", json={"customers": [VALID_CUSTOMER]}).json()
        assert body["total_customers"] == 1
        assert body["predicted_churners"] == 1

    def test_batch_multiple_customers(self):
        from mlops.app.api import app, model_store
        probs = [0.9, 0.2, 0.6, 0.1, 0.75]
        model_store["pipeline"] = _mock_batch(probs)
        with TestClient(app) as c:
            body = c.post("/predict/batch",
                          json={"customers": [VALID_CUSTOMER] * 5}).json()
        assert body["total_customers"] == 5
        assert body["predicted_churners"] == 3  
        assert body["churn_rate_predicted"] == pytest.approx(0.6, abs=0.01)

    def test_batch_all_churners(self):
        from mlops.app.api import app, model_store
        model_store["pipeline"] = _mock_batch([0.8, 0.9])
        with TestClient(app) as c:
            body = c.post("/predict/batch",
                          json={"customers": [VALID_CUSTOMER, VALID_CUSTOMER]}).json()
        assert body["churn_rate_predicted"] == pytest.approx(1.0)

    def test_batch_empty_list_returns_422(self, client):
        r = client.post("/predict/batch", json={"customers": []})
        assert r.status_code == 422

    def test_batch_model_not_loaded_returns_503(self, client_no_model):
        r = client_no_model.post("/predict/batch",
                                  json={"customers": [VALID_CUSTOMER]})
        assert r.status_code == 503


# Tests: lógica de niveles de riesgo 
class TestRiskLevel:
    def test_risk_levels_boundaries(self):
        from mlops.app.api import _risk_level
        assert _risk_level(0.00) == "BAJO"
        assert _risk_level(0.39) == "BAJO"
        assert _risk_level(0.40) == "MEDIO"
        assert _risk_level(0.69) == "MEDIO"
        assert _risk_level(0.70) == "ALTO"
        assert _risk_level(1.00) == "ALTO"