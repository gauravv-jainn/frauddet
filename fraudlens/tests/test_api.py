from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts"
pytestmark = pytest.mark.skipif(not (ART / "model.joblib").exists(), reason="no trained model")


@pytest.fixture
def client(monkeypatch):
    from fastapi.testclient import TestClient

    from api.main import app

    monkeypatch.setenv("FRAUDLENS_API_KEY", "test-key-123")
    return TestClient(app)


def _payload():
    s = pd.read_parquet(ART / "test_scored.parquet").nlargest(2, "p")
    cols = ["trans_date_trans_time", "cc_num", "merchant", "category", "amt", "gender", "city", "state", "lat", "long",
            "city_pop", "job", "dob", "merch_lat", "merch_long", "trans_num"]
    rows = s[cols].copy()
    for c in ["trans_date_trans_time", "dob"]:
        rows[c] = rows[c].dt.strftime("%Y-%m-%dT%H:%M:%S")
    return {"transactions": rows.to_dict(orient="records")}, s


def test_requires_api_key(client):
    r = client.post("/score", json=_payload()[0])
    assert r.status_code == 401
    r = client.post("/score", json=_payload()[0], headers={"X-API-Key": "wrong"})
    assert r.status_code == 401


def test_score_matches_offline_scores(client):
    payload, s = _payload()
    r = client.post("/score", json=payload, headers={"X-API-Key": "test-key-123"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert len(body) == 2
    for item, p in zip(body, s["p"]):
        assert item["probability"] == pytest.approx(p, abs=1e-12)
        assert item["band"] in {"Low", "Medium", "High", "Critical"}
        assert item["reasons"]


def test_validation_error(client):
    payload, _ = _payload()
    payload["transactions"][0]["amt"] = -5
    r = client.post("/score", json=payload, headers={"X-API-Key": "test-key-123"})
    assert r.status_code == 422
