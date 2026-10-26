from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts"
pytestmark = pytest.mark.skipif(not (ART / "model.joblib").exists(), reason="no trained model")
ENV_KEY = "env-bootstrap-key-for-tests-0001"


@pytest.fixture
def client(monkeypatch, tmp_path):
    from fastapi.testclient import TestClient

    import api.main as api

    monkeypatch.setenv("FRAUDLENS_DB", str(tmp_path / "api.sqlite"))
    monkeypatch.setenv("FRAUDLENS_API_KEY", ENV_KEY)
    api.settings.cache_clear()
    api.database.cache_clear()
    api.limiter.hits.clear()
    return TestClient(api.app)


@pytest.fixture
def db_key(client):
    import api.main as api
    from fraudkit.auth import ApiKeyStore

    rec, token = ApiKeyStore(api.database()).create("test-client", actor="pytest")
    return rec, token


def _payload(n=2):
    s = pd.read_parquet(ART / "test_scored.parquet").nlargest(n, "p")
    cols = [
        "trans_date_trans_time",
        "cc_num",
        "merchant",
        "category",
        "amt",
        "gender",
        "city",
        "state",
        "lat",
        "long",
        "city_pop",
        "job",
        "dob",
        "merch_lat",
        "merch_long",
        "trans_num",
    ]
    rows = s[cols].copy()
    for c in ["trans_date_trans_time", "dob"]:
        rows[c] = rows[c].dt.strftime("%Y-%m-%dT%H:%M:%S")
    return {"transactions": rows.to_dict(orient="records")}, s


def test_health_and_ready_need_no_key(client):
    assert client.get("/v1/health").json() == {"status": "ok"}
    r = client.get("/v1/ready")
    assert r.status_code == 200 and "model_version" in r.json()


def test_requires_api_key(client):
    assert client.post("/v1/score", json=_payload()[0]).status_code == 401
    assert client.post("/v1/score", json=_payload()[0], headers={"X-API-Key": "wrong"}).status_code == 401
    assert client.get("/v1/model").status_code == 401


def test_score_matches_offline_scores_with_db_key(client, db_key):
    rec, token = db_key
    payload, s = _payload()
    r = client.post("/v1/score", json=payload, headers={"X-API-Key": token, "X-Request-ID": "abc-123"})
    assert r.status_code == 200, r.text
    assert r.headers["X-Request-ID"] == "abc-123"
    body = r.json()
    assert body["request_id"] == "abc-123" and body["model_version"]
    assert len(body["results"]) == 2
    for item, p, amt in zip(body["results"], s["p"], s["amt"], strict=True):
        assert item["probability"] == pytest.approx(p, abs=1e-12)
        assert item["band"] in {"Low", "Medium", "High", "Critical"}
        assert item["decision"] in {"approve", "step_up", "review", "block"}
        assert 0 <= item["risk_score"] <= 999
        assert item["expected_loss"] == pytest.approx(p * amt, abs=1e-9)
        assert item["reasons"]
    from fraudkit.auth import ApiKeyStore
    import api.main as api

    assert ApiKeyStore(api.database()).get(rec.key_id).use_count == 1


def test_revoked_key_is_rejected(client, db_key):
    import api.main as api
    from fraudkit.auth import ApiKeyStore

    rec, token = db_key
    ApiKeyStore(api.database()).revoke(rec.key_id, actor="pytest")
    assert client.post("/v1/score", json=_payload(1)[0], headers={"X-API-Key": token}).status_code == 401


def test_env_bootstrap_key_and_legacy_route(client):
    r = client.post("/score", json=_payload(1)[0], headers={"X-API-Key": ENV_KEY})
    assert r.status_code == 200 and isinstance(r.json(), list)


def test_validation_error(client):
    payload, _ = _payload()
    payload["transactions"][0]["amt"] = -5
    r = client.post("/v1/score", json=payload, headers={"X-API-Key": ENV_KEY})
    assert r.status_code == 422


def test_rate_limit(client, monkeypatch):
    import api.main as api

    monkeypatch.setattr(api.settings().api, "rate_limit_per_minute", 2)
    h = {"X-API-Key": ENV_KEY}
    assert client.get("/v1/model", headers=h).status_code == 200
    assert client.get("/v1/model", headers=h).status_code == 200
    r = client.get("/v1/model", headers=h)
    assert r.status_code == 429 and int(r.headers["Retry-After"]) >= 1


def test_batch_size_limit(client, monkeypatch):
    import api.main as api

    monkeypatch.setattr(api.settings().api, "max_batch", 1)
    r = client.post("/v1/score", json=_payload(2)[0], headers={"X-API-Key": ENV_KEY})
    assert r.status_code == 413
