from __future__ import annotations

import io
import sys

import pytest


@pytest.fixture
def client(tmp_path, monkeypatch):
    sys.modules.pop("model_utils", None)
    sys.modules.pop("webapp.model_utils", None)

    from webapp import app as app_module

    monkeypatch.setitem(app_module.app.config, "TESTING", True)
    monkeypatch.setitem(app_module.app.config, "UPLOAD_FOLDER", str(tmp_path))
    return app_module, app_module.app.test_client()


def test_import_and_homepage_do_not_load_model(client):
    _, test_client = client

    response = test_client.get("/")

    assert response.status_code == 200
    assert b"Banana Feature Predictor" in response.data
    assert "model_utils" not in sys.modules
    assert "webapp.model_utils" not in sys.modules


@pytest.mark.parametrize(
    "payload",
    ({}, {"image": (io.BytesIO(b""), "")}),
)
def test_missing_upload_redirects_to_home_without_loading_model(client, payload):
    _, test_client = client

    response = test_client.post("/upload", data=payload)

    assert response.status_code == 302
    assert response.headers["Location"].endswith("/")
    assert "model_utils" not in sys.modules
    assert "webapp.model_utils" not in sys.modules


def test_valid_upload_invokes_prediction_lazily(client, monkeypatch):
    app_module, test_client = client
    calls = []

    def fake_predict(image_path):
        calls.append(image_path)
        return {"seeds": 3, "curvature": 42.0}

    monkeypatch.setattr(app_module, "_predict_from_image", fake_predict)

    response = test_client.post(
        "/upload",
        data={"image": (io.BytesIO(b"fake image bytes"), "banana.jpg")},
        content_type="multipart/form-data",
    )

    assert response.status_code == 200
    assert len(calls) == 1
    assert calls[0].endswith("banana.jpg")
