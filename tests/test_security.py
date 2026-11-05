"""Authentication contract tests."""

from __future__ import annotations

import uuid

from tests.conftest import make_client, make_services


def test_health_does_not_require_api_key() -> None:
    with make_client() as client:
        response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "ocr_loaded": True,
        "tts_ready": True,
    }


def test_health_returns_503_while_not_ready() -> None:
    services = make_services(ocr_loaded=False, tts_ready=True)
    with make_client(services=services) as client:
        response = client.get("/health")
    assert response.status_code == 503
    assert response.json() == {
        "status": "starting",
        "ocr_loaded": False,
        "tts_ready": True,
    }


def test_ocr_missing_api_key_returns_401() -> None:
    with make_client() as client:
        response = client.post(
            "/ocr", json={"file_key": "uploads/abc/source.png"}
        )
    assert response.status_code == 401
    assert isinstance(response.json()["detail"], str)


def test_ocr_wrong_api_key_returns_401() -> None:
    with make_client() as client:
        response = client.post(
            "/ocr",
            json={"file_key": "uploads/abc/source.png"},
            headers={"x-api-key": "wrong"},
        )
    assert response.status_code == 401


def test_tts_missing_api_key_returns_401() -> None:
    with make_client() as client:
        response = client.post(
            "/tts",
            json={"history_id": str(uuid.uuid4()), "text": "Halo"},
        )
    assert response.status_code == 401


def test_tts_wrong_api_key_returns_401() -> None:
    with make_client() as client:
        response = client.post(
            "/tts",
            json={"history_id": str(uuid.uuid4()), "text": "Halo"},
            headers={"x-api-key": "wrong"},
        )
    assert response.status_code == 401


def test_production_default_api_key_is_rejected() -> None:
    from tests.conftest import make_settings

    settings = make_settings(
        app_env="production", ocr_service_api_key="change-me"
    )
    services = make_services(settings)
    try:
        with make_client(settings=settings, services=services):
            pass
    except RuntimeError as exc:
        assert "OCR_SERVICE_API_KEY" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("production startup accepted the default API key")


def test_unknown_route_returns_404_without_auth() -> None:
    with make_client() as client:
        response = client.get("/nope")
    assert response.status_code == 404
    assert response.json()["detail"] == "Not Found"
