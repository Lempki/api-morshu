import os

import pytest
from fastapi.testclient import TestClient

SECRET = "test-secret-0123456789"
os.environ["DISCORD_API_SECRET"] = SECRET
os.environ.setdefault("TTS_SOURCE_WAV", "assets/morshu.wav")

from tts_api.main import VERSION, app  # noqa: E402
from tts_api.service import service_version  # noqa: E402

client = TestClient(app)
AUTH = {"Authorization": f"Bearer {SECRET}"}
WRONG = {"Authorization": "Bearer wrong"}


def test_health_needs_no_token() -> None:
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {
        "status": "ok",
        "service": "discord-api-morshu",
        "version": VERSION,
    }


def test_version_comes_from_package_metadata() -> None:
    assert VERSION == service_version("discord-api-morshu") != "0.0.0"
    assert service_version("not-an-installed-project") == "0.0.0"


@pytest.mark.parametrize(
    "headers",
    [
        {},
        {"Authorization": "Bearer wrong"},
        {"Authorization": f"Bearer {SECRET}x"},
        {"Authorization": f"Basic {SECRET}"},
    ],
    ids=["missing", "wrong", "longer", "wrong-scheme"],
)
def test_synthesize_rejects_without_valid_token(headers: dict[str, str]) -> None:
    r = client.post("/tts/synthesize", json={"text": "hello"}, headers=headers)
    assert r.status_code == 401
    assert r.headers["WWW-Authenticate"] == "Bearer"


def test_synthesize_text_too_long() -> None:
    r = client.post("/tts/synthesize", json={"text": "a" * 501}, headers=AUTH)
    assert r.status_code == 422


def test_phonemes_requires_auth() -> None:
    r = client.get("/tts/phonemes")
    assert r.status_code == 401


def test_phonemes_wrong_auth() -> None:
    r = client.get("/tts/phonemes", headers=WRONG)
    assert r.status_code == 401


def test_synthesize_empty_text_rejected() -> None:
    # min_length=1 on SynthesizeRequest.text.
    r = client.post("/tts/synthesize", json={"text": ""}, headers=AUTH)
    assert r.status_code == 422


def test_synthesize_speed_too_low_rejected() -> None:
    # ge=0.5; 0.4 is below the minimum.
    r = client.post(
        "/tts/synthesize", json={"text": "hello", "speed": 0.4}, headers=AUTH
    )
    assert r.status_code == 422


def test_synthesize_speed_too_high_rejected() -> None:
    # le=2.0; 2.1 exceeds the maximum.
    r = client.post(
        "/tts/synthesize", json={"text": "hello", "speed": 2.1}, headers=AUTH
    )
    assert r.status_code == 422


def test_synthesize_invalid_format_rejected() -> None:
    # format must be Literal["wav", "video"]; "ogg" is not accepted.
    r = client.post(
        "/tts/synthesize", json={"text": "hello", "format": "ogg"}, headers=AUTH
    )
    assert r.status_code == 422
