import os

from fastapi.testclient import TestClient

os.environ.setdefault("DISCORD_API_SECRET", "test-secret")
os.environ.setdefault("TTS_SOURCE_WAV", "assets/morshu.wav")

from tts_api.main import app  # noqa: E402

client = TestClient(app)
AUTH = {"Authorization": "Bearer test-secret"}
WRONG = {"Authorization": "Bearer wrong"}


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"
    assert r.json()["service"] == "discord-api-morshu"


def test_synthesize_requires_auth():
    r = client.post("/tts/synthesize", json={"text": "hello"})
    assert r.status_code == 403


def test_synthesize_text_too_long():
    r = client.post("/tts/synthesize", json={"text": "a" * 501}, headers=AUTH)
    assert r.status_code == 422


def test_phonemes_requires_auth():
    r = client.get("/tts/phonemes")
    assert r.status_code == 403


def test_health_includes_version():
    r = client.get("/health")
    assert "version" in r.json()


def test_synthesize_wrong_auth():
    r = client.post("/tts/synthesize", json={"text": "hello"}, headers=WRONG)
    assert r.status_code == 401


def test_phonemes_wrong_auth():
    r = client.get("/tts/phonemes", headers=WRONG)
    assert r.status_code == 401


def test_synthesize_empty_text_rejected():
    # min_length=1 on SynthesizeRequest.text.
    r = client.post("/tts/synthesize", json={"text": ""}, headers=AUTH)
    assert r.status_code == 422


def test_synthesize_speed_too_low_rejected():
    # ge=0.5; 0.4 is below the minimum.
    r = client.post(
        "/tts/synthesize", json={"text": "hello", "speed": 0.4}, headers=AUTH
    )
    assert r.status_code == 422


def test_synthesize_speed_too_high_rejected():
    # le=2.0; 2.1 exceeds the maximum.
    r = client.post(
        "/tts/synthesize", json={"text": "hello", "speed": 2.1}, headers=AUTH
    )
    assert r.status_code == 422


def test_synthesize_invalid_format_rejected():
    # format must be Literal["wav", "video"]; "ogg" is not accepted.
    r = client.post(
        "/tts/synthesize", json={"text": "hello", "format": "ogg"}, headers=AUTH
    )
    assert r.status_code == 422
