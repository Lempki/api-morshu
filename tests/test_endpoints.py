import logging
import os
import subprocess
from typing import Literal

import numpy as np
import pytest
from fastapi import Response
from fastapi.testclient import TestClient
from pydub import AudioSegment

SECRET = "test-secret-0123456789"
os.environ["DISCORD_API_SECRET"] = SECRET

from tts_api import main  # noqa: E402
from tts_api.main import VERSION, app  # noqa: E402
from tts_api.models import SynthesizeRequest  # noqa: E402
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
    # SynthesizeRequest.text has min_length=1.
    r = client.post("/tts/synthesize", json={"text": ""}, headers=AUTH)
    assert r.status_code == 422


def test_synthesize_speed_too_low_rejected() -> None:
    # 0.4 is below the minimum of 0.5.
    r = client.post(
        "/tts/synthesize", json={"text": "hello", "speed": 0.4}, headers=AUTH
    )
    assert r.status_code == 422


def test_synthesize_speed_too_high_rejected() -> None:
    # 2.1 exceeds the maximum of 2.0.
    r = client.post(
        "/tts/synthesize", json={"text": "hello", "speed": 2.1}, headers=AUTH
    )
    assert r.status_code == 422


def test_synthesize_invalid_format_rejected() -> None:
    # format accepts only "wav" and "video", so "ogg" is rejected.
    r = client.post(
        "/tts/synthesize", json={"text": "hello", "format": "ogg"}, headers=AUTH
    )
    assert r.status_code == 422


def test_phonemes_does_not_expose_the_source_path() -> None:
    r = client.get("/tts/phonemes", headers=AUTH)
    assert r.status_code == 200
    assert set(r.json()) == {"phonemes"}
    assert "L" in r.json()["phonemes"]


@pytest.fixture
def fake_synthesis(monkeypatch: pytest.MonkeyPatch) -> dict[str, bytes]:
    files = {"wav": b"RIFF fake wav", "video": b"fake mp4"}
    monkeypatch.setattr(main, "_synthesize_blocking", lambda *_: files["wav"])
    monkeypatch.setattr(main, "_synthesize_video_blocking", lambda _: files["video"])
    return files


@pytest.mark.parametrize(
    ("fmt", "media_type", "filename"),
    [("wav", "audio/wav", "morshu.wav"), ("video", "video/mp4", "morshu.mp4")],
)
def test_synthesize_returns_the_file_with_its_media_type(
    fake_synthesis: dict[str, bytes], fmt: str, media_type: str, filename: str
) -> None:
    r = client.post(
        "/tts/synthesize", json={"text": "lamp oil", "format": fmt}, headers=AUTH
    )
    assert r.status_code == 200
    assert r.headers["content-type"] == media_type
    assert r.headers["content-disposition"] == f'attachment; filename="{filename}"'
    # A plain Response knows its length up front, unlike the StreamingResponse it replaced.
    assert r.headers["content-length"] == str(len(fake_synthesis[fmt]))
    assert r.content == fake_synthesis[fmt]


@pytest.mark.parametrize("fmt", ["wav", "video"])
async def test_synthesize_builds_a_plain_response(
    fake_synthesis: dict[str, bytes], fmt: Literal["wav", "video"]
) -> None:
    body = SynthesizeRequest(text="lamp oil", format=fmt)
    response = await main.synthesize(body, main.get_settings())
    assert type(response) is Response
    assert response.body == fake_synthesis[fmt]


class _FakeMorshu:
    """Returns a short silent clip, so the video path reaches ffmpeg without the G2p model."""

    def __init__(self) -> None:
        self.audio_segment_timings = np.rec.fromrecords(
            [(0, 0)], names=("output", "morshu")
        )

    def load_text(self, text: str) -> AudioSegment:
        return AudioSegment.silent(duration=250)


_LONG_STDERR = b"x" * 10_000 + b"final ffmpeg line"


@pytest.mark.parametrize(
    "error",
    [
        subprocess.CalledProcessError(1, ["ffmpeg"], output=b"", stderr=_LONG_STDERR),
        subprocess.TimeoutExpired(["ffmpeg"], 120, output=b"", stderr=_LONG_STDERR),
    ],
    ids=["failed", "timed-out"],
)
def test_ffmpeg_failure_answers_500_with_a_fixed_detail(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    error: subprocess.SubprocessError,
) -> None:
    def fail(*args: object, **kwargs: object) -> None:
        raise error

    monkeypatch.setattr(main, "Morshu", _FakeMorshu)
    monkeypatch.setattr(subprocess, "run", fail)
    with caplog.at_level(logging.ERROR, logger=main.logger.name):
        r = client.post(
            "/tts/synthesize",
            json={"text": "lamp oil", "format": "video"},
            headers=AUTH,
        )
    assert r.status_code == 500
    assert r.json() == {"detail": main.VIDEO_ENCODING_FAILED}
    [record] = [rec for rec in caplog.records if rec.name == main.logger.name]
    message = record.getMessage()
    assert "final ffmpeg line" in message
    assert len(message) < 2500


def _quiet_speech_with_padding() -> AudioSegment:
    """Two quiet tones with a pause between them and silence around them."""
    from pydub.generators import Sine

    tone = Sine(440).to_audio_segment(duration=1500, volume=-24.0)
    pause = AudioSegment.silent(duration=300)
    return (
        AudioSegment.silent(duration=500)
        + tone
        + pause
        + tone
        + AudioSegment.silent(duration=500)
    )


def test_trim_edges_keeps_quiet_speech_and_inner_pauses() -> None:
    from tts_api.main import trim_edges

    trimmed = trim_edges(_quiet_speech_with_padding())

    # pydub's strip_silence() returned nothing for a clip like this.
    # Its fixed threshold of -16 dBFS counted the -24 dBFS tones as silence.
    # Now only the edges go, and the pause between the tones stays.
    assert 3250 <= len(trimmed) <= 3350


def test_trim_edges_leaves_an_all_silent_clip_unchanged() -> None:
    from tts_api.main import trim_edges

    silence = AudioSegment.silent(duration=800)

    assert len(trim_edges(silence)) == 800
