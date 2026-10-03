"""Tests for how the service finds morshu.wav, the recording that every clip is cut from."""

import os
import shutil
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

SECRET = "test-secret-0123456789"
os.environ["API_SECRET"] = SECRET

import tts_api.morshutalk.morshu as morshu_module  # noqa: E402
from tts_api import main  # noqa: E402
from tts_api.config import Settings  # noqa: E402
from tts_api.morshutalk.audio import Clip  # noqa: E402


class _StubG2p:
    """Stands in for G2p, so init runs without loading the dictionary or the model."""


@pytest.fixture
def stub_engine(monkeypatch: pytest.MonkeyPatch) -> None:
    # Setting the globals to their current values makes monkeypatch restore them afterwards.
    for name in ("_g2p", "_morshu_wav", "_wav_path"):
        monkeypatch.setattr(morshu_module, name, getattr(morshu_module, name))
    monkeypatch.setattr(morshu_module, "G2p", _StubG2p)


def _settings() -> Settings:
    # _env_file=None keeps a developer's .env out of the test.
    return Settings(_env_file=None)


def test_packaged_wav_ships_inside_the_package() -> None:
    assert morshu_module.PACKAGED_WAV.name == "morshu.wav"
    assert morshu_module.PACKAGED_WAV.parent == Path(morshu_module.__file__).parent
    assert len(Clip.from_wav(morshu_module.PACKAGED_WAV)) > 0


@pytest.mark.parametrize("value", [None, "", "  "], ids=["unset", "empty", "blank"])
@pytest.mark.usefixtures("stub_engine")
def test_packaged_wav_is_used_when_the_override_is_unset(
    monkeypatch: pytest.MonkeyPatch, value: str | None
) -> None:
    if value is None:
        monkeypatch.delenv("TTS_SOURCE_WAV", raising=False)
    else:
        monkeypatch.setenv("TTS_SOURCE_WAV", value)
    settings = _settings()
    assert settings.tts_source_wav is None

    morshu_module.init(settings.tts_source_wav)
    assert morshu_module._wav_path == morshu_module.PACKAGED_WAV
    assert morshu_module._morshu_wav is not None
    assert len(morshu_module._morshu_wav) > 0


@pytest.mark.usefixtures("stub_engine")
def test_tts_source_wav_overrides_the_packaged_wav(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    custom = tmp_path / "custom.wav"
    shutil.copyfile(morshu_module.PACKAGED_WAV, custom)
    monkeypatch.setenv("TTS_SOURCE_WAV", str(custom))
    settings = _settings()
    assert settings.tts_source_wav == str(custom)

    morshu_module.init(settings.tts_source_wav)
    assert morshu_module._wav_path == custom


def test_lifespan_passes_the_setting_to_init(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TTS_SOURCE_WAV", raising=False)
    received: list[str | None] = []
    monkeypatch.setattr(main, "get_settings", _settings)
    monkeypatch.setattr(main, "init", received.append)
    with TestClient(main.app):
        pass
    assert received == [None]
