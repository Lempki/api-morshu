import math
import wave
from pathlib import Path

import numpy as np
import pytest

from tts_api.morshutalk.audio import Clip, detect_leading_silence

RATE = 18900


def _clip(*values: int) -> Clip:
    return Clip(np.array(values, dtype=np.int16), RATE)


def _write_wav(path: Path, data: bytes, channels: int, width: int) -> None:
    with wave.open(str(path), "wb") as writer:
        writer.setnchannels(channels)
        writer.setsampwidth(width)
        writer.setframerate(RATE)
        writer.writeframes(data)


def test_length_and_slices_are_in_milliseconds() -> None:
    clip = Clip(np.arange(RATE, dtype=np.int16), RATE)
    assert len(clip) == 1000
    part = clip[100:250]
    assert len(part) == 150
    # 100 ms at 18900 Hz lands on sample 1889 the way pydub computed it, not on 1890.
    assert part.samples[0] == int(100 * (RATE / 1000.0))


def test_open_ended_slices() -> None:
    clip = Clip(np.arange(RATE, dtype=np.int16), RATE)
    assert len(clip[:300]) == 300
    assert len(clip[700:]) == 300


def test_slices_with_a_step_are_refused() -> None:
    with pytest.raises(ValueError, match="step"):
        _clip(1, 2, 3)[0:10:2]


def test_joining_requires_the_same_frame_rate() -> None:
    assert (_clip(1, 2) + _clip(3)).samples.tolist() == [1, 2, 3]
    with pytest.raises(ValueError, match="Hz"):
        _clip(1) + Clip(np.zeros(1, dtype=np.int16), 8000)


def test_silence_and_empty_clips() -> None:
    assert len(Clip.silent(20, RATE).samples) == int(20 * (RATE / 1000.0))
    assert not Clip.silent(20, RATE).samples.any()
    assert len(Clip.empty(RATE)) == 0


def test_dbfs() -> None:
    assert Clip.silent(10, RATE).dbfs == -math.inf
    assert Clip.empty(RATE).dbfs == -math.inf
    assert _clip(16384, -16384).dbfs == pytest.approx(20 * math.log10(0.5))


def test_reverse() -> None:
    assert _clip(1, 2, 3).reverse().samples.tolist() == [3, 2, 1]


@pytest.mark.parametrize(("factor", "expected"), [(2.0, 500), (0.5, 2000), (1.5, 667)])
def test_change_speed_scales_the_length(factor: float, expected: int) -> None:
    clip = Clip(np.zeros(RATE, dtype=np.int16), RATE)
    assert len(clip.change_speed(factor)) == expected


def test_wav_round_trip(tmp_path: Path) -> None:
    clip = _clip(0, 1000, -1000, 32767, -32768)
    path = tmp_path / "clip.wav"
    path.write_bytes(clip.to_wav())
    loaded = Clip.from_wav(path)
    assert loaded.frame_rate == RATE
    assert loaded.samples.tolist() == clip.samples.tolist()


def test_stereo_is_mixed_down_to_mono(tmp_path: Path) -> None:
    path = tmp_path / "stereo.wav"
    _write_wav(path, np.array([100, 300, -200, 0], dtype="<i2").tobytes(), 2, 2)
    assert Clip.from_wav(path).samples.tolist() == [200, -100]


def test_wav_files_that_are_not_16_bit_are_refused(tmp_path: Path) -> None:
    path = tmp_path / "eight_bit.wav"
    _write_wav(path, bytes(10), 1, 1)
    with pytest.raises(ValueError, match="16-bit"):
        Clip.from_wav(path)


def test_detect_leading_silence() -> None:
    loud = Clip(np.full(int(0.2 * RATE), 8000, dtype=np.int16), RATE)
    clip = Clip.silent(300, RATE) + loud
    assert detect_leading_silence(clip, silence_threshold=-40.0) == 300
    assert detect_leading_silence(Clip.silent(55, RATE), -40.0) == 55
