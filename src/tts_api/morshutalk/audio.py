"""A small mono audio clip type built on numpy and the standard library's wave module.

It replaces pydub, which is unmaintained and depends on the audioop module that Python 3.13 removed.
The engine only needs to load, slice, join, measure, reverse, resample, and save 16-bit audio.
Positions and lengths are in milliseconds, as they were with pydub.
"""

import io
import math
import wave
from dataclasses import dataclass
from os import PathLike

import numpy as np
import numpy.typing as npt

__all__ = ["Clip", "detect_leading_silence"]

Samples = npt.NDArray[np.int16]

# The largest magnitude a 16-bit sample can have, which is 0 dBFS.
_FULL_SCALE = 32768.0


@dataclass(frozen=True)
class Clip:
    """Mono 16-bit audio.

    Attributes:
        samples: The samples as 16-bit integers.
        frame_rate: The samples per second.
    """

    samples: Samples
    frame_rate: int

    @classmethod
    def from_wav(cls, path: str | PathLike[str]) -> "Clip":
        """Loads a 16-bit PCM WAV file. A stereo file is mixed down to mono.

        Args:
            path: The WAV file.

        Returns:
            The clip.

        Raises:
            ValueError: When the file is not 16-bit PCM with one or two channels.
        """
        with wave.open(str(path), "rb") as reader:
            channels = reader.getnchannels()
            if reader.getsampwidth() != 2 or channels not in (1, 2):
                raise ValueError(
                    f"{path} must be 16-bit PCM audio with one or two channels."
                )
            frame_rate = reader.getframerate()
            data = np.frombuffer(reader.readframes(reader.getnframes()), dtype="<i2")
        if channels == 2:
            mixed = data.reshape(-1, 2).astype(np.int32).sum(axis=1) // 2
            data = mixed.astype(np.int16)
        return cls(data.astype(np.int16), frame_rate)

    @classmethod
    def empty(cls, frame_rate: int) -> "Clip":
        """Returns a clip without samples."""
        return cls(np.zeros(0, dtype=np.int16), frame_rate)

    @classmethod
    def silent(cls, duration_ms: float, frame_rate: int) -> "Clip":
        """Returns silence of the given length."""
        return cls(
            np.zeros(_frames(duration_ms, frame_rate), dtype=np.int16), frame_rate
        )

    def __len__(self) -> int:
        """Returns the length in whole milliseconds."""
        return round(1000 * len(self.samples) / self.frame_rate)

    def __getitem__(self, span: slice) -> "Clip":
        """Returns the part between two positions in milliseconds."""
        if span.step is not None:
            raise ValueError("A clip slice cannot have a step.")
        start = _frames(span.start or 0, self.frame_rate)
        stop = (
            len(self.samples)
            if span.stop is None
            else _frames(span.stop, self.frame_rate)
        )
        return Clip(self.samples[start:stop], self.frame_rate)

    def __add__(self, other: "Clip") -> "Clip":
        """Returns this clip followed by the other one."""
        if other.frame_rate != self.frame_rate:
            raise ValueError(
                f"Cannot join clips at {self.frame_rate} Hz and {other.frame_rate} Hz."
            )
        return Clip(np.concatenate((self.samples, other.samples)), self.frame_rate)

    @property
    def dbfs(self) -> float:
        """Returns the loudness as RMS in decibels relative to full scale, or -inf for silence."""
        if len(self.samples) == 0:
            return -math.inf
        rms = math.sqrt(float(np.mean(self.samples.astype(np.float64) ** 2)))
        return -math.inf if rms == 0 else 20 * math.log10(rms / _FULL_SCALE)

    def reverse(self) -> "Clip":
        """Returns the clip played backwards."""
        return Clip(self.samples[::-1].copy(), self.frame_rate)

    def change_speed(self, factor: float) -> "Clip":
        """Plays the clip faster or slower, which raises or lowers its pitch too.

        Args:
            factor: 2.0 plays twice as fast, and 0.5 half as fast.

        Returns:
            The resampled clip at the same frame rate.
        """
        count = len(self.samples)
        new_count = max(round(count / factor), 0)
        if count == 0 or new_count == 0:
            return Clip.empty(self.frame_rate)
        positions = np.arange(new_count) * factor
        resampled = np.interp(
            positions, np.arange(count), self.samples.astype(np.float64)
        )
        return Clip(np.round(resampled).astype(np.int16), self.frame_rate)

    def to_wav(self) -> bytes:
        """Returns the clip as a 16-bit mono PCM WAV file."""
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as writer:
            writer.setnchannels(1)
            writer.setsampwidth(2)
            writer.setframerate(self.frame_rate)
            writer.writeframes(self.samples.astype("<i2").tobytes())
        return buffer.getvalue()


def detect_leading_silence(
    clip: Clip, silence_threshold: float, chunk_size_ms: int = 10
) -> int:
    """Returns how long a clip stays below a loudness threshold from its start.

    Args:
        clip: The clip to scan.
        silence_threshold: The loudness in dBFS below which a chunk counts as silence.
        chunk_size_ms: The length of each scanned chunk, in milliseconds.

    Returns:
        The length of the leading silence in milliseconds, at most the clip's length.
    """
    position = 0
    length = len(clip)
    while position < length and clip[position : position + chunk_size_ms].dbfs < (
        silence_threshold
    ):
        position += chunk_size_ms
    return min(position, length)


def _frames(position_ms: float, frame_rate: int) -> int:
    """Converts a position in milliseconds to a sample index, exactly as pydub did.

    Dividing the frame rate first can land a hair below a whole number, which then rounds down.
    Keeping pydub's order of operations keeps every cut on the same sample as before.
    """
    return int(position_ms * (frame_rate / 1000.0))
