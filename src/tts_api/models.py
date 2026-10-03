"""Request and response models."""

from typing import Literal

from pydantic import BaseModel, Field


class SynthesizeRequest(BaseModel):
    """The body of POST /tts/synthesize.

    Attributes:
        text: The text to speak, at least one character long.
            TTS_MAX_TEXT_LENGTH caps its length.
        speed: The playback speed from 0.5 to 2.0, which shifts the pitch too.
            Video output ignores it.
        trim_silence: Whether to strip silence from the audio. Video output ignores it.
        format: "wav" for a WAV file or "video" for a lip-synced MP4.
    """

    text: str = Field(..., min_length=1)
    speed: float = Field(default=1.0, ge=0.5, le=2.0)
    trim_silence: bool = False
    format: Literal["wav", "video"] = "wav"


class PhonemesResponse(BaseModel):
    """The body of GET /tts/phonemes, the sorted phoneme tokens of the source recording."""

    phonemes: list[str]


class HealthResponse(BaseModel):
    """The body of GET /health."""

    status: str
    service: str
    version: str
