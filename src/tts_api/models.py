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
        trim_silence: Whether to strip the silence at the start and the end of the audio.
            Video output ignores it.
        format: "wav" for a WAV file or "video" for a lip-synced MP4.
        phrase_matching: Whether word runs that morshu.wav says verbatim play as recorded.
            A run needs at least two words and eight characters.
            The phoneme engine speaks everything else.
    """

    text: str = Field(..., min_length=1)
    speed: float = Field(default=1.0, ge=0.5, le=2.0)
    trim_silence: bool = False
    format: Literal["wav", "video"] = "wav"
    phrase_matching: bool = True


class PhonemesResponse(BaseModel):
    """The body of GET /tts/phonemes, the sorted phoneme tokens of the source recording."""

    phonemes: list[str]


class HealthResponse(BaseModel):
    """The body of GET /health."""

    status: str
    service: str
    version: str
