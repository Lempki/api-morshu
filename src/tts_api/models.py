from typing import Literal

from pydantic import BaseModel, Field


class SynthesizeRequest(BaseModel):
    text: str = Field(..., min_length=1)
    speed: float = Field(default=1.0, ge=0.5, le=2.0)
    trim_silence: bool = False
    format: Literal["wav", "video"] = "wav"


class PhonemesResponse(BaseModel):
    phonemes: list[str]


class HealthResponse(BaseModel):
    status: str
    service: str
    version: str
