"""This service's settings, read from the environment or from .env."""

from functools import lru_cache

from pydantic import field_validator

from .service import ServiceSettings

__all__ = ["Settings", "get_settings"]


class Settings(ServiceSettings):
    """The shared settings plus this service's own.

    Each field reads the environment variable of the same name in upper case.

    Attributes:
        tts_source_wav: The path to a replacement for morshu.wav, or None for the packaged copy.
            Every clip is cut from this recording.
        tts_max_text_length: The longest text, in characters, that one synthesis request accepts.
    """

    tts_source_wav: str | None = None
    tts_max_text_length: int = 500

    @field_validator("tts_source_wav")
    @classmethod
    def _blank_means_packaged(cls, path: str | None) -> str | None:
        # An empty TTS_SOURCE_WAV= line in .env reads as "", which means the same as unset.
        return path if path and path.strip() else None


@lru_cache
def get_settings() -> Settings:
    """Returns the settings, read once and then cached for the process."""
    return Settings()
