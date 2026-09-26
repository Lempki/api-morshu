"""This service's settings, read from the environment or from .env."""

from functools import lru_cache

from .service import ServiceSettings

__all__ = ["Settings", "get_settings"]


class Settings(ServiceSettings):
    """The shared settings plus this service's own.

    Each field reads the environment variable of the same name in upper case.

    Attributes:
        tts_source_wav: The path to morshu.wav, the source recording that every clip is cut from.
        tts_max_text_length: The longest text, in characters, that one synthesis request accepts.
    """

    tts_source_wav: str = "/data/morshu.wav"
    tts_max_text_length: int = 500


@lru_cache
def get_settings() -> Settings:
    """Returns the settings, read once and then cached for the process."""
    return Settings()
