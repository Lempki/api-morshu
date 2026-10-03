"""The FastAPI application, its lifespan, and its routes."""

import asyncio
import io
import logging
import subprocess
import tempfile
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, suppress
from pathlib import Path
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, Response, status
from pydub import AudioSegment
from pydub.silence import detect_leading_silence

from .auth import require_auth
from .config import Settings, get_settings
from .logging_config import configure_logging
from .models import HealthResponse, PhonemesResponse, SynthesizeRequest
from .morshutalk import Morshu, init
from .morshutalk.morshu import morshu_rec
from .service import service_version

# The service name is also the project name in pyproject.toml, which the version is read from.
SERVICE = "discord-api-morshu"
VERSION = service_version(SERVICE)

# Logging is set up on import, before uvicorn prints its startup lines, so every line is JSON.
configure_logging(get_settings().log_level)
logger = logging.getLogger(__name__)

_FFMPEG_TIMEOUT_SECONDS = 120
# ffmpeg writes a long banner and progress output, so a failure logs only the end of its stderr.
_FFMPEG_STDERR_LIMIT = 2000
# The 500 answer never includes ffmpeg's output, because that output can hold server paths.
VIDEO_ENCODING_FAILED = "Could not encode the video."


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Loads the G2p model and the source WAV before the first request.

    TTS_SOURCE_WAV overrides the source WAV. When it is unset, the packaged morshu.wav is used.
    """
    settings = get_settings()
    await asyncio.to_thread(init, settings.tts_source_wav)
    yield


app = FastAPI(title=SERVICE, version=VERSION, lifespan=lifespan)


class VideoEncodingError(Exception):
    """Raised when ffmpeg fails or runs past its timeout while it encodes a video."""


# Silence is measured against the clip's own loudness, because recordings differ in level.
# A fixed threshold such as pydub's default of -16 dBFS treats quiet speech as silence.
_SILENCE_BELOW_AVERAGE_DB = 20.0


def trim_edges(audio: AudioSegment) -> AudioSegment:
    """Removes the silence before the first sound and after the last one.

    Silence in the middle of the clip stays, so the pauses between words are kept.

    Args:
        audio: The synthesized clip.

    Returns:
        The clip without leading and trailing silence, or the clip unchanged when it is all silence.
    """
    if audio.dBFS == float("-inf"):
        return audio
    threshold = audio.dBFS - _SILENCE_BELOW_AVERAGE_DB
    start = detect_leading_silence(audio, silence_threshold=threshold)
    end = len(audio) - detect_leading_silence(
        audio.reverse(), silence_threshold=threshold
    )
    return audio[start:end] if start < end else audio


def _synthesize_blocking(text: str, speed: float, trim_silence: bool) -> bytes:
    m = Morshu()
    result = m.load_text(text)
    if result is False or len(result) == 0:
        return b""
    audio = result
    if speed != 1.0:
        audio = audio._spawn(
            audio.raw_data, overrides={"frame_rate": int(audio.frame_rate * speed)}
        )
        audio = audio.set_frame_rate(result.frame_rate)
    if trim_silence:
        audio = trim_edges(audio)
    buf = io.BytesIO()
    audio.export(buf, format="wav")
    return buf.getvalue()


_SPRITES_DIR = Path(__file__).parent / "morshutalk" / "sprites"
_MAX_FRAME = 153


def _stderr_tail(stderr: bytes | str | None) -> str:
    """Returns the end of a captured stderr stream as text.

    Args:
        stderr: The captured stream. Its type depends on how the process was run.

    Returns:
        At most the last _FFMPEG_STDERR_LIMIT characters of the stream.
    """
    if stderr is None:
        return ""
    text = (
        stderr.decode("utf-8", errors="replace")
        if isinstance(stderr, bytes)
        else stderr
    )
    return text[-_FFMPEG_STDERR_LIMIT:]


def _run_ffmpeg(args: list[str]) -> None:
    """Runs ffmpeg and turns a failure or a timeout into a VideoEncodingError.

    Args:
        args: The full command line, starting with "ffmpeg".

    Raises:
        VideoEncodingError: ffmpeg exited with an error or did not finish in time.
    """
    try:
        subprocess.run(
            args, capture_output=True, check=True, timeout=_FFMPEG_TIMEOUT_SECONDS
        )
    except subprocess.CalledProcessError as error:
        logger.error(
            "ffmpeg exited with code %d. The end of its stderr follows. %s",
            error.returncode,
            _stderr_tail(error.stderr),
        )
        raise VideoEncodingError from error
    except subprocess.TimeoutExpired as error:
        logger.error(
            "ffmpeg did not finish within %d seconds. The end of its stderr follows. %s",
            _FFMPEG_TIMEOUT_SECONDS,
            _stderr_tail(error.stderr),
        )
        raise VideoEncodingError from error


def _synthesize_video_blocking(text: str) -> bytes:
    m = Morshu()
    result = m.load_text(text)
    if result is False or len(result) == 0:
        return b""

    audio = result
    timings = m.audio_segment_timings
    total_ms = len(audio)
    output_times = timings["output"].tolist()
    morshu_times = timings["morshu"].tolist()

    # Each entry is one 100 ms video frame, picked with the same formula as the MorshuTalk GUI.
    # The source time is morshu_start + (output_t - output_start), and the frame is time // 100.
    # Silence segments use frame 0.
    # Each entry holds the frame index and the frame's duration in milliseconds.
    frame_entries: list[tuple[int, int]] = []
    seg_idx = 0
    t = 0
    while t < total_ms:
        while seg_idx + 1 < len(output_times) and output_times[seg_idx + 1] <= t:
            seg_idx += 1
        morshu_start = int(morshu_times[seg_idx])
        output_start = int(output_times[seg_idx])
        if morshu_start < 0:
            frame_idx = 0
        else:
            frame_idx = min((morshu_start + (t - output_start)) // 100, _MAX_FRAME)
        frame_entries.append((frame_idx, min(100, total_ms - t)))
        t += 100

    if not frame_entries:
        return b""

    wav_buf = io.BytesIO()
    audio.export(wav_buf, format="wav")
    wav_bytes = wav_buf.getvalue()

    tmp_files: list[str] = []
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(wav_bytes)
            wav_path = f.name
            tmp_files.append(wav_path)

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, encoding="utf-8"
        ) as f:
            concat_path = f.name
            tmp_files.append(concat_path)
            f.write("ffconcat version 1.0\n")
            for frame_idx, duration_ms in frame_entries:
                sprite = (_SPRITES_DIR / f"{frame_idx}.png").as_posix()
                f.write(f"file '{sprite}'\n")
                f.write(f"duration {duration_ms / 1000:.3f}\n")
            # ffconcat drops the final frame unless the last file is repeated without a duration.
            last_sprite = (_SPRITES_DIR / f"{frame_entries[-1][0]}.png").as_posix()
            f.write(f"file '{last_sprite}'\n")

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            out_path = f.name
            tmp_files.append(out_path)

        _run_ffmpeg(
            [
                "ffmpeg",
                "-y",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                concat_path,
                "-i",
                wav_path,
                "-vf",
                "scale=trunc(iw/2)*2:trunc(ih/2)*2,format=yuv420p",
                "-c:v",
                "libx264",
                "-preset",
                "fast",
                "-c:a",
                "aac",
                "-shortest",
                out_path,
            ]
        )

        with Path(out_path).open("rb") as f:
            return f.read()

    finally:
        for p in tmp_files:
            with suppress(OSError):
                Path(p).unlink()


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Reports that the service is up. It needs no token, so monitors and Docker can call it."""
    return HealthResponse(status="ok", service=SERVICE, version=VERSION)


@app.get(
    "/tts/phonemes",
    response_model=PhonemesResponse,
    dependencies=[Depends(require_auth)],
)
async def phonemes() -> PhonemesResponse:
    """Lists the phoneme tokens that the source recording contains."""
    unique = sorted({p for p in morshu_rec["phoneme"].tolist() if p})
    return PhonemesResponse(phonemes=unique)


@app.post("/tts/synthesize", dependencies=[Depends(require_auth)])
async def synthesize(
    body: SynthesizeRequest,
    settings: Annotated[Settings, Depends(get_settings)],
) -> Response:
    """Synthesizes the text as a WAV file or as a lip-synced MP4 video.

    Args:
        body: The text and the output options.
        settings: The service settings, which hold the text length limit.

    Returns:
        The whole file in one response, with the media type of the requested format.

    Raises:
        HTTPException: The status is 422 when the text is too long or matches no phonemes.
            It is 500 when ffmpeg fails to encode the video.
    """
    if len(body.text) > settings.tts_max_text_length:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=f"Text exceeds maximum length of {settings.tts_max_text_length} characters.",
        )

    if body.format == "video":
        try:
            mp4_bytes = await asyncio.to_thread(_synthesize_video_blocking, body.text)
        except VideoEncodingError as error:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=VIDEO_ENCODING_FAILED,
            ) from error
        if not mp4_bytes:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail="Could not generate video. No phoneme matches were found for the text.",
            )
        return Response(
            content=mp4_bytes,
            media_type="video/mp4",
            headers={"Content-Disposition": 'attachment; filename="morshu.mp4"'},
        )

    wav_bytes = await asyncio.to_thread(
        _synthesize_blocking, body.text, body.speed, body.trim_silence
    )
    if not wav_bytes:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="Could not generate audio. No phoneme matches were found for the text.",
        )
    return Response(
        content=wav_bytes,
        media_type="audio/wav",
        headers={"Content-Disposition": 'attachment; filename="morshu.wav"'},
    )
