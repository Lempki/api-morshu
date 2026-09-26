"""The FastAPI application, its lifespan, and its routes."""

import asyncio
import io
import subprocess
import tempfile
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, suppress
from pathlib import Path
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.responses import StreamingResponse

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


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Loads the G2p model and the source WAV before the first request."""
    settings = get_settings()
    await asyncio.to_thread(init, settings.tts_source_wav)
    yield


app = FastAPI(title=SERVICE, version=VERSION, lifespan=lifespan)


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
        audio = audio.strip_silence()
    buf = io.BytesIO()
    audio.export(buf, format="wav")
    return buf.getvalue()


_SPRITES_DIR = Path(__file__).parent / "morshutalk" / "sprites"
_MAX_FRAME = 153


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

    # Build one entry per 100 ms frame using the same formula as the MorshuTalk
    # GUI: effective_morshu_t = morshu_start + (output_t - output_start),
    # frame = effective_morshu_t // 100. Silence segments use frame 0.
    frame_entries: list[tuple[int, int]] = []  # (frame_idx, duration_ms)
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
            # Repeat the last file entry without a duration to prevent ffconcat
            # from dropping the final frame.
            last_sprite = (_SPRITES_DIR / f"{frame_entries[-1][0]}.png").as_posix()
            f.write(f"file '{last_sprite}'\n")

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            out_path = f.name
            tmp_files.append(out_path)

        subprocess.run(
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
            ],
            capture_output=True,
            check=True,
            timeout=120,
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
async def phonemes(
    settings: Annotated[Settings, Depends(get_settings)],
) -> PhonemesResponse:
    unique = sorted({p for p in morshu_rec["phoneme"].tolist() if p})
    return PhonemesResponse(phonemes=unique, source_wav=settings.tts_source_wav)


@app.post("/tts/synthesize", dependencies=[Depends(require_auth)])
async def synthesize(
    body: SynthesizeRequest,
    settings: Annotated[Settings, Depends(get_settings)],
) -> StreamingResponse:
    if len(body.text) > settings.tts_max_text_length:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Text exceeds maximum length of {settings.tts_max_text_length} characters.",
        )

    if body.format == "video":
        mp4_bytes = await asyncio.to_thread(_synthesize_video_blocking, body.text)
        if not mp4_bytes:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Could not generate video — no phoneme matches found for the given text.",
            )
        return StreamingResponse(
            io.BytesIO(mp4_bytes),
            media_type="video/mp4",
            headers={"Content-Disposition": 'attachment; filename="morshu.mp4"'},
        )

    wav_bytes = await asyncio.to_thread(
        _synthesize_blocking, body.text, body.speed, body.trim_silence
    )
    if not wav_bytes:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Could not generate audio — no phoneme matches found for the given text.",
        )
    return StreamingResponse(
        io.BytesIO(wav_bytes),
        media_type="audio/wav",
        headers={"Content-Disposition": 'attachment; filename="morshu.wav"'},
    )
