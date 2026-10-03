# discord-api-morshu

This is a REST API that synthesizes speech in Morshu's voice and returns the result as an audio or video file. It hosts the TTS engine adapted from [MorshuTalk](https://github.com/n0spaces/MorshuTalk) by [n0spaces](https://github.com/n0spaces), converting arbitrary text into audio by stitching phoneme segments from Morshu's original Zelda CD-i dialogue. Discord bots call this API to generate and play Morshu audio without bundling the TTS engine or its dependencies locally. This project is based on the [discord-api-template](https://github.com/Lempki/discord-api-template) repository, which provides the core architecture.

## Endpoints

| Method | Path | Description |
|---|---|---|
| `POST` | `/tts/synthesize` | Generate audio or video from text. Returns a WAV or MP4 file depending on the `format` field. |
| `GET` | `/tts/phonemes` | List the phoneme tokens available in the loaded source audio. |
| `GET` | `/health` | Returns the service name and the version set in `pyproject.toml`. Uptime monitors and the Docker health check call it. |

All endpoints except `/health` require a bearer token in the `Authorization` header.
A request without the header or with a wrong token gets `401 Unauthorized` with a `WWW-Authenticate: Bearer` header.
The token is compared in constant time.

### POST /tts/synthesize

```json
{
  "text": "lamp oil, rope, bombs?",
  "speed": 1.0,
  "trim_silence": false,
  "format": "wav"
}
```

`speed` accepts values between `0.5` and `2.0`. `trim_silence` removes leading and trailing silence from the output. `format` accepts `"wav"` (default) or `"video"`.

When `format` is `"wav"`, returns a binary WAV file with `Content-Type: audio/wav`. When `format` is `"video"`, generates an MP4 by compositing MorshuTalk sprite frames at 10 fps in sync with the synthesised audio and returns the file with `Content-Type: video/mp4`. The `speed` and `trim_silence` fields are ignored for video output. If the text exceeds the configured maximum length or no phoneme matches are found, a `422` response is returned.

The whole file arrives in one response with a `Content-Length` header.
If FFmpeg fails or runs for more than 120 seconds, the video request gets a `500` answer.
Its body is always `{"detail": "Could not encode the video."}`.
The service logs the end of FFmpeg's error output, but the response never includes it.

### GET /tts/phonemes

Returns the phoneme tokens that the source recording contains, as `{"phonemes": ["AA", "AE", ...]}`.
The response holds only that list.

## Prerequisites

* [Docker](https://docs.docker.com/get-docker/) and Docker Compose.

Running without Docker requires Python 3.12, [uv](https://docs.astral.sh/uv/), and FFmpeg available in the system PATH. On Windows, install uv with `winget install --id astral-sh.uv`.

## Assets

The TTS engine cuts every clip from one source recording, `morshu.wav`, which holds Morshu's CD-i dialogue.
It ships inside the package at `src/tts_api/morshutalk/morshu.wav`, next to the sprite frames.
The wheel and the Docker image include it, so the container needs no volume.
To use a different recording, set `TTS_SOURCE_WAV` to its path.

The 154 sprite frames used for video synthesis are adapted from [MorshuTalk](https://github.com/n0spaces/MorshuTalk) and are bundled with the application in `src/tts_api/morshutalk/sprites/`. They do not need to be provided separately.

## Setup

You can use the included setup script to prepare the project in a single step.

On Windows, run the following command:

```
setup.bat
```

On macOS or Linux, run the following commands:

```
chmod +x setup.sh
./setup.sh
```

The script runs `uv sync`, which creates the `.venv` virtual environment if needed and installs the package with its locked dependencies. It copies `.env.template` to `.env` on the first run. You must edit `.env` and set `DISCORD_API_SECRET` before starting the API.

If you prefer to perform the setup manually, follow these steps:

```bash
uv sync
cp .env.template .env
# Edit .env and set DISCORD_API_SECRET and other values as needed.
uv run uvicorn tts_api.main:app --port 8002
```

### Docker

Alternatively, you can run the API as a Docker container.

1. Copy `.env.template` to `.env` and set `DISCORD_API_SECRET`.
2. Build and start the container:

   ```
   docker compose up --build
   ```

After starting with Docker Compose, the API is available at `http://localhost:8002`. The container itself listens on port `8000`, and Docker Compose maps host port `8002` to it. The service keeps no state between requests, so the container needs no volume.

The image has a health check that calls `/health` every 30 seconds.
Startup loads the grapheme-to-phoneme model, so the check allows 60 seconds before it counts a failure.
`docker ps` shows the container as `healthy` once the API answers.

The image downloads the NLTK data that g2p-en and the engine need at build time.
It lives in `/usr/local/share/nltk_data`, which the `NLTK_DATA` variable points to.
A container start downloads nothing and needs no network access.
Without Docker, the first start downloads any missing NLTK data into the default NLTK data directory.

## Configuration

All configuration is read from environment variables or from a `.env` file in the project root.

| Variable | Required | Default | Description |
|---|---|---|---|
| `DISCORD_API_SECRET` | Yes | None | Shared bearer token of at least 16 characters. All Discord bots must send this value in the `Authorization` header. Generate one with `python -c "import secrets; print(secrets.token_urlsafe(32))"`. |
| `TTS_SOURCE_WAV` | No | None | Path to a replacement for the source WAV file. When it is unset or empty, the service uses the `morshu.wav` that ships in the package. |
| `LOG_LEVEL` | No | `INFO` | Log verbosity. Accepts `DEBUG`, `INFO`, `WARNING`, `ERROR`, or `CRITICAL`. |
| `TTS_MAX_TEXT_LENGTH` | No | `500` | Maximum number of characters accepted per synthesis request. |

The service refuses to start when `DISCORD_API_SECRET` is shorter than 16 characters or is a placeholder such as `changeme`.
The error names the variable but never repeats its value.

Logs are structured JSON.
Every line is one JSON object, including uvicorn's access log, so log collectors can parse it without guessing.

## Project structure

```
discord-api-morshu/
├── src/tts_api/
│   ├── main.py         # FastAPI application and route definitions.
│   ├── config.py       # This service's settings on top of the shared ones.
│   ├── service.py      # Shared settings, secret validation, and the version lookup.
│   ├── logging_config.py  # JSON log formatter for the app and uvicorn.
│   ├── auth.py         # Bearer token dependency.
│   ├── models.py       # Pydantic request and response models.
│   └── morshutalk/     # TTS engine adapted from MorshuTalk by n0spaces.
│       ├── morshu.py   # Core phoneme matching and audio stitching logic.
│       ├── g2p.py      # Grapheme-to-phoneme conversion wrapper.
│       ├── morshu.wav  # Source recording that every clip is cut from.
│       └── sprites/    # 154 sprite frames for video synthesis (0.png to 153.png).
├── tests/
├── Dockerfile
├── docker-compose.yml
├── pyproject.toml      # Project metadata and dependencies.
├── uv.lock             # Locked dependency versions.
├── ruff.toml           # Lint and format settings on top of the shared baseline.
├── setup.bat           # Windows setup script.
├── setup.sh            # macOS and Linux setup script.
└── .env.template       # Template for environment variables.
```

## Running tests

```bash
uv run pytest
```

Run every lint and format check with `uvx pre-commit run --all-files`, or install the hooks once with `uvx pre-commit install` so they run on each commit.
The coding, prose, and commit conventions are documented in [discord-dev-standards](https://github.com/Lempki/discord-dev-standards).

## Known risks

The grapheme-to-phoneme step depends on g2p-en, which is unmaintained.
Its last release, 2.1.0, was uploaded to PyPI on 2019-12-31.
It still expects the NLTK behavior of that time.
For example, it checks for NLTK's old `averaged_perceptron_tagger` package.
NLTK 3.9 and later tag with `averaged_perceptron_tagger_eng` instead, so the image carries both.
If a future NLTK release breaks g2p-en, the service fails at startup or on every synthesis request.
Pinning NLTK to the last working version would be the quick fix.
The lasting fix is to vendor g2p-en's Apache-2.0 code and model into `morshutalk`, as `g2p.py` already does in part.
Its pronunciations would then come from the maintained `cmudict` package.

## Credits

The TTS engine in `src/tts_api/morshutalk/` is adapted from [MorshuTalk](https://github.com/n0spaces/MorshuTalk) by [n0spaces](https://github.com/n0spaces), released under the [MIT License](https://github.com/n0spaces/MorshuTalk/blob/main/LICENSE.txt).
