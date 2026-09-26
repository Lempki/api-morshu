# discord-api-morshu

This is a REST API that synthesizes speech in Morshu's voice and returns the result as an audio or video file. It hosts the TTS engine adapted from [MorshuTalk](https://github.com/n0spaces/MorshuTalk) by [n0spaces](https://github.com/n0spaces), converting arbitrary text into audio by stitching phoneme segments from Morshu's original Zelda CD-i dialogue. Discord bots call this API to generate and play Morshu audio without bundling the TTS engine or its dependencies locally. This project is based on the [discord-api-template](https://github.com/Lempki/discord-api-template) repository, which provides the core architecture.

## Endpoints

| Method | Path | Description |
|---|---|---|
| `POST` | `/tts/synthesize` | Generate audio or video from text. Returns a WAV or MP4 file depending on the `format` field. |
| `GET` | `/tts/phonemes` | List the phoneme tokens available in the loaded source audio. |
| `GET` | `/health` | Returns the service name and version. Used for uptime monitoring. |

All endpoints except `/health` require a bearer token in the `Authorization` header.

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

## Prerequisites

* [Docker](https://docs.docker.com/get-docker/) and Docker Compose.
* The source audio file described in the [Assets](#assets) section below.

Running without Docker requires Python 3.12, [uv](https://docs.astral.sh/uv/), and FFmpeg available in the system PATH. On Windows, install uv with `winget install --id astral-sh.uv`.

## Assets

The TTS engine requires one source file. `morshu.wav` is committed to this repository in the `assets/` directory.

| File | Required | Description |
|---|---|---|
| `assets/morshu.wav` | Yes | The source audio file containing Morshu's CD-i dialogue. Committed to the repository. |

The `docker-compose.yml` mounts the `assets/` directory at `/data` inside the container. The default configuration expects `morshu.wav` at `/data/morshu.wav`.

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
   docker-compose up --build
   ```

After starting with docker-compose, the API is available at `http://localhost:8002`. The container itself listens on port `8000`; docker-compose maps `8002` on the host to `8000` inside the container.

## Configuration

All configuration is read from environment variables or from a `.env` file in the project root.

| Variable | Required | Default | Description |
|---|---|---|---|
| `DISCORD_API_SECRET` | Yes | — | Shared bearer token. All Discord bots must send this value in the `Authorization` header. |
| `TTS_SOURCE_WAV` | No | `/data/morshu.wav` | Absolute path to the source WAV file inside the container. |
| `LOG_LEVEL` | No | `INFO` | Log verbosity. Accepts standard Python logging levels. |
| `TTS_MAX_TEXT_LENGTH` | No | `500` | Maximum number of characters accepted per synthesis request. |

## Project structure

```
discord-api-morshu/
├── assets/             # Source files. morshu.wav is committed; generated output is not.
│   └── morshu.wav      # Source audio file. Required.
├── src/tts_api/
│   ├── main.py         # FastAPI application and route definitions.
│   ├── config.py       # Environment variable reader.
│   ├── auth.py         # Bearer token dependency.
│   ├── models.py       # Pydantic request and response models.
│   └── morshutalk/     # TTS engine adapted from MorshuTalk by n0spaces.
│       ├── morshu.py   # Core phoneme matching and audio stitching logic.
│       ├── g2p.py      # Grapheme-to-phoneme conversion wrapper.
│       └── sprites/    # 154 sprite frames for video synthesis (0.png–153.png).
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

## Credits

The TTS engine in `src/tts_api/morshutalk/` is adapted from [MorshuTalk](https://github.com/n0spaces/MorshuTalk) by [n0spaces](https://github.com/n0spaces), released under the [MIT License](https://github.com/n0spaces/MorshuTalk/blob/main/LICENSE.txt).
